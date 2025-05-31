import copy
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':16:8'       

import dgl
import gym
import numpy as np
import torch as th
import torch.nn as nn

from .distributions import SquashedDiagGaussianDistribution
from .policies import GraphContinuousCritic
from .torch_layers import (
    create_mlp,
    GraphExtractor,
    get_actor_critic_arch,
)
from .utils import Schedule, get_device

# CAP the standard deviation of the actor
LOG_STD_MAX = 2
LOG_STD_MIN = -20


class GraphActor(nn.Module):
    """
    Actor network (policy) for SAC.

    :param observation_space: Obervation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE.
    :param use_expln: Use ``expln()`` function instead of ``exp()`` when using gSDE to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param clip_mean: Clip the mean output when using gSDE to avoid numerical instability.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    """

    optimizer: th.optim.Optimizer
    features_extractor: GraphExtractor

    def __init__(
        self,
        action_space: gym.spaces.Box,
        net_arch: List[int],
        features_extractor: GraphExtractor,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        full_std: bool = True,
        use_expln: bool = False,
        clip_mean: float = 2.0,
    ) -> None:
        super().__init__()
        self.features_extractor = features_extractor
        self._squash_output = True

        # Save arguments to re-create object at loading
        self.use_sde = use_sde
        self.sde_features_extractor = None
        self.net_arch = net_arch
        self.features_dim = features_dim
        self.activation_fn = activation_fn
        self.log_std_init = log_std_init
        self.use_expln = use_expln
        self.full_std = full_std
        self.clip_mean = clip_mean

        action_dim = int(np.prod(action_space.shape))
        latent_pi_net = create_mlp(features_dim, -1, net_arch, activation_fn)
        self.latent_pi = nn.Sequential(*latent_pi_net)
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else features_dim

        if use_sde:
            pass  # TODO:
        else:
            self.action_dist = SquashedDiagGaussianDistribution(action_dim)  # type: ignore[assignment]
            self.mu = nn.Linear(last_layer_dim, action_dim)
            self.log_std = nn.Linear(last_layer_dim, action_dim)

    def get_action_dist_params(self, obs: dgl.DGLHeteroGraph) -> Tuple[th.Tensor, th.Tensor, Dict[str, th.Tensor]]:
        """
        Get the parameters for the action distribution.

        :param obs:
        :return:
            Mean, standard deviation and optional keyword arguments.
        """
        features = self.extract_features(obs, self.features_extractor)
        latent_pi = self.latent_pi(features)
        mean_actions = self.mu(latent_pi)

        if self.use_sde:
            return mean_actions, self.log_std, dict(latent_sde=latent_pi)
        # Unstructured exploration (Original implementation)
        log_std = self.log_std(latent_pi)  # type: ignore[operator]
        # Original Implementation to cap the standard deviation
        log_std = th.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean_actions, log_std, {}

    def forward(self, obs: dgl.DGLHeteroGraph,  deterministic: bool = False) -> th.Tensor:
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs)
        # Note: the action is squashed
        return self.action_dist.actions_from_params(mean_actions, log_std, deterministic=deterministic, **kwargs)

    def action_log_prob(self, obs: dgl.DGLHeteroGraph) -> Tuple[th.Tensor, th.Tensor]:
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs)
        # return action and associated log prob
        return self.action_dist.log_prob_from_params(mean_actions, log_std, **kwargs)

    def extract_features(self, obs: dgl.DGLHeteroGraph, features_extractor: GraphExtractor) -> th.Tensor:
        """ Embed state into a latent space. Take the first row of the feature matrix as state representation.
        """
        cur_state = copy.deepcopy(obs)
        # exploration or test phase
        if cur_state.batch_size == 1:
            robot_state = cur_state.ndata['h'][0, 4:9].clone()
            return th.cat((robot_state, features_extractor(cur_state)[0, :]), dim=0)
        # batch training phase
        else:
            batch_features = cur_state.ndata['h']
            num_nodes = cur_state._batch_num_nodes['_N'].cpu().numpy()
            robot_ids = []
            robot_id = 0
            for i in range(num_nodes.shape[0]):
                robot_ids.append(robot_id)
                robot_id += num_nodes[i]

            robot_ids = th.LongTensor(robot_ids).to(self.device)
            robot_features = th.index_select(batch_features, 0, robot_ids)
            robot_states = robot_features[:, 4:9].clone()
            features = th.index_select(features_extractor(cur_state), 0, robot_ids)
            return th.cat((robot_states, features), dim=1)  # dim 0 is the batch dim, 1 is the feature dim

    @property
    def device(self) -> th.device:
        """Infer which device this policy lives on by inspecting its parameters.
        If it has no parameters, the 'cpu' device is used as a fallback.

        :return:"""
        for param in self.parameters():
            return param.device
        return get_device("cpu")

    @property
    def squash_output(self) -> bool:
        """(bool) Getter for squash_output."""
        return self._squash_output

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.

        This affects certain modules, such as batch normalisation and dropout.

        :param mode: if true, set to training mode, else set to evaluation mode
        """
        self.train(mode)


class GraphSACPolicy(nn.Module):
    """
    Policy class (with both actor and critic) for SAC.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param use_expln: Use ``expln()`` function instead of ``exp()`` when using gSDE to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param clip_mean: Clip the mean output when using gSDE to avoid numerical instability.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether to share or not the features extractor
        between the actor and the critic (this saves computation time)
    """

    actor: GraphActor
    critic: GraphContinuousCritic
    critic_target: GraphContinuousCritic

    def __init__(
        self,
        action_space: gym.spaces.Box,
        lr_schedule: Schedule,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[GraphExtractor] = GraphExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
    ):
        super().__init__()

        if optimizer_kwargs is None:
            optimizer_kwargs = {}

        if features_extractor_kwargs is None:
            features_extractor_kwargs = {}

        self.action_space = action_space

        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs

        self.features_extractor_class = features_extractor_class
        self.features_extractor_kwargs = features_extractor_kwargs

        self._squash_output = True

        if net_arch is None:
            net_arch = [256, 256]

        actor_arch, critic_arch = get_actor_critic_arch(net_arch)

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.net_args = {
            "action_space": self.action_space,
            "net_arch": actor_arch,
            "activation_fn": self.activation_fn,
        }
        self.actor_kwargs = self.net_args.copy()
        self.critic_kwargs = self.net_args.copy()
        self.critic_kwargs.update(
            {
                "n_critics": n_critics,
                "net_arch": critic_arch,
                "share_features_extractor": share_features_extractor,
            }
        )

        self.share_features_extractor = share_features_extractor
        self._build(lr_schedule)

    def _build(self, lr_schedule: Schedule) -> None:
        self.actor = self.make_actor()
        self.actor.optimizer = self.optimizer_class(
            self.actor.parameters(),
            lr=lr_schedule(1),  # type: ignore[call-arg]
            **self.optimizer_kwargs,
        )

        if self.share_features_extractor:
            self.critic = self.make_critic(features_extractor=self.actor.features_extractor)
            # Do not optimize the shared features extractor with the critic loss
            # otherwise, there are gradient computation issues
            critic_parameters = [param for name, param in self.critic.named_parameters() if "features_extractor" not in name]
        else:
            # Create a separate features extractor for the critic
            # this requires more memory and computation
            self.critic = self.make_critic(features_extractor=None)
            critic_parameters = list(self.critic.parameters())

        # Critic target should not share the features extractor with critic
        self.critic_target = self.make_critic(features_extractor=None)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.critic.optimizer = self.optimizer_class(
            critic_parameters,
            lr=lr_schedule(1),  # type: ignore[call-arg]
            **self.optimizer_kwargs,
        )

        # Target networks should always be in eval mode
        self.critic_target.set_training_mode(False)

    def _update_features_extractor(
        self,
        net_kwargs: Dict[str, Any],
        features_extractor: Optional[GraphExtractor] = None,
    ) -> Dict[str, Any]:
        """
        Update the network keyword arguments and create a new features extractor object if needed.
        If a ``features_extractor`` object is passed, then it will be shared.

        :param net_kwargs: the base network keyword arguments, without the ones
            related to features extractor
        :param features_extractor: a features extractor object.
            If None, a new object will be created.
        :return: The updated keyword arguments
        """
        net_kwargs = net_kwargs.copy()
        if features_extractor is None:
            # The features extractor is not shared, create a new one
            features_extractor = self.make_features_extractor()
        net_kwargs.update(dict(
            features_extractor=features_extractor,
            features_dim=features_extractor.features_dim + 5),  # concat with robot state
        )
        return net_kwargs

    def make_features_extractor(self) -> GraphExtractor:
        """Helper method to create a features extractor."""
        return self.features_extractor_class(**self.features_extractor_kwargs)

    def make_actor(self, features_extractor: Optional[GraphExtractor] = None) -> GraphActor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return GraphActor(**actor_kwargs).to(self.device)

    def make_critic(self, features_extractor: Optional[GraphExtractor] = None) -> GraphContinuousCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return GraphContinuousCritic(**critic_kwargs).to(self.device)

    @property
    def device(self) -> th.device:
        """Infer which device this policy lives on by inspecting its parameters.
        If it has no parameters, the 'cpu' device is used as a fallback.

        :return:"""
        for param in self.parameters():
            return param.device
        return get_device("cpu")

    @property
    def squash_output(self) -> bool:
        """(bool) Getter for squash_output."""
        return self._squash_output

    def forward(self, obs: dgl.DGLHeteroGraph, deterministic: bool = False) -> th.Tensor:
        return self.actor(obs, deterministic)

    def predict(self, observation: dgl.DGLHeteroGraph, deterministic: bool = False) -> np.ndarray:
        # Switch to eval mode (this affects batch norm / dropout)
        self.set_training_mode(False)
        with th.no_grad():
            actions = self.actor(observation, deterministic=deterministic)
        # Convert to numpy, and reshape to the original action shape
        actions = actions.cpu().numpy().reshape((-1, *self.action_space.shape))

        if isinstance(self.action_space, gym.spaces.Box):
            if self.squash_output:
                # Rescale to proper domain when using squashing
                actions = self.unscale_action(actions)
            else:
                # Actions could be on arbitrary scale, so clip the actions to avoid
                # out of bound error (e.g. if sampling from a Gaussian distribution)
                actions = np.clip(actions, self.action_space.low, self.action_space.high)

        return actions

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.

        This affects certain modules, such as batch normalisation and dropout.

        :param mode: if true, set to training mode, else set to evaluation mode
        """
        self.actor.set_training_mode(mode)
        self.critic.set_training_mode(mode)
        self.training = mode

    def scale_action(self, action: np.ndarray) -> np.ndarray:
        """
        Rescale the action from [low, high] to [-1, 1]
        (no need for symmetric action space)

        :param action: Action to scale
        :return: Scaled action
        """
        low, high = self.action_space.low, self.action_space.high
        return 2.0 * ((action - low) / (high - low)) - 1.0

    def unscale_action(self, scaled_action: np.ndarray) -> np.ndarray:
        """
        Rescale the action from [-1, 1] to [low, high]
        (no need for symmetric action space)

        :param scaled_action: Action to un-scale
        """
        low, high = self.action_space.low, self.action_space.high
        return low + (0.5 * (scaled_action + 1.0) * (high - low))
