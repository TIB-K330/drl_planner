from typing import Any, Dict, List, Optional, Tuple, Type, Union

import copy
import dgl
import gym
import numpy as np
import torch
import torch as th
from torch import nn
from torch.nn import functional as F

from .graph_sac_core import GraphActor, GraphSACPolicy, GraphExtractor
from .type_aliases import Schedule
from .torch_layers import get_actor_critic_arch
from .utils import get_device


def create_mlp(
    input_dim: int,
    output_dim: int,
    net_arch: List[int],
    activation_fn: Type[nn.Module] = nn.ReLU,
    target_drop_rate: float = 0.0,
    layer_norm: bool = False,
) -> List[nn.Module]:

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0])]
        if target_drop_rate > 0.0:
            modules.append(nn.Dropout(p=target_drop_rate))
        if layer_norm:
            modules.append(nn.LayerNorm(net_arch[0]))
        modules.append(activation_fn())
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        if target_drop_rate > 0.0:
            modules.append(nn.Dropout(p=target_drop_rate))
        if layer_norm:
            modules.append(nn.LayerNorm(net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))

    return modules


class GraphDroQCritic(nn.Module):
    """
    Graph Critic network(s) for DroQ with obs as Graph.

    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether the features extractor is shared or not
        between the actor and the critic (this saves computation time)
    """

    optimizer: th.optim.Optimizer
    ext_optimizer: th.optim.Optimizer
    features_extractor: GraphExtractor

    def __init__(
        self,
        action_space: gym.spaces.Box,
        net_arch: List[int],
        features_extractor: GraphExtractor,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        share_features_extractor: bool = True,
        n_critics: int = 2,
        target_drop_rate: float = 0.01,
        layer_norm: bool = True,
    ):
        super().__init__()
        action_dim = int(np.prod(action_space.shape))
        self.features_extractor = features_extractor
        self.share_features_extractor = share_features_extractor
        self.n_critics = n_critics
        self.q_networks = []

        for idx in range(n_critics):
            q_net = create_mlp(features_dim + action_dim, 1, net_arch, activation_fn, target_drop_rate, layer_norm)
            q_net = nn.Sequential(*q_net)
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)

    def forward(self, obs: dgl.DGLHeteroGraph, actions: th.Tensor) -> Tuple[th.Tensor, ...]:
        # Learn the features extractor using the policy loss only
        # when the features_extractor is shared with the actor
        with th.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(obs, self.features_extractor)
        qvalue_input = th.cat([features, actions], dim=1)
        return tuple(q_net(qvalue_input) for q_net in self.q_networks)

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

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.

        This affects certain modules, such as batch normalisation and dropout.

        :param mode: if true, set to training mode, else set to evaluation mode
        """
        self.train(mode)


class GraphDroQPolicy(GraphSACPolicy):

    actor: GraphActor
    critic: GraphDroQCritic
    critic_target: GraphDroQCritic

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
        droq_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        nn.Module.__init__(self)  # fully override GraphSACPolicy's constructor

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

        if droq_kwargs is None:
            droq_kwargs = {}

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
                **droq_kwargs,
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
        else:
            # Create a separate features extractor for the critic
            # this requires more memory and computation
            self.critic = self.make_critic(features_extractor=None)

        critic_parameters = [
            param for name, param in self.critic.named_parameters() if "features_extractor" not in name
        ]

        # Critic target should not share the features extractor with critic
        self.critic_target = self.make_critic(features_extractor=None)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.critic.optimizer = self.optimizer_class(
            critic_parameters,
            lr=lr_schedule(1),  # type: ignore[call-arg]
            **self.optimizer_kwargs,
        )

        # Update the feature extractor of the critic separatedly
        self.critic.ext_optimizer = self.optimizer_class(
            self.critic.features_extractor.parameters(),
            lr = lr_schedule(1),  # type: ignore[call-arg]
            ** self.optimizer_kwargs,
        )

        # Target networks should always be in eval mode
        self.critic_target.set_training_mode(False)

    def make_critic(self, features_extractor: Optional[GraphExtractor] = None) -> GraphDroQCritic:
        if "target_drop_rate" in self.critic_kwargs:
            critic_kwargs = self.critic_kwargs.copy()
            if features_extractor is None:
                features_extractor_kwargs = self.features_extractor_kwargs.copy()
                features_extractor_kwargs.update(dict(target_drop_rate=critic_kwargs["target_drop_rate"]))
                features_extractor = self.features_extractor_class(**features_extractor_kwargs)

            critic_kwargs.update(dict(
                features_extractor=features_extractor,
                features_dim=features_extractor.features_dim + 5)
            )
        else:
            critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)

        return GraphDroQCritic(**critic_kwargs).to(self.device)
