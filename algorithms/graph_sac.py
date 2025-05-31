import io
import pathlib
import sys
import time
from collections import deque
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, TypeVar, Union

import dgl
import gym
import numpy as np
import torch as th
from torch.nn import functional as F

from modules.base_class import BaseAlgorithm
from modules.callbacks import BaseCallback
# from modules.envs.actions import ActionDiff
# from modules.envs.states import JointState
# from modules.envs.infos import *

from crowd_sim.envs.utils.action import ActionDiff
from crowd_sim.envs.utils.state import JointState
from crowd_sim.envs.utils.info import *

from modules.graph_sac_core import GraphActor, GraphContinuousCritic, GraphSACPolicy
from modules.policies import GraphReplayBuffer, ActionNoise
from modules.save_util import load_from_pkl, save_to_pkl
from modules.type_aliases import MaybeCallback, RolloutReturn, Schedule, TrainFreq, TrainFrequencyUnit
from modules.utils import (
    configure_logger,
    get_schedule_fn,
    get_parameters_by_name,
    joint_state_as_graph,
    polyak_update,
    safe_mean,
    should_collect_more_steps,
)

SelfGraphSAC = TypeVar("SelfGraphSAC", bound="GraphSAC")


class GraphSAC(BaseAlgorithm):
    """
    Soft Actor-Critic (SAC)
    Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor,
    This implementation borrows code from original implementation (https://github.com/haarnoja/sac)
    from OpenAI Spinning Up (https://github.com/openai/spinningup), from the softlearning repo
    (https://github.com/rail-berkeley/softlearning/)
    and from Stable Baselines (https://github.com/hill-a/stable-baselines)
    Paper: https://arxiv.org/abs/1801.01290
    Introduction to SAC: https://spinningup.openai.com/en/latest/algorithms/sac.html

    Note: we use double q target and not value target as discussed
    in https://github.com/hill-a/stable-baselines/issues/270

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: learning rate for adam optimizer,
        the same learning rate will be used for all networks (Q-Values, Actor and Value function)
        it can be a function of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param action_noise: the action noise type (None by default), this can help
        for hard exploration problem. Cf common.noise for the different action noise type.
    :param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
        If ``None``, it will be automatically selected.
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param ent_coef: Entropy regularization coefficient. (Equivalent to
        inverse of reward scale in the original SAC paper.)  Controlling exploration/exploitation trade-off.
        Set it to 'auto' to learn it automatically (and 'auto_0.1' for using 0.1 as initial value)
    :param target_update_interval: update the target network every ``target_network_update_freq``
        gradient steps.
    :param target_entropy: target entropy when learning ``ent_coef`` (``ent_coef = 'auto'``)
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param use_sde_at_warmup: Whether to use gSDE instead of uniform sampling
        during the warm up phase (before learning starts)
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: ClassVar[Dict[str, Any]] = {"GraphPolicy": GraphSACPolicy}

    policy: GraphSACPolicy
    actor: GraphActor
    critic: GraphContinuousCritic
    critic_target: GraphContinuousCritic

    def __init__(
        self,
        policy: Union[str, Type[GraphSACPolicy]],
        env: gym.Env,
        learning_rate: Union[float, Schedule] = 3e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 1,
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[Type[GraphReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        ent_coef: Union[str, float] = "auto",
        target_update_interval: int = 1,
        target_entropy: Union[str, float] = "auto",
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy,
            env,
            learning_rate,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
        )

        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.tau = tau
        self.gamma = gamma
        self.gradient_steps = gradient_steps
        self.action_noise = action_noise
        self.replay_buffer: Optional[GraphReplayBuffer] = None
        self.replay_buffer_class = replay_buffer_class
        self.replay_buffer_kwargs = replay_buffer_kwargs or {}
        self._episode_storage = None

        # Save train freq parameter, will be converted later to TrainFreq object
        self.train_freq = train_freq
        self.rewards = []

        # SAC hyper-parameter
        self.target_entropy = target_entropy
        self.log_ent_coef = None  # type: Optional[th.Tensor]
        # Entropy coefficient / Entropy temperature
        # Inverse of the reward scale
        self.ent_coef = ent_coef
        self.target_update_interval = target_update_interval
        self.ent_coef_optimizer: Optional[th.optim.Adam] = None

        if _init_setup_model:
            self._setup_model()

    def _convert_train_freq(self) -> None:
        """
        Convert `train_freq` parameter (int or tuple)
        to a TrainFreq object.
        """
        if not isinstance(self.train_freq, TrainFreq):
            train_freq = self.train_freq

            # The value of the train frequency will be checked later
            if not isinstance(train_freq, tuple):
                train_freq = (train_freq, "step")

            try:
                train_freq = (train_freq[0], TrainFrequencyUnit(train_freq[1]))
            except ValueError as e:
                raise ValueError(
                    f"The unit of the `train_freq` must be either 'step' or 'episode' not '{train_freq[1]}'!"
                ) from e

            if not isinstance(train_freq[0], int):
                raise ValueError(f"The frequency of `train_freq` must be an integer and not {train_freq[0]}")

            self.train_freq = TrainFreq(*train_freq)

    def _setup_model(self) -> None:
        self.lr_schedule = get_schedule_fn(self.learning_rate)
        self.set_random_seed(self.seed)

        if self.replay_buffer_class is None:
            self.replay_buffer_class = GraphReplayBuffer

        if self.replay_buffer is None:
            replay_buffer_kwargs = self.replay_buffer_kwargs.copy()
            self.replay_buffer = self.replay_buffer_class(
                self.buffer_size,
                self.action_space,
                device=self.device,
                **replay_buffer_kwargs,  # pytype:disable=wrong-keyword-args
            )

        self.policy = self.policy_class(  # pytype:disable=not-instantiable
            self.action_space,
            self.lr_schedule,
            **self.policy_kwargs,  # pytype:disable=not-instantiable
        )
        self.policy = self.policy.to(self.device)

        self._convert_train_freq()  # Convert train freq parameter to TrainFreq object
        self._create_aliases()
        # Running mean and running var
        self.batch_norm_stats = get_parameters_by_name(self.critic, ["running_"])
        self.batch_norm_stats_target = get_parameters_by_name(self.critic_target, ["running_"])
        # Target entropy is used when learning the entropy coefficient
        if self.target_entropy == "auto":
            # automatically set target entropy if needed
            self.target_entropy = float(-np.prod(self.env.action_space.shape).astype(np.float32))  # type: ignore
        else:
            # Force conversion
            # this will also throw an error for unexpected string
            self.target_entropy = float(self.target_entropy)

        # The entropy coefficient or entropy can be learned automatically
        # see Automating Entropy Adjustment for Maximum Entropy RL section
        # of https://arxiv.org/abs/1812.05905
        if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
            # Default initial value of ent_coef when learned
            init_value = 1.0
            if "_" in self.ent_coef:
                init_value = float(self.ent_coef.split("_")[1])
                assert init_value > 0.0, "The initial value of ent_coef must be greater than 0"

            # Note: we optimize the log of the entropy coeff which is slightly different from the paper
            # as discussed in https://github.com/rail-berkeley/softlearning/issues/37
            self.log_ent_coef = th.log(th.ones(1, device=self.device) * init_value).requires_grad_(True)
            self.ent_coef_optimizer = th.optim.Adam([self.log_ent_coef], lr=self.lr_schedule(1))
        else:
            # Force conversion to float
            # this will throw an error if a malformed string (different from 'auto')
            # is passed
            self.ent_coef_tensor = th.tensor(float(self.ent_coef), device=self.device)

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

    def collect_rollouts(
        self,
        env: gym.Env,
        callback: BaseCallback,
        train_freq: TrainFreq,
        replay_buffer: GraphReplayBuffer,
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        log_interval: Optional[int] = None,
    ) -> RolloutReturn:
        """
        Collect experiences and store them into a ``ReplayBuffer``.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param train_freq: How much experience to collect
            by doing rollouts of current policy.
            Either ``TrainFreq(<n>, TrainFrequencyUnit.STEP)``
            or ``TrainFreq(<n>, TrainFrequencyUnit.EPISODE)``
            with ``<n>`` being an integer greater than 0.
        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param log_interval: Log data every ``log_interval`` episodes
        :param replay_buffer:
        :return:
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        num_collected_steps, num_collected_episodes = 0, 0
        rewards = self.rewards

        assert isinstance(env, gym.Env), "You must pass a VecEnv"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        callback.on_rollout_start()
        continue_training = True
        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            # Select action randomly or according to policy
            action, buffer_action = self._sample_action(learning_starts, action_noise)

            # Rescale and perform action
            new_ob, reward, done, info = env.step(ActionDiff(action[0][0], action[0][1]))
            new_robot_state = env.robot.get_full_state()
            new_state = joint_state_as_graph(JointState(new_robot_state, new_ob), device=self.device)

            rewards.append(float(reward))
            if done:
                returns = []
                for step in range(len(rewards)):
                    step_return = sum([pow(self.gamma, t) * reward for t, reward in enumerate(rewards[step:])])
                    returns.append(step_return)

                ep_info = {"r": round(safe_mean(returns), 6), "l": len(self.rewards)}
                self.ep_info_buffer.append(ep_info)
                self.ep_success_buffer.append(True if isinstance(info, ReachGoal) else False)

            self.num_timesteps += 1
            num_collected_steps += 1

            # Give access to local variables
            callback.update_locals(locals())
            # Only stop training if return value is False, not when it is None.
            if callback.on_step() is False:
                return RolloutReturn(num_collected_steps, num_collected_episodes, continue_training=False)

            # Store data in replay buffer (normalized action and unnormalized observation)
            replay_buffer.add(
                self._last_state,
                new_state,
                buffer_action,
                reward,
                done,
                True if isinstance(info, Timeout) else False,
            )

            if done:
                # Update stats
                num_collected_episodes += 1
                self._episode_num += 1

                if action_noise is not None:
                    kwargs = {}
                    action_noise.reset(**kwargs)

                if log_interval is not None and self._episode_num % log_interval == 0:
                    self._dump_logs()

                self.rewards = []
                new_ob = env.reset(phase="train")
                new_robot_state = env.robot.get_full_state()
                new_state = joint_state_as_graph(JointState(new_robot_state, new_ob), device=self.device)

            self._last_state = new_state

        callback.on_rollout_end()

        return RolloutReturn(num_collected_steps, num_collected_episodes, continue_training)

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size)  # type: ignore[union-attr]

            # We need to sample because `log_std` may have changed between two gradient steps
            # if self.use_sde:
            #     pass  # TODO

            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                # Important: detach the variable from the graph,
                # so we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            # Optimize entropy coefficient, also called
            # entropy temperature or alpha in the paper
            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                # Select action according to policy
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                # Compute the next Q values: min over all critics targets
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                # add entropy term
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                # td error + entropy term
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates for each critic network
            # using action from the replay buffer
            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            assert isinstance(critic_loss, th.Tensor)  # for type checker
            critic_losses.append(critic_loss.item())  # type: ignore[union-attr]

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Compute actor loss
            # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
            # Min over all critic networks
            q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_losses.append(actor_loss.item())

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                # Copy running stats, see GH issue #996
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps

    def _sample_action(
        self,
        learning_starts: int,
        action_noise: Optional[ActionNoise] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample an action according to the exploration policy.
        This is either done by sampling the probability distribution of the policy,
        or sampling a random action (from a uniform distribution over the action space)
        or by adding noise to the deterministic output.

        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param n_envs:
        :return: action to take in the environment
            and scaled action that will be stored in the replay buffer.
            The two differs when the action space is not normalized (bounds are not [-1, 1]).
        """
        # Select action randomly or according to policy
        if self.num_timesteps < learning_starts:
            # Warmup phase
            unscaled_action = np.array([self.action_space.sample()])
        else:
            # Note: when using continuous actions,
            # we assume that the policy uses tanh to scale the action
            unscaled_action = self.predict(self._last_state, deterministic=False)

        # Rescale the action from [low, high] to [-1, 1]
        scaled_action = self.policy.scale_action(unscaled_action)

        # Add noise to the action (improve exploration)
        if action_noise is not None:
            scaled_action = np.clip(scaled_action + action_noise(), -1, 1)

        # We store the scaled action in the buffer
        buffer_action = scaled_action
        action = self.policy.unscale_action(scaled_action)

        return action, buffer_action

    def _dump_logs(self) -> None:
        """
        Write log.
        """
        time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
        self.logger.record("time/episodes", self._episode_num, exclude="tensorboard")
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            self.logger.record("rollout/ep_ret_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
            self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
        self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
        self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")

        if len(self.ep_success_buffer) > 0:
            self.logger.record("rollout/success_rate", safe_mean(self.ep_success_buffer))
        # Pass the number of timesteps for tensorboard
        self.logger.dump(step=self.num_timesteps)

    def _setup_learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        reset_num_timesteps: bool = True,
        tb_log_name: str = "run",
    ) -> Tuple[int, BaseCallback]:
        """
        Initialize different variables needed for training.

        :param total_timesteps: The total number of samples (env steps) to train on
        :param callback: Callback(s) called at every step with state of the algorithm.
        :param reset_num_timesteps: Whether to reset or not the ``num_timesteps`` attribute
        :param tb_log_name: the name of the run for tensorboard log
        :return: Total timesteps and callback(s)
        """
        self.start_time = time.time_ns()

        if self.ep_info_buffer is None or reset_num_timesteps:
            # Initialize buffers if they don't exist, or reinitialize if resetting counters
            self.ep_info_buffer = deque(maxlen=self._stats_window_size)
            self.ep_success_buffer = deque(maxlen=self._stats_window_size)

        if self.action_noise is not None:
            self.action_noise.reset()

        if reset_num_timesteps:
            self.num_timesteps = 0
            self._episode_num = 0
        else:
            total_timesteps += self.num_timesteps  # Make sure training timesteps are ahead of the internal counter

        # Avoid resetting the environment when calling ``.learn()`` consecutive times
        if reset_num_timesteps or self._last_state is None:
            assert self.env is not None
            # pytype: disable=annotation-type-mismatch
            ob = self.env.reset(phase="train")
            robot_state = self.env.robot.get_full_state()
            self._last_state = joint_state_as_graph(JointState(robot_state, ob), device=self.device)

        # Configure logger's outputs if no logger was passed
        if not self._custom_logger:
            self._logger = configure_logger(self.verbose, self.tensorboard_log, tb_log_name, reset_num_timesteps)

        # Create eval callback if needed
        callback = self._init_callback(callback)

        return total_timesteps, callback

    def learn(
        self: SelfGraphSAC,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "SAC",
        reset_num_timesteps: bool = True,
    ) -> SelfGraphSAC:
        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
        )

        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:
            rollout = self.collect_rollouts(
                self.env,
                train_freq=self.train_freq,
                action_noise=self.action_noise,
                callback=callback,
                learning_starts=self.learning_starts,
                replay_buffer=self.replay_buffer,
                log_interval=log_interval,
            )

            if rollout.continue_training is False:
                break

            if self.num_timesteps > 0 and self.num_timesteps > self.learning_starts:
                # If no `gradient_steps` is specified,
                # do as many gradients steps as steps performed during the rollout
                gradient_steps = self.gradient_steps if self.gradient_steps >= 0 else rollout.episode_timesteps
                # Special case when the user passes `gradient_steps=0`
                if gradient_steps > 0:
                    self.train(batch_size=self.batch_size, gradient_steps=gradient_steps)

        callback.on_training_end()

        return self

    def predict(self, observation: dgl.DGLHeteroGraph, deterministic: bool = False) -> np.ndarray:
        if isinstance(observation, JointState):
            observation = joint_state_as_graph(observation, device=self.device)
        return self.policy.predict(observation, deterministic)

    def _excluded_save_params(self) -> List[str]:
        return [
            "policy",
            "device",
            "env",
            "replay_buffer",
            "_logger",
            "_custom_logger",
            "actor",
            "critic",
            "critic_target"
        ]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer"]
        if self.ent_coef_optimizer is not None:
            saved_pytorch_variables = ["log_ent_coef"]
            state_dicts.append("ent_coef_optimizer")
        else:
            saved_pytorch_variables = ["ent_coef_tensor"]
        return state_dicts, saved_pytorch_variables

    def save_replay_buffer(self, path: Union[str, pathlib.Path, io.BufferedIOBase]) -> None:
        """
        Save the replay buffer as a pickle file.

        :param path: Path to the file where the replay buffer should be saved.
            if path is a str or pathlib.Path, the path is automatically created if necessary.
        """
        assert self.replay_buffer is not None, "The replay buffer is not defined"
        save_to_pkl(path, self.replay_buffer, self.verbose)

    def load_replay_buffer(self, path: Union[str, pathlib.Path, io.BufferedIOBase]) -> None:
        """
        Load a replay buffer from a pickle file.

        :param path: Path to the pickled replay buffer.
        """
        self.replay_buffer = load_from_pkl(path, self.verbose)
        assert isinstance(self.replay_buffer, GraphReplayBuffer), "The replay buffer must be a GraphReplayBuffer class"

        # Update saved replay buffer device to match current setting, see GH#1561
        self.replay_buffer.device = self.device
