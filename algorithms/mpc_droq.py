from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, Union

import io
import pathlib

import dgl
import gym
import numpy as np
import torch as th
from torch.nn import functional as F

from modules.graph_droq_core import GraphDroQPolicy, GraphDroQCritic
from modules.policies import ActionNoise, GraphReplayBuffer, GraphReplayBufferSamples
from modules.save_util import load_from_pkl
from modules.type_aliases import Schedule
from modules.utils import polyak_update
from algorithms.graph_sac import GraphSAC


class MpcDroQ(GraphSAC):

    policy_aliases: ClassVar[Dict[str, Any]] = {"GraphPolicy": GraphDroQPolicy}

    policy: GraphDroQPolicy
    critic: GraphDroQCritic
    critic_target: GraphDroQCritic

    def __init__(
        self,
        policy: Union[str, Type[GraphDroQPolicy]],
        env: gym.Env,
        learning_rate: Union[float, Schedule] = 3e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 1,
        gradient_steps: int = 1,
        utd_ratio: int = 20,
        sample_ratio: float = 0.5,
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
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise,
            replay_buffer_class,
            replay_buffer_kwargs,
            ent_coef,
            target_update_interval,
            target_entropy,
            tensorboard_log,
            policy_kwargs,
            verbose,
            seed,
            device,
            _init_setup_model,
        )

        self.utd_ratio = utd_ratio
        self.sample_ratio = sample_ratio
        self.source_buffer = None  # type: Optional[GraphReplayBuffer]

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        num_update = self.utd_ratio  # 'G' in original paper, tend to update the critic not the policy

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        for gradient_step in range(gradient_steps):
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = th.exp(self.log_ent_coef.detach())  # 'alpha' in the original paper
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            for i_update in range(num_update):
                # Sample a mini-batch replay buffer
                # replay_data = self.replay_buffer.sample(batch_size)  # type: ignore[union-attr]
                replay_data = self.sample_replay_data(batch_size=batch_size, ratio=self.sample_ratio)

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
                critic_losses.append(critic_loss.item())

                # Optimize the critic
                self.critic.optimizer.zero_grad()
                self.critic.ext_optimizer.zero_grad()

                critic_loss.backward()
                self.critic.optimizer.step()

                if not self.policy.share_features_extractor:
                    self.critic.ext_optimizer.step()

                # Update target networks
                if gradient_step % self.target_update_interval == 0:
                    polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                    # Copy running stats, see GH issue #996
                    polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

            # Update the actor
            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)
            q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            ave_qf_pi = th.mean(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - ave_qf_pi).mean()
            actor_losses.append(actor_loss.item())

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Update the entropy value
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

    def sample_replay_data(self, batch_size: int, ratio: float = 0.5) -> GraphReplayBufferSamples:
        if ratio > 0.0:
            src_batch_size = round(float(batch_size) * ratio)
            online_batch_size = batch_size - src_batch_size
            assert online_batch_size >= 0
            assert self.source_buffer is not None, "The source buffer is not loaded"

            src_data = self.source_buffer.sample(src_batch_size)
            online_data = self.replay_buffer.sample(online_batch_size)
            data = (
                dgl.batch([src_data.observations, online_data.observations]),
                th.cat((src_data.actions, online_data.actions), dim=0),
                dgl.batch([src_data.next_observations, online_data.next_observations]),
                th.cat((src_data.dones, online_data.dones), dim=0),
                th.cat((src_data.rewards, online_data.rewards), dim=0),
            )
            return GraphReplayBufferSamples(*data)
        else:
            return self.replay_buffer.sample(batch_size)

    def _excluded_save_params(self) -> List[str]:
        return super()._excluded_save_params() + ["source_buffer"]

    def load_source_buffer(self, path: Union[str, pathlib.Path, io.BufferedIOBase]) -> None:
        """
        Load a source buffer from a pickle file.

        :param path: Path to the pickled source buffer.
        """
        self.source_buffer = load_from_pkl(path, self.verbose)
        assert isinstance(self.source_buffer, GraphReplayBuffer), "The source buffer must be a GraphReplayBuffer class"

        # Update saved replay buffer device to match current setting, see GH#1561
        self.source_buffer.device = self.device

        if self.verbose >= 1:
            print("Source buffer loaded successfully")

        # print("DEBUG: Source buffer size is", self.source_buffer.pos)
