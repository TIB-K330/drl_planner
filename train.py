import argparse
import gym
import importlib.util
import logging
import numpy as np
import os
import shutil
import sys
import torch

from crowd_sim.envs.utils.robot import Robot
from modules.policies import ExternalPolicy

from algorithms.mpc_droq import MpcDroQ
from modules.callbacks import CurriculumCallback, EvalCallback
from modules.evaluation import evaluate_policy


def main(args):
    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)

    os.makedirs(args.output_dir)
    # copy config file from ${args.config} to ${args.output_dir} as config.py
    shutil.copy(args.config, os.path.join(args.output_dir, 'config.py'))
    args.config = os.path.join(args.output_dir, 'config.py')

    # load config module of policy, env and trainer
    spec = importlib.util.spec_from_file_location('config', args.config)
    if spec is None:
        parser.error('Config file not found.')

    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    # configure logging
    log_file = os.path.join(args.output_dir, 'output.log')
    file_handler = logging.FileHandler(log_file, mode='a' if args.resume else 'w')
    stdout_handler = logging.StreamHandler(sys.stdout)
    level = logging.INFO if not args.debug else logging.DEBUG
    logging.basicConfig(level=level, handlers=[stdout_handler, file_handler],
                        format='%(asctime)s, %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")

    logging.info('Current random seed: {}'.format(sys_args.randomseed))
    logging.info('Current safe_weight: {}'.format(sys_args.safe_weight))
    logging.info('Current re_rvo_weight: {}'.format(sys_args.re_rvo))
    logging.info('Current re_theta_weight: {}'.format(sys_args.re_theta))
    logging.info('Current goal_weight: {}'.format(sys_args.goal_weight))
    logging.info('Current re_collision: {}'.format(sys_args.re_collision))
    logging.info('Current re_arrival: {}'.format(sys_args.re_arrival))
    logging.info('Current config content is :{}'.format(config))
    device = torch.device('cuda:0' if torch.cuda.is_available() and args.gpu else 'cpu')
    logging.info('Using device: %s', device)

    # configure environment
    env_config = config.EnvConfig(args.debug)
    env_config.reward.success_reward = args.re_arrival
    env_config.reward.goal_factor = args.goal_weight
    env_config.reward.collision_penalty = args.re_collision
    env_config.reward.discomfort_penalty_factor = args.safe_weight
    env_config.reward.re_theta = args.re_theta
    env_config.reward.re_rvo = args.re_rvo

    env = gym.make('CrowdSim-v0')
    env.configure(env_config)
    env.set_phase(3)

    robot = Robot(env_config, 'robot')
    robot.time_step = env.time_step
    robot.set_policy(ExternalPolicy())
    env.set_robot(robot)

    model = MpcDroQ(
        "GraphPolicy",
        env,
        buffer_size=300000,
        learning_starts=5000,
        batch_size=256,
        tau=0.005,
        gamma=pow(0.95, 0.25),
        utd_ratio=1,
        sample_ratio=0,
        seed=args.randomseed,
        device=device,
        policy_kwargs=dict(
            net_arch=[256, 256],
            share_features_extractor=True,
            droq_kwargs=dict(target_drop_rate=0.005)
        ),
        tensorboard_log=args.output_dir
    )

    curriculum_callback = CurriculumCallback(verbose=0)

    eval_callback = EvalCallback(eval_env=env, n_eval_episodes=100, eval_freq=500,
                                 best_model_save_path=args.output_dir, verbose=1)

    # model.load_source_buffer("data/mpc_rollouts.pkl")

    model.learn(10000000, callback=[eval_callback, curriculum_callback], log_interval=10)
    del model

    env.set_phase(10)

    try:
        model = MpcDroQ.load(os.path.join(args.output_dir, "best_model"), env=env, device=device)
    except ValueError as e:
        print(f"Caught an exception: {e}")
        exit()

    sr, cr, nt, dn, rt = evaluate_policy(model, env, 500, phase="test")
    print(sr, cr, nt, dn, rt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--config', type=str, default='configs/mpc_rl.py')
    parser.add_argument('--output_dir', type=str, default='data/output')
    parser.add_argument('--resume', default=True, action='store_true')
    parser.add_argument('--gpu', default=True, action='store_true')
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--randomseed', type=int, default=2)
    parser.add_argument('--safe_weight', type=float, default=0.5)
    parser.add_argument('--goal_weight', type=float, default=0.1)
    parser.add_argument('--re_collision', type=float, default=-0.25)
    parser.add_argument('--re_arrival', type=float, default=0.25)
    parser.add_argument('--re_rvo', type=float, default=0.01)
    parser.add_argument('--re_theta', type=float, default=0.01)

    sys_args = parser.parse_args()
    main(sys_args)
