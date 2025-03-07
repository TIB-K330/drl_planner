from typing import Dict, List, Optional, Tuple
import logging
import gym
import torch
from gym import spaces
import matplotlib.lines as mlines
from matplotlib import patches
import numpy as np
from numpy.linalg import norm
from modules.rvo.rvo_inter import rvo_inter

from .policy.policy_factory import policy_factory
from .utils.state import ObservableState, ObstacleState, WallState
from .utils.action import ActionDiff
from .utils.human import Human
from .utils.obstacle import Obstacle
from .utils.wall import Wall
from .utils.info import *
from .utils.utils import point_to_segment_dist, counterclockwise, point_in_poly, map_action_to_goal
from .utils.robot import Robot

EnvObs = Tuple[List[ObservableState], List[ObstacleState], List[WallState]]
EnvStepReturn = Tuple[EnvObs, float, bool, object]


class CrowdSim(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        """
        Movement simulation for n+1 agents
        Agent can either be human or robot.
        humans are controlled by an unknown and fixed policy.
        robot is controlled by a known and learnable policy.

        """
        self.time_limit: Optional[float] = None
        self.time_step: Optional[float] = None
        self.robot: Optional[Robot] = None
        self.humans: Optional[List[Human]] = None
        self.obstacles: Optional[List[Obstacle]] = None
        self.walls: Optional[List[Wall]] = None
        self.poly_obstacles = []
        self.global_time: Optional[float] = None
        self.robot_sensor_range: Optional[float] = None

        # reward function
        self.success_reward: Optional[float] = None
        self.collision_penalty: Optional[float] = None
        self.goal_factor: Optional[float] = None
        self.discomfort_dist: Optional[float] = None
        self.discomfort_penalty_factor: Optional[float] = None
        self.re_rvo: Optional[float] = None
        self.re_theta: Optional[float] = None

        # simulation configuration
        self.config = None
        self.case_capacity = None
        self.case_size = None
        self.case_counter = None
        self.randomize_attributes = None
        self.train_val_scenario = None
        self.test_scenario = None
        self.current_scenario = None
        self.square_width = None
        self.circle_radius = None
        self.human_num: Optional[int] = None
        self.nonstop_human: Optional[bool] = None
        self.static_obstacle_num = 3
        self.centralized_planning = None
        self.centralized_planner = None
        self.test_changing_size = False

        # for visualization
        self.states = None
        self.action_mask_vis: Optional[List[Tuple[float, float]]] = None
        self.panel_width: float = 10
        self.panel_height: float = 10
        self.fig = None
        self.ax = None

        self.rvo_inter = rvo_inter(
            neighbor_region=6,
            neighbor_num=20,
            vxmax=1,
            vymax=1,
            acceler=1.0,
            env_train=True,
            exp_radius=0.0,
            ctime_threshold=3.0,
            ctime_line_threshold=3.0
        )

        self.action_space: Optional[gym.Space] = None
        self.action_mask: Optional[torch.Tensor] = None
        self.random_seed: int = 0
        self.phase_num: int = 0
        self.phase: Optional[str] = None

    def configure(self, config) -> None:
        """
        Configure the crowd navigation environment based on the config file
        """
        self.config = config
        self.time_limit = config.env.time_limit
        self.time_step = config.env.time_step
        self.randomize_attributes = config.env.randomize_attributes
        self.robot_sensor_range = config.env.robot_sensor_range

        self.success_reward = config.reward.success_reward
        self.collision_penalty = config.reward.collision_penalty
        self.goal_factor = config.reward.goal_factor
        self.re_rvo = config.reward.re_rvo
        self.re_theta = config.reward.re_theta
        self.discomfort_penalty_factor = config.reward.discomfort_penalty_factor
        self.discomfort_dist = config.reward.discomfort_dist

        self.case_capacity = {'train': np.iinfo(np.uint32).max - 2000, 'val': 1000, 'test': 1000}
        self.case_size = {'train': config.env.train_size, 'val': config.env.val_size, 'test': config.env.test_size}
        self.case_counter = {'train': 0, 'test': 0, 'val': 0}

        self.train_val_scenario = config.sim.train_val_scenario
        self.test_scenario = config.sim.test_scenario
        self.square_width = config.sim.square_width
        self.circle_radius = config.sim.circle_radius

        if config.sim.obstacle_num:
            self.static_obstacle_num = config.sim.obstacle_num

        self.human_num = config.sim.human_num
        self.nonstop_human = config.sim.nonstop_human
        self.centralized_planning = config.sim.centralized_planning

        human_policy = config.humans.policy
        if self.centralized_planning:
            if human_policy == 'socialforce':
                logging.warning('Current socialforce policy only works in decentralized way with visible robot!')
            self.centralized_planner = policy_factory['centralized_' + human_policy]()

        logging.info('human number: {}'.format(self.human_num))
        if self.randomize_attributes:
            logging.info("Randomize human's radius and preferred speed")
        else:
            logging.info("Not randomize human's radius and preferred speed")

        logging.info(
            'Training simulation: {}, test simulation: {}'.format(self.train_val_scenario, self.test_scenario))

        logging.info('Square width: {}, circle width: {}'.format(self.square_width, self.circle_radius))

    def set_phase(self, phase_num: int) -> None:
        self.phase_num = phase_num
        if self.phase_num == 0:
            self.static_obstacle_num = 1
            self.human_num = 1
        elif self.phase_num == 1:
            self.static_obstacle_num = 3
            self.human_num = 1
        elif self.phase_num == 2:
            self.static_obstacle_num = 3
            self.human_num = 3
        elif self.phase_num == 3:
            self.static_obstacle_num = 3
            self.human_num = 5
        elif self.phase_num == 4:
            self.static_obstacle_num = np.random.randint(2, 5)
            self.human_num = np.random.randint(5, 10)
        elif self.phase_num == 10 or self.phase_num == 11:  # for test, differ in center polygon obstacle
            self.static_obstacle_num = 3
            self.human_num = 5

    def set_robot(self, robot):
        self.robot = robot

        if self.robot.kinematics == "holonomic":
            # action space: speed, heading
            self.action_space = spaces.Box(
                low=np.array([0, -np.pi]),
                high=np.array([1, np.pi]),
                dtype=np.float32
            )
        elif self.robot.kinematics == "unicycle":
            self.action_space = spaces.Box(
                low=np.array([0, -self.robot.rotation_constraint]),
                high=np.array([1, self.robot.rotation_constraint]),
                dtype=np.float32
            )
        elif self.robot.kinematics == "differential":
            self.action_space = spaces.Box(
                low=np.array([-1.0, -1.0]),
                high=np.array([1.0, 1.0]),
                dtype=np.float32
            )

        logging.info('rotation constraint: {}'.format(self.robot.rotation_constraint))

    def set_human_num(self, human_num: int):
        self.human_num = human_num
        logging.info('human number is {}'.format(self.human_num))

    def generate_human(self, human: Human = None, non_stop: bool = False, square: bool = False):
        if human is None:
            human = Human(self.config, 'humans')
        if self.randomize_attributes:
            human.sample_random_attributes()
        if square is False and non_stop is False:
            for sample_count in range(200):
                angle = np.random.random() * np.pi * 2
                # add some noise to simulate all the possible cases robot could meet with human
                px_noise = (np.random.random() - 0.5) * human.v_pref
                py_noise = (np.random.random() - 0.5) * human.v_pref
                ex_noise = (np.random.random() - 0.5) * human.v_pref
                ey_noise = (np.random.random() - 0.5) * human.v_pref
                px = (self.circle_radius + 0.5) * np.cos(angle) + px_noise
                py = (self.circle_radius + 0.5) * np.sin(angle) + py_noise
                gx = -(self.circle_radius + 0.5) * np.cos(angle) + ex_noise
                gy = -(self.circle_radius + 0.5) * np.sin(angle) + ey_noise
                collide = False
                for agent in [self.robot]:
                    min_dist = human.radius + agent.radius + self.discomfort_dist + 0.5
                    if (norm((px - agent.px, py - agent.py)) < min_dist or
                        norm((px - agent.gx, py - agent.gy)) < min_dist
                    ):
                        collide = True
                        break
                for agent in self.humans:
                    min_dist = human.radius + agent.radius + self.discomfort_dist
                    if (norm((px - agent.px, py - agent.py)) < min_dist or
                        norm((px - agent.gx, py - agent.gy)) < min_dist
                    ):
                        collide = True
                        break
                for wall in self.walls:
                    if (point_to_segment_dist(wall.sx, wall.sy, wall.ex, wall.ey, px, py) < human.radius + 0.3 and
                        point_to_segment_dist(wall.sx, wall.sy, wall.ex, wall.ey, gx, gy) < human.radius + 0.3
                    ):
                        collide = True
                        break
                for poly_obs in self.poly_obstacles:
                    if point_in_poly(px, py, poly_obs) or point_in_poly(gx, gy, poly_obs):
                        collide = True
                        break
                if not collide:
                    break

            human.start_pos.append((px, py))
            human.set(px, py, gx, gy, 0, 0, 0)
        elif square is False and non_stop is True:
            for sample_count in range(200):
                angle = np.random.random() * np.pi * 2
                # add some noise to simulate all the possible cases robot could meet with human
                px = human.px
                py = human.py
                gx_noise = (np.random.random() - 0.5) * human.v_pref
                gy_noise = (np.random.random() - 0.5) * human.v_pref
                gx = self.circle_radius * np.cos(angle) + gx_noise
                gy = self.circle_radius * np.sin(angle) + gy_noise
                collide = False
                for agent in [self.robot] + self.humans:
                    min_dist = human.radius + agent.radius + self.discomfort_dist + 0.5
                    if norm((gx - agent.gx, gy - agent.gy)) < min_dist:
                        collide = True
                        break
                for agent in [self.robot]:
                    min_dist = human.radius + agent.radius + self.discomfort_dist
                    if norm((gx - agent.gx, gy - agent.gy)) < min_dist:
                        collide = True
                        break
                for wall in self.walls:
                    if (point_to_segment_dist(wall.sx, wall.sy, wall.ex, wall.ey, gx, gy) < human.radius +
                        self.robot.radius
                    ):
                        collide = True
                        break
                for poly_obs in self.poly_obstacles:
                    if point_in_poly(gx, gy, poly_obs):
                        collide = True
                        break
                if not collide:
                    break

            human.start_pos.append((px, py))
            human.set(px, py, gx, gy, 0, 0, 0)
        elif square is True and non_stop is False:
            if np.random.random() > 0.5:
                sign = -1
            else:
                sign = 1
            for sample_count in range(200):
                px = np.random.random() * self.square_width * 0.5 * sign
                py = (np.random.random() - 0.5) * self.square_width
                collide = False
                for agent in [self.robot] + self.humans:
                    min_dist = human.radius + agent.radius + self.discomfort_dist + 0.5
                    if norm((px - agent.px, py - agent.py)) < min_dist:
                        collide = True
                        break
                for wall in self.walls:
                    if (point_to_segment_dist(wall.sx, wall.sy, wall.ex, wall.ey, px, py) < human.radius +
                        self.robot.radius
                    ):
                        collide = True
                        break
                for poly_obs in self.poly_obstacles:
                    if point_in_poly(px, py, poly_obs):
                        collide = True
                        break
                if not collide:
                    break
            for sample_count in range(200):
                gx = np.random.random() * self.square_width * 0.5 * (- sign)
                gy = (np.random.random() - 0.5) * self.square_width
                collide = False
                for agent in [self.robot] + self.humans:
                    min_dist = human.radius + agent.radius + self.discomfort_dist + 0.5
                    if norm((gx - agent.gx, gy - agent.gy)) < min_dist:
                        collide = True
                        break
                for wall in self.walls:
                    if (point_to_segment_dist(wall.sx, wall.sy, wall.ex, wall.ey, gx, gy) < human.radius +
                        self.robot.radius
                    ):
                        collide = True
                        break
                for poly_obs in self.poly_obstacles:
                    if point_in_poly(gx, gy, poly_obs) or point_in_poly(gx, gy, poly_obs):
                        collide = True
                        break
                if not collide:
                    break
            human.start_pos.append((px, py))
            human.set(px, py, gx, gy, 0, 0, 0)
        elif square is True and non_stop is True:
            if np.random.random() > 0.5:
                sign = -1
            else:
                sign = 1
            goal_count = 0
            for sample_count in range(200):
                goal_count = goal_count + 1
                px = human.px
                py = human.py
                gx = np.random.random() * self.square_width * 0.5 * (- sign)
                gy = (np.random.random() - 0.5) * self.square_width
                collide = False
                for agent in [self.robot] + self.humans:
                    min_dist = human.radius + agent.radius + self.discomfort_dist
                    if norm((gx - agent.gx, gy - agent.gy)) < min_dist:
                        collide = True
                        break
                for wall in self.walls:
                    if (point_to_segment_dist(wall.sx, wall.sy, wall.ex, wall.ey, gx, gy) <
                        human.radius + self.robot.radius
                    ):
                        collide = True
                        break
                for poly_obs in self.poly_obstacles:
                    if point_in_poly(gx, gy, poly_obs) or point_in_poly(gx, gy, poly_obs):
                        collide = True
                        break
                if not collide:
                    break
            human.start_pos.append((px, py))
            human.set(px, py, gx, gy, 0, 0, 0)
        return human

    def generate_static_obstacle(self, obstacle: Obstacle = None):
        self.obstacles = []
        for i in range(self.static_obstacle_num):
            obstacle = Obstacle()
            obstacle.sample_random_attributes()
            sample_count = 0
            for sample_count in range(200):
                px = (np.random.random() - 0.5) * self.square_width * 0.8
                py = (np.random.random() - 0.5) * self.circle_radius * 2
                obstacle.set(px, py, obstacle.radius)
                collide = False
                for agent in [self.robot] + self.humans:
                    if (norm((px - agent.px, py - agent.py)) < obstacle.radius + agent.radius + 0.5 or
                        norm((px - agent.gx, py - agent.gy)) < obstacle.radius + agent.radius + 0.5
                    ):
                        collide = True
                        break
                for agent in self.obstacles:
                    if norm((px - agent.px, py - agent.py)) < obstacle.radius + agent.radius + 0.5:
                        collide = True
                        break
                for wall in self.walls:
                    if point_to_segment_dist(wall.sx, wall.sy, wall.ex, wall.ey, px, py) < obstacle.radius + 0.8:
                        collide = True
                        break
                for poly_obs in self.poly_obstacles:
                    if point_in_poly(px, py, poly_obs):
                        collide = True
                        break
                if not collide:
                    break

            if sample_count < 200:
                self.obstacles.append(obstacle)

    def generate_wall(self, start_position, end_position, wall: Wall = None):
        wall = Wall(self.config)
        wall.set_position(start_position, end_position)
        return wall

    def generate_center_obstacle(self):
        center_x = (np.random.random() - 0.5) * 2  # gen random num in [-1, 1]
        center_y = (np.random.random() - 0.5) * 2

        # the width and length of the rect rand in [1, 3]
        width = np.clip(np.random.normal(2, 1.0), 1, 3)
        length = np.clip(np.random.normal(2, 1.0), 1, 3)

        x1 = center_x - width / 2.0
        x2 = center_x + width / 2.0
        y1 = center_y - length / 2.0
        y2 = center_y + length / 2.0

        # The rectangle obstacle is fixed with the center, width, and length of
        # (0.0, 0.0), 2.5 m, and 1.0 m in the test setting.
        if self.phase_num == 10:
            y1 = -0.5
            y2 = 0.5
            x1 = -1.25
            x2 = 1.25
        if self.phase_num == 11:
            y1 = -0.5
            y2 = 1.5
            x1 = -0.5
            x2 = 0.5

        transfer_vertex = ([x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1])

        # gen walls based on the vertex of the rect
        for i in range(len(transfer_vertex) - 1):
            self.walls.append(self.generate_wall(transfer_vertex[i], transfer_vertex[i + 1]))

        self.poly_obstacles.clear()
        self.poly_obstacles.append(transfer_vertex)

    def generate_corridor_scenario(self):
        if self.phase_num >= 0:
            corridor_width = self.square_width
            corridor_length = self.square_width * 2.0  # as a large num
            self.walls.append(self.generate_wall(
                [-corridor_width / 2, corridor_length / 2],
                [-corridor_width / 2, -corridor_length / 2])
            )

            self.walls.append(self.generate_wall(
                [corridor_width / 2, -corridor_length / 2],
                [corridor_width / 2, corridor_length / 2])
            )

    def reset(self, phase: str = 'test', test_case: Optional[int] = None) -> EnvObs:
        assert phase in ['train', 'val', 'test'], "Invalid phase set!!"
        self.phase = phase
        if self.robot is None:
            raise AttributeError('Robot has to be set!')

        if test_case is not None:
            self.case_counter[phase] = test_case

        self.global_time = 0  # reset global time
        # update random seed
        train_seed_begin = [0, 10, 100, 1000, 10000]
        val_seed_begin = [0, 10, 100, 1000, 10000]
        test_seed_begin = [0, 10, 100, 1000, 10000]
        base_seed = {
            'val': 0 + val_seed_begin[1],
            'test': self.case_capacity['val'] + test_seed_begin[2],
            'train': self.case_capacity['val'] + self.case_capacity['test'] + train_seed_begin[1],
        }

        self.random_seed = base_seed[phase] + self.case_counter[phase]
        np.random.seed(self.random_seed)

        if self.phase_num == 10 or self.phase_num == 11:
            target_x = 0
            target_y = self.circle_radius
            robot_theta = np.pi / 2 + np.random.random() * np.pi / 4.0 - np.pi / 8.0
        elif self.phase_num == 0:
            target_x = (np.random.random() - 0.5) * self.square_width * 0.8  # rand in [-4, 4]
            target_y = self.circle_radius - np.random.random() * 6  # rand in [-2, 4]
            robot_theta = (np.random.random() - 0.5) * 2 * np.pi  # rand in [-pi, pi]
        else:
            target_x = (np.random.random() - 0.5) * self.square_width * 0.8  # rand in [-4, 4]
            target_y = self.circle_radius  # fixed to 4.0
            robot_theta = np.pi / 2 + np.random.random() * np.pi / 4.0 - np.pi / 8.0

        if self.phase_num == 4:
            self.static_obstacle_num = np.random.randint(2, 5)
            self.human_num = np.random.randint(5, 10)

        self.robot.set(0, -self.circle_radius, target_x, target_y, 0, 0, robot_theta)  # fixed init position (0, -4)

        if self.case_counter[phase] >= 0:
            self.walls = []
            self.generate_corridor_scenario()
            if self.phase_num >= 1:
                self.generate_center_obstacle()

            if phase == 'test':
                logging.debug('current test seed is:{}'.format(base_seed[phase] + self.case_counter[phase]))

            if not self.robot.policy.multiagent_training and phase in ['train', 'val']:
                self.current_scenario = 'circle_crossing'
            else:
                self.current_scenario = self.test_scenario

            self.humans = []
            for i in range(self.human_num):
                if self.current_scenario == 'circle_crossing' and i < 5:
                    self.humans.append(self.generate_human())
                else:
                    self.humans.append(self.generate_human(square=True))

            self.generate_static_obstacle()
            self.case_counter[phase] = (self.case_counter[phase] + 1) % self.case_size[phase]  # update case_counter
        else:
            assert phase == 'test'
            if self.case_counter[phase] == -1:
                # for debugging purposes
                self.human_num = 3
                self.humans = [Human(self.config, 'humans') for _ in range(self.human_num)]
                self.humans[0].set(0, -6, 0, 5, 0, 0, np.pi / 2)
                self.humans[1].set(-5, -5, -5, 5, 0, 0, np.pi / 2)
                self.humans[2].set(5, -5, 5, 5, 0, 0, np.pi / 2)

                self.obstacles = []
                self.walls = []
            else:
                raise NotImplementedError

        for agent in [self.robot] + self.humans:
            agent.time_step = self.time_step
            agent.policy.time_step = self.time_step

        if self.centralized_planning:
            self.centralized_planner.reset()
            self.centralized_planner.time_step = self.time_step

        self.states = list()  # historical states of dynamic agents, for visualization
        self._update_action_mask()

        # get current observation
        if self.robot.sensor == 'coordinates':
            ob_human = self.compute_observation_for(self.robot)
            if self.obstacles is not None:
                ob_obstacles = [obstacle.get_observable_state() for i, obstacle in enumerate(self.obstacles)]
            else:
                ob_obstacles = []

            if self.walls is not None:
                ob_walls = [wall.get_observable_state() for i, wall in enumerate(self.walls)]
            else:
                ob_walls = []

            ob = (ob_human, ob_obstacles, ob_walls)
        elif self.robot.sensor == 'RGB':
            raise NotImplementedError

        return ob

    def get_case_counter(self) -> Dict[str, int]:
        return self.case_counter

    def set_case_counter(self, case_counter: Dict[str, int]) -> None:
        assert all(phase in ['train', 'val', 'test'] for phase in case_counter), "Invalid case counter set!!"
        self.case_counter = case_counter

    def reward_cal(self, action, human_actions):
        """
        Compute actions for all agents, detect collision, update environment and return (ob, reward, done, info)
        """
        """ Firstly, predict the robot position """
        pre_robot_pos_x, pre_robot_pos_y = self.robot.compute_position(action, self.time_step)
        weight_goal = self.goal_factor
        weight_safe = self.discomfort_penalty_factor
        re_collision = self.collision_penalty
        re_arrival = self.success_reward
        re_theta = self.re_theta
        # collision detection
        dmin = float('inf')
        collision = False
        safety_penalty = 0.0
        num_discom = 0
        """ Secondly, deal with humans """
        for i, human in enumerate(self.humans):
            px = human.px - self.robot.px
            py = human.py - self.robot.py
            end_human_x = human_actions[i].vx * self.time_step + human.px
            end_human_y = human_actions[i].vy * self.time_step + human.py
            ex = end_human_x - pre_robot_pos_x
            ey = end_human_y - pre_robot_pos_y
            if self.phase == 'test':
                closest_dist = point_to_segment_dist(px, py, ex, ey, 0, 0) - human.radius - self.robot.radius + 0.1
            else:
                closest_dist = point_to_segment_dist(px, py, ex, ey, 0, 0) - human.radius - self.robot.radius

            if closest_dist < 0:
                collision = True
                logging.info("Collision: distance between robot and pedestrian{} is {:.2E} at time {:.2E}".format(
                    i, closest_dist, self.global_time))

            if closest_dist < dmin:
                dmin = closest_dist
            if closest_dist < self.discomfort_dist:
                safety_penalty = safety_penalty + (closest_dist - self.discomfort_dist)
                num_discom = num_discom + 1

        """ Thirdly, deal with obstacles """
        for i, obstacle in enumerate(self.obstacles):
            px = obstacle.px - self.robot.px
            py = obstacle.py - self.robot.py
            ex = obstacle.px - pre_robot_pos_x
            ey = obstacle.py - pre_robot_pos_y
            if self.phase == 'test':
                closest_dist = point_to_segment_dist(px, py, ex, ey, 0, 0) - obstacle.radius - self.robot.radius + 0.1
            else:
                closest_dist = point_to_segment_dist(px, py, ex, ey, 0, 0) - obstacle.radius - self.robot.radius
            if closest_dist < 0:
                collision = True
                logging.info("Collision: distance between robot and obstacle{} is {:.2E} at time {:.2E}".format(
                    i, closest_dist, self.global_time))

                num_discom = num_discom + 1
            # if closest_dist < dmin:
            # dmin = closest_dist
            if closest_dist < self.discomfort_dist * 0.5:
                safety_penalty = safety_penalty + (closest_dist - self.discomfort_dist * 0.5) * 0.5
                num_discom = num_discom + 1

        """ Then, deal with walls """
        for i, wall in enumerate(self.walls):
            # across the wall #
            if ((counterclockwise(wall.sx, wall.sy, wall.ex, wall.ey, self.robot.px, self.robot.py) !=
                counterclockwise(wall.sx, wall.sy, wall.ex, wall.ey, pre_robot_pos_x, pre_robot_pos_y)) and
                (counterclockwise(self.robot.px, self.robot.py, pre_robot_pos_x, pre_robot_pos_y, wall.sx, wall.sy) !=
                 counterclockwise(self.robot.px, self.robot.py, pre_robot_pos_x, pre_robot_pos_y, wall.ex, wall.ey))
            ):
                closest_dist = 0.0
            else:
                min_dis_start = point_to_segment_dist(wall.sx, wall.sy, wall.ex, wall.ey, self.robot.px, self.robot.py)
                min_dis_end = point_to_segment_dist(wall.sx, wall.sy, wall.ex, wall.ey, pre_robot_pos_x,
                                                    pre_robot_pos_y)

                closest_dist = min_dis_end if min_dis_end < min_dis_start else min_dis_start

            if self.phase == 'test':
                closest_dist = closest_dist - self.robot.radius + 0.1
            else:
                closest_dist = closest_dist - self.robot.radius

            if closest_dist < 0:
                collision = True
                logging.info("Collision: distance between robot and wall {} is {:.2E} at time {:.2E}".format(
                    i, closest_dist, self.global_time))

                num_discom = num_discom + 1
            # if closest_dist < dmin:
            #     dmin = closest_dist
            if closest_dist < self.discomfort_dist * 0.5:
                safety_penalty = safety_penalty + (closest_dist - self.discomfort_dist * 0.5) * 0.5
                num_discom = num_discom + 1

        """check if reaching the goal"""
        end_position = np.array([pre_robot_pos_x, pre_robot_pos_y])  # predicted next robot position
        cur_position = np.array((self.robot.px, self.robot.py))
        goal_position = np.array(self.robot.get_goal_position())
        reward_goal = norm(cur_position - goal_position) - norm(end_position - goal_position)
        reaching_goal = norm(end_position - goal_position) < self.robot.radius

        robot2goal = goal_position - cur_position
        theta_r2g = np.math.atan2(robot2goal[1], robot2goal[0])
        reward_theta = np.cos(self.robot.theta) * np.cos(theta_r2g) + np.sin(self.robot.theta) * np.sin(theta_r2g) - 1
        reward_theta = reward_theta / (norm(cur_position - goal_position) + 5.0)  # d_theta = 5.0 m
        reward_col = 0.0
        reward_arrival = 0.0

        if self.global_time >= self.time_limit - 1:
            done = True
            info = Timeout()
        elif collision:
            reward_col = re_collision
            done = True
            info = Collision()
        elif reaching_goal:
            reward_arrival = re_arrival
            done = True
            info = ReachGoal()
        elif dmin < self.discomfort_dist:
            done = False
            info = Discomfort(dmin)
            info.num = num_discom
        else:
            done = False
            info = Nothing()

        reward = reward_arrival + weight_goal * reward_goal + re_theta * reward_theta
        constraint = reward_col + weight_safe * safety_penalty
        return reward, constraint, done, info

    def rvo_reward_cal(self, ob) -> float:
        robot_state, human_state, obstacle_state, wall_state = self.robot.get_state(ob)

        robot_state_array = robot_state.numpy()
        human_state_array = human_state.numpy()
        obstacle_state_array = obstacle_state.numpy()
        wall_state_array = wall_state.numpy()

        vo_flag_array, min_exp_time_array, min_dis_array = self.rvo_inter.config_vo_reward(
            robot_state_array,
            human_state_array,
            obstacle_state_array,
            wall_state_array
        )

        rvo_reward = 0.0
        for i in range(len(vo_flag_array)):
            vo_flag = vo_flag_array[i]
            min_exp_time = min_exp_time_array[i]
            min_dis = min_dis_array[i]
            min_exp_time = max(min_exp_time, 0.0)
            exp_time_reward = 1.0 / (min_exp_time + 1.0)

            if vo_flag is True:
                if min_exp_time < 1.0:
                    rvo_reward += 3.0 * (1 / (3.0 + 1.0) - exp_time_reward) - 0.1
                else:
                    rvo_reward += 1 / (3.0 + 1.0) - exp_time_reward - 0.1

        return rvo_reward

    def _update_action_mask(self) -> None:
        self.action_mask = torch.ones(81)  # reset mask to default
        self.action_mask_vis = []
        rotation_matrix = np.array([[np.cos(self.robot.theta), -np.sin(self.robot.theta)],
                                    [np.sin(self.robot.theta), np.cos(self.robot.theta)]])

        trans_vec = np.array([[self.robot.px], [self.robot.py]])
        safety_dis = self.robot.radius + 0.1
        for action_idx in range(len(self.action_mask)):
            is_valid = True
            gx, gy = map_action_to_goal(action_idx)
            goal_in_map = np.dot(rotation_matrix, np.array([[gx], [gy]])) + trans_vec
            goal_in_map = (goal_in_map[0][0], goal_in_map[1][0])
            self.action_mask_vis.append(goal_in_map)
            for obst in self.obstacles:
                closest_dist = np.sqrt((goal_in_map[0] - obst.px)**2 + (goal_in_map[1] - obst.py)**2)
                if closest_dist < obst.radius + safety_dis:
                    self.action_mask[action_idx] = 0.0
                    is_valid = False
                    break
            if not is_valid:
                continue
            for wall in self.walls:
                closest_dist = point_to_segment_dist(wall.sx, wall.sy, wall.ex, wall.ey, *goal_in_map)
                if closest_dist < safety_dis:
                    self.action_mask[action_idx] = 0.0
                    is_valid = False
                    break
            if not is_valid:
                continue
            for poly in self.poly_obstacles:
                if point_in_poly(*goal_in_map, poly):
                    self.action_mask[action_idx] = 0.0
                    is_valid = False
                    break
            if not is_valid:
                continue
            # boundary check
            if (goal_in_map[0] > self.square_width / 2.0 or goal_in_map[0] < -self.square_width / 2.0 or
                goal_in_map[1] > self.square_width - safety_dis or goal_in_map[1] < -self.square_width + safety_dis):
                self.action_mask[action_idx] = 0.0

    def step(self, action: ActionDiff, update: bool = True) -> EnvStepReturn:
        """
        Compute actions for all agents, detect collision, update environment and return (ob, reward, done, info)
        """
        if self.centralized_planning:
            agent_states = [human.get_full_state() for human in self.humans]
            self.centralized_planner.set_walls(self.walls)
            self.centralized_planner.set_static_obstacles(self.obstacles)
            if self.robot.visible:
                if self.robot.kinematics == 'differential':
                    robot_state = self.robot.get_full_state()
                    linear_vel = 0.5 * (robot_state.vx + robot_state.vy)
                    robot_state.vx = linear_vel * np.cos(robot_state.theta)
                    robot_state.vy = linear_vel * np.sin(robot_state.theta)
                    agent_states.append(robot_state)
                else:
                    agent_states.append(self.robot.get_full_state())
                human_actions = self.centralized_planner.predict(agent_states)[:-1]
            else:
                human_actions = self.centralized_planner.predict(agent_states)
        else:
            human_actions = []
            for human in self.humans:
                ob = self.compute_observation_for(human)
                human_actions.append(human.act(ob))

        reward, constraint, done, info = self.reward_cal(action, human_actions)

        if update:
            # update all agents, both robot and humans
            self.robot.step(action)
            self._update_action_mask()

            for human, human_action in zip(self.humans, human_actions):
                human.step(human_action)
                if not self.nonstop_human or not human.reached_destination():
                    continue
                # generate new goals for non-stop pedestrians
                human.reach_count = human.reach_count + 1
                if (human.reach_count > 2 and norm((human.px - self.robot.px, human.py - self.robot.py)) >
                    human.radius + self.robot.radius + 0.5):
                    if self.current_scenario == 'circle_crossing':
                        self.generate_human(human, non_stop=True)
                        human.reach_count = 0
                    else:
                        self.generate_human(human, non_stop=True, square=True)
                        human.reach_count = 0

            self.global_time += self.time_step
            self.states.append([
                self.robot.get_full_state(),
                [human.get_full_state() for human in self.humans],
                [human.id for human in self.humans]
            ])  # save states for playback or visualization

            # compute the observation
            if self.robot.sensor == 'coordinates':
                ob_human = [human.get_observable_state() for human in self.humans]
                ob_obstacles = [obstacle.get_observable_state() for i, obstacle in enumerate(self.obstacles)]
                ob_walls = [wall.get_observable_state() for i, wall in enumerate(self.walls)]
                ob = (ob_human, ob_obstacles, ob_walls)
            elif self.robot.sensor == 'RGB':
                raise NotImplementedError
        else:
            if self.robot.sensor == 'coordinates':
                ob_human = [
                    human.get_next_observable_state(action) for human, action in zip(self.humans, human_actions)
                ]
                ob_obstacles = [obstacle.get_observable_state() for i, obstacle in enumerate(self.obstacles)]
                ob_walls = [wall.get_observable_state() for i, wall in enumerate(self.walls)]
                ob = (ob_human, ob_obstacles, ob_walls)
            elif self.robot.sensor == 'RGB':
                raise NotImplementedError

        rvo_reward = self.rvo_reward_cal(ob)
        reward = 100 * (reward + constraint + self.re_rvo * rvo_reward)  # 100 times scale

        return ob, reward, done, info

    def compute_observation_for(self, agent):
        if agent == self.robot:
            ob = []
            for human in self.humans:
                if self.test_changing_size is False:
                    ob.append(human.get_observable_state())
                else:
                    dis2 = ((human.px - agent.px) * (human.px - agent.px) +
                            (human.py - agent.py) * (human.py - agent.py))

                    if dis2 < self.robot_sensor_range * self.robot_sensor_range:
                        ob.append(human.get_observable_state())
        else:
            ob = [other_human.get_observable_state() for other_human in self.humans if other_human != agent]
            if self.robot.visible:
                ob += [self.robot.get_observable_state()]
        return ob

    def render(self, mode: str = "video", output_file: Optional[str] = None,
               local_goal_vis: Optional[Tuple[float, float]] = None):
        from matplotlib import animation
        import matplotlib.pyplot as plt
        # plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
        x_offset = 0.3
        y_offset = 0.4
        cmap = plt.cm.get_cmap('terrain', 200)
        cmap2 = plt.cm.get_cmap('tab10', 10)
        robot_color = 'red'
        arrow_style = patches.ArrowStyle.Fancy(head_length=4, head_width=2, tail_width=.6)
        display_numbers = True

        if mode == 'traj':
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.tick_params(labelsize=16)
            ax.set_xlim(-self.panel_width / 2 - 1, self.panel_width / 2 + 1)
            ax.set_ylim(-self.panel_height / 2 - 0.5, self.panel_height / 2 + 0.5)
            ax.set_xlabel('x(m)', fontsize=16)
            ax.set_ylabel('y(m)', fontsize=16)

            # add human start positions and goals
            human_colors = [cmap2(i) for i in range(len(self.humans))]
            for i in range(len(self.humans)):
                human = self.humans[i]
                human_goal = mlines.Line2D(
                    [human.get_goal_position()[0]],
                    [human.get_goal_position()[1]],
                    color=human_colors[i],
                    marker='*',
                    linestyle='None',
                    markersize=10
                )
                ax.add_artist(human_goal)

                for j in range(len(human.start_pos)):
                    pos = human.start_pos[j]
                    if j == 0:
                        test_start = mlines.Line2D(
                            [pos[0]],
                            [pos[1]],
                            color=human_colors[i],
                            marker=4,
                            linestyle='None',
                            markersize=10
                        )
                    else:
                        test_start = mlines.Line2D(
                            [pos[0]],
                            [pos[1]],
                            color=human_colors[i],
                            marker='s',
                            linestyle='None',
                            markersize=10
                        )
                    ax.add_artist(test_start)
                # human_start = mlines.Line2D([human.get_start_position()[0]], [human.get_start_position()[1]],
                #                             color=human_colors[i],
                #                             marker='s', linestyle='None', markersize=10)
                # ax.add_artist(human_start)
            for i in range(len(self.obstacles)):
                obstacle = self.obstacles[i]
                obstacle_mark = plt.Circle(obstacle.get_position(), obstacle.radius, fill=True, color='grey')
                ax.add_artist(obstacle_mark)

            for i in range(len(self.walls)):
                wall = self.walls[i]
                wall_line = mlines.Line2D(
                    [wall.sx, wall.ex],
                    [wall.sy, wall.ey],
                    color='black',
                    marker='.',
                    linestyle='solid',
                    markersize=5
                )
                ax.add_artist(wall_line)

            robot_positions = [self.states[i][0].position for i in range(len(self.states))]
            human_positions = [
                [self.states[i][1][j].position for j in range(len(self.humans))] for i in range(len(self.states))
            ]
            goal = mlines.Line2D(
                [self.robot.get_goal_position()[0]],
                [self.robot.get_goal_position()[1]],
                color='black',
                marker='*',
                linestyle='None',
                markersize=15,
                label='Goal'
            )
            ax.add_artist(goal)

            for k in range(len(self.states)):
                if k % 4 == 0 or k == len(self.states) - 1:
                    robot = plt.Circle(robot_positions[k], self.robot.radius / 4, fill=True, color=robot_color)
                    humans = [
                        plt.Circle(
                            human_positions[k][i],
                            self.humans[i].radius / 4,
                            fill=True,
                            color=human_colors[i]
                        ) for i in range(len(self.humans))
                    ]
                    ax.add_artist(robot)
                    for human in humans:
                        ax.add_artist(human)

                # add time annotation
                global_time = k * self.time_step
                if global_time % 5 == 0 or k == len(self.states) - 1:
                    agents = humans + [robot]
                    times = [
                        plt.text(
                            agents[i].center[0] + 0.1,
                            agents[i].center[1] + 0.1,
                            '{:.1f}'.format(global_time),
                            color='black',
                            fontsize=12
                        ) for i in range(self.human_num + 1)
                    ]
                    for time in times:
                        ax.add_artist(time)

                if k != 0:
                    nav_direction = plt.arrow(
                        self.states[k - 1][0].px,
                        self.states[k - 1][0].py,
                        self.states[k][0].px - self.states[k - 1][0].px,
                        self.states[k][0].py - self.states[k - 1][0].py,
                        length_includes_head=True,
                        head_width=0.08,
                        lw=0.8,
                        color='black'
                    )
                    human_directions = [
                        plt.arrow(
                            self.states[k - 1][1][i].px,
                            self.states[k - 1][1][i].py,
                            self.states[k][1][i].px - self.states[k - 1][1][i].px,
                            self.states[k][1][i].py - self.states[k - 1][1][i].py,
                            length_includes_head=True,
                            head_width=0.08,
                            lw=0.5,
                            color=human_colors[i]
                        ) for i in range(self.human_num)
                    ]
                    ax.add_artist(nav_direction)
                    for human_direction in human_directions:
                        ax.add_artist(human_direction)

            plt.legend([robot], ['Robot'], fontsize=16)
            plt.show()
            # plt.ion()
            # plt.pause(1.0)
            # plt.close()
        elif mode == 'video':
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.tick_params(labelsize=12)
            ax.set_xlim(-self.panel_width / 2 - 1, self.panel_width / 2 + 1)
            ax.set_ylim(-self.panel_height / 2 - 0.5, self.panel_height / 2 + 0.5)
            ax.set_xlabel('x(m)', fontsize=14)
            ax.set_ylabel('y(m)', fontsize=14)
            show_human_start_goal = False
            robot_color = 'black'
            # add human start positions and goals
            human_colors = [cmap(20) for i in range(len(self.humans))]

            # add robot start position
            robot_start = mlines.Line2D(
                [self.robot.get_start_position()[0]],
                [self.robot.get_start_position()[1]],
                color=robot_color,
                marker='o',
                linestyle='None',
                markersize=15
            )

            ax.add_artist(robot_start)

            # add robot and its goal
            robot_positions = [state[0].position for state in self.states]
            robot = plt.Circle(robot_positions[0], self.robot.radius, fill=False, color=robot_color)
            ax.add_artist(robot)

            robot_goal = mlines.Line2D(
                [self.robot.get_goal_position()[0]],
                [self.robot.get_goal_position()[1]],
                color='red',
                marker='*',
                linestyle='None',
                markersize=15,
                label='Goal'
            )
            ax.add_artist(robot_goal)

            # add static circular obstacles
            for i in range(len(self.obstacles)):
                obstacle = self.obstacles[i]
                obstacle_mark = plt.Circle(obstacle.get_position(), obstacle.radius, fill=True, color='grey')
                ax.add_artist(obstacle_mark)

            # add static polygon obstacles
            for i in range(len(self.walls)):
                wall = self.walls[i]
                wall_line = mlines.Line2D(
                    [wall.sx, wall.ex],
                    [wall.sy, wall.ey],
                    color='black',
                    marker='.',
                    linestyle='solid',
                    markersize=5
                )
                ax.add_artist(wall_line)

            direction_length = 1.0
            if len(self.humans) == 0:
                # add time annotation
                time = plt.text(0.4, 1.02, 'Time: {}'.format(0), fontsize=16, transform=ax.transAxes)
                ax.add_artist(time)
                radius = self.robot.radius
                orientations = []

                # concat the humans and the robot
                for i in range(self.human_num + 1):
                    orientation = []
                    for state in self.states:
                        agent_state = state[0] if i == 0 else state[1][i - 1]
                        if self.robot.kinematics =='holonomic' or i != 0:
                            theta = np.arctan2(agent_state.vy, agent_state.vx)
                            direction = (
                                (agent_state.px, agent_state.py),
                                (agent_state.px + direction_length * radius * np.cos(theta),
                                 agent_state.py + direction_length * radius * np.sin(theta))
                            )
                        else:
                            direction = (
                                (agent_state.px, agent_state.py),
                                (agent_state.px + direction_length * radius * np.cos(agent_state.theta),
                                 agent_state.py + direction_length * radius * np.sin(agent_state.theta))
                            )
                        orientation.append(direction)
                    orientations.append(orientation)
                    if i == 0:
                        robot_arrow_color = 'red'
                        arrows = [patches.FancyArrowPatch(
                            *orientation[0],
                            color=robot_arrow_color,
                            arrowstyle=arrow_style
                        )]
                    else:
                        human_arrow_color = 'red'
                        arrows.extend([
                            patches.FancyArrowPatch(*orientation[0], color=human_arrow_color, arrowstyle=arrow_style)
                        ])

                for arrow in arrows:
                    ax.add_artist(arrow)
                global_step = 0
            else:
                # add humans and their numbers
                human_positions = [[state[1][j].position for j in range(len(self.humans))] for state in self.states]
                humans = [
                    plt.Circle(
                        human_positions[0][i],
                        self.humans[i].radius,
                        fill=False,
                        color=human_colors[i]
                    ) for i in range(len(self.humans))
                ]

                plt.legend([robot, humans[0], robot_goal], ['Robot', 'Human', 'Goal'], fontsize=14)

                # disable showing human numbers
                if display_numbers:
                    human_numbers = [
                        plt.text(
                            humans[i].center[0] - x_offset,
                            humans[i].center[1] + y_offset,
                            str(i + 1),
                            color='black',
                            fontsize=12
                        ) for i in range(len(self.humans))
                    ]

                for i, human in enumerate(humans):
                    ax.add_artist(human)
                    if display_numbers:
                        ax.add_artist(human_numbers[i])

                # add time annotation
                time = plt.text(0.4, 1.02, 'Time: {}'.format(0), fontsize=16, transform=ax.transAxes)
                ax.add_artist(time)

                # compute orientation in each step and use arrow to show the direction
                radius = self.robot.radius
                orientations = []
                for i in range(self.human_num + 1):
                    orientation = []
                    for state in self.states:
                        agent_state = state[0] if i == 0 else state[1][i - 1]
                        if self.robot.kinematics =='holonomic' or i != 0:
                            # assume that the heading of holonomic agents is inconsistent with their velocity vector
                            theta = np.arctan2(agent_state.vy, agent_state.vx)
                            direction = (
                                (agent_state.px, agent_state.py),
                                (agent_state.px + direction_length * radius * np.cos(theta),
                                 agent_state.py + direction_length * radius * np.sin(theta))
                            )
                        else:
                            direction = (
                                (agent_state.px, agent_state.py),
                                (agent_state.px + direction_length * radius * np.cos(agent_state.theta),
                                 agent_state.py + direction_length * radius * np.sin(agent_state.theta))
                            )
                        orientation.append(direction)
                    orientations.append(orientation)
                    if i == 0:
                        robot_arrow_color = 'red'
                        arrows = [patches.FancyArrowPatch(
                            *orientation[0],
                            color=robot_arrow_color,
                            arrowstyle=arrow_style
                        )]
                    else:
                        human_arrow_color = 'red'
                        arrows.extend([patches.FancyArrowPatch(
                            *orientation[0],
                            color=human_arrow_color,
                            arrowstyle=arrow_style
                        )])
                for arrow in arrows:
                    ax.add_artist(arrow)

                global_step = 0

            def update(frame_num):
                nonlocal global_step
                nonlocal arrows
                # nonlocal scores
                global_step = frame_num
                robot.center = robot_positions[frame_num]  # update the position of robot in the animation
                if self.human_num > 0:
                    for i, human in enumerate(humans):
                        human.center = human_positions[frame_num][i]  # update the position of humans
                        if display_numbers:
                            human_numbers[i].set_position((human.center[0], human.center[1]))  # update the position of human numbers

                # update the headings of the agents
                for arrow in arrows:
                    arrow.remove()

                for i in range(self.human_num + 1):
                    orientation = orientations[i]
                    if i == 0:
                        arrows = [patches.FancyArrowPatch(
                            *orientation[frame_num],
                            color=robot_arrow_color,
                            arrowstyle=arrow_style
                        )]
                    else:
                        arrows.extend([patches.FancyArrowPatch(
                            *orientation[frame_num],
                            color=human_arrow_color,
                            arrowstyle=arrow_style
                        )])
                for arrow in arrows:
                    ax.add_artist(arrow)

                # update simulation time
                time.set_text('Time: {:.2f}'.format(frame_num * self.time_step))

            def on_click(event):
                if anim.running:
                    anim.event_source.stop()
                else:
                    anim.event_source.start()
                anim.running ^= True

            fig.canvas.mpl_connect('key_press_event', on_click)
            anim = animation.FuncAnimation(fig, update, frames=len(self.states), interval=self.time_step * 1000)
            anim.running = True

            if output_file is not None:
                # save as video
                ffmpeg_writer = animation.FFMpegWriter(fps=8, metadata=dict(artist='Me'), bitrate=1800)
                anim.save(output_file, writer=ffmpeg_writer)

                # save as gif
                # pillow_writer = animation.PillowWriter(fps=8, metadata=dict(artist='Me'), bitrate=1800)
                # anim.save(output_file, writer=pillow_writer)
            else:
                plt.show()
        elif mode == 'debug':
            def update():
                self.ax.clear()
                self.ax.tick_params(labelsize=12)
                self.ax.set_xlim(-self.panel_width / 2 - 1, self.panel_width / 2 + 1)
                self.ax.set_ylim(-self.panel_height / 2 - 0.5, self.panel_height / 2 + 0.5)
                self.ax.set_xlabel('x(m)', fontsize=14)
                self.ax.set_ylabel('y(m)', fontsize=14)
                robot_color = 'black'
                # add human positions and goals
                human_colors = [cmap(20) for i in range(len(self.humans))]
                # add robot and its goal
                goal = mlines.Line2D(
                    [self.robot.get_goal_position()[0]],
                    [self.robot.get_goal_position()[1]],
                    color='red',
                    marker='*',
                    linestyle='None',
                    markersize=15,
                    label='Goal'
                )
                robot = plt.Circle(self.robot.get_position(), self.robot.radius, fill=False, color=robot_color)
                self.ax.add_artist(robot)
                self.ax.add_artist(goal)

                for i in range(len(self.obstacles)):
                    obstacle = self.obstacles[i]
                    obstacle_mark = plt.Circle(obstacle.get_position(), obstacle.radius, fill=True, color='grey')
                    self.ax.add_artist(obstacle_mark)

                for i in range(len(self.walls)):
                    wall = self.walls[i]
                    wall_line = mlines.Line2D(
                        [wall.sx, wall.ex],
                        [wall.sy, wall.ey],
                        color='black',
                        marker='.',
                        linestyle='solid',
                        markersize=5
                    )
                    self.ax.add_artist(wall_line)

                direction_length = 1.0

                if len(self.humans) >= 0:
                    # add humans and their numbers
                    human_positions = [human.get_position() for human in self.humans]
                    humans = [
                        plt.Circle(
                            human_positions[i],
                            self.humans[i].radius,
                            fill=False,
                            color=human_colors[i]
                        ) for i in range(len(self.humans))
                    ]
                    # disable showing human numbers
                    if display_numbers:
                        human_numbers = [
                            plt.text(
                                humans[i].center[0] - x_offset,
                                humans[i].center[1] + y_offset,
                                str(i + 1),
                                color='black',
                                fontsize=12
                            ) for i in range(len(self.humans))
                        ]

                    for i, human in enumerate(humans):
                        self.ax.add_artist(human)
                        if display_numbers:
                            self.ax.add_artist(human_numbers[i])

                    # add time annotation
                    time = plt.text(0.4, 6.02, 'Time: {}'.format(self.global_time), fontsize=16)
                    self.ax.add_artist(time)

                    # compute orientation in each step and use arrow to show the direction
                    radius = self.robot.radius
                    orientation = []
                    for i in range(self.human_num + 1):
                        agent_state = self.robot if i == 0 else self.humans[i - 1]
                        if agent_state.kinematics == 'holonomic' or i != 0:
                            theta = np.arctan2(agent_state.vy, agent_state.vx)
                            direction = (
                                (agent_state.px, agent_state.py),
                                (agent_state.px + direction_length * radius * np.cos(theta),
                                 agent_state.py + direction_length * radius * np.sin(theta))
                            )
                            orientation.append(direction)
                        else:
                            direction = (
                                (agent_state.px, agent_state.py),
                                (agent_state.px + direction_length * radius * np.cos(agent_state.theta),
                                 agent_state.py + direction_length * radius * np.sin(agent_state.theta))
                            )
                            orientation.append(direction)
                        if i == 0:
                            robot_arrow_color = 'red'
                            arrows = [patches.FancyArrowPatch(
                                *orientation[0],
                                color=robot_arrow_color,
                                arrowstyle=arrow_style
                            )]
                        else:
                            human_arrow_color = 'black'
                            arrows.extend([patches.FancyArrowPatch(
                                *orientation[i],
                                color=human_arrow_color,
                                arrowstyle=arrow_style
                            )])
                    for arrow in arrows:
                        self.ax.add_artist(arrow)

                for candidate_goal, mask_value in zip(self.action_mask_vis, self.action_mask):
                    if mask_value == 0.0:
                        self.ax.plot(candidate_goal[0], candidate_goal[1], 'ro')
                    else:
                        self.ax.plot(candidate_goal[0], candidate_goal[1], 'go')

                if local_goal_vis is not None:
                    self.ax.plot(local_goal_vis[0], local_goal_vis[1], 'bo')

            # end def update

            if self.fig is None:
                self.fig, self.ax = plt.subplots(figsize=(7, 7))
                self.ax.tick_params(labelsize=12)
                self.ax.set_xlim(-self.panel_width / 2, self.panel_width / 2)
                self.ax.set_ylim(-self.panel_height / 2 - 0.5, self.panel_height / 2 + 0.5)
                self.ax.set_xlabel('x(m)', fontsize=14)
                self.ax.set_ylabel('y(m)', fontsize=14)

                plt.ion()
                show_human_start_goal = False

                robot_color = 'black'
                human_colors = [cmap(20) for i in range(len(self.humans))]

                # add robot and its goal
                robot = plt.Circle(self.robot.get_position(), self.robot.radius, fill=False, color=robot_color)
                self.ax.add_artist(robot)

                goal = mlines.Line2D(
                    [self.robot.get_goal_position()[0]],
                    [self.robot.get_goal_position()[1]],
                    color='red',
                    marker='*',
                    linestyle='None',
                    markersize=15,
                    label='Goal'
                )

                self.ax.add_artist(goal)

                for i in range(len(self.obstacles)):
                    obstacle = self.obstacles[i]
                    obstacle_mark = plt.Circle(obstacle.get_position(), obstacle.radius, fill=True, color='grey')
                    self.ax.add_artist(obstacle_mark)

                for i in range(len(self.walls)):
                    wall = self.walls[i]
                    wall_line = mlines.Line2D(
                        [wall.sx, wall.ex],
                        [wall.sy, wall.ey],
                        color='black',
                        marker='.',
                        linestyle='solid',
                        markersize=5
                    )
                    self.ax.add_artist(wall_line)
                direction_length = 1.0

                if len(self.humans) == 0:
                    print('no human')
                else:
                    # add humans and their numbers
                    human_positions = [human.get_position() for human in self.humans]
                    humans = [
                        plt.Circle(
                            human_positions[i],
                            self.humans[i].radius,
                            fill=False,
                            color=human_colors[i]
                        ) for i in range(len(self.humans))
                    ]
                    # disable showing human numbers
                    if display_numbers:
                        human_numbers = [
                            plt.text(
                                humans[i].center[0] - x_offset,
                                humans[i].center[1] + y_offset,
                                str(i + 1),
                                color='black',
                                fontsize=12
                            ) for i in range(len(self.humans))
                        ]

                    for i, human in enumerate(humans):
                        self.ax.add_artist(human)
                        if display_numbers:
                            self.ax.add_artist(human_numbers[i])

                    # add time annotation
                    time = plt.text(0.4, 1.02, 'Time: {}'.format(self.global_time), fontsize=16)
                    self.ax.add_artist(time)

                    # compute orientation in each step and use arrow to show the direction
                    radius = self.robot.radius
                    orientation = []
                    for i in range(self.human_num + 1):
                        agent_state = self.robot if i == 0 else self.humans[i - 1]
                        if agent_state.kinematics == 'holonomic' or i != 0:
                            theta = np.arctan2(agent_state.vy, agent_state.vx)
                            direction = (
                                (agent_state.px, agent_state.py),
                                (agent_state.px + direction_length * radius * np.cos(theta),
                                 agent_state.py + direction_length * radius * np.sin(theta))
                            )
                            orientation.append(direction)
                        else:
                            direction = (
                                (agent_state.px, agent_state.py),
                                (agent_state.px + direction_length * radius * np.cos(agent_state.theta),
                                 agent_state.py + direction_length * radius * np.sin(agent_state.theta))
                            )
                            orientation.append(direction)
                        if i == 0:
                            robot_arrow_color = 'red'
                            arrows = [patches.FancyArrowPatch(
                                *orientation[0],
                                color=robot_arrow_color,
                                arrowstyle=arrow_style
                            )]
                        else:
                            human_arrow_color = 'black'
                            arrows.extend([patches.FancyArrowPatch(
                                *orientation[i],
                                color=human_arrow_color,
                                arrowstyle=arrow_style
                            )])
                    for arrow in arrows:
                        self.ax.add_artist(arrow)

                for candidate_goal, mask_value in zip(self.action_mask_vis, self.action_mask):
                    if mask_value == 0.0:
                        self.ax.plot(candidate_goal[0], candidate_goal[1], 'ro')
                    else:
                        self.ax.plot(candidate_goal[0], candidate_goal[1], 'go')

                if local_goal_vis is not None:
                    self.ax.plot(local_goal_vis[0], local_goal_vis[1], 'bo')

                plt.pause(0.001)
            else:
                update()
                plt.pause(0.001)
        else:
            raise NotImplementedError
