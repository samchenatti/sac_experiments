import random
import time
from typing import List, Optional, Tuple

import numpy as np
# from gym import Env, spaces
from gymnasium import Env, spaces
from mysac.envs.nao.constants import STATES_TO_DROP
from pyrep import PyRep
from pyrep.backend.sim import simGetObjectVelocity
from pyrep.objects.force_sensor import ForceSensor
from pyrep.objects.joint import Joint
from pyrep.objects.shape import Shape

NAO_JOINT_NAMES = [
    "HipYawPitch3",
    "HipRoll3",
    "HipPitch3",
    "KneePitch3",
    "AnklePitch3",
    "AnkleRoll3",
    "ShoulderPitch3",
    "ShoulderRoll3",
    "ElbowYaw3",
    "ElbowRoll3"
]

MAX_VELOCITY = 50e-3

NAO_DEG_JOINT_LIMITS = [
    (-6.562e+01, 1.081e+02),
    (-2.174e+01, 6.607e+01),
    (-1.016e+02, 1.263e+02),
    (-5.291e+00, 1.263e+02),
    (-6.798e+01, 1.208e+02),
    (-2.227e+01, 6.730e+01),
    (-1.195e+02, 2.390e+02),
    (-1.800e+01, 9.401e+01),
    (-1.195e+02, 2.390e+02),
    (-8.850e+01, 8.650e+01)
]

NAO_JOINT_LIMITS = [
    (np.radians(limit), np.radians(joint_range))
    for limit, joint_range in NAO_DEG_JOINT_LIMITS
]

NAO_JOINT_LIMIT_MAP = {
    name: limits for name, limits in zip(NAO_JOINT_NAMES, NAO_JOINT_LIMITS)
}

FOOT_SENSOR_NAMES = [
    # Left foot
    'NAO_LFsrFL',
    'NAO_LFsrFR',
    'NAO_LFsrRL',
    'NAO_LFsrRR',

    # Right foot
    'NAO_RFsrFL',
    'NAO_RFsrFR',
    'NAO_RFsrRL',
    'NAO_RFsrRR',
]

ENERGY_COST_THRESHOLD = 1


def vectorized_to_interval(limits: np.array, actions: np.array) -> np.array:
    """
    Converts a vector of actions in the range (-1, 1) to the range given by
    limits array

    Args:
        limits: the limits for which the actions will be scaled
        actions: the action in the range (-1, 1)
    """
    # Actions ranges in (-1, 1)
    min_, max_ = -1, 1
    a, b = limits[:, 0], limits[:, 1]

    return ((b - a)*(actions - min_)/(max_ - min_)) + a


class NAO:
    """
    Basic interface for NAO Robot in the PyRep lib

    Args:
        headless: if True, will not display the CoppeliaSim interface
    """
    SCENES_FOLDER = (
        '/home/figo/Develop/IC/sac_experiments/mysac/envs/coppelia_scenes/')

    SCENE_FILE = 'nao_walk_original.ttt'

    def __init__(self, headless: bool = True,
                 energy_cost_threshold: float = ENERGY_COST_THRESHOLD,
                 pomdp_states: Optional[List[str]] = None,
                 pomdp_mode: str = 'drop',
                 *args, **kwargs):
        print('Energy cost:', energy_cost_threshold)

        print('POMDP:', pomdp_mode, pomdp_states)

        self.all_joints: List[Joint] = []
        self.foot_sensors: List[ForceSensor] = []
        self.joint_limits = []
        self.energy_cost_threshold = energy_cost_threshold

        print('Before PyRep start')
        self.pr = PyRep()
        self.pr.launch(self.SCENES_FOLDER + self.SCENE_FILE, headless=headless)
        self.pr.start()
        print('After PyRep start')

        self.head: Shape = None
        self.chest: Shape = None

        self.total_steps = 0

        self.load_joints()
        self.load_shapes()

        self.action_space = spaces.Box(
            low=np.array(len(self.joint_limits) * [-1]),
            high=np.array(len(self.joint_limits) * [1]),
        )

        self.observation_space = spaces.Box(
            low=np.array(30 * [np.inf]),
            high=np.array(30 * [np.inf])
        )

        self.pomdp_states = pomdp_states
        self.pomdp_mode = pomdp_mode

    def load_force_sensors(self) -> None:
        """
        Load force sensors into foot_sensors attribute
        """
        for sensor_name in FOOT_SENSOR_NAMES:
            self.foot_sensors.append(
                ForceSensor(name_or_handle=sensor_name)
            )

    def load_joints(self) -> None:
        """
        Load the joints from NAO_JOINT_NAMES, building also the
        NAO_JOINT_LIMITS attribute
        """
        for joint_name, limits in zip(NAO_JOINT_NAMES, NAO_JOINT_LIMITS):
            for side in ['L', 'R']:
                joint_full_name = side + joint_name

                joint = Joint(name_or_handle=joint_full_name)

                setattr(self, joint_full_name, joint)

                self.joint_limits.append((limits[0], limits[1] + limits[0]))

                self.all_joints.append(joint)

        self.joint_limits = np.array(self.joint_limits)

    def load_shapes(self) -> None:
        """
        Load some of NAO shapes for retrieving states
        """
        self.head = Shape(name_or_handle='HeadPitch_link_respondable')
        self.chest = Shape(name_or_handle='imported_part_20_sub0')

        self.l_foot = Shape(
            name_or_handle='l_sole_link_pure3'
        )

        self.r_foot = Shape(
            name_or_handle='r_sole_link_pure9'
        )


class WalkingNao(NAO, Env):
    """
    Environment based on NAO Robot where the objective is to move forward
    """

    def __init__(self, use_energy_cost: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.position_history: List[Tuple[float, float]] = []
        self.delayed_energy_cost = []
        self.past_joint_velocities = []

        self.use_delayed_energy_cost = True

    def get_observation(self) -> np.array:
        """
        Returns an Numpy Array with the env observations
        """
        _, _, head_z = self.head.get_position()

        orientation_x, orientation_y, orientation_z = \
            self.chest.get_orientation()

        linear, angular = simGetObjectVelocity(objectHandle=self.chest._handle)

        joint_positions = [
            joint.get_joint_position() for joint in self.all_joints
        ]

        observation = np.array(
            (
                head_z,
                orientation_x,
                orientation_y,
                orientation_z,
                *linear,
                *angular,
                *joint_positions,
                *self.get_foot_sensor_signal()
            ),
            dtype='float32'
        )

        if self.pomdp_states:
            if self.pomdp_mode == 'drop':
                for state_to_drop in self.pomdp_states:
                    observation[STATES_TO_DROP[state_to_drop]] = 0

            elif self.pomdp_mode == 'noise':
                for state_to_drop in self.pomdp_states:
                    slice_to_drop: slice = STATES_TO_DROP[state_to_drop]

                    noise = np.random.rand(
                        slice_to_drop.stop - slice_to_drop.start)

                    noise = observation[slice_to_drop] * noise

                    observation[slice_to_drop] += noise

        return observation

    def reset(self, random_initialization: bool = True, seed=None) -> np.array:
        """
        Reset the env to the initial state, optionally doing a random action

        Args:
            random_initialization: if True, will execute a random action from
                the action_space attribute
        """
        self.position_history = []
        self.past_joint_velocities = []
        self.delayed_energy_cost = []
        self.random_initialization = random_initialization

        self.pr.stop()
        self.pr.start()

        if self.random_initialization:
            self.step(action=self.action_space.sample())

        self.total_steps = 0

        return self.get_observation(), {}

    def get_foot_sensor_signal(self) -> List[float]:
        """
        Returns the signal for all the foot force sensors
        """
        readings = []

        for sensor in self.foot_sensors:
            force, torque = sensor.read()

            readings.extend(force)
            readings.extend(torque)

        return readings

    def knees_at_limit(self) -> int:
        """
        Retuns the number of knees whose speed or position is at least at 99%
        of its range
        """
        KNEE_JOINT_NAME = 'KneePitch3'

        knees_at_limit = 0

        position_min, position_range = NAO_JOINT_LIMIT_MAP[KNEE_JOINT_NAME]

        for side in ['L', 'R']:
            joint_name = side + KNEE_JOINT_NAME

            joint: Joint = getattr(self, joint_name)

            # Relative speed
            joint_velocity_limit = joint.get_joint_upper_velocity_limit()
            joint_velocity = joint.get_joint_velocity()
            joint_relative_speed = joint_velocity/joint_velocity_limit

            # Relative position
            joint_position = joint.get_joint_position() - position_min
            joint_relative_position = joint_position / position_range

            if joint_relative_speed > 0.95:
                print(f'{joint_name} is over 95% velocity:',
                      joint.get_joint_velocity(),
                      ', limit:',
                      joint_velocity_limit)
                knees_at_limit += 1

            if joint_relative_position > 0.95:
                print(f'{joint_name} is over 95% position:',
                      joint.get_joint_position())
                knees_at_limit += 1

        return knees_at_limit

    def get_reward(self) -> float:
        """
        Returns the reward signal in the current simulation state
        """
        last_position = getattr(
            self,
            'last_position',
            (-2.32, -1.19)
        )

        self.position_history.append(last_position)

        x, y, _ = self.chest.get_position()

        delta_x = np.clip(x - last_position[0], -MAX_VELOCITY, MAX_VELOCITY)
        delta_y = np.clip(y - last_position[1], -MAX_VELOCITY, MAX_VELOCITY)

        delta_x *= 500 if delta_x > 0 else 1000
        delta_y = -250 * np.abs(delta_y)

        reward = delta_x + delta_y

        x_orientation, _, z_orientation = self.chest.get_orientation()

        if x_orientation < -0.4 or x_orientation > 0.4:
            reward *= 0.8

        if z_orientation < -0.4 or z_orientation > 0.4:
            reward *= 0.8

        if self.foots_collision():
            reward *= 0.8

        self.last_position = (x, y)

        return reward

    def foots_collision(self) -> bool:
        """
        Returns true if the NAO foot respondable parts collide
        """
        return self.l_foot.check_collision(self.r_foot)

    def compute_delayed_energy_cost(self) -> float:
        """
        Returns the energy cost of the >last< action
        """
        if not self.past_joint_velocities:
            return

        # The current velocity is due to the last action
        current_velocity = np.array(
            [j.get_joint_velocity() for j in self.all_joints]
        )

        past_joint_velocities = np.array(
            self.past_joint_velocities
        )

        cost = np.abs(current_velocity - past_joint_velocities).mean()

        self.delayed_energy_cost.append(cost)

    def step(self, action: np.array) -> Tuple[np.array, float, bool, str]:
        """
        Executes an action in the environment

        Args:
            action: an array of shape action_space.shape

        Returns:
            A tuple containing the current state, the reward signal, the done
            signal and an optional information string
        """
        self.total_steps += 1

        x, y, z = self.head.get_position()

        done = z < 0.35
        done |= 2.15 < y < -2.15
        done |= -2.32 > x > 2.5

        action = vectorized_to_interval(
            limits=self.joint_limits,
            actions=action
        )

        for target_position, joint in zip(action, self.all_joints):
            joint.set_joint_target_position(target_position)

        self.pr.step()

        self.compute_delayed_energy_cost()

        # We use this for energy cost
        self.past_joint_velocities = [
            j.get_joint_velocity() for j in self.all_joints
        ]

        return self.get_observation(), self.get_reward(), False, done, {}

    def post_episode_sampling_callback(
            self, rewards: List[np.array]) -> List[np.array]:
        """
        Called by the sampler if the reward must be calculated with a delayed
        effect

        Args:
            rewards: the rewards originally collected for the episode
        """
        if not self.use_delayed_energy_cost:
            return rewards

        self.compute_delayed_energy_cost()

        delayed_energy_cost = self.delayed_energy_cost
        if self.random_initialization:
            delayed_energy_cost = delayed_energy_cost[1:]

        new_rewards = []
        for reward, energy_cost in zip(rewards, self.delayed_energy_cost):
            if energy_cost > self.energy_cost_threshold:
                new_rewards.append(0.8 * reward)

            else:
                new_rewards.append(reward)

        return new_rewards


if __name__ == '__main__':
    env = WalkingNao(headless=False)
    print("Action space:", env.action_space)

    for episode in range(2):
        env.reset()

        for step in range(100):
            env.step(action=env.action_space.sample())
