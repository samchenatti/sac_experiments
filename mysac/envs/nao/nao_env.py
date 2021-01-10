from typing import List, Tuple

import numpy as np
from gym import Env, spaces
from mysac.samplers.sampler import BasicTrajectorySampler
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

NAO_JOINT_LIMITS = [
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
    SCENES_FOLDER = ('/home/samuel/Develop/IC/my_own_sac/mysac/envs/'
                     'coppelia_scenes/')

    SCENE_FILE = 'nao_walk_original.ttt'

    def __init__(self, headless: bool = True, *args, **kwargs):
        self.all_joints: List[Joint] = []
        self.foot_sensors: List[ForceSensor] = []
        self.joint_limits = []

        self.pr = PyRep()
        self.pr.launch(self.SCENES_FOLDER + self.SCENE_FILE, headless=headless)
        self.pr.start()

        self.head: Shape = None
        self.chest: Shape = None

        self.load_joints()
        self.load_shapes()

        self.action_space = spaces.Box(
            low=np.array(len(self.joint_limits) * [-1]),
            high=np.array(len(self.joint_limits) * [1]),
        )

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


class WalkingNao(NAO, Env):
    """
    Environment based on NAO Robot where the objective is to move forward
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.position_history: List[Tuple[float, float]] = []

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

        return np.array(
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

    def reset(self, random_initialization: bool = True) -> np.array:
        """
        Reset the env to the initial state, optionally doing a random action

        Args:
            random_initialization: if True, will execute a random action from
                the action_space attribute
        """
        self.position_history = []

        self.pr.stop()
        self.pr.start()

        if random_initialization:
            self.step(action=self.action_space.sample())

        return self.get_observation()

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

    def get_reward(self) -> float:
        """
        Returns the reward signal in the current simulation state
        """
        last_position = getattr(
            self,
            'last_position',
            (-6.2544e-08, -1.1951e+01)
        )

        self.position_history.append(last_position)

        x, y, _ = self.chest.get_position()

        delta_x = x - last_position[0]
        delta_y = y - last_position[1]

        delta_x *= 500 if delta_x > 0 else 1000
        delta_y = -250 * np.abs(delta_y)

        reward = delta_x + delta_y

        x_orientation, _, z_orientation = self.chest.get_orientation()

        if x_orientation < -0.4 or x_orientation > 0.4:
            reward *= 0.8

        if z_orientation < -0.4 or z_orientation > 0.4:
            reward *= 0.8

        self.last_position = (x, y)

        return reward

    def step(self, action: np.array) -> Tuple[np.array, float, bool, str]:
        """
        Executes an action in the environment

        Args:
            action: an array of shape action_space.shape

        Returns:
            A tuple containing the current state, the reward signal, the done
            signal and an optional information string
        """
        done = self.head.get_position()[-1] < 0.35

        action = vectorized_to_interval(
            limits=self.joint_limits,
            actions=action
        )

        for target_position, joint in zip(action, self.all_joints):
            joint.set_joint_target_position(target_position)

        self.pr.step()

        return self.get_observation(), self.get_reward(), done, ''


if __name__ == '__main__':
    env = WalkingNao(headless=False)
    print("Action space:", env.action_space)

    for episode in range(2):
        env.reset()

        for step in range(100):
            env.step(action=env.action_space.sample())
