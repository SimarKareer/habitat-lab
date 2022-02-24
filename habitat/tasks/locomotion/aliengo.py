import math

import magnum as mn
import numpy as np

from habitat.utils.geometry_utils import wrap_heading
from habitat_sim.physics import JointMotorSettings


def attribute_to_str(attr):
    return attr.__repr__().split(" at ")[0].split(".")[-1]


class VectorCachableProperty(property):
    def __init__(self, attribute):
        super().__init__(attribute)
        # Use the name of the attribute as the cache key
        self.cache_key = attribute_to_str(attribute)


class IterateAll:
    def __init__(self, attr):
        self.function = attr
        self.attr_str = attribute_to_str(attr)

    def __call__(self, *args, **kwargs):
        # return self.function
        self.function(*args, **kwargs)


class AlienGo:
    def __init__(
        self,
        robot_id,
        sim,
        fixed_base,
        robot_cfg,
        reset_position=mn.Vector3(0.0, 0.5, 0.0),
    ):
        self.robot_id = robot_id
        self._sim = sim
        self.fixed_base = fixed_base

        self.jmsIdxToJoint = [
            "FL_hip",
            "FL_thigh",
            "FL_calf",
            "FR_hip",
            "FR_thigh",
            "FR_calf",
            "RL_hip",
            "RL_thigh",
            "RL_calf",
            "RR_hip",
            "RR_thigh",
            "RR_calf",
        ]

        # joint position limits
        self.standing_pose = np.array(robot_cfg.STANDING_POSE * 4)
        self.joint_limits_stand = np.array(robot_cfg.JOINT_LIMITS_STAND * 4)
        self.joint_limits_upper = np.array(robot_cfg.JOINT_LIMITS.UPPER * 4)
        self.joint_limits_lower = np.array(robot_cfg.JOINT_LIMITS.LOWER * 4)
        self.reset_position = reset_position

    @VectorCachableProperty
    def height(self):
        # Translation is [y, z, x]
        return self.robot_id.rigid_state.translation[1]

    @VectorCachableProperty
    def joint_velocities(self) -> np.ndarray:
        return np.array(self.robot_id.joint_velocities, dtype=np.float32)

    @VectorCachableProperty
    def joint_positions(self) -> np.ndarray:
        return np.array(self.robot_id.joint_positions, dtype=np.float32)

    @VectorCachableProperty
    def position(self) -> np.ndarray:
        """translation in global frame"""
        return self.robot_id.transformation.translation

    @VectorCachableProperty
    def velocity(self) -> mn.Vector3:
        """velocity in global frame"""
        return self.robot_id.root_linear_velocity

    @VectorCachableProperty
    def local_velocity(self):
        """returns local velocity and corrects for initial rotation of aliengo
        [forward, right, up]
        """
        local_vel = self.robot_id.transformation.inverted().transform_vector(
            self.velocity
        )
        return np.array([local_vel[0], local_vel[2], -local_vel[1]])

    @VectorCachableProperty
    def joint_torques(self) -> np.ndarray:
        phys_ts = self._sim.get_physics_time_step()
        torques = np.array(self.robot_id.get_joint_motor_torques(phys_ts))
        return torques

    @VectorCachableProperty
    def rpy(self):
        """Given a numpy quaternion we'll return the roll pitch yaw

        :return: rpy: tuple of roll, pitch yaw
        """
        quat = self.robot_id.rotation.normalized()
        undo_rot = mn.Quaternion(
            ((np.sin(np.deg2rad(45)), 0.0, 0.0), np.cos(np.deg2rad(45)))
        ).normalized()
        quat = quat * undo_rot

        x, y, z = quat.vector
        w = quat.scalar

        roll, pitch, yaw = self._euler_from_quaternion(x, y, z, w)
        rpy = wrap_heading(np.array([roll, pitch, yaw]))
        return rpy

    @property
    def rp(self):
        return self.rpy[..., :2]

    @property
    def forward_velocity(self) -> float:
        """local forward velocity"""
        return self.local_velocity[..., 0]

    @property
    def side_velocity(self) -> float:
        """local side_velocity"""
        return self.local_velocity[..., 2]

    @IterateAll
    def set_pose_jms(self, pose, kinematic_snap=True, **kwargs):
        """Sets a robot's pose and changes the jms to that pose (rests at
        given position)
        """
        # Snap joints kinematically
        if kinematic_snap:
            self.robot_id.joint_positions = pose

        # Make motor controllers maintain this position
        for idx, p in enumerate(pose):
            self.robot_id.update_joint_motor(idx, self._new_jms(p))

    @IterateAll
    def reset(self, yaw=0, **kwargs):
        """Resets robot's movement, moves it back to center of platform"""
        # Zero out the link and root velocities
        self.robot_id.clear_joint_states()
        self.robot_id.root_angular_velocity = mn.Vector3(0.0, 0.0, 0.0)
        self.robot_id.root_linear_velocity = mn.Vector3(0.0, 0.0, 0.0)

        # Roll robot 90 deg
        base_transform = mn.Matrix4.rotation(
            mn.Rad(np.deg2rad(-90)), mn.Vector3(1.0, 0.0, 0.0)
        ) @ mn.Matrix4.rotation(
            mn.Rad(np.deg2rad(yaw)), mn.Vector3(0.0, 0.0, 1.0)
        )

        # Position above center of platform; slightly higher if fixed_base
        base_transform.translation = (
            self.reset_position + mn.Vector3(0.0, 0.3, 0.0)
            if self.fixed_base
            else self.reset_position
        )
        self.robot_id.transformation = base_transform

    @IterateAll
    def add_jms_pos(self, joint_pos):
        """Updates existing joint positions by adding each position in array of
        joint_positions

        :param joint_pos: array of delta joint positions
        """
        for i, new_pos in enumerate(joint_pos):
            jms = self._jms_copy(self.robot_id.get_joint_motor_settings(i))
            jms.position_target = np.clip(
                wrap_heading(jms.position_target + new_pos),
                self.joint_limits_lower[i],
                self.joint_limits_upper[i],
            )
            self.robot_id.update_joint_motor(i, jms)

    def prone(self, **kwargs):
        self.set_pose_jms(np.array([0, 1.3, -2.5] * 4), **kwargs)

    def stand(self, **kwargs):
        self.set_pose_jms(self.standing_pose, **kwargs)

    def get_feet_contacts(self):
        """THIS ASSUMES THAT THERE IS ONLY ONE ROBOT IN THE SIM
        Returns np.array size 4, either 1s or 0s, for FL FR RL RR feet.
        """
        contacts = self._sim.get_physics_contact_points()
        contacting_feet = set()
        for c in contacts:
            for link in [c.link_id_a, c.link_id_b]:
                contacting_feet.add(self.robot_id.get_link_name(link))
        feet = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        return np.array([1 if foot in contacting_feet else 0 for foot in feet])

    def _jms_copy(self, jms):
        """Returns a deep copy of a jms

        :param jms: the jms to copy
        """
        return JointMotorSettings(
            jms.position_target,
            jms.position_gain,
            jms.velocity_target,
            jms.velocity_gain,
            jms.max_impulse,
        )

    def _new_jms(self, pos):
        """Returns a new jms with default settings at a given position

        :param pos: the new position to set to
        """
        return JointMotorSettings(
            pos,  # position_target
            0.03,  # position_gain
            0.0,  # velocity_target
            1.8,  # velocity_gain
            1.0,  # max_impulse
        )

    @staticmethod
    def _euler_from_quaternion(x, y, z, w):
        """Convert a quaternion into euler angles (roll, yaw, pitch)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = 2.0 * (w * x + y * z)
        t1 = 1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)

        t2 = 2.0 * (w * y - z * x)
        t2 = 1.0 if t2 > 1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)

        t3 = 2.0 * (w * z + x * y)
        t4 = 1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)

        return roll_x, -yaw_z, pitch_y  # in radians
