import math
import magnum as mn
import numpy as np
from habitat_sim.physics import JointMotorSettings


from habitat.utils.geometry_utils import wrap_heading

class AlienGo:
    def __init__(self, robot_id, sim, fixed_base, task_config):
        self.robot_id = robot_id
        self._sim = sim
        self.fixed_base = fixed_base
        self.task_config = task_config

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
        self.joint_limits_lower = np.array(
            [-0.1, -np.pi / 3, -5 / 6 * np.pi] * 4
        )
        self.joint_limits_upper = np.array([0.1, np.pi / 2.1, -np.pi / 4] * 4)

    @property
    def height(self):
        # Translation is [y, z, x]
        return self.robot_id.rigid_state.translation[1]

    @property
    def joint_velocities(self) -> np.ndarray:
        return np.array(self.robot_id.joint_velocities, dtype=np.float32)

    @property
    def joint_positions(self) -> np.ndarray:
        return np.array(self.robot_id.joint_positions, dtype=np.float32)

    def set_joint_positions(self, pose):
        """This is kinematic! Not dynamic."""
        self.robot_id.joint_positions = wrap_heading(pose)

    def get_feet_contacts(self):
        """THIS ASSUMES THAT THERE IS ONLY ONE ROBOT IN THE SIM
        Returns np.array size 4, either 1s or 0s, for FL FR RL RR feet.
        """
        contacts = self._sim.get_physics_contact_points()
        contacting_feet = set()
        for c in contacts:
            for link in [c.link_id_a, c.link_id_b]:
                contacting_feet.add(self.robot_id.get_link_name(link))
        return np.array(
            [
                1 if foot in contacting_feet else 0
                for foot in ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
            ]
        )

    def _jms_copy(self, jms):
        """Returns a deep copy of a jms

        Args:
            jms: the jms to copy
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

        Args:
            pos: the new position to set to
        """
        return JointMotorSettings(
            pos,  # position_target
            0.6,  # position_gain
            0.0,  # velocity_target
            1.5,  # velocity_gain
            1.0,  # max_impulse
        )

    def set_pose_jms(self, pose):
        f""" Sets a robot's pose and changes the jms to that pose (rests at given position)
        """
        # Snap joints kinematically
        self.robot_id.joint_positions = pose

        # Make motor controllers maintain this position
        for idx, p in enumerate(pose):
            self.robot_id.update_joint_motor(idx, self._new_jms(p))

    def prone(self):
        self.set_pose_jms(np.array([0, 0.432, -0.77] * 4))
        
    def stand(self):
        self.set_pose_jms(np.array([0, 1.3, -2.5] * 4))

    def reset(self):
        f"""
            Reset's all the robots movemement, moves it back to center of platform, 
        """
        # Zero out the link and root velocities
        self.robot_id.clear_joint_states()
        self.robot_id.root_angular_velocity = mn.Vector3(0.0, 0.0, 0.0)
        self.robot_id.root_linear_velocity = mn.Vector3(0.0, 0.0, 0.0)

        # Roll robot 90 deg
        base_transform = mn.Matrix4.rotation(
            mn.Rad(np.deg2rad(-90)), mn.Vector3(1.0, 0.0, 0.0)
        )

        # Position above center of platform
        base_transform.translation = (
            mn.Vector3(0.0, 0.8, 0.0)
            if self.fixed_base
            else mn.Vector3(0.0, 0.3, 0.0)
        )
        self.robot_id.transformation = base_transform

        # Reset all jms
        # base_jms = self._new_jms(0)
        # for i in range(12):
        #     self.robot_id.update_joint_motor(i, base_jms)

        # self._set_joint_type_pos("thigh", self.task_config.TASK.START.THIGH)
        # self._set_joint_type_pos("calf", self.task_config.TASK.START.CALF)

    def _set_joint_type_pos(self, joint_type, joint_pos):
        """Sets all joints of a given type to a given position

        Args:
            joint_type: type of joint ie hip, thigh or calf
            joint_pos: position to set these joints to
        """
        for idx, joint_name in enumerate(self.jmsIdxToJoint):
            if joint_type in joint_name:
                self.robot_id.update_joint_motor(idx, self._new_jms(joint_pos))

    def add_jms_pos(self, joint_pos):
        """
        Updates existing joint positions by adding each position in array of
        joint_positions
        Args
            joint_pos: array of delta joint positions
        """
        for i, new_pos in enumerate(joint_pos):
            jms = self._jms_copy(self.robot_id.get_joint_motor_settings(i))
            jms.position_target = np.clip(
                wrap_heading(jms.position_target + new_pos),
                self.joint_limits_lower[i],
                self.joint_limits_upper[i],
            )
            self.robot_id.update_joint_motor(i, jms)

    def get_rpy(self):
        """Given a numpy quaternion we'll return the roll pitch yaw

        Returns:
            rpy: tuple of roll, pitch yaw
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

    def _euler_from_quaternion(self, x, y, z, w):
        """
        Convert a quaternion into euler angles (roll, yaw, pitch)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)

        return roll_x, -yaw_z, pitch_y  # in radians
