import argparse
from habitat_baselines.common.environments import LocomotionRLEnvStand, LocomotionRLEnvEnergy
from habitat_baselines.config.default import get_config
import numpy as np
import cv2
import magnum as mn

RENDER = False

def test_forward():
    """
        assert that local vx and global fx are both aligned (positive = forward).  assert that forward velocity reward is close to 0 (good)
    """
    parser = argparse.ArgumentParser()

    config = get_config("./habitat_baselines/config/locomotion/ddppo_energy.yaml")
    config.defrost()
    config.TASK_CONFIG.DEBUG.FIXED_BASE = False
    config.TASK_CONFIG.TASK.TARGET_VELOCITY = 1
    config.freeze()
    env = LocomotionRLEnvEnergy(config=config, render=RENDER)
    obs = env.reset()
    env._sim.set_gravity([0., 0., 0.])
        # Position above center of platform

    base_transform = mn.Matrix4.rotation(
        mn.Rad(np.deg2rad(-90)), mn.Vector3(1.0, 0.0, 0.0)
    )

    base_transform.translation = (
        mn.Vector3(0.0, 0.6, 0.0)
    )

    env.robot.robot_id.transformation = base_transform

    global_vel = np.zeros((40,3))
    local_vel = np.zeros((40,3))
    vx_rewards = np.zeros(40)
    for i in range(40):
        action = np.zeros(12)
        env.robot.robot_id.root_linear_velocity = env.robot.robot_id.transformation.inverted().transform_vector(mn.Vector3(1., 0., 0.))
        env.robot.robot_id.root_angular_velocity = mn.Vector3(0., 0., 0.)
        # env.robot.robot_id.root_linear_velocity = mn.Vector3(1., 0., 0.)

        action_args = {"action_args": {"joint_deltas": action}}
        action_args = {"joint_deltas": action}
        if i == 39:
            obs, reward, done, info = env.step("null", action_args, step_render=RENDER)
        obs, reward, done, info = env.step("null", action_args, step_render=False)

        global_vel[i] = env.robot.robot_id.root_linear_velocity
        local_vel[i] = env.robot.local_velocity
        vx_rewards[i] = env.named_rewards["forward_velocity_reward"]

    print("v global:", global_vel)
    print("v local:", local_vel)
    assert(np.all(global_vel[1:,0] > 0))
    assert(np.all(local_vel[1:,0] > 0))
    assert(np.all(np.abs(global_vel[1:,1]) < 0.1))
    assert(np.all(np.abs(local_vel[1:,1]) < 0.1))
    assert(np.all(np.abs(global_vel[1:,2]) < 0.1))
    assert(np.all(np.abs(local_vel[1:,2]) < 0.1))

    print("vx rewards: ", vx_rewards)
    assert(np.all(vx_rewards[1:] > -0.5))

def test_forward_rotated():
    """
        assert local velocity works when not aligned with global.  assert forward velocity is close to 0 (good)
    """
    parser = argparse.ArgumentParser()

    config = get_config("./habitat_baselines/config/locomotion/ddppo_energy.yaml")
    config.defrost()
    config.TASK_CONFIG.DEBUG.FIXED_BASE = False
    config.TASK_CONFIG.TASK.TARGET_VELOCITY = 1
    config.freeze()
    env = LocomotionRLEnvEnergy(config=config, render=RENDER)
    obs = env.reset()
    env._sim.set_gravity([0., 0., 0.])

    base_transform = mn.Matrix4.rotation(
        mn.Rad(np.deg2rad(-90)), mn.Vector3(1.0, 0.0, 0.0)
    ) @ mn.Matrix4.rotation(
        mn.Rad(np.deg2rad(90)), mn.Vector3(0.0, 0.0, 1.0)
    )
    base_transform.translation = (
        mn.Vector3(0.0, 0.6, 0.0)
    )
    
    env.robot.robot_id.transformation = base_transform


    global_vel = np.zeros((40,3))
    local_vel = np.zeros((40,3))
    vx_rewards = np.zeros(40)
    for i in range(40):
        action = np.zeros(12)
        env.robot.robot_id.root_linear_velocity = mn.Vector3(0., 0., 1.) #env.robot.robot_id.transformation.inverted().transform_vector(mn.Vector3(1., 0., 0.))
        env.robot.robot_id.root_angular_velocity = mn.Vector3(0., 0., 0.)
        # env.robot.robot_id.root_linear_velocity = mn.Vector3(1., 0., 0.)

        action_args = {"action_args": {"joint_deltas": action}}
        action_args = {"joint_deltas": action}
        if i == 39:
            obs, reward, done, info = env.step("null", action_args, step_render=RENDER)
        obs, reward, done, info = env.step("null", action_args, step_render=False)

        global_vel[i] = env.robot.velocity
        local_vel[i] = env.robot.local_velocity
        vx_rewards[i] = env.named_rewards["forward_velocity_reward"]

    print("v global:", global_vel)
    print("v local:", local_vel)
    assert(np.all(global_vel[1:,2] > 0.9))
    assert(np.all(local_vel[1:,0] < -0.9))
    assert(np.all(np.abs(global_vel[1:,0]) < 0.1))
    assert(np.all(np.abs(local_vel[1:,1]) < 0.1))
    assert(np.all(np.abs(global_vel[1:,1]) < 0.1))
    assert(np.all(np.abs(local_vel[1:,2]) < 0.1))

    print("vx rewards: ", vx_rewards)
    assert(np.all(vx_rewards[1:] < 30))

def test_side_reward():
    """
    assert that side local and global are aligned.  Assert that side reward is negative (bad)
    """
    parser = argparse.ArgumentParser()

    config = get_config("./habitat_baselines/config/locomotion/ddppo_energy.yaml")
    config.defrost()
    config.TASK_CONFIG.DEBUG.FIXED_BASE = False
    config.TASK_CONFIG.TASK.TARGET_VELOCITY = 1
    config.freeze()
    env = LocomotionRLEnvEnergy(config=config, render=RENDER)
    obs = env.reset()
    env._sim.set_gravity([0., 0., 0.])
        # Position above center of platform

    base_transform = mn.Matrix4.rotation(
        mn.Rad(np.deg2rad(-90)), mn.Vector3(1.0, 0.0, 0.0)
    )

    base_transform.translation = (
        mn.Vector3(0.0, 0.6, 0.0)
    )

    env.robot.robot_id.transformation = base_transform

    global_vel = np.zeros((40,3))
    local_vel = np.zeros((40,3))
    vx_rewards = np.zeros(40)
    for i in range(40):
        action = np.zeros(12)
        env.robot.robot_id.root_linear_velocity = mn.Vector3(0., 0., 1.)
        env.robot.robot_id.root_angular_velocity = mn.Vector3(0., 0., 0.)
        # env.robot.robot_id.root_linear_velocity = mn.Vector3(1., 0., 0.)

        action_args = {"action_args": {"joint_deltas": action}}
        action_args = {"joint_deltas": action}
        if i == 39:
            obs, reward, done, info = env.step("null", action_args, step_render=RENDER)
        obs, reward, done, info = env.step("null", action_args, step_render=False)

        global_vel[i] = env.robot.robot_id.root_linear_velocity
        local_vel[i] = env.robot.local_velocity
        vx_rewards[i] = env.named_rewards["side_velocity_reward"]

    print("v global:", global_vel)
    print("v local:", local_vel)
    assert(np.all(global_vel[1:,2] > 0.9))
    assert(np.all(local_vel[1:,2] > 0.9))
    assert(np.all(np.abs(global_vel[1:,1]) < 0.1))
    assert(np.all(np.abs(local_vel[1:,1]) < 0.1))
    assert(np.all(np.abs(global_vel[1:,0]) < 0.1))
    assert(np.all(np.abs(local_vel[1:,0]) < 0.1))

    print("vx rewards: ", vx_rewards)
    assert(np.all(vx_rewards[1:] < -0.5))

def test_angular_reward():
    parser = argparse.ArgumentParser()

    config = get_config("./habitat_baselines/config/locomotion/ddppo_energy.yaml")
    config.defrost()
    config.TASK_CONFIG.DEBUG.FIXED_BASE = False
    config.freeze()
    env = LocomotionRLEnvEnergy(config=config, render=RENDER)
    obs = env.reset()

    ang_rewards = np.zeros(50)
    for i in range(50):
        # action1 = np.array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1], dtype=np.float32)
        # action2 = np.array([0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0], dtype=np.float32)
        # action = action1 - action2*0.5
        action = np.zeros(12)

        env.add_force(0, 0, 100, link=1)

        action_args = {"action_args": {"joint_deltas": action}}
        action_args = {"joint_deltas": action}
        obs, reward, done, info = env.step("null", action_args)

        ang_rewards[i] = env.named_rewards["side_velocity_reward"]
        # print("Reward2: ", ang_rewards[i])
        # print("Reward: ", reward)
        # print("INFO: ", info)
        # print("OBS: ", obs)
    assert np.any(ang_rewards < -5.0)

def test_energy_reward():
    parser = argparse.ArgumentParser()

    config = get_config("./habitat_baselines/config/locomotion/ddppo_energy.yaml")
    env = LocomotionRLEnvEnergy(config=config, render=RENDER)
    obs = env.reset()

    energy_rewards = np.zeros(50)
    for i in range(50):
        # action1 = np.array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1], dtype=np.float32)
        # action2 = np.array([0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0], dtype=np.float32)
        # action = action1 - action2*0.5
        action = np.zeros(12)

        env.add_force(0, 0, 100, link=1)

        action_args = {"action_args": {"joint_deltas": action}}
        action_args = {"joint_deltas": action}
        obs, reward, done, info = env.step("null", action_args)

        energy_rewards[i] = env.named_rewards["energy_reward"]
        # print("Reward2: ", energy_rewards[i])
        # print("Reward: ", reward)
        # print("INFO: ", info)
        # print("OBS: ", obs)
    print("ER: ", energy_rewards)
    assert np.any(energy_rewards < 1e-1)

def test_energy_reward_2():
    parser = argparse.ArgumentParser()

    config = get_config("./habitat_baselines/config/locomotion/ddppo_energy.yaml")
    env = LocomotionRLEnvEnergy(config=config, render=RENDER)
    obs = env.reset()
    env.robot.prone()

    energy_rewards = np.zeros(50)
    for i in range(50):
        # action1 = np.array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1], dtype=np.float32)
        # action2 = np.array([0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0], dtype=np.float32)
        # action = action1 - action2*0.5
        # action = np.array([0, 1.3, -2.5] * 4)
        action = np.zeros(12)

        action_args = {"joint_deltas": action}
        if i == 49:
            obs, reward, done, info = env.step("null", action_args, step_render=RENDER)
        obs, reward, done, info = env.step("null", action_args, step_render=False)

        energy_rewards[i] = env.named_rewards["energy_reward"]
        # print("Reward2: ", energy_rewards[i])
        # print("Reward: ", reward)
        # print("INFO: ", info)
        print("OBS: ", obs)
    print("ER: ", energy_rewards)
    assert np.all(energy_rewards < 0)

# test_side_reward()
# test_local_coordinates()
# test_forward_rotated()
# test_forward()
# test_forward_local()