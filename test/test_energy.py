import argparse
from habitat_baselines.common.environments import LocomotionRLEnvStand, LocomotionRLEnvEnergy
from habitat_baselines.config.default import get_config
import numpy as np
import cv2


def test_forward_reward():
    parser = argparse.ArgumentParser()

    config = get_config("./habitat_baselines/config/locomotion/ddppo_energy.yaml")
    config.defrost()
    config.TASK_CONFIG.DEBUG.FIXED_BASE = False
    config.freeze()
    env = LocomotionRLEnvEnergy(config=config, render=False)
    obs = env.reset()

    vx_rewards = np.zeros(50)
    for i in range(50):
        # action1 = np.array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1], dtype=np.float32)
        # action2 = np.array([0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0], dtype=np.float32)
        # action = action1 - action2*0.5
        action = np.zeros(12)

        env.add_force(30, 0, 0, link=0)

        action_args = {"action_args": {"joint_deltas": action}}
        action_args = {"joint_deltas": action}
        obs, reward, done, info = env.step("null", action_args)

        vx_rewards[i] = env.named_rewards["forward_velocity_reward"]
        print("Reward: ", vx_rewards[i])
        # print("Reward: ", reward)
        # print("INFO: ", info)
        # print("OBS: ", obs)
    
    assert np.all(np.diff(vx_rewards) > 0)


def test_side_reward():
    parser = argparse.ArgumentParser()

    config = get_config("./habitat_baselines/config/locomotion/ddppo_energy.yaml")
    config.defrost()
    config.TASK_CONFIG.DEBUG.FIXED_BASE = False
    config.freeze()
    env = LocomotionRLEnvEnergy(config=config, render=False)
    obs = env.reset()

    vy_rewards = np.zeros(50)
    for i in range(50):
        # action1 = np.array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1], dtype=np.float32)
        # action2 = np.array([0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0], dtype=np.float32)
        # action = action1 - action2*0.5
        action = np.zeros(12)

        env.add_force(0, 0, 30, link=0)

        action_args = {"action_args": {"joint_deltas": action}}
        action_args = {"joint_deltas": action}
        obs, reward, done, info = env.step("null", action_args)

        vy_rewards[i] = env.named_rewards["side_velocity_reward"]
        print("Reward2: ", vy_rewards[i])
        # print("Reward: ", reward)
        # print("INFO: ", info)
        # print("OBS: ", obs)
    print(np.diff(vy_rewards))
    assert np.all(np.diff(vy_rewards) < 0)

def test_angular_reward():
    parser = argparse.ArgumentParser()

    config = get_config("./habitat_baselines/config/locomotion/ddppo_energy.yaml")
    config.defrost()
    config.TASK_CONFIG.DEBUG.FIXED_BASE = False
    config.freeze()
    env = LocomotionRLEnvEnergy(config=config, render=False)
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
        print("Reward2: ", ang_rewards[i])
        # print("Reward: ", reward)
        # print("INFO: ", info)
        # print("OBS: ", obs)
    assert np.any(ang_rewards < -5.0)

def test_energy_reward():
    parser = argparse.ArgumentParser()

    config = get_config("./habitat_baselines/config/locomotion/ddppo_energy.yaml")
    env = LocomotionRLEnvEnergy(config=config, render=False)
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
        print("Reward2: ", energy_rewards[i])
        # print("Reward: ", reward)
        # print("INFO: ", info)
        # print("OBS: ", obs)
    assert np.any(energy_rewards < -5.0)

test_angular_reward()