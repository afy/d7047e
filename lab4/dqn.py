import os
import random
import time

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Imports all our hyperparameters from the other file
from hyperparams import Hyperparameters as params

# stable_baselines3 have wrappers that simplifies
# the preprocessing a lot, read more about them here:
# https://stable-baselines3.readthedocs.io/en/master/common/atari_wrappers.html
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv, # clips the reward to reduce the effect of large reward differences
    EpisodicLifeEnv, # ends episodes when a life is lost, making it more challenging
    FireResetEnv, # automatically teaks "FIRE" action at the start of an episode if required by the game
    MaxAndSkipEnv, # skips frames to reduce computation and uses the max max pixel value over the skipped frames
    NoopResetEnv, # Introduces random inital no-op steps for diversity
)
from stable_baselines3.common.buffers import ReplayBuffer


# Creates our gym environment and with all our wrappers.
def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}", episode_trigger=lambda episode_id: True)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk



class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, env.single_action_space.n)

    def forward(self, x):
        x = x.float() / 255.0  # Convert to float and normalize
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

if __name__ == "__main__":
    run_name = f"{params.env_id}__{params.exp_name}__{params.seed}__{int(time.time())}"

    random.seed(params.seed)
    np.random.seed(params.seed)
    torch.manual_seed(params.seed)
    torch.backends.cudnn.deterministic = params.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv([make_env(params.env_id, params.seed, 0, params.capture_video, run_name)])
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    q_network = QNetwork(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=params.learning_rate)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())

    # We’ll be using experience replay memory for training our DQN.
    # It stores the transitions that the agent observes, allowing us to reuse this data later.
    # By sampling from it randomly, the transitions that build up a batch are decorrelated.
    # It has been shown that this greatly stabilizes and improves the DQN training procedure.
    rb = ReplayBuffer(
        params.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        optimize_memory_usage=False,
        handle_timeout_termination=True,
    )

    obs = envs.reset()
    for global_step in range(params.total_timesteps):
        # Here we get epsilon for our epislon greedy.
        epsilon = linear_schedule(params.start_e, params.end_e, params.exploration_fraction * params.total_timesteps, global_step)

        if random.random() < epsilon:
            actions = envs.action_space.sample()  # Sample a random action
        else:
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, device=device).float()  # convert obs to tensor and float
                q_values = q_network(obs_tensor)
                actions = torch.argmax(q_values, dim=1).cpu().numpy()

        # Take a step in the environment
        next_obs, rewards, dones, infos = envs.step(actions)

        # Here we print our reward.
        for info in infos:
            if "episode" in info.keys():
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                break

        # Save data to replay buffer
        real_next_obs = next_obs.copy()
        for idx, d in enumerate(dones):
            if d:
                real_next_obs[idx] = infos[idx]["terminal_observation"]

        # Here we store the transitions in D
        rb.add(obs, real_next_obs, actions, rewards, dones, infos)

        obs = next_obs
         # Training
        if global_step > params.learning_starts:
            if global_step % params.train_frequency == 0:
                # Sample random minibatch of transitions from D
                data = rb.sample(params.batch_size)
                # Debugging print statements
                #print(f"data.observations shape: {data.observations.shape}")
                #print(f"data.next_observations shape: {data.next_observations.shape}")
                #print(f"data.actions shape: {data.actions.shape}")

                # Convert observations and actions to appropriate types
                next_obs_tensor = torch.tensor(data.next_observations, device=device).clone().detach().float()  # Convert next_observations to tensor and float
                obs_tensor = torch.tensor(data.observations, device=device).clone().detach().float()  # Convert observations to tensor and float
                actions_tensor = data.actions.long().squeeze(-1)  # Ensure actions have the correct shape

                # Debugging print statements
                #print(f"next_obs_tensor shape: {next_obs_tensor.shape}")
                #print(f"obs_tensor shape: {obs_tensor.shape}")
                #print(f"actions_tensor shape: {actions_tensor.shape}")

                with torch.no_grad():
                    next_state_values = target_network(next_obs_tensor).max(dim=1)[0]
                    td_target = data.rewards + params.gamma * next_state_values * (1 - data.dones)

                state_action_values = q_network(obs_tensor).gather(1, actions_tensor.unsqueeze(-1)).squeeze(-1)
                loss = F.mse_loss(state_action_values, td_target)

                # Perform our gradient descent step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if global_step % 1000 == 0:
                    print(f"global_step={global_step}, loss={loss.item()}")

            # Update target network
            if global_step % params.target_network_update_freq == 0:
                target_network.load_state_dict(q_network.state_dict())


    if params.save_model:
        model_path = f"runs/{run_name}/{params.exp_name}_model"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)  # Ensure the directory exists
        torch.save(q_network.state_dict(), model_path)
        print(f"model saved to {model_path}")

    envs.close()
