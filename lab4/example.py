import gym
import time

env = gym.make("ALE/Breakout-v5", render_mode="human")
observation = env.reset(seed=42)

for _ in range(1000):
    action = env.action_space.sample()  # this is where you would insert your policy
    observation, reward, terminated, truncated = env.step(action)

    if terminated or truncated:
        observation = env.reset()

env.close()
print("Executed environment successfully.")
