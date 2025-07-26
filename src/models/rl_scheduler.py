import gym
import numpy as np


class MaintenanceEnv(gym.Env):
    """Simple environment where the state is remaining useful life."""

    def __init__(self):
        super().__init__()
        self.max_rul = 100
        self.rul = self.max_rul
        self.action_space = gym.spaces.Discrete(2)  # 0: do nothing, 1: maintain
        self.observation_space = gym.spaces.Box(
            low=0, high=self.max_rul, shape=(1,), dtype=np.int32
        )
        self.time = 0

    def reset(self):
        self.rul = self.max_rul
        self.time = 0
        return np.array([self.rul], dtype=np.int32)

    def step(self, action):
        reward = 0.0
        self.time += 1
        self.rul -= 1
        done = self.rul <= 0

        if action == 1:
            reward = self.rul
            self.rul = self.max_rul
        return np.array([self.rul], dtype=np.int32), reward, done, {}


def schedule_maintenance(rul_predictions, episodes: int = 1000):
    """Use a simple Q-learning scheduler for maintenance."""
    env = MaintenanceEnv()
    q_table = np.zeros((env.max_rul + 1, env.action_space.n))
    alpha = 0.1
    gamma = 0.95
    epsilon = 0.1

    for _ in range(episodes):
        state = env.reset()[0]
        done = False
        while not done:
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(q_table[state]))
            next_state, reward, done, _ = env.step(action)
            q_table[state, action] += alpha * (
                reward + gamma * np.max(q_table[next_state[0]]) - q_table[state, action]
            )
            state = next_state[0]

    schedule = []
    for pred in rul_predictions:
        state = min(int(pred), env.max_rul)
        action = int(np.argmax(q_table[state]))
        schedule.append(action)
    return schedule
