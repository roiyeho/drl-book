import numpy as np
import matplotlib.pyplot as plt
import time
import os
import gym

class EnvRunner:
    def __init__(self,
                 env,
                 agent,
                 n_episodes=1000,
                 test_env=None,
                 test_episode_max_len=10000,
                 check_solved=True,
                 win_trials=100,
                 win_mean_reward=195,
                 stats_interval=100,
                 ):
        """
        :param env: an instance of gym environment
        :param agent: the agent instance
        :param n_episodes: number of episodes used for training
        :param test_env: the environment used for testing. If None, use the same
                         environment used for training.
        :param test_episode_max_len: maximum number of steps used for evaluation
        :param check_solved: whether to check if the environment was solved
        :param win_trials: number of consecutive trials considered for
                           checking if the environment was solved
        :param win_mean_reward: mean of rewards that needs to be achieved
                                for the environment to be considered solved
        :param stats_interval: how frequently to compute statistics
        """
        self.env = env
        self.agent = agent
        self.n_episodes = n_episodes
        self.test_env = env if test_env is None else test_env
        self.test_episode_max_len = test_episode_max_len
        self.check_solved = check_solved
        self.win_trials = win_trials
        self.win_mean_reward = win_mean_reward
        self.stats_interval = stats_interval
        self.create_results_folder()

    def create_results_folder(self, **kwargs):
        params_str = ''
        for key, value in kwargs.items():
            params_str += f'-{key}{value}'

        time_str = time.strftime("%Y%m%d-%H%M%S")
        self.results_folder = os.path.join('results', f'{self.env.spec.id}-{time_str}{params_str}')
        os.makedirs(self.results_folder)

        # Create the results file
        filename = os.path.join(self.results_folder, 'results.txt')
        self.results_file = open(filename, 'w')
        print('#Episode Reward', file=self.results_file)

    def run(self):
        total_rewards = []  # stores the total reward per episode
        start_time = time.time()
        total_steps = 0

        for episode in range(self.n_episodes):
            done = False
            total_reward = 0
            step = 0  # counts the time steps in this episode
            states, actions, rewards = [], [], []

            state = self.env.reset()

            # Run an episode
            while not done:
                action = self.agent.select_action(state)
                next_state, reward, done, info = self.env.step(action)
                states.append(state)
                actions.append(action)
                rewards.append(reward)

                state = next_state
                total_reward += reward
                step += 1
                total_steps += 1

            # Train the agent at the end of the episode
            self.agent.train(states, actions, rewards)

            # Store the total reward
            total_rewards.append(total_reward)
            print(f'Episode: {episode + 1}, steps: {step}, reward: {total_reward}')
            self.save_result(episode, total_reward)

            # Check if the environment was solved
            if self.check_solved and self.is_env_solved(episode, total_rewards, total_steps, start_time):
                break

            # Compute the mean total reward in the last 100 episodes
            if (episode + 1) % self.stats_interval == 0:
                self.compute_stats(episode, total_rewards, total_steps, start_time)

        self.end_experiment(total_rewards)

    def is_env_solved(self, episode, total_rewards, total_steps, start_time):
        mean_reward = np.mean(total_rewards[-self.win_trials:])
        if mean_reward >= self.win_mean_reward and \
                episode >= self.win_trials:
            elapsed_time = int(time.time() - start_time)
            print('=' * 95)
            print(f'Solved in episode {episode}, '
                  f'mean reward: {mean_reward}, '
                  f'total steps: {total_steps}, '
                  f'elapsed time: {elapsed_time} sec')
            self.evaluate_agent(episode)
            print('=' * 95)
            return True
        return False

    def compute_stats(self, episode, total_rewards, total_steps, start_time):
        mean_reward = np.mean(total_rewards[-self.win_trials:])
        elapsed_time = int(time.time() - start_time)
        print('=' * 85)
        print(f'Episode {episode + 1}: mean reward = {mean_reward}, '
              f'total steps = {total_steps}, elapsed time = {elapsed_time} sec')
        self.evaluate_agent(episode)
        print('=' * 85)
        self.agent.save_model(self.results_folder, self.env.spec.id)

    def end_experiment(self, total_rewards):
        self.results_file.close()
        self.agent.save_model(self.results_folder, self.env.spec.id)
        self.plot_rewards(total_rewards)

    def save_result(self, episode, reward):
        print(episode + 1, reward, file=self.results_file)
        self.results_file.flush()

    def plot_rewards(self, rewards):
        x = range(0, len(rewards))
        plt.plot(x, rewards)
        plt.xlabel('Episode', fontsize=12)
        plt.ylabel('Total reward', fontsize=12)
        plt.title('Learning curve', fontsize=14)
        graph_file = os.path.join(self.results_folder, f'{self.env.spec.id}.png')
        plt.savefig(graph_file)

    def plot_results_from_file(self, file_path):
        rewards = np.loadtxt(file_path)
        self.plot_rewards(rewards)

    def evaluate_agent(self, episode):
        # Add a Monitor wrapper for recording a video
        video_folder = os.path.join(self.results_folder, f'videos/e{episode + 1}')
        env = gym.wrappers.Monitor(self.test_env, video_folder)

        done = False
        total_reward = 0
        step = 0
        state = env.reset()

        # Run an episode on the wrapped environment with exploration turned off
        while not done and step < self.test_episode_max_len:
            action = self.agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            step += 1
            total_reward += reward

        print(f'Agent evaluation: steps = {step}, reward = {total_reward}')