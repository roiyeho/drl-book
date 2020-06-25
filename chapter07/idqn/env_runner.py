import numpy as np
import time
import os
import cv2
from pprint import pprint

import idqn.plot_utils as plot_utils
from idqn.dqn_team import DQNTeam

class EnvRunner:
    def __init__(self,
                 env,
                 team_config,
                 agent_config,
                 n_episodes=2000,
                 stats_interval=100,
                 check_solved=False,
                 solved_episodes=0,
                 solved_mean_reward=0,
                 eval_interval=0,
                 test_env=None,
                 n_eval_episodes=30,
                 show_policy_interval=0,
                 record_policy_interval=100,
                 ):
        """
        :param env: an instance of gym environment
        :param team_config: configuration of the team
        :param agent_config: configuration of an agent
        :param n_episodes: number of episodes used for training
        :param stats_interval: number of episodes between statistics updates
        :param check_solved: whether to check if the environment was solved
        :param solved_episodes: number of consecutive episodes to check for solution
        :param solved_mean_reward: mean of rewards that needs to be achieved
                                for the environment to be considered solved
        :param eval_interval: number of episodes between two evaluations
        :param test_env: the environment used for testing. If None, use the same
                         environment used for training.
        :param n_eval_episodes: number of episodes to use for an evaluation
        :param show_policy_interval: number of episodes between policy displays
        :param record_policy_interval: number of episodes between two policy records
        """
        self.env = env
        self.team_config = team_config
        self.agent_config = agent_config

        # Create the team
        self.team = DQNTeam(env, team_config, agent_config)

        self.n_episodes = n_episodes

        self.stats_interval = stats_interval

        self.check_solved = check_solved
        self.solved_episodes = solved_episodes
        self.solved_mean_reward = solved_mean_reward

        self.eval_interval = eval_interval
        self.test_env = env if test_env is None else test_env
        self.n_eval_episodes = n_eval_episodes

        self.show_policy_interval = show_policy_interval
        self.record_policy_interval = record_policy_interval

        self.create_results_folder()

    def create_results_folder(self, **kwargs):
        params_str = ''
        for key, value in kwargs.items():
            params_str += f'-{key}{value}'

        time_str = time.strftime("%d-%b-%Y %H-%M-%S")
        self.results_folder = os.path.join('results', self.env.spec.id, f'{time_str}{params_str}')
        os.makedirs(self.results_folder)

        # Create the results file
        results_file_path = os.path.join(self.results_folder, 'results.csv')
        self.results_file = open(results_file_path, 'w')
        print('Episode, Total Reward, Steps', file=self.results_file)

        # Save the team and agent configurations
        team_config_file_path = os.path.join(self.results_folder, 'team_config.txt')
        s = str(vars(self.team_config))
        with open(team_config_file_path, 'w') as file:
            pprint(vars(self.team_config), stream=file)

        agent_config_file_path = os.path.join(self.results_folder, 'agent_config.txt')
        with open(agent_config_file_path, 'w') as file:
            pprint(vars(self.agent_config), stream=file)

    def run(self):
        episodes_rewards = []  # stores the total reward per episode
        episodes_steps = []    # stores the number of steps per episode

        start_time = time.time()
        total_steps = 0

        for episode in range(self.n_episodes):
            time_step = 0  # counts the time steps in this episode
            total_reward = 0
            done = False

            observations = self.env.reset()

            # Run an episode
            while not done:
                actions = self.team.get_actions(observations)
                next_observations, rewards, done, _ = self.env.step(actions)
                self.team.store_transition(observations, actions, rewards, next_observations, done)

                total_reward += self.get_team_reward(rewards)

                time_step += 1
                total_steps += 1

                # Train the team
                self.team.train(total_steps)

                observations = next_observations
                print('\rTime step:', time_step, end='')

            self.team.update_exploration_rate()

            # Store episode stats
            episodes_rewards.append(total_reward)
            episodes_steps.append(time_step)
            self.save_result(episode, total_reward, time_step)

            print()
            elapsed_time = time.time() - start_time
            print(f'Episode: {episode + 1}, reward: {total_reward:.2f}, steps: {time_step}, '
                  f'eps: {self.team.epsilon:.3f}, elapsed time: {elapsed_time:.2f}')
            self.print_env_stats()

            # Compute statistics for the last X episodes
            if self.stats_interval and (episode + 1) % self.stats_interval == 0:
                self.compute_stats(episode, episodes_rewards, episodes_steps,
                                   total_steps, start_time)

            # Check if the environment was solved
            if self.check_solved and episode > self.solved_episodes and \
                    self.is_env_solved(episode, episodes_rewards):
                break

            # Evaluate the current policy on X test trials
            if self.eval_interval and (episode + 1) % self.eval_interval == 0:
                self.evaluate()

            # Show the current policy
            if self.show_policy_interval and (episode + 1) % self.show_policy_interval == 0:
                self.show_policy()

            # Record the current policy
            if self.record_policy_interval and (episode + 1) % self.record_policy_interval == 0:
                self.record_policy(episode)

        self.end_run(episodes_rewards, episodes_steps, total_steps, start_time)

    def get_team_reward(self, rewards):
        # Check if the reward is a scalar or a list
        if hasattr(rewards, '__iter__'):
            return np.sum(rewards)
        else:
            return rewards

    def print_env_stats(self):
        if not hasattr(self.env, 'get_stats'):
            return
        stats = self.env.get_stats()
        for key, item in stats.items():
            print(f'{key}: {item}, ', end='')
        print('\b\b')

    def save_result(self, episode, total_reward, steps):
        print(episode + 1, f'{total_reward:.2f}', steps, file=self.results_file, sep=', ')
        self.results_file.flush()

    def compute_stats(self, episode, episodes_rewards, episodes_steps, total_steps, start_time):
        mean_reward = np.mean(episodes_rewards[-self.stats_interval:])
        mean_steps = np.mean(episodes_steps[-self.stats_interval:])
        elapsed_time = int(time.time() - start_time)
        print('=' * 85)
        print(f'Episode: {episode + 1}, mean reward: {mean_reward:.2f}, '
              f'mean steps: {mean_steps:.2f}')
        print(f'Total steps: {total_steps}, elapsed time: {elapsed_time} sec')
        print('=' * 85)
        self.team.save_models(self.results_folder)

    def is_env_solved(self, episode, episodes_rewards):
        mean_reward = np.mean(episodes_rewards[-self.solved_episodes:])
        if mean_reward >= self.solved_mean_reward:
            print('=' * 95)
            print(f'Solved in episode {episode}, mean reward: {mean_reward:.2f}')
            print('=' * 95)
            return True
        return False

    def evaluate(self):
        print('Evaluating the policy...')
        episodes_rewards = []
        episodes_steps = []

        for episode in range(self.n_eval_episodes):
            print('Evaluation episode ')
            total_reward = 0
            time_step = 0  # counts the time steps in this episode
            done = False

            observations = self.test_env.reset()

            # Run an episode without exploration
            while not done:
                actions = self.team.get_best_actions(observations)
                next_observations, rewards, done, _ = self.env.step(actions)
                total_reward += self.get_team_reward(rewards)
                time_step += 1
                observations = next_observations

            episodes_rewards.append(total_reward)
            episodes_steps.append(time_step)

            print(f'Evaluation episode: {episode + 1}, reward: {total_reward:.2f}, steps: {time_step}')

        episodes_rewards = np.array(episodes_rewards)
        episodes_steps = np.array(episodes_steps)

        print('Evaluation completed')
        print(f'Average reward: {np.mean(episodes_rewards):.3f}, std: {np.std(episodes_rewards):.3f}')
        print(f'Average steps: {np.mean(episodes_steps):.3f}, std: {np.std(episodes_steps):.3f}')

    def show_policy(self):
        time_step = 0
        total_reward = 0
        done = False
        observations = self.env.reset()

        self.env.render()

        while not done:
            actions = self.team.get_actions(observations)
            next_observations, rewards, done, _ = self.env.step(actions)
            self.env.render()

            time_step += 1
            total_reward += self.env.get_team_reward(rewards)

            print(f'\rTime step: {time_step}, reward: {total_reward:.2f}', end='')
            observations = next_observations

        print(f'\rPolicy run has finished. Steps: {time_step}, reward: {total_reward:.2f}')

    def end_run(self, episodes_rewards, episodes_steps, total_steps, start_time):
        self.results_file.close()
        self.team.save_models(self.results_folder)

        elapsed_time = int(time.time() - start_time)
        print('=' * 70)
        print(f'End of run. Total steps: {total_steps}, '
              f'elapsed time: {elapsed_time} sec')
        print('=' * 70)

        plot_utils.plot_rewards(self.results_folder, episodes_rewards)
        plot_utils.plot_steps(self.results_folder, episodes_steps)

    def record_policy(self, episode):
        print('=' * 70)
        print('Recording the policy...')

        # Add a Monitor wrapper for recording a video
        video_folder = os.path.join(self.results_folder, f'videos')
        if not os.path.exists(video_folder):
            os.makedirs(video_folder)
        video_path = os.path.join(video_folder, f'e{episode + 1}.mp4')

        time_step = 0
        total_reward = 0
        done = False
        observations = self.env.reset()

        frames = []
        frames.append(self.env.render(mode=None))

        while not done:
            actions = self.team.get_actions(observations)
            next_observations, rewards, done, _ = self.env.step(actions)
            frames.append(self.env.render(mode=None))

            time_step += 1
            total_reward = self.get_team_reward(rewards)
            observations = next_observations

        # Define the codec and create the VideoWrite object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 20
        size = frames[0].shape
        out = cv2.VideoWriter(video_path, fourcc, fps, size, False)

        # Write the frames array to the video
        for frame in frames:
            out.write(frame)
        out.release()

        print(f'Policy recording has finished. Steps: {time_step}, reward: {total_reward:.2f}')
        print('=' * 70)





