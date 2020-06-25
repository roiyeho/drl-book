import time
from env_runner import EnvRunner

class ACEnvRunner(EnvRunner):
    def __init__(self,
                 env,
                 agent,
                 n_episodes=1000,
                 test_env=None,
                 test_episode_max_len=10000,
                 check_win=True,
                 win_trials=100,
                 win_mean_reward=195,
                 stats_interval=100):
        super().__init__(env, agent, n_episodes, test_env, test_episode_max_len,
                         check_win, win_trials, win_mean_reward, stats_interval)

    def run(self):
        total_rewards = []  # stores the total reward per episode
        start_time = time.time()
        total_steps = 0

        for episode in range(self.n_episodes):
            done = False
            total_reward = 0
            step = 0  # counts the time steps in this episode

            state = self.env.reset()

            # Run an episode
            while not done:
                action = self.agent.select_action(state)
                next_state, reward, done, info = self.env.step(action)

                self.agent.train(state, action, reward, next_state, done)

                state = next_state
                total_reward += reward
                step += 1
                total_steps += 1

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