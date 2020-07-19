import numpy as np

class RLAgent:
    def __init__(self, env, alpha, gamma, epsilon):
        """
        :param env: the gym environment instance
        :param alpha: learning rate
        :param gamma: discount factor
        :param epsilon: initial exploration rate
        """
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        # Initialize the Q table
        self.n_states = self.env.observation_space.n
        self.n_actions = self.env.action_space.n
        self.Q = np.zeros((self.n_states, self.n_actions))

    def select_action(self, state):
        """
        An epsilon-greedy action selection
        :param state: the current state of the environment
        :return: an action
        """
        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.n_actions)  # exploration
        else:
            action = np.argmax(self.Q[state])  # exploitation
        return action

    def update_Q(self, state, action, reward, next_state):
        raise NotImplementedError

    def simulate_episode(self):
        state = self.env.reset()
        done = False
        total_reward = 0

        while not done:
            # Choose an action based on our current policy
            action = self.select_action(state)

            # Execute the action and observe the outcome state and the reward
            next_state, reward, done, _ = self.env.step(action)

            # Update the total reward
            total_reward += reward

            # Update the Q table
            self.update_Q(state, action, reward, next_state)

            # Update the state to the next time step
            state = next_state
        return total_reward

    def train(self, n_episodes=50000, interval=100, epsilon_decay=0.995):
        rewards = []
        best_avg_reward = -np.inf

        for episode in range(n_episodes):
            total_reward = self.simulate_episode()
            rewards.append(total_reward)
            self.epsilon *= epsilon_decay

            if episode % interval == 0 and episode > 0:
                avg_reward = np.mean(rewards[episode - interval:episode])
                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward

                print(f'Episode: {episode}, best average reward: {best_avg_reward}')
        return rewards

    def display_policy(self):
        state = self.env.reset()
        print('Initial state')
        self.env.render()

        done = False
        step = 0

        while not done:
            step += 1

            # Choose an action based on our current policy
            action = self.select_action(state)

            # Execute the action and observe the outcome state and the reward
            next_state, reward, done, _ = self.env.step(action)

            # Render the environment
            self.env.render()
            print(f'Timestep: {step}')
            print(f'Reward: {reward}')
            print()

            # Update the state to the next time step
            state = next_state





