import numpy as np
from collections import deque

class DQNTeam:
    """A team consists of n agents, each running some type of a deep Q-network."""
    def __init__(self,
                 env,
                 team_config,
                 agent_config,
                 ):
        """
        :param env: the gym environment where the team will run
        :param team_config: hyperparameters of the team
        :param agent_config: the configuration of the agents
        """
        self.env = env
        self.config = team_config

        # Instantiate the agents
        self.n_agents = self.config.n_agents
        self.create_agents(agent_config)

        # Create the centralized replay buffer
        self.replay_buffer = deque(maxlen=self.config.replay_buffer_size)

        # Set up the initial exploration rate
        self.epsilon = self.config.initial_epsilon

    def create_agents(self, agent_config):
        agent_class = self.get_agent_class(agent_config.agent_type)
        self.agents = [
            agent_class(self.env, agent_config) for _ in range(self.n_agents)
        ]

    def get_agent_class(self, agent_type):
        """Importing the requested agent class"""
        if agent_type == 'dqn':
            from idqn.dqn_agent import DQNAgent as agent_class
        elif agent_type == 'ddqn':
            from idqn.ddqn_agent import DoubleDQNAgent as agent_class
        elif agent_type == 'hdqn':
            from idqn.hdqn_agent import HystereticDQNAgent as agent_class
        return agent_class

    def get_actions(self, observations):
        """Invoke the action selection of each agent
        :param observations: the agents' observations
        :return: a list of actions
        """
        actions = []
        for i in range(self.n_agents):
            action = self.agents[i].select_action(observations[i], self.epsilon)
            actions.append(action)
        return actions

    def store_transition(self, observations, actions, rewards, next_observations, done):
        """Store a transition in the replay buffer."""
        self.replay_buffer.append((observations, actions, rewards, next_observations, done))

    def sample_transitions(self):
        indices = np.random.randint(len(self.replay_buffer), size=self.config.batch_size)
        mini_batch = [self.replay_buffer[index] for index in indices]

        observations, actions, rewards, next_observations, dones = [
            np.array([transition[field_index] for transition in mini_batch])
            for field_index in range(5)
        ]
        return observations, actions, rewards, next_observations, dones

    def train(self, steps_from_start):
        """Execute a training step of the agents
        :param steps_from_start: Number of steps from the beginning of this run
        :return:
        """
        # Check that we've reached the next round of training
        if steps_from_start % self.config.train_interval != 0:
            return

        # Check that we have enough transitions in the replay buffer
        if len(self.replay_buffer) < max(self.config.batch_size, self.config.replay_start_size):
            return

        # Sample transitions from the replay buffer
        observations, actions, rewards, next_observations, dones = self.sample_transitions()

        # Provide each agent with its own observations, actions and rewards
        for i in range(self.n_agents):
            # Check if the reward is global or per agent
            if rewards.ndim == 2:
                agent_reward = rewards[:, i]
            else:
                agent_reward = rewards
            self.agents[i].train(observations[:, i], actions[:, i], agent_reward,
                                 next_observations[:, i], dones)

        # Update the target networks
        if steps_from_start % self.config.target_update_interval == 0:
            for agent in self.agents:
                agent.update_target_network()

    def update_exploration_rate(self):
        """Update epsilon after each episode"""

        # Check if learning has started
        if len(self.replay_buffer) < self.config.replay_start_size:
            return

        if self.epsilon > self.config.final_epsilon:
            self.epsilon *= self.config.epsilon_decay

    def save_models(self, folder):
        for i in range(self.n_agents):
            self.agents[i].save_model(folder, i)

    def load_models(self, folder):
        for i in range(self.n_agents):
            self.agents[i].load_model(folder, i)