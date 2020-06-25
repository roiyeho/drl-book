class TeamConfig:
    def __init__(self,
                 n_agents=2,
                 initial_epsilon=1.0,
                 final_epsilon=0.1,
                 epsilon_decay=0.999,
                 replay_buffer_size=250000,
                 replay_start_size=200000,
                 batch_size=32,
                 train_interval=4,
                 target_update_interval=5000,
                 ):
        """
        :param n_agents: number of agents in the team
        :param initial_epsilon: initial exploration rate
        :param final_epsilon: minimum exploration rate
        :param replay_buffer_size: the size of the shared replay buffer
        :param replay_start_size: the initial size of the replay memory before learning starts
        :param batch_size: the size of a minibatch
        :param train_interval: number of steps between two training steps
                               if 0, then train after an episode ends
        :param target_update_interval: number of steps between two updates of the target network
        """

        self.n_agents = n_agents

        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.epsilon_decay = epsilon_decay

        self.replay_buffer_size = replay_buffer_size
        self.replay_start_size = replay_start_size

        self.batch_size = batch_size
        self.train_interval = train_interval
        self.target_update_interval = target_update_interval