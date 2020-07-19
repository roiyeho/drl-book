class AgentConfig:
    def __init__(self,
                 agent_type='dqn',
                 alpha=0.0001,
                 gamma=0.95,
                 hysteretic_beta=0.5,
                 filters=[32, 64],
                 kernel_size=[4, 2],
                 strides=[2, 1],
                 fc_size=1024
                 ):
        """
        :param agent_type: options: 'dqn', 'ddqn', 'hdqn'
        :param alpha: learning rate
        :param gamma: discount factor
        :param beta: hysteretic beta
        :param filters: number of filters in each conv layer in the CNN
        :param kernel_size: size of the kernels
        :param stride: the stride used in each conv layer
        :param fc_size: the size of the fully connected layer
        """
        self.agent_type = agent_type
        self.alpha = alpha
        self.gamma = gamma
        self.hysteretic_beta = hysteretic_beta
        self.cnn = self.CNNConfig(filters, kernel_size, strides, fc_size)

    class CNNConfig:
        """Convolutional network's hyperparameters"""
        def __init__(self, filters, kernel_size, strides, fc_size):
            self.filters = filters
            self.kernel_size = kernel_size
            self.strides = strides
            self.fc_size = fc_size

        def __repr__(self):
            return str(vars(self))
