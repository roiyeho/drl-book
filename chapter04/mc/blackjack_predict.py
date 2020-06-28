import gym
from mc_predict import MCPredict
from plot_utils import plot_blackjack_values

STICK = 0
HIT = 1

class BlackjackSimplePolicy():
    def get_action(self, state):
        if state[0] >= 18:
            return STICK
        return HIT

def blackjack_predict():
    env = gym.make('Blackjack-v0')

    policy = BlackjackSimplePolicy()
    blackjack_predict = MCPredict(env, policy, gamma=1)

    Q = blackjack_predict.predict(n_episodes=100000)

    # Get the state value function V of the policy
    V = {s: q[policy.get_action(s)] for s, q in Q.items()}
    plot_blackjack_values(V, filename='MC_Prediction.png')

blackjack_predict()





