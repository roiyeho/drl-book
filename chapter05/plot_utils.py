import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches

from computing_pi import is_inside_circle

def plot_circle():
    circle = plt.Circle((0, 0), 0.5, fill=False, lw=2)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.add_artist(circle)
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    n_points = 500
    x = np.random.random(n_points) - 0.5
    y = np.random.random(n_points) - 0.5
    c = ['red' if is_inside_circle(p_x, p_y) else 'blue' for p_x, p_y in zip(x, y)]

    plt.scatter(x, y, color=c)
    fig.savefig('figures/computing_pi.png')

def plot_blackjack_values(V):
    def get_figure(ax, usable_ace):
        x_range = np.arange(11, 22)
        y_range = np.arange(1, 11)
        X, Y = np.meshgrid(x_range, y_range)
        Z = [V[(x, y, usable_ace)] if (x, y, usable_ace) in V else 0
             for x, y in zip(np.ravel(X), np.ravel(Y))]

        Z = np.array(Z).reshape(X.shape)

        ax.plot_surface(X, Y, Z, cmap=plt.cm.coolwarm)
        ax.set_xlabel("Player's current sum", fontsize=12)
        ax.set_ylabel("Dealer's showing card", fontsize=12)
        ax.set_zlabel('State value', fontsize=12)
        ax.view_init(ax.elev, -120)

    #fig = plt.figure(figsize=(12, 12))
    fig = plt.figure(figsize=(14, 14))
    ax = fig.add_subplot(211, projection='3d')
    ax.set_title('No usable ace', fontsize=12)
    get_figure(ax, False)

    ax = fig.add_subplot(212, projection='3d')
    ax.set_title('Usable ace', fontsize=12)
    get_figure(ax, True)

    output_file = f'figures/blackjack_state_values.png'
    plt.savefig(output_file)

def plot_policy(Q):
    def get_figure(usable_ace, ax):
        x_range = np.arange(11, 22)
        y_range = np.arange(10, 0, -1)

        Z = [[np.argmax(Q[(x, y, usable_ace)]) for x in x_range] for y in y_range]
        Z = np.array(Z)
        print(Z)

        im = ax.imshow(Z, extent=[10.5, 21.5, 0.5, 10.5], cmap=plt.cm.coolwarm)
        ax.grid()
        ax.set_xticks(x_range)
        ax.set_yticks(y_range)
        ax.set_xlabel("Player's current sum")
        ax.set_ylabel("Dealer's showing card")

        # Create a legend for the colors used by imshow
        values = [0, 1]
        labels = ['STICK', 'HIT']
        colors = [im.cmap(im.norm(value)) for value in values]
        patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(len(values))]
        ax.legend(handles=patches)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.set_title('Usable ace')
    get_figure(True, ax1)
    ax2.set_title('No usable ace')
    get_figure(False, ax2)

    output_file = f'figures/{filename}'
    plt.savefig(output_file)