import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches

def plot_blackjack_values(V, filename):
    def get_figure(ax, usable_ace):
        x_range = np.arange(11, 22)
        y_range = np.arange(1, 11)
        X, Y = np.meshgrid(x_range, y_range)
        Z = [V[(x, y, usable_ace)] if (x, y, usable_ace) in V else 0
             for x, y in zip(np.ravel(X), np.ravel(Y))]

        Z = np.array(Z).reshape(X.shape)

        ax.plot_surface(X, Y, Z, cmap=plt.cm.coolwarm)
        ax.set_xlabel("Player's current sum")
        ax.set_ylabel("Dealer's showing card")
        ax.set_zlabel('State value')
        ax.view_init(ax.elev, -120)

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(211, projection='3d')
    ax.set_title('Usable ace')
    get_figure(ax, True)

    ax = fig.add_subplot(212, projection='3d')
    ax.set_title('No usable ace')
    get_figure(ax, False)
    output_file = f'output/{filename}'
    plt.savefig(output_file)

def plot_policy(Q, filename):
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

    output_file = f'output/{filename}'
    plt.savefig(output_file)