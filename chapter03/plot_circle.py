import matplotlib.pyplot as plt
import numpy as np
from computing_pi import within_circle

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
c = ['red' if within_circle(p_x, p_y) else 'blue' for p_x, p_y in zip(x, y)]

plt.scatter(x, y, color=c)
fig.savefig('output/computing_pi.png')