import numpy as np

def compute_pi(n_samples=1000000):
    """Estimate the value of pi by sampling points inside a unit square and
    checking how many of them fall inside the enclosed circle
    :param n_samples: number of samples to use
    :return: estimated value of pi
    """
    n, n_inner = 0, 0

    for i in range(n_samples):
        x = np.random.random() - 0.5
        y = np.random.random() - 0.5
        if is_inside_circle(x, y):
            n_inner += 1
        n += 1

    pi = 4 * n_inner / n
    return pi

def is_inside_circle(x, y):
    """Check if a given point falls inside the enclosed circle
    :param x: x coordinate of the point
    :param y: y coordinate of the point
    :return: a boolean True/False
    """
    r = 0.5  # radius of the circle
    if x ** 2 + y ** 2 < r ** 2:
        return True
    else:
        return False

if __name__ == '__main__':
    pi = compute_pi()
    print('Approximate value of pi:', pi)
    print('Exact value of pi:', np.pi)