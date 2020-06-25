import numpy as np

def within_circle(x, y):
    R = 0.5
    if x ** 2 + y ** 2 < R ** 2:
        return True
    else:
        return False

def compute_pi(n_samples=1000000):
    n = 0
    n_inside = 0

    for i in range(n_samples):
        x = np.random.random() - 0.5
        y = np.random.random() - 0.5
        if within_circle(x, y):
            n_inside += 1
        n += 1

    pi = 4 * n_inside / n
    return pi

if __name__ == '__main__':
    pi = compute_pi()
    print('Approximate value of pi:', pi)
    print('Exact value of pi:', np.pi)