import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    filename = "output.csv" # substitute it with your filename here
    signal = np.loadtxt(filename)

    plt.figure()
    n = signal.shape[1] - 5

    c = 1
    for channel in range(n):

        plt.subplot(n, 1, c)
        plt.plot((signal[:, channel]))
        plt.title(channel)
        plt.grid()
        c += 1

    plt.show()
