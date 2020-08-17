from matplotlib import pyplot as plt
import numpy as np


def sines_plot(data, n_x, n_y):
    f, a = plt.subplots(n_x, n_y, figsize=(4*n_x, 2*n_y), sharey = True)
    if type(a) is np.ndarray:
        if len(a.shape) == 1:
            for i in range(max(n_x,n_y)):
                img = data[i]
                a[i].plot(img)
        else:
            for i in range(n_x):
                for j in range(n_y):
                    img = data[i*n_x + j]
                    a[i,j].plot(img)
    else:
        img = data[0]
        a.plot(img)
            
    plt.show()
