import numpy as np
import chainer


def show(x):
    s = 28
    img = x.reshape((s, s))
    print(' '+'='*s)
    for i in range(s):
        print('|', end='')
        for j in range(s):
            p = img[i, j]
            if p > 0.8:
                print('#', end='')
            else:
                print(' ', end='')
        print('|')
    print(' '+'='*s)


def load_mnist():
    return chainer.datasets.get_mnist(withlabel=False, ndim=1)
