import os
import sys

import numpy as np
import chainer

import utils
from stateless_model import StatelessModel


def main():
    os.makedirs("images", exist_ok=True)
    model = StatelessModel()
    dataset, _ = utils.load_mnist()
    opt = chainer.optimizers.Adam(1e-3)
    opt.setup(model)

    for i, img in enumerate(dataset[:100]):
        x = np.zeros((1, 3, 28, 28), np.float32)
        x[0, 1, :, :] = img.reshape((28, 28))  # ref
        x[0, 2, 0, 0] = 1.0  # initial pen position
        images = []

        for t in range(30):
            model.cleargrads()
            loss = model(x)
            loss.backward()
            opt.update()
            if t%30==29:
                print("loss: ", loss.data)
                sys.stdout.flush()
            canvas = model.canvas
            x[0, 0, :, :] = canvas.data
            x[0, 2, :, :] = model.current_pos.data[0, 0, :, :]
            outim = np.zeros((28, 28*3+2), np.float32)
            outim[:, 28] = 1.0
            outim[:, 57] = 1.0
            for c in range(3):
                outim[:, 28*c+c:28*(c+1)+c] = x[0, c, :, :].data
            images.append(np.clip(outim, 0.0, 1.0))
        utils.save_gif("images/test{0:03d}.gif".format(i), images)
        if model.tau > 0.1:
            model.tau *= 0.99
        # break


if __name__ == '__main__':
    main()
