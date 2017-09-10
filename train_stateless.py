import os

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
        utils.show(img)
        x = np.zeros((1, 3, 28, 28), np.float32)
        x[0, 1, :, :] = img.reshape((28, 28))  # ref
        x[0, 2, 0, 0] = 1.0  # initial pen position
        images = []

        for _ in range(30):
            # print(x)
            model.cleargrads()
            loss = model(x)
            # opt.update(model, x)
            # loss = model.loss
            print(loss)
            loss.backward()
            opt.update()
            canvas = model.canvas
            x[0, 0, :, :] = canvas.data
            x[0, 2, :, :] = model.current_pos.data[0, 0, :, :]
            images.append(np.clip(canvas.data, 0.0, 1.0))
            # utils.show(canvas.data)
        utils.show(canvas.data)
        utils.save_gif("images/test{0:03d}.gif".format(i), images)
        if model.tau > 0.1:
            model.tau *= 0.99
        # break


if __name__ == '__main__':
    main()
