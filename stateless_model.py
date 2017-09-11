import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L

import chainer_extensions as E


class StatelessModel(chainer.Chain):
    """static model"""

    def __init__(self):
        super().__init__(
            l1=L.Convolution2D(None, 4, 3, pad=1),
            l2=L.Convolution2D(None, 4, 3, pad=1),
            lout=L.Convolution2D(None, 1, 3, pad=1),
            l_str=L.Linear(None, 1),
        )
        self.pen = np.zeros((3, 3), np.float32)  # pen shape
        self.pen += 0.1
        self.pen[1, 1] = 0.9
        self.pen = self.pen.reshape((1, 1, 3, 3))
        self.move_cost = np.zeros((5, 5), np.float32)
        for i in range(5):
            for j in range(5):
                self.move_cost[i, j] = -5 + np.sqrt((i-2)**2+(j-2)**2)
        self.move_cost[2, 2] = 4.0  # don't stop!
        self.move_cost = self.move_cost.reshape((1, 1, 5, 5))
        self.tau = 1.0

    def calc(self, x):
        h = x
        h = self.l1(h)
        h = F.tanh(h)
        h = self.l2(h)
        h = F.tanh(h)
        self.strength = F.sigmoid(self.l_str(h))
        self.strength = F.reshape(self.strength, (-1, 1, 1))
        h = self.lout(h)
        return h

    def __call__(self, x):
        # ch: canvas, ref, prev_pen
        pred = self.calc(x)  # b, 1, w, h
        shape = pred.shape
        pred = F.reshape(pred, (shape[0], -1))
        # pred = F.softmax(pred)
        pred = E.gumbel_softmax(pred, tau=self.tau)
        pred = F.reshape(pred, (shape[0], 1)+shape[2:])
        self.current_pos = pred  # pen position
        mv_cost = F.sum(
            0.5*self.current_pos*(
                F.convolution_2d(
                    x[:, 2:3, :, :], self.move_cost, pad=2)+4))
        print(mv_cost.data)
        draw = F.convolution_2d(pred, self.pen, pad=1)  # pen stroke
        strength, draw = F.broadcast(self.strength, draw[:, 0, :, :])
        self.draw = strength*draw
        canvas = x[:, 0, :, :] + self.draw
        self.canvas = E.leaky_clip(canvas[0, :, :], 0, 1, leak=0.001)
        ref = x[:, 1, :, :]
        diff = F.sum((canvas-ref)**2)
        self.loss = diff+mv_cost
        return self.loss
