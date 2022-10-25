import math

import numpy as np


class Neuron:
    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self.weights = np.random.randn(dimension)
        self.y = 0
        self.delta = 0

    def activate(self, x: list) -> None:
        v = np.dot(np.array(x), self.weights)
        self.y = 1 if v < -20 else 1 / (1 + math.exp(-v))

    def updateDelta(self, error: float = None, delta_w_ls: np.ndarray = None) -> None:
        if error is not None:
            self.delta = error * self.y * (1 - self.y)
        else:
            self.delta = np.sum(delta_w_ls)
            self.delta *= self.y * (1 - self.y)

    def updateWeights(self, eta: float, last_y: np.ndarray) -> None:
        self.weights += eta * self.delta * last_y
