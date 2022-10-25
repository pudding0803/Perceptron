import numpy as np

from Neuron import Neuron


class Perceptron:
    def __init__(self, eta: float, dimension: int, layer_num: list) -> None:
        self.eta = eta
        self.dimension = dimension
        self.layers = [[Neuron(0) for _ in range(self.dimension)]]
        for i in range(1, len(layer_num)):
            self.layers.append([Neuron(layer_num[i - 1] + 1) for _ in range(layer_num[i])])

    def forward(self, inputs: np.ndarray) -> None:
        curr_x = [-1]
        for i in range(len(inputs)):
            self.layers[0][i].y = inputs[i]
            curr_x.append(self.layers[0][i].y)
        for i in range(1, len(self.layers)):
            next_x = [-1]
            for j in range(len(self.layers[i])):
                self.layers[i][j].activate(curr_x)
                next_x.append(self.layers[i][j].y)
            curr_x = next_x.copy()
            next_x.clear()

    def backward(self, output: int) -> None:
        err = 0 if round(self.layers[-1][0].y) == output else output - self.layers[-1][0].y
        self.layers[-1][0].updateDelta(error=err)
        self.layers[-1][0].updateWeights(self.eta, np.array([-1] + [neuron.y for neuron in self.layers[-2]]))
        curr_delta_w, next_delta_w = [], []
        curr_delta_w.append(self.layers[-1][0].delta * self.layers[-1][0].weights)
        for i in range(len(self.layers) - 2, 0, -1):
            for j in range(len(self.layers[i])):
                self.layers[i][j].updateDelta(delta_w_ls=np.array(curr_delta_w)[:, j + 1])
                self.layers[i][j].updateWeights(self.eta, np.array([-1] + [neuron.y for neuron in self.layers[i - 1]]))
                next_delta_w.append(self.layers[i][j].delta * self.layers[i][j].weights)
            curr_delta_w = next_delta_w.copy()
            next_delta_w.clear()

    def getOutputWeights(self) -> np:
        return self.layers[-1][0].weights.copy()

    def tryAccuracy(self, inputs: np.ndarray, outputs: np.ndarray) -> str:
        if inputs.size == 0:
            return 'No Data'
        correct = 0
        for i in range(len(inputs)):
            self.forward(inputs[i])
            if round(self.layers[-1][0].y) == outputs[i]:
                correct += 1
        return str(round(correct / len(inputs) * 100, 4)) + ' %'
