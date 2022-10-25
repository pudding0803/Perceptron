import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QFileDialog

from Perceptron import Perceptron
from UI import Ui_MainWindow


class MainWindowController(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_control()
        self.data = []
        self.dimension = 0
        self.categories = {}

    def setup_control(self):
        self.ui.data_button.clicked.connect(self.chooseData)
        self.ui.train_button.clicked.connect(self.training)

    def chooseData(self):
        filename, filetype = QFileDialog.getOpenFileName(self, '選擇資料集', './', '文字文件 (*.txt)')
        self.ui.path_label.setText('...')
        self.ui.train_result.setText('...')
        self.ui.test_result.setText('...')
        self.ui.weight_result.setText('...')
        self.ui.hidden_line.setText('')
        self.ui.alert_label.setText('')
        self.ui.progressBar.setValue(0)
        if filename == '':
            self.ui.train_button.setEnabled(False)
            return
        with open(filename, 'r') as file:
            self.ui.path_label.setText(filename.split('/')[-1])
            line = file.readline().split()
            self.dimension = len(line) - 1
            self.data = np.array([[float(i) for i in line]])
            for line in file:
                self.data = np.append(self.data, [[float(i) for i in line.split()]], axis=0)
        self.categories = {}
        for i in self.data[:, -1]:
            if i not in self.categories.keys():
                self.categories[i] = len(self.categories)
        if len(self.categories) > 2:
            self.ui.alert_label.setText('Sorry, we only support binary classification :(')
        else:
            self.ui.train_button.setEnabled(True)

    def training(self):
        self.ui.train_button.setEnabled(False)
        self.ui.progressBar.setValue(0)
        self.ui.alert_label.setText('')
        layer_num = self.ui.hidden_line.text().split()
        for i in range(len(layer_num)):
            if not layer_num[i].isdecimal() or int(layer_num[i]) <= 0:
                self.ui.alert_label.setText('Hidden Layers should be all positive integers or empty :(')
                self.ui.train_button.setEnabled(True)
                return
            layer_num[i] = int(layer_num[i])
        if len(layer_num) != 0 and layer_num[-1] != self.dimension:
            self.ui.alert_label.setText(
                f'The last layer of Hidden Layers should be {self.dimension} (dimensions) Neurons :(')
            self.ui.train_button.setEnabled(True)
            return
        layer_num = [self.dimension] + layer_num + [1]
        perceptron = Perceptron(self.ui.rate_box.value(), self.dimension, layer_num)
        ratio = int(len(self.data) * self.ui.ratio_box.value())
        np.random.shuffle(self.data)
        train_data, test_data = self.data[:ratio, :], self.data[ratio:, :]
        c = []
        epoch = self.ui.epoch_box.value()
        stage = 10 if epoch >= 10 else epoch
        for i in range(epoch):
            if self.dimension == 2 and i % (epoch // stage) == 0:
                c.append(perceptron.getOutputWeights())
            self.ui.progressBar.setValue((i + 1) * 100 / epoch)
            for d in train_data:
                perceptron.forward(d[:-1])
                perceptron.backward(self.categories[d[-1]])
        self.ui.train_result.setText(
            perceptron.tryAccuracy(train_data[:, :-1], np.array(list(map(self.categories.get, train_data[:, -1])))))
        self.ui.test_result.setText(
            perceptron.tryAccuracy(test_data[:, :-1], np.array(list(map(self.categories.get, test_data[:, -1])))))
        c.append(perceptron.getOutputWeights())
        self.ui.weight_result.setText(f'{np.array2string(c[-1], precision=2, suppress_small=True)}')
        color = np.array(self.data[:, -1]).astype(int).astype(str)
        symbol = ['train' if np.any(np.all(i == train_data, axis=1)) else 'test' for i in self.data]
        if self.dimension == 2:
            fig = px.scatter(x=self.data[:, 0], y=self.data[:, 1], color=color, symbol=symbol)
            x = np.linspace(np.min(self.data[:, 0]) - np.ptp(self.data[:, 0]) / 2,
                            np.max(self.data[:, 0]) + np.ptp(self.data[:, 0]) / 2, 2)
            for i in range(len(c)):
                y = (c[i][0] - x * c[i][1]) / c[i][2]
                fig.add_trace(
                    go.Scatter(x=x, y=y, name=f'{round(i * 100 / stage, 2)}% trained',
                               line=dict() if i == len(c) - 1 else dict(dash='dot')))
            fig.update_xaxes(range=[np.min(self.data[:, 0]) - np.ptp(self.data[:, 0]) / 2,
                                    np.max(self.data[:, 0]) + np.ptp(self.data[:, 0]) / 2])
            fig.update_yaxes(range=[np.min(self.data[:, 1]) - np.ptp(self.data[:, 1]) / 2,
                                    np.max(self.data[:, 1]) + np.ptp(self.data[:, 1]) / 2])
            fig.show()
        elif self.dimension == 3:
            fig = px.scatter_3d(x=self.data[:, 0], y=self.data[:, 1], z=self.data[:, 2], color=color, symbol=symbol)
            xs = np.linspace(np.min(self.data[:, 0]) - np.ptp(self.data[:, 0]) / 2,
                             np.max(self.data[:, 0]) + np.ptp(self.data[:, 0]) / 2, 5)
            ys = np.linspace(np.min(self.data[:, 1]) - np.ptp(self.data[:, 1]) / 2,
                             np.max(self.data[:, 1]) + np.ptp(self.data[:, 1]) / 2, 5)
            x, y = np.meshgrid(xs, ys)
            z = (c[-1][0] - c[-1][1] * x - c[-1][2] * y) / c[-1][3]
            fig.add_trace(go.Surface(x=x, y=y, z=z, opacity=0.5, colorscale='Picnic', showscale=False))
            fig.update_layout(scene=dict(
                xaxis=dict(range=[np.min(self.data[:, 0]) - np.ptp(self.data[:, 0]) / 2,
                                  np.max(self.data[:, 0]) + np.ptp(self.data[:, 0]) / 2]),
                yaxis=dict(range=[np.min(self.data[:, 1]) - np.ptp(self.data[:, 1]) / 2,
                                  np.max(self.data[:, 1]) + np.ptp(self.data[:, 1]) / 2]),
                zaxis=dict(range=[np.min(self.data[:, 2]) - np.ptp(self.data[:, 2]) / 2,
                                  np.max(self.data[:, 2]) + np.ptp(self.data[:, 2]) / 2])
            ))
            fig.show()
        else:
            self.ui.alert_label.setText('Sorry, we only support two and three dimension data visualization :(')
        self.ui.train_button.setEnabled(True)
