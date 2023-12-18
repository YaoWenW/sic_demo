import sys
import time
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout
from PyQt5.QtCore import QThread, pyqtSignal
import main
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier

class TrainingThread(QThread):
    finished = pyqtSignal()  # 发送信号以指示训练完成
    interrupted = pyqtSignal()  # 发送信号以指示训练被中断

    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name

    def run(self):
        if self.model_name == 'BSR_model':
            self.train_random_forest()
        elif self.model_name == '线性回归':
            self.train_linear_regression()
        elif self.model_name == '神经网络':
            self.train_neural_network()

    def train_random_forest(self):
        try:
            # 模拟训练过程
            main.BSR_train()
        except KeyboardInterrupt:
            return
        self.finished.emit()

    def train_linear_regression(self):
        try:
            # 模拟训练过程
            main.ANN_train()
        except KeyboardInterrupt:
            return
        self.finished.emit()

    def train_neural_network(self):
        try:
            # 模拟训练过程
            for i in range(10):
                print(f"Epoch {i} - Training 神经网络")
                time.sleep(1)  # 模拟训练操作
                if self.isInterruptionRequested():
                    self.interrupted.emit()
                    return
        except KeyboardInterrupt:
            return
        self.finished.emit()

class TrainingController:
    def __init__(self):
        self.training_threads = {}

    def start_training(self, model_name):
        self.training_threads[model_name] = TrainingThread(model_name)
        self.training_threads[model_name].finished.connect(self.training_finished)
        self.training_threads[model_name].start()

    def interrupt_training(self, model_name):
        if model_name in self.training_threads:
            self.training_threads[model_name].terminate()

    def training_finished(self):
        print("训练完成！")

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.training_controller = TrainingController()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        models = ['随机森林', '线性回归', '神经网络']

        for model in models:
            train_button = QPushButton(f'开始训练 {model}', self)
            train_button.clicked.connect(lambda _, m=model: self.training_controller.start_training(m))

            interrupt_button = QPushButton(f'中断训练 {model}', self)
            interrupt_button.clicked.connect(lambda _, m=model: self.training_controller.interrupt_training(m))

            layout.addWidget(train_button)
            layout.addWidget(interrupt_button)

        self.setLayout(layout)
        self.setWindowTitle('模型训练')
        self.show()

def run_app():
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())

if __name__ == '__main__':
    run_app()
