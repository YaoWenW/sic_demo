import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QMessageBox

class PredictionWindow(QWidget):
    def __init__(self, x_test_columns):
        super().__init__()
        self.x_test_columns = x_test_columns
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        self.entry_fields = []
        value_list = [2, 15, 1500, 1.5, 1.1, 10000, 20, 20, 2, 15]
        for i,column in enumerate(self.x_test_columns):
            label = QLabel(column)
            layout.addWidget(label)
            entry = QLineEdit()
            entry.setText(str(value_list[i]))  # 设置默认值
            layout.addWidget(entry)
            self.entry_fields.append(entry)

        self.predict_button = QPushButton('推荐')
        self.label_res = QLabel()
        layout.addWidget(self.predict_button)
        layout.addWidget(self.label_res)

        # self.save_button = QPushButton('关闭')
        # layout.addWidget(self.save_button)

        self.setLayout(layout)
        self.setWindowTitle('实时推荐')

    def perform_prediction(self):
        try:
            input_values = [[float(entry.text()) for entry in self.entry_fields]]

            # 在此处添加模型预测的代码
            # model.predict(input_values)
            # 替换上面的注释行为实际的模型预测代码
            QMessageBox.information(self, '预测结果', '这里是预测结果')  # 暂时用消息框代替预测结果
        except Exception as e:
            QMessageBox.critical(self, '错误', f'发生错误: {e}')

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        self.predict_button = QPushButton('预测新窗口')
        self.predict_button.clicked.connect(self.show_prediction_window)
        layout.addWidget(self.predict_button)

        self.setLayout(layout)
        self.setWindowTitle('主界面')

    def show_prediction_window(self):
        x_test_columns = ['特征1', '特征2', '特征3']  # 替换为你实际的列名列表
        self.prediction_window = PredictionWindow(x_test_columns)
        self.prediction_window.show()

def run_app():
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
# 运行主界面应用
    run_app()
