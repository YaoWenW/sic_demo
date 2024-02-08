import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QMessageBox, QDialog
from PyQt5.QtGui import QFont
from PyQt5 import QtCore
import qdarkstyle
#

class LoginWindow(QDialog):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.flag = 0
    def init_ui(self):
        self.setWindowTitle('登录 - 盾构智能化掘进系统')  # 在这里修改软件标题
        self.setGeometry(300, 200, 400, 300)
        self.setFixedSize(400, 300)  # 固定窗口大小
        # self.setStyleSheet("background-color: #f0f0f0;")

        self.title_label = QLabel('欢迎登录', self)
        self.title_label.setFont(QFont("Arial", 18))
        self.title_label.setGeometry(140, 30, 120, 30)

        self.username_input = QLineEdit(self)
        self.username_input.setGeometry(50, 100, 300, 30)
        self.username_input.setPlaceholderText('用户名')
        # self.username_input.setStyleSheet("background-color: #ffffff; border: 1px solid #ccc; border-radius: 5px;")

        self.password_input = QLineEdit(self)
        self.password_input.setGeometry(50, 150, 300, 30)
        self.password_input.setPlaceholderText('密码')
        self.password_input.setEchoMode(QLineEdit.Password)
        # self.password_input.setStyleSheet("background-color: #ffffff; border: 1px solid #ccc; border-radius: 5px;")

        self.login_button = QPushButton('登录', self)
        self.login_button.setGeometry(50, 210, 300, 40)
        self.login_button.setStyleSheet("background-color: #4CAF50; color: white; border-radius: 5px;")
        self.login_button.clicked.connect(self.login_attempt)

    def login_attempt(self):
        # 这里可以添加登录验证逻辑，暂时使用简单的示例
        username = self.username_input.text()
        password = self.password_input.text()

        # 假设用户名是"user"，密码是"password"，用于示例登录
        if username == 'wyw' and password == '123':
            self.successful_login()
        else:
            QMessageBox.warning(self, '登录失败', '用户名或密码错误')

    def successful_login(self):
        # self.close()  #/ 登录成功后关闭登录窗口，可在此处打开主窗口
        # self.flag == 1
        self.accept()

if __name__ == '__main__':

    app = QApplication(sys.argv)
    # app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    login_window = LoginWindow()
    login_window.show()
    sys.exit(app.exec_())
