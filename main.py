import sys

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QMainWindow, QDialog
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtCore import QUrl

import DB
import ML_win
import login


class main_win_Ui(QMainWindow, ML_win.Ui_Dialog):
    def __init__(self):
        super(main_win_Ui, self).__init__()
        self.setupUi(self)

#
# class qsm_w_Ui(QMainWindow, qsm_w.Ui_Dialog):
#     def __init__(self):
#         super(qsm_w_Ui, self).__init__()
#         self.setupUi(self)
#
#
# class table_Ui(QMainWindow, table.Ui_Dialog):
#     def __init__(self):
#         super(table_Ui, self).__init__()
#         self.setupUi(self)

def open_url(self):
    # 在这里添加你要打开的 URL
    url = QUrl('https://www.example.com')
    QDesktopServices.openUrl(url)

def go_to_page_1(self):
    # 在这里添加跳转到页面1的代码
    print('跳转到页面1')

def go_to_page_2(self):
    # 在这里添加跳转到页面2的代码
    print('跳转到页面2')
# de
class db_Ui(QMainWindow, DB.Ui_MainWindow):
    def __init__(self):
        super(db_Ui, self).__init__()
        self.setupUi(self)

if __name__ == '__main__':

    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    app = QtWidgets.QApplication(sys.argv)
    # app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    dialog = login.LoginWindow()
    MainWindow = QtWidgets.QMainWindow()
    if dialog.exec_() == QDialog.Accepted:
        # if True:
        ui = main_win_Ui()
        ui.setupUi(MainWindow)

        db = db_Ui() # 数据处理软件

        MainWindow.show()

        ui.pushButton_db.clicked.connect(
            lambda: {db.show()}
        )


        sys.exit(app.exec_())
