import sys

import qdarkstyle
from qdarkstyle.light.palette import LightPalette
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QMainWindow, QDialog
from PyQt5.QtGui import QDesktopServices, QIcon
from PyQt5.QtCore import QUrl
from qt_material import apply_stylesheet
from QCandyUi.CandyWindow import colorful



import DB
import ML_win
import login
import ml

# @colorful('blueDeep')
class main_win_Ui(QWidget, ML_win.Ui_Dialog):
    def __init__(self):
        super(main_win_Ui, self).__init__()
        self.setupUi(self)




# de
class db_Ui(QMainWindow, DB.Ui_MainWindow):
    def __init__(self):
        super(db_Ui, self).__init__()
        self.setupUi(self)

class ml_Ui(QWidget, ml.Ui_ML_Dialog):
    def __init__(self):
        super(ml_Ui, self).__init__()
        self.setupUi(self)


if __name__ == '__main__':

    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    app = QtWidgets.QApplication(sys.argv)
    app.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling,True)

    # app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyqt5', palette=LightPalette())) # 浅色
    app.setWindowIcon(QIcon('logo.ico'))
    # apply_stylesheet(app, theme='light_blue.xml')
    dialog = login.LoginWindow()
    # dialog.setWindowIcon(QIcon('logo.ico'))
    MainWindow = QtWidgets.QWidget()


    def open_url():
        # 在这里添加你要打开的 URL
        print('1')
        url = QUrl('https://www.custai.top/login/')
        QDesktopServices.openUrl(url)
        return

    if dialog.exec_() == QDialog.Accepted:
    # if True:
        ui = main_win_Ui()
        # ui = ML_win.Ui_Dialog()
        ui.setupUi(MainWindow)
        MainWindow.show()

        db = db_Ui() # 数据处理软件
        ml = ml_Ui()

        ui.pushButton_db.clicked.connect(
            lambda: {db.show()}
        )
        ui.pushButton_db2.clicked.connect(
            open_url
        )
        ui.pushButton_ml.clicked.connect(
            lambda: {ml.show()}
        )


        sys.exit(app.exec_())
