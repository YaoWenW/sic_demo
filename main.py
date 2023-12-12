import sys

import qdarkstyle
from qdarkstyle.light.palette import LightPalette
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QMainWindow, QDialog, QTableWidgetItem
from PyQt5.QtGui import QDesktopServices, QIcon
from PyQt5.QtCore import QUrl
from qt_material import apply_stylesheet
from QCandyUi.CandyWindow import colorful
from PyQt5.QtWidgets import QFileDialog
import pandas as pd
from PyQt5.Qt import *
import numpy as np
import DB
import ML_win
import login
import ml
from sklearn.model_selection import train_test_split
import ML_demo

# @colorful('blueDeep')
class main_win_Ui(QWidget, ML_win.Ui_Dialog):
    def __init__(self):
        super(main_win_Ui, self).__init__()
        self.setupUi(self)


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

    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    # app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyqt5', palette=LightPalette())) # 浅色
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

    # 打开路径

    def input(input_table):
        input_table_rows = input_table.shape[0]
        input_table_colunms = input_table.shape[1]
        print(input_table_rows, input_table_colunms)
        input_table_header = input_table.columns.values.tolist()

        ml.tableWidget.setColumnCount(input_table_colunms)
        ml.tableWidget.setRowCount(input_table_rows)
        ml.tableWidget.setHorizontalHeaderLabels(input_table_header)
        for i in range(input_table_rows):
            input_table_rows_values = input_table.iloc[[i]]
            input_table_rows_values_array = np.array(input_table_rows_values)
            input_table_rows_values_list = input_table_rows_values_array.tolist()[0]
            for j in range(input_table_colunms):
                input_table_items_list = input_table_rows_values_list[j]
                input_table_items = str(input_table_items_list)
                newItem = QTableWidgetItem(input_table_items)
                newItem.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
                ml.tableWidget.setItem(i, j, newItem)

    def upload_excel():
        try:
            _translate = QtCore.QCoreApplication.translate
            directory1 = QFileDialog.getOpenFileName(None, "选择文件", "H:/")
            path = directory1[0]
            if path:
                if "csv" in path:
                    ori_data = pd.read_csv(path)
                elif "xls" in path:
                    ori_data = pd.read_excel(path)
                else:
                    print("文件格式有误，请上传.csv或.xlsx格式文件")
                # if ori_data:
                t = ""  # 读取列名
                for i in ori_data.columns:
                    t = t + i if i == ori_data.columns[0] else t + ',' + i
                m, n = ori_data.shape[0], ori_data.shape[1]
                ml.label_upload.setText(f'上传成功！一共{m}行{n}列')
                input(ori_data)
        except:
            print('导入文件异常')
        return

    def minmax_norm(df_input):
        return (df_input - df_input.min()) / (df_input.max() - df_input.min())

    def data_split(data):
        global x_train
        global x_test
        global y_train
        global y_test
        try:
            if ml.checkBox_minmax.isChecked():
                print('归一化')
            if ml.checkBox_std.isChecked():
                print('标准化')
            goal_col = ml.lineEdit_goal.text()
            # goal_col = '标注'
            print(goal_col)
            if data and goal_col not in data.columns:
                print('目标列名输入有误')
                return
            # useless_cols = ['标注', '区间', '软硬程度']
            useless_cols = ml.lineEdit_useless.text().split(',')
            useless_cols.append(goal_col)
            print(useless_cols)
            train_cols = [col for col in data.columns if col not in useless_cols]
            print(train_cols)

            X = data[train_cols]
            y = data[goal_col]

            test_size = ml.spinBox_split.value() / 100
            if ml.comboBox_split.currentText() == '随机划分':
                x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=2000, shuffle=True)
            else:
                x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=2000, shuffle=False)
        except:
            print('error')

        return

    # if dialog.exec_() == QDialog.Accepted:
    if True:
        ui = main_win_Ui()
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

        ml.pushButton_Upload.clicked.connect(
            upload_excel
        )

        ml.pushButton_split.clicked.connect(
            data_split
        )

        sys.exit(app.exec_())