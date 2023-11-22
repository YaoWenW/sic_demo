import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PyQt5 import QtCore, QtWidgets
from PyQt5.Qt import *
from PyQt5.QtWidgets import QFileDialog
from scipy import stats
import sys
import qdarkstyle
# 设置显示中文字体

matplotlib.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

ori_data = None
path = None
save_path = None
data_nozero = None
data_nospecial = None
data_yuzhi = None
data_denoise = None
data = None
data_rings = None


# 切分
def split(string):  # 遍历字符串
    cols = []
    for col in string.split(','):
        cols.append(col)
    return cols


# 归一化数据
def minmax_norm(df_input):
    return (df_input - df_input.min()) / (df_input.max() - df_input.min())


def pivot(df_input, x="环号", df='mean'):
    for colum in df_input.columns:
        if "环" in colum:
            x = colum
    return df_input.pivot_table(index=x,  # 透视的行，分组依据
                                # values=df_input[df_input.columns],  # 值
                                # aggfunc=df
                                )  # 聚合函数


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(547, 444)  # 主窗口大小
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_40 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_40.setObjectName("gridLayout_40")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.tabWidget.setElideMode(QtCore.Qt.ElideNone)
        self.tabWidget.setDocumentMode(True)
        self.tabWidget.setTabsClosable(False)
        self.tabWidget.setMovable(False)
        self.tabWidget.setTabBarAutoHide(False)
        self.tabWidget.setObjectName("tabWidget")
        self.tab_7 = QtWidgets.QWidget()
        self.tab_7.setObjectName("tab_7")

        self.gridLayout_27 = QtWidgets.QGridLayout(self.tab_7)
        self.gridLayout_27.setObjectName("gridLayout_27")
        self.groupBox_13 = QtWidgets.QGroupBox(self.tab_7)
        self.groupBox_13.setEnabled(True)
        self.groupBox_13.setObjectName("groupBox_13")

        self.gridLayout_28 = QtWidgets.QGridLayout(self.groupBox_13)
        self.gridLayout_28.setObjectName("gridLayout_28")
        self.label_7 = QtWidgets.QLabel(self.groupBox_13)
        self.label_7.setObjectName("label_7")
        self.gridLayout_28.addWidget(self.label_7, 2, 0, 1, 1)
        self.label_7 = QtWidgets.QLabel(self.groupBox_13)
        self.label_7.setObjectName("label_7")
        self.gridLayout_28.addWidget(self.label_7, 2, 0, 1, 1)
        self.plainTextEdit_4 = QtWidgets.QPlainTextEdit(self.groupBox_13)
        self.plainTextEdit_4.setPlainText("")
        self.plainTextEdit_4.setObjectName("plainTextEdit_4")
        self.gridLayout_28.addWidget(self.plainTextEdit_4, 1, 0, 1, 1)
        self.groupBox_14 = QtWidgets.QGroupBox(self.tab_7)
        # self.groupBox_14 = QtWidgets.QGroupBox(self.groupBox_13)
        self.groupBox_14.setObjectName("groupBox_14")

        self.gridLayout_29 = QtWidgets.QGridLayout(self.groupBox_14)
        self.gridLayout_29.setObjectName("gridLayout_29")
        self.lineEdit_17 = QtWidgets.QLineEdit(self.groupBox_14)
        self.lineEdit_17.setObjectName("lineEdit_17")
        self.gridLayout_29.addWidget(self.lineEdit_17, 1, 0, 1, 1)
        self.pushButton_20 = QtWidgets.QPushButton(self.groupBox_14)
        self.pushButton_20.setAutoDefault(False)
        self.pushButton_20.setObjectName("pushButton_20")
        self.pushButton_20.setText("确定")
        self.gridLayout_29.addWidget(self.pushButton_20, 1, 2, 1, 1)
        self.plainTextEdit_5 = QtWidgets.QPlainTextEdit(self.groupBox_14)
        self.plainTextEdit_5.setObjectName("plainTextEdit_5")
        self.gridLayout_29.addWidget(self.plainTextEdit_5, 3, 0, 1, 1)
        self.label_8 = QtWidgets.QLabel(self.groupBox_14)
        self.label_8.setObjectName("label_8")
        self.gridLayout_29.addWidget(self.label_8, 0, 0, 1, 1)
        self.label_81 = QtWidgets.QLabel(self.groupBox_14)
        self.label_81.setObjectName("label_81")
        self.label_81.setText("更改为标准数据格式（直接替换为当前名称，不改变顺序）")

        self.gridLayout_29.addWidget(self.label_81, 2, 0, 1, 1)
        self.pushButton_21 = QtWidgets.QPushButton(self.groupBox_14)
        self.pushButton_21.setAutoDefault(False)
        self.pushButton_21.setObjectName("pushButton_21")
        self.gridLayout_29.addWidget(self.pushButton_21, 3, 2, 1, 1)

        self.gridLayout_28.addWidget(self.groupBox_14, 4, 0, 1, 1)
        self.label_9 = QtWidgets.QLabel(self.groupBox_13)
        self.label_9.setObjectName("label_9")
        self.gridLayout_28.addWidget(self.label_9, 0, 0, 1, 1)
        self.describe1_3 = QtWidgets.QTableWidget(self.groupBox_13)
        self.describe1_3.setObjectName("describe1_3")
        self.describe1_3.setColumnCount(0)
        self.describe1_3.setRowCount(0)
        self.gridLayout_28.addWidget(self.describe1_3, 3, 0, 1, 1)
        self.gridLayout_27.addWidget(self.groupBox_13, 1, 0, 1, 1)
        self.gridLayout_30 = QtWidgets.QGridLayout()
        self.gridLayout_30.setObjectName("gridLayout_30")
        self.lineEdit_18 = QtWidgets.QLineEdit(self.tab_7)
        self.lineEdit_18.setText("")
        self.lineEdit_18.setObjectName("lineEdit_18")
        self.gridLayout_30.addWidget(self.lineEdit_18, 0, 0, 1, 1)
        self.pushButton_22 = QtWidgets.QPushButton(self.tab_7)
        self.pushButton_22.setObjectName("pushButton_22")
        self.gridLayout_30.addWidget(self.pushButton_22, 0, 2, 1, 1)
        self.pushButton_23 = QtWidgets.QPushButton(self.tab_7)
        self.pushButton_23.setObjectName("pushButton_23")
        self.gridLayout_30.addWidget(self.pushButton_23, 0, 1, 1, 1)
        self.gridLayout_27.addLayout(self.gridLayout_30, 0, 0, 1, 1)

        self.tabWidget.addTab(self.tab_7, "")
        self.tab_8 = QtWidgets.QWidget()
        self.tab_8.setObjectName("tab_8")
        self.gridLayout_31 = QtWidgets.QGridLayout(self.tab_8)
        self.gridLayout_31.setObjectName("gridLayout_31")
        self.groupBox_15 = QtWidgets.QGroupBox(self.tab_8)
        self.groupBox_15.setCheckable(True)
        self.groupBox_15.setObjectName("groupBox_15")
        self.gridLayout_32 = QtWidgets.QGridLayout(self.groupBox_15)
        self.gridLayout_32.setObjectName("gridLayout_32")
        self.gridLayout_33 = QtWidgets.QGridLayout()
        self.gridLayout_33.setObjectName("gridLayout_33")
        self.lineEdit_19 = QtWidgets.QLineEdit(self.groupBox_15)
        self.lineEdit_19.setText("")
        self.lineEdit_19.setObjectName("lineEdit_19")
        self.gridLayout_33.addWidget(self.lineEdit_19, 0, 1, 1, 1)
        self.pushButton_24 = QtWidgets.QPushButton(self.groupBox_15)
        self.pushButton_24.setAutoDefault(False)
        self.pushButton_24.setObjectName("pushButton_24")
        self.gridLayout_33.addWidget(self.pushButton_24, 0, 2, 1, 1)
        self.label_nozero = QtWidgets.QLabel(self.groupBox_15)
        self.label_nozero.setObjectName("label_nozero")
        self.gridLayout_33.addWidget(self.label_nozero, 0, 3, 1, 1)
        self.label_nozero.hide()  # _nozero
        self.label_sx = QtWidgets.QLabel(self.groupBox_15)
        self.label_sx.setObjectName("label_sx")
        self.gridLayout_33.addWidget(self.label_sx, 0, 0, 1, 1)

        self.gridLayout_32.addLayout(self.gridLayout_33, 0, 0, 1, 1)
        self.gridLayout_31.addWidget(self.groupBox_15, 0, 0, 1, 1)
        self.groupBox_16 = QtWidgets.QGroupBox(self.tab_8)
        self.groupBox_16.setCheckable(True)
        self.groupBox_16.setObjectName("groupBox_16")
        self.gridLayout_34 = QtWidgets.QGridLayout(self.groupBox_16)
        self.gridLayout_34.setObjectName("gridLayout_34")
        self.gridLayout_35 = QtWidgets.QGridLayout()
        self.gridLayout_35.setObjectName("gridLayout_35")
        self.comboBox_5 = QtWidgets.QComboBox(self.groupBox_16)
        self.comboBox_5.setObjectName("comboBox_5")
        self.comboBox_5.addItem("")
        self.comboBox_5.addItem("")
        # self.comboBox_5.addItem("")
        # self.comboBox_5.addItem("")
        self.gridLayout_35.addWidget(self.comboBox_5, 0, 0, 1, 1)
        self.lineEdit_20 = QtWidgets.QLineEdit(self.groupBox_16)
        self.lineEdit_20.setText("")
        self.lineEdit_20.setObjectName("lineEdit_20")
        self.gridLayout_35.addWidget(self.lineEdit_20, 0, 1, 1, 3)
        self.pushButton_25 = QtWidgets.QPushButton(self.groupBox_16)
        self.pushButton_25.setAutoDefault(False)
        self.pushButton_25.setObjectName("pushButton_25")
        self.gridLayout_35.addWidget(self.pushButton_25, 0, 4, 1, 1)
        self.label_nospecial = QtWidgets.QLabel(self.groupBox_16)
        self.label_nospecial.setObjectName("label_nospecial")
        self.gridLayout_35.addWidget(self.label_nospecial, 0, 5, 1, 1)
        self.label_nospecial.hide()  # nospecial
        self.label_yz = QtWidgets.QLabel(self.groupBox_16)
        self.label_yz.setObjectName("label_yuzhi")
        self.label_yz.setText("设置阈值")
        self.gridLayout_35.addWidget(self.label_yz, 1, 0, 1, 1)
        self.lineEdit_yz = QtWidgets.QLineEdit(self.groupBox_16)
        self.lineEdit_yz.setText("")
        self.lineEdit_yz.setPlaceholderText("输入分析列")
        self.gridLayout_35.addWidget(self.lineEdit_yz, 1, 1, 1, 1)
        self.lineEdit_min = QtWidgets.QLineEdit(self.groupBox_16)
        self.lineEdit_min.setText("")
        self.lineEdit_min.setPlaceholderText("min")
        self.gridLayout_35.addWidget(self.lineEdit_min, 1, 2, 1, 1)
        self.lineEdit_max = QtWidgets.QLineEdit(self.groupBox_16)
        self.lineEdit_max.setText("")
        self.lineEdit_max.setPlaceholderText("max")
        self.gridLayout_35.addWidget(self.lineEdit_max, 1, 3, 1, 1)
        self.pushButton_yz = QtWidgets.QPushButton(self.groupBox_16)
        self.pushButton_yz.setAutoDefault(False)
        self.pushButton_yz.setObjectName("pushButton_yz")
        self.pushButton_yz.setText("确定")
        self.gridLayout_35.addWidget(self.pushButton_yz, 1, 4, 1, 1)
        self.label_yz1 = QtWidgets.QLabel(self.groupBox_16)
        self.label_yz1.setObjectName("label_yz1")
        self.gridLayout_35.addWidget(self.label_yz1, 1, 5, 1, 1)
        self.label_yz1.hide()  # nospecial

        self.gridLayout_34.addLayout(self.gridLayout_35, 0, 0, 1, 1)
        self.gridLayout_31.addWidget(self.groupBox_16, 1, 0, 1, 1)
        self.groupBox_17 = QtWidgets.QGroupBox(self.tab_8)
        self.groupBox_17.setFlat(False)
        self.groupBox_17.setCheckable(True)
        self.groupBox_17.setObjectName("groupBox_17")
        self.gridLayout_36 = QtWidgets.QGridLayout(self.groupBox_17)
        self.gridLayout_36.setObjectName("gridLayout_36")
        self.gridLayout_37 = QtWidgets.QGridLayout()
        self.gridLayout_37.setObjectName("gridLayout_37")
        self.lineEdit_21 = QtWidgets.QLineEdit(self.groupBox_17)
        self.lineEdit_21.setText("")
        self.lineEdit_21.setObjectName("lineEdit_21")
        self.gridLayout_37.addWidget(self.lineEdit_21, 0, 2, 1, 1)
        self.comboBox_6 = QtWidgets.QComboBox(self.groupBox_17)
        self.comboBox_6.setObjectName("comboBox_6")
        self.comboBox_6.addItem("")
        self.comboBox_6.addItem("")
        # self.comboBox_6.addItem("")
        # self.comboBox_6.addItem("")
        # self.comboBox_6.addItem("")
        self.gridLayout_37.addWidget(self.comboBox_6, 0, 1, 1, 1)
        self.pushButton_26 = QtWidgets.QPushButton(self.groupBox_17)
        self.pushButton_26.setAutoDefault(False)
        self.pushButton_26.setObjectName("pushButton_26")
        self.gridLayout_37.addWidget(self.pushButton_26, 0, 3, 1, 1)
        self.label_denoise = QtWidgets.QLabel(self.groupBox_17)
        self.label_denoise.setObjectName("label_denoise")
        self.gridLayout_37.addWidget(self.label_denoise, 0, 4, 1, 1)
        self.label_denoise.hide()  # denoise
        self.gridLayout_36.addLayout(self.gridLayout_37, 0, 0, 1, 1)
        self.gridLayout_31.addWidget(self.groupBox_17, 2, 0, 1, 1)

        self.groupBox_18 = QtWidgets.QGroupBox(self.tab_8)
        self.groupBox_18.setMouseTracking(False)
        self.groupBox_18.setStyleSheet("font: 9pt \"黑体\";")
        self.groupBox_18.setFlat(True)
        self.groupBox_18.setCheckable(False)
        self.groupBox_18.setObjectName("groupBox_18")

        self.gridLayout_38 = QtWidgets.QGridLayout(self.groupBox_18)
        self.gridLayout_38.setObjectName("gridLayout_38")
        self.label_x0 = QtWidgets.QLabel(self.groupBox_18)
        self.label_x0.setText('1111')
        self.gridLayout_38.addWidget(self.label_x0, 0, 0, 1, 1)
        self.label_x0.hide()
        self.label_x1 = QtWidgets.QLabel(self.groupBox_18)
        self.label_x1.setText('1111')
        self.gridLayout_38.addWidget(self.label_x1, 1, 0, 1, 1)
        self.label_x1.hide()
        self.label_x2 = QtWidgets.QLabel(self.groupBox_18)
        self.label_x2.setText('1111')
        self.gridLayout_38.addWidget(self.label_x2, 2, 0, 1, 1)
        self.label_x2.hide()

        self.gridLayout_31.addWidget(self.groupBox_18, 3, 0, 1, 1)
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.lineEdit_22 = QtWidgets.QLineEdit(self.tab_8)
        self.lineEdit_22.setObjectName("lineEdit_22")
        self.horizontalLayout_8.addWidget(self.lineEdit_22)
        self.pushButton_27 = QtWidgets.QPushButton(self.tab_8)
        self.pushButton_27.setObjectName("pushButton_27")
        self.comboBox_1 = QtWidgets.QComboBox(self.tab_8)
        self.comboBox_1.setObjectName("comboBox_1")
        self.comboBox_1.addItem("")
        self.comboBox_1.addItem("")
        self.comboBox_1.addItem("")
        self.comboBox_1.addItem("")
        self.comboBox_1.addItem("")
        self.horizontalLayout_8.addWidget(self.comboBox_1)
        self.horizontalLayout_8.addWidget(self.pushButton_27)
        self.pushButton_28 = QtWidgets.QPushButton(self.tab_8)
        self.pushButton_28.setObjectName("pushButton_28")
        self.horizontalLayout_8.addWidget(self.pushButton_28)
        self.gridLayout_31.addLayout(self.horizontalLayout_8, 4, 0, 1, 1)

        self.label_x3 = QtWidgets.QLabel(self.tab_8)
        self.label_x3.setText('导出成功!')
        self.horizontalLayout_8.addWidget(self.label_x3)
        self.gridLayout_31.addLayout(self.horizontalLayout_8, 5, 0, 1, 1)
        self.label_x3.hide()

        self.tabWidget.addTab(self.tab_8, "")
        self.tab_9 = QtWidgets.QWidget()
        self.tab_9.setObjectName("tab_9")
        self.gridLayout_39 = QtWidgets.QGridLayout(self.tab_9)
        self.gridLayout_39.setObjectName("gridLayout_39")
        self.formLayout_3 = QtWidgets.QFormLayout()
        self.formLayout_3.setObjectName("formLayout_3")
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.lineEdit_23 = QtWidgets.QLineEdit(self.tab_9)
        self.lineEdit_23.setObjectName("lineEdit_23")
        self.horizontalLayout_9.addWidget(self.lineEdit_23)
        self.pushButton_29 = QtWidgets.QPushButton(self.tab_9)
        self.pushButton_29.setObjectName("pushButton_29")
        self.horizontalLayout_9.addWidget(self.pushButton_29)
        self.formLayout_3.setLayout(2, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_9)
        self.horizontalLayout_10 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_10.setObjectName("horizontalLayout_10")
        self.lineEdit_24 = QtWidgets.QLineEdit(self.tab_9)
        self.lineEdit_24.setObjectName("lineEdit_24")
        self.horizontalLayout_10.addWidget(self.lineEdit_24)
        self.pushButton_30 = QtWidgets.QPushButton(self.tab_9)
        self.pushButton_30.setObjectName("pushButton_30")
        self.horizontalLayout_10.addWidget(self.pushButton_30)
        self.formLayout_3.setLayout(0, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_10)
        self.horizontalLayout_11 = QtWidgets.QHBoxLayout()
        self.lineEdit_25 = QtWidgets.QLineEdit(self.tab_9)
        self.horizontalLayout_11.addWidget(self.lineEdit_25)
        self.pushButton_31 = QtWidgets.QPushButton(self.tab_9)
        self.horizontalLayout_11.addWidget(self.pushButton_31)
        self.formLayout_3.setLayout(3, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_11)
        self.horizontalLayout_12 = QtWidgets.QHBoxLayout()
        self.lineEdit_26 = QtWidgets.QLineEdit(self.tab_9)
        self.horizontalLayout_12.addWidget(self.lineEdit_26)
        self.pushButton_32 = QtWidgets.QPushButton(self.tab_9)
        self.horizontalLayout_12.addWidget(self.pushButton_32)
        self.formLayout_3.setLayout(4, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_12)
        self.horizontalLayout_13 = QtWidgets.QHBoxLayout()
        self.lineEdit_27 = QtWidgets.QLineEdit(self.tab_9)
        self.horizontalLayout_13.addWidget(self.lineEdit_27)
        self.pushButton_33 = QtWidgets.QPushButton(self.tab_9)
        self.horizontalLayout_13.addWidget(self.pushButton_33)
        self.formLayout_3.setLayout(5, QtWidgets.QFormLayout.FieldRole, self.horizontalLayout_13)
        self.gridLayout_39.addLayout(self.formLayout_3, 0, 0, 1, 1)
        self.tabWidget.addTab(self.tab_9, "")
        self.gridLayout_40.addWidget(self.tabWidget, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "盾构掘进数据处理系统"))
        self.tabWidget.setWhatsThis(_translate("MainWindow", "<html><head/><body><p><br/></p></body></html>"))
        self.groupBox_13.setTitle(_translate("MainWindow", "数据描述"))
        self.label_7.setText(_translate("MainWindow", "数据预览"))
        self.plainTextEdit_4.setPlaceholderText(_translate("MainWindow", "所有列名"))  ##
        self.groupBox_14.setTitle(_translate("MainWindow", "特征选择"))
        self.lineEdit_17.setPlaceholderText(_translate("MainWindow", "选择分析列，形如 1,2,3"))
        col = "环号(r),刀盘转矩(kN*m),刀盘转速(r/min),顶部土压(bar),总推进力(kN),推进速度(mm/min),设备侧滚(mm)," \
              "设备倾角(°),贯入度(mm),土压2#(bar),土压3#(bar),土压4#(bar),土压5#(bar),土压6#(bar),土压平均值(bar)," \
              "1#铰接油缸行程(mm),3#铰接油缸行程(mm),5#铰接油缸行程(mm),7#铰接油缸行程(mm),右A组油缸推进压力(bar)," \
              "下B组油缸推进压力(bar),左C组油缸推进压力(bar),上D组油缸推进压力(bar),螺旋机上卸料门开度(mm)," \
              "螺旋机下卸料门开度(mm),螺旋机压力测量值(bar),螺旋机土压测量值后(bar),螺旋机转矩(kN*m),螺旋机速度(r/min)," \
              "1#膨润土流量(L),2#膨润土流量(L),泡沫原液流量(L),A组油缸行程(mm),B组油缸行程(mm),C组油缸行程(mm),D组油缸行程(mm)"

        self.plainTextEdit_5.setPlainText(_translate("MainWindow", col))
        self.label_8.setText(_translate("MainWindow", "选择分析特征"))
        self.pushButton_21.setText(_translate("MainWindow", "确定"))
        self.label_9.setText(_translate("MainWindow", "所有列名"))  ##
        self.lineEdit_18.setPlaceholderText(_translate("MainWindow", "CSV或XLSX文件"))  ##
        self.pushButton_22.setText(_translate("MainWindow", "导入"))  ##
        self.pushButton_23.setText(_translate("MainWindow", "选择文件"))  ##
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_7), _translate("MainWindow", "文件导入"))
        self.groupBox_15.setTitle(_translate("MainWindow", "去除无效值"))
        self.lineEdit_19.setPlaceholderText(_translate("MainWindow", "选择筛选目标列，形如 1,2,3"))
        self.pushButton_24.setText(_translate("MainWindow", "确定"))
        self.label_sx.setText(_translate("MainWindow", "去空值/零值行"))
        self.groupBox_16.setTitle(_translate("MainWindow", "去除异常值"))
        self.comboBox_5.setItemText(0, _translate("MainWindow", "2倍箱型图"))
        self.comboBox_5.setItemText(1, _translate("MainWindow", "3倍箱型图"))
        self.lineEdit_20.setPlaceholderText(_translate("MainWindow", "选择去除异常值目标列，形如 1,2,3"))
        self.pushButton_25.setText(_translate("MainWindow", "确定"))
        self.groupBox_17.setTitle(_translate("MainWindow", "数据平滑"))
        self.lineEdit_21.setPlaceholderText(_translate("MainWindow", "选择降噪目标列，形如 1,2,3"))
        self.comboBox_6.setItemText(0, _translate("MainWindow", "5倍滤波降噪"))
        self.comboBox_6.setItemText(1, _translate("MainWindow", "2倍滤波降噪"))
        # self.comboBox_6.setItemText(2, _translate("MainWindow", "傅里叶变化"))
        # self.comboBox_6.setItemText(3, _translate("MainWindow", "小波降噪"))
        self.pushButton_26.setText(_translate("MainWindow", "确定"))
        self.groupBox_18.setTitle(_translate("MainWindow", "数据描述"))
        self.lineEdit_22.setPlaceholderText(_translate("MainWindow", "保存地址"))  ##
        self.pushButton_27.setText(_translate("MainWindow", "选择地址"))  ##
        self.pushButton_28.setText(_translate("MainWindow", "导出"))  ##
        self.comboBox_1.setItemText(0, _translate("MainWindow", "有效数据"))
        self.comboBox_1.setItemText(1, _translate("MainWindow", "标准格式数据"))
        self.comboBox_1.setItemText(2, _translate("MainWindow", "处理后数据"))
        self.comboBox_1.setItemText(4, _translate("MainWindow", "每环平均数据"))
        self.comboBox_1.setItemText(3, _translate("MainWindow", "降噪后数据"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_8), _translate("MainWindow", "数据处理"))
        self.lineEdit_23.setPlaceholderText(_translate("MainWindow", "选择曲线绘制分析列，形如 1,2,3"))
        self.pushButton_29.setText(_translate("MainWindow", "曲线绘制"))
        self.lineEdit_24.setPlaceholderText(_translate("MainWindow", "选择相关性分析列，形如 1,2,3"))
        self.pushButton_30.setText(_translate("MainWindow", "相关性分析"))
        self.lineEdit_25.setPlaceholderText(_translate("MainWindow", "选择参数分布分析列，形如 1,2,3"))
        self.pushButton_31.setText(_translate("MainWindow", "参数分布"))
        self.lineEdit_26.setPlaceholderText(_translate("MainWindow", "选择箱型图分析列，形如 1,2,3"))
        self.pushButton_32.setText(_translate("MainWindow", "箱型图"))
        self.lineEdit_27.setPlaceholderText(_translate("MainWindow", "选择正态回归分析列，形如 1,2,3"))
        self.pushButton_33.setText(_translate("MainWindow", "正态回归"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_9), _translate("MainWindow", "数据挖掘"))
        # self.menu.setTitle(_translate("MainWindow", "数据处理"))
        # self.menu_2.setTitle(_translate("MainWindow", "地层识别"))
        # self.menu_3.setTitle(_translate("MainWindow", "机器学习"))
        # 信号槽
        # 导入导出文件
        self.pushButton_23.clicked.connect(self.open_event)  # 打开路径
        self.pushButton_22.clicked.connect(self.open_csv)  # 打开csv
        self.pushButton_27.clicked.connect(self.save_event)  # 保存路径
        self.pushButton_28.clicked.connect(self.save_csv)  # 保存csv
        # 特征选择
        self.pushButton_20.clicked.connect(self.creat_table_show0)  # 特征选择
        self.pushButton_21.clicked.connect(self.creat_table_show)  # 标准特征选择
        self.pushButton_24.clicked.connect(self.nozero)  # 筛选零值空值
        self.pushButton_25.clicked.connect(self.nospecial)  # 筛选异常值
        self.pushButton_yz.clicked.connect(self.yuzhi)  # 筛选阈值
        self.pushButton_26.clicked.connect(self.denoise)  # 噪声处理
        self.pushButton_30.clicked.connect(self.correlation)  # 相关性分析
        self.pushButton_29.clicked.connect(self.plot1)  # 曲线绘制
        self.pushButton_31.clicked.connect(self.plot2)  # 参数分布
        self.pushButton_32.clicked.connect(self.box)  # 参数分布
        self.pushButton_33.clicked.connect(self.norm)  # 参数分布

    # 打开路径
    def open_event(self):
        global path
        _translate = QtCore.QCoreApplication.translate
        directory1 = QFileDialog.getOpenFileName(None, "选择文件", "H:/")
        path = directory1[0]
        self.lineEdit_18.setText(_translate("MainWindow", path))

    # 打开文件
    def open_csv(self):
        global ori_data
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
            self.plainTextEdit_4.setPlainText(t)

    # 保存路径
    def save_event(self):
        global save_path
        _translate = QtCore.QCoreApplication.translate
        fileName2, ok2 = QFileDialog.getSaveFileName(None, "选择地址", "H:/")
        # print(fileName2)  # 打印保存文件的全部路径（包括文件名和后缀名）
        save_path = fileName2
        self.lineEdit_22.setText(_translate("MainWindow", save_path))

    # 保存csv
    def save_csv(self):
        global data
        global data_std
        global save_path
        global data_nozero
        save_path0 = save_path + "_youxiao.xlsx"
        save_path1 = save_path + "_clean.xlsx"
        save_path2 = save_path + "_std.xlsx"
        save_path3 = save_path + "_rings.xlsx"
        save_path4 = save_path + "_denoise.xlsx"

        if save_path is not None and data is not None:
            if self.comboBox_1.currentText() == "有效数据":
                data_nozero.to_excel(save_path0, index=False)
            if self.comboBox_1.currentText() == "处理后数据":
                data.to_excel(save_path1, index=False)
            if self.comboBox_1.currentText() == "标准格式数据":
                data_std.to_excel(save_path2, index=False)
            if self.comboBox_1.currentText() == "每环平均数据":
                DATA = data
                cols = ['环号(r)', '刀盘转矩(kN*m)', '刀盘转速(r/min)', '顶部土压(bar)', '总推进力(kN)', '推进速度(mm/min)',
                        '设备侧滚(mm)', '设备倾角(°)', '贯入度(mm)', '土压2#(bar)', '土压3#(bar)', '土压4#(bar)', '土压5#(bar)',
                        '土压6#(bar)',
                        '土压平均值(bar)', '1#铰接油缸行程(mm)', '3#铰接油缸行程(mm)', '5#铰接油缸行程(mm)', '7#铰接油缸行程(mm)',
                        '右A组油缸推进压力(bar)', '下B组油缸推进压力(bar)', '左C组油缸推进压力(bar)', '上D组油缸推进压力(bar)',
                        '螺旋机上卸料门开度(mm)', '螺旋机下卸料门开度(mm)', '螺旋机压力测量值(bar)', '螺旋机土压测量值后(bar)',
                        '螺旋机转矩(kN*m)', '螺旋机速度(r/min)', '1#膨润土流量(L)', '2#膨润土流量(L)', '泡沫原液流量(L)',
                        'A组油缸行程(mm)', 'B组油缸行程(mm)', 'C组油缸行程(mm)', 'D组油缸行程(mm)']  # 标准列名格式

                sum_cols = ['1#膨润土流量(L)', '2#膨润土流量(L)', '泡沫原液流量(L)']
                max_cols = ['A组油缸行程(mm)', 'B组油缸行程(mm)', 'C组油缸行程(mm)', 'D组油缸行程(mm)']
                if set(sum_cols).issubset(data.columns) and set(max_cols).issubset(data.columns):
                    # 每环取平均--数据透视表
                    data_ring = DATA.pivot_table(index='环号(r)',  # 透视的行，分组依据
                                                 values=DATA[DATA.columns],  # 值
                                                 aggfunc='mean')  # 聚合函数
                    # 每环取求和--数据透视表
                    datasum = DATA.pivot_table(index='环号(r)',  # 透视的行，分组依据
                                               values=DATA[sum_cols],  # 值
                                               aggfunc='sum')  # 聚合函数
                    # 每环取求和--数据透视表
                    datamax = DATA.pivot_table(index='环号(r)',  # 透视的行，分组依据
                                               values=DATA[max_cols],  # 值
                                               aggfunc='max')  # 聚合函数
                    data_ring[sum_cols] = datasum[sum_cols]
                    data_ring[max_cols] = datamax[max_cols]

                    DATA_rings = pd.DataFrame(columns=cols)
                    col = data_ring.reset_index().columns
                    DATA_rings[col] = data_ring.reset_index()
                else:
                    DATA_rings = pivot(DATA).reset_index()
                DATA_rings.to_excel(save_path3, index=False)
            if self.comboBox_1.currentText() == "降噪后数据":
                data_denoise.to_excel(save_path4, index=False)
            self.label_x3.show()
        else:
            pass

    # 特征选择
    def creat_table_show0(self, MainWindow):
        if path is not None:
            global cols
            global ori_data
            global data
            cols = []
            cols = split(self.lineEdit_17.text())
            if set(cols).issubset(ori_data.columns) and cols is not None:
                data = ori_data[cols]

                def input(input_table):
                    input_table_rows = input_table.shape[0]
                    input_table_colunms = input_table.shape[1]
                    print(input_table_rows, input_table_colunms)
                    input_table_header = input_table.columns.values.tolist()
                    self.describe1_3.setColumnCount(input_table_colunms)
                    self.describe1_3.setRowCount(input_table_rows)
                    self.describe1_3.setHorizontalHeaderLabels(input_table_header)
                    for i in range(input_table_rows):
                        input_table_rows_values = input_table.iloc[[i]]
                        input_table_rows_values_array = np.array(input_table_rows_values)
                        input_table_rows_values_list = input_table_rows_values_array.tolist()[0]
                        for j in range(input_table_colunms):
                            input_table_items_list = input_table_rows_values_list[j]
                            input_table_items = str(input_table_items_list)
                            newItem = QTableWidgetItem(input_table_items)
                            newItem.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
                            self.describe1_3.setItem(i, j, newItem)

                input(data)
            else:
                self.label.setText("输入有误，请重新输入")
                self.label.show()

        else:
            pass

    # 特征选择(建议特征)
    def creat_table_show(self, MainWindow):
        if path is not None:
            global cols
            global ori_data
            global data_std
            global data
            cols_new = ['环号(r)', '刀盘转矩(kN*m)', '刀盘转速(r/min)', '顶部土压(bar)', '总推进力(kN)', '推进速度(mm/min)',
                        '设备侧滚(mm)', '设备倾角(°)', '贯入度(mm)', '土压2#(bar)', '土压3#(bar)', '土压4#(bar)', '土压5#(bar)',
                        '土压6#(bar)',
                        '土压平均值(bar)', '1#铰接油缸行程(mm)', '3#铰接油缸行程(mm)', '5#铰接油缸行程(mm)', '7#铰接油缸行程(mm)',
                        '右A组油缸推进压力(bar)', '下B组油缸推进压力(bar)', '左C组油缸推进压力(bar)', '上D组油缸推进压力(bar)',
                        '螺旋机上卸料门开度(mm)', '螺旋机下卸料门开度(mm)', '螺旋机压力测量值(bar)', '螺旋机土压测量值后(bar)',
                        '螺旋机转矩(kN*m)', '螺旋机速度(r/min)', '1#膨润土流量(L)', '2#膨润土流量(L)', '泡沫原液流量(L)',
                        'A组油缸行程(mm)', 'B组油缸行程(mm)', 'C组油缸行程(mm)', 'D组油缸行程(mm)']
            cols = []
            cols = split(self.plainTextEdit_5.toPlainText())
            # if ori_data:
            col_total = list(ori_data.columns) + cols_new
            if set(cols).issubset(col_total):
                data_std = pd.DataFrame(columns=cols)
                for col in cols:
                    if col in ori_data.columns:
                        data_std[col] = ori_data[col]
                    else:
                        pass
                data_std.columns = cols_new
                data = data_std

                def input(input_table):
                    input_table_rows = input_table.shape[0]
                    input_table_colunms = input_table.shape[1]
                    print(input_table_rows, input_table_colunms)
                    input_table_header = input_table.columns.values.tolist()
                    self.describe1_3.setColumnCount(input_table_colunms)
                    self.describe1_3.setRowCount(input_table_rows)
                    self.describe1_3.setHorizontalHeaderLabels(input_table_header)
                    for i in range(input_table_rows):
                        input_table_rows_values = input_table.iloc[[i]]
                        input_table_rows_values_array = np.array(input_table_rows_values)
                        input_table_rows_values_list = input_table_rows_values_array.tolist()[0]
                        for j in range(input_table_colunms):
                            input_table_items_list = input_table_rows_values_list[j]
                            input_table_items = str(input_table_items_list)
                            newItem = QTableWidgetItem(input_table_items)
                            newItem.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
                            self.describe1_3.setItem(i, j, newItem)

                input(data)
                self.label_x0.setText('原始数据:{}行{}列'.format(data.shape[0], data.shape[1]))
                self.label_x0.show()
                t = ""  # 读取列名
                for i in data.columns:
                    t = t + i if i == data.columns[0] else t + ',' + i
                self.plainTextEdit_4.clear()
                self.plainTextEdit_4.setPlainText(t)
            else:
                self.label.setText("输入有误，请重新输入")
                self.label.show()
        else:
            pass

    # 数据处理
    def nozero(self):
        global cols
        global data_nozero
        global data
        nozero_cols = []
        nozero_cols = split(self.lineEdit_19.text())
        if set(nozero_cols).issubset(data.columns) and nozero_cols != ['']:
            def delnull(df, cols):
                for col in cols:
                    #         df = df.drop(df[df[col]== 0].index)# 不改变索引
                    df = df.drop(df[df[col] == 0].index).reset_index(drop=True)  # 改变索引
                return df

            data_nozero = delnull(data, nozero_cols)

            count_nozero = data.shape[0] - data_nozero.shape[0]
            data = data_nozero
            self.label_nozero.setText("筛选{}行,当前{}行".format(count_nozero, data_nozero.shape[0]))
            self.label_x1.setText('有效数据:{}行{}列'.format(data_nozero.shape[0], data_nozero.shape[1]))
            self.label_x1.show()
        else:
            self.label_nozero.setText("输入有误，请重新输入")

        self.label_nozero.show()

    def nospecial(self):
        global cols
        global data_nozero
        global data_nospecial
        global data
        nospecial_cols = []
        nospecial_cols = split(self.lineEdit_20.text())
        if set(nospecial_cols).issubset(data.columns):

            def outliers_proc(data, col_name, scale=2):
                """
                用于清洗异常值，默认用 box_plot（scale=3）进行清洗
                :param data: 接收 pandas 数据格式
                :param col_name: pandas 列名
                :param scale: 尺度
                """

                def box_plot_outliers(data_ser, box_scale):
                    """
                    利用箱线图去除异常值
                    :param data_ser: 接收 pandas.Series 数据格式
                    :param box_scale: 箱线图尺度，
                    """
                    iqr = box_scale * (data_ser.quantile(0.75) - data_ser.quantile(0.25))
                    val_low = data_ser.quantile(0.25) - iqr
                    val_up = data_ser.quantile(0.75) + iqr
                    rule_low = (data_ser < val_low)
                    rule_up = (data_ser > val_up)
                    return (rule_low, rule_up), (val_low, val_up)

                data_n = data.copy()
                data_series = data_n[col_name]
                if 'float' in str(data_series.dtypes) or 'int' in str(data_series.dtypes):
                    rule, value = box_plot_outliers(data_series, box_scale=scale)
                    index = np.arange(data_series.shape[0])[rule[0] | rule[1]]
                    data_n = data_n.drop(index)
                    data_n.reset_index(drop=True, inplace=True)
                    # print("Now column number is: {}".format(data_n.shape[0]))
                    index_low = np.arange(data_series.shape[0])[rule[0]]
                    outliers = data_series.iloc[index_low]
                    index_up = np.arange(data_series.shape[0])[rule[1]]
                    outliers = data_series.iloc[index_up]
                    fig, ax = plt.subplots(1, 2, figsize=(10, 7))
                    sns.boxplot(y=data[col_name], data=data, palette="Set1", ax=ax[0])
                    sns.boxplot(y=data_n[col_name], data=data_n, palette="Set1", ax=ax[1])
                    plt.show()
                    return data_n
                else:
                    self.lineEdit_20.clear()
                    self.lineEdit_20.setText('类型有误!必须是数值型！')
                    return data_n

            data_nospecial = data.copy()
            if self.comboBox_5.currentText() == "2倍箱型图":
                for col in nospecial_cols:
                    data_nospecial = outliers_proc(data, col, 2)
            if self.comboBox_5.currentText() == "3倍箱型图":
                for col in nospecial_cols:
                    data_nospecial = outliers_proc(data, col, 3)

            count_nozero = data.shape[0] - data_nospecial.shape[0]
            data = data_nospecial
            self.label_nospecial.setText("筛选{}行,当前{}行".format(count_nozero, data_nospecial.shape[0]))
            self.label_x2.setText('处理后数据:{}行{}列'.format(data.shape[0], data.shape[1]))
            self.label_x2.show()
        else:
            self.label_nospecial.setText("输入有误，请重新输入")
        self.label_nospecial.show()

    def yuzhi(self):
        global cols
        global data_yuzhi
        # global data_nospecial
        global data
        yuzhi_cols = []
        yuzhi_cols = split(self.lineEdit_yz.text())
        if set(yuzhi_cols).issubset(data.columns):
            data_yuzhi = data
            print('1')
            if self.lineEdit_min.text():
                def delmin(df, cols):
                    for col in cols:
                        #         df = df.drop(df[df[col]== 0].index)# 不改变索引
                        df = df.drop(df[df[col] < float(self.lineEdit_min.text())].index).reset_index(drop=True)  # 改变索引
                    return df

                data_yuzhi = delmin(data_yuzhi, yuzhi_cols)
            if self.lineEdit_max.text():
                def delmax(df, cols):
                    for col in cols:
                        #         df = df.drop(df[df[col]== 0].index)# 不改变索引
                        df = df.drop(df[df[col] > float(self.lineEdit_max.text())].index).reset_index(drop=True)  # 改变索引
                    return df

                data_yuzhi = delmax(data_yuzhi, yuzhi_cols)
            count_yuzhi = data.shape[0] - data_yuzhi.shape[0]
            data = data_yuzhi
            self.label_yz1.setText("筛选{}行,当前{}行".format(count_yuzhi, data_yuzhi.shape[0]))
            self.label_x2.setText('处理后数据:{}行{}列'.format(data.shape[0], data.shape[1]))
            self.label_x2.show()
        else:
            self.label_yz1.setText("输入有误，请重新输入")

        self.label_yz1.show()

    def denoise(self):
        global cols
        global data_nozero
        global data_denoise
        global data
        # while data_nospecial == None:
        #     data_nozero = ori_data[cols]
        denoise_cols = []
        denoise_cols = split(self.lineEdit_21.text())
        if set(denoise_cols).issubset(data.columns):
            # 均值滤波降噪
            # 函数ava_filter用于单次计算给定窗口长度的均值滤波

            def ava_filter(x, filt_length):
                N = len(x)
                res1 = []
                for i in range(N):
                    if i <= filt_length // 2 or i >= N - (filt_length // 2):
                        temp = x[i]
                    else:
                        sum = 0
                        for j in range(filt_length):
                            sum += x[i - filt_length // 2 + j]
                        temp = sum * 1.0 / filt_length
                    res1.append(temp)
                return res1

            # 函数denoise用于指定次数调用ava_filter函数，进行降噪处理

            def de_noise(x, n, filt_length):
                for i in range(n):
                    res = ava_filter(x, filt_length)
                    # x = res
                return res

            data_denoise = data.copy()

            if self.comboBox_6.currentText() == "5倍滤波降噪":
                '''
                均值滤波降噪：
                    函数ava_filter用于单次计算给定窗口长度的均值滤波
                    函数denoise用于指定次数调用ava_filter函数，进行降噪处理
                '''
                for col_1 in denoise_cols:
                    data_denoise[col_1] = de_noise(data[col_1], 3, 5)
                    data_denoise.loc[:, [col_1]].plot()  # 降噪后
                    plt.title("降噪后")
                    # plt.show()
                    data.loc[:, [col_1]].plot()  # 降噪后
                    plt.title("降噪前")
                plt.show()
            elif self.comboBox_6.currentText() == "2倍滤波降噪":
                for col_1 in denoise_cols:
                    data_denoise[col_1] = de_noise(data[col_1], 3, 2)
                    data_denoise.loc[:, [col_1]].plot()  # 降噪后
                    plt.title("降噪后")
                    # plt.show()
                    data.loc[:, [col_1]].plot()  # 降噪后
                    plt.title("降噪前")
                plt.show()
            else:
                pass

            self.label_denoise.setText("降噪成功！")
        elif denoise_cols == ['']:
            data_denoise = data_nozero
            self.label_denoise.setText("输入为空")
        else:
            self.label_denoise.setText("输入有误，请重新输入")
        # data = data_denoise
        self.label_denoise.show()

    # 数据挖掘
    # 相关性
    def correlation(self):
        global cols
        corr_cols = []
        corr_cols = split(self.lineEdit_24.text())
        if set(corr_cols).issubset(data.columns):
            data_corr = data[corr_cols]
            correlation = data_corr.corr()
            self.fig = sns.heatmap(correlation, square=True, vmax=0.8, annot=True)
            plt.show()
        else:
            pass

    # 折线图
    def plot1(self):
        global cols
        corr_cols = []
        corr_cols = split(self.lineEdit_23.text())
        if set(corr_cols).issubset(data.columns):
            for col in corr_cols:
                data.loc[:, [col]].plot()
            plt.show()
        else:
            pass

    # 参数分布图
    def plot2(self):
        global cols
        corr_cols = []
        corr_cols = split(self.lineEdit_25.text())
        if set(corr_cols).issubset(data.columns):
            # data_corr = data[corr_cols]
            data_corr = data[corr_cols]
            data_corr.hist(bins=50)
            plt.show()
        else:
            pass

    def box(self):
        # 画箱式图
        global cols
        column = []
        column = split(self.lineEdit_26.text())
        if set(column).issubset(data.columns):
            fig = plt.figure()  # 指定绘图对象宽度和高度
            for i in range(len(column)):
                plt.subplot(len(column) // 2 + 1, 2, i + 1)  #

                sns.boxplot(data[column[i]], orient="v", width=0.5)  # 箱式图
                plt.ylabel(column[i])
            fig.tight_layout(h_pad=5)
            plt.show()
        else:
            pass

    def norm(self):
        # return
        # # 拟正态分布
        global cols
        column = []
        column = split(self.lineEdit_27.text())
        if set(column).issubset(data.columns):
            train_cols = 4
            train_rows = len(column) // 2 + 1
            fig = plt.figure(figsize=(3 * train_cols, 3 * train_rows))  # 指定绘图对象宽度和高度
            i = 0
            for col in column:
                i += 1
                self.ax = plt.subplot(train_rows, train_cols, i)
                sns.distplot(data[col], fit=stats.norm)
                i += 1
                self.ax = plt.subplot(train_rows, train_cols, i)
                self.res = stats.probplot(data[col], plot=plt)
            fig.tight_layout(h_pad=5)
            plt.show()
        else:
            pass


# 智能掘进，放在另一个GUI里面，
# 沉降预测
# 异常识别
# 参数预测
# 地层识别

class logindialog(QDialog):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setWindowTitle('登录界面')
        self.resize(230, 180)
        self.setFixedSize(self.width(), self.height())
        self.setWindowFlags(Qt.WindowCloseButtonHint)

        ###### 设置界面控件
        self.frame = QFrame(self)
        self.verticalLayout = QVBoxLayout(self.frame)

        self.lineEdit_account = QLineEdit()
        self.lineEdit_account.setPlaceholderText("请输入账号")
        self.verticalLayout.addWidget(self.lineEdit_account)

        self.lineEdit_password = QLineEdit()
        self.lineEdit_password.setPlaceholderText("请输入密码")
        self.verticalLayout.addWidget(self.lineEdit_password)

        self.pushButton_enter = QPushButton()
        self.pushButton_enter.setText("确定")
        self.verticalLayout.addWidget(self.pushButton_enter)

        self.pushButton_quit = QPushButton()
        self.pushButton_quit.setText("取消")
        self.verticalLayout.addWidget(self.pushButton_quit)

        ###### 绑定按钮事件
        self.pushButton_enter.clicked.connect(self.on_pushButton_enter_clicked)
        self.pushButton_quit.clicked.connect(QCoreApplication.instance().quit)

    def on_pushButton_enter_clicked(self):
        # 账号判断
        if self.lineEdit_account.text() != "wyw":
            print("用户名不存在")
        # 密码判断
        elif self.lineEdit_password.text() != "123":
            print("密码错误")

        # 通过验证，关闭对话框并返回1
        else:
            self.accept()


if __name__ == '__main__':  # 直接测试 运行的时候会执行，导入的时候不会执行
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    dialog = logindialog()
    MainWindow = QtWidgets.QMainWindow()
    if dialog.exec_() == QDialog.Accepted:
        # if True:
        ui = Ui_MainWindow()
        ui.setupUi(MainWindow)
        MainWindow.show()
        sys.exit(app.exec_())
