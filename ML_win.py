# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ML_win.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(464, 309)
        self.gridLayout = QtWidgets.QGridLayout(Dialog)
        self.gridLayout.setObjectName("gridLayout")
        self.textBrowser_3 = QtWidgets.QTextBrowser(Dialog)
        self.textBrowser_3.setObjectName("textBrowser_3")
        self.gridLayout.addWidget(self.textBrowser_3, 3, 2, 1, 1)
        self.pushButton_db2 = QtWidgets.QPushButton(Dialog)
        self.pushButton_db2.setObjectName("pushButton_db2")
        self.gridLayout.addWidget(self.pushButton_db2, 2, 1, 1, 1)
        self.pushButton_ml = QtWidgets.QPushButton(Dialog)
        self.pushButton_ml.setObjectName("pushButton_ml")
        self.gridLayout.addWidget(self.pushButton_ml, 2, 2, 1, 1)
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.textBrowser_2 = QtWidgets.QTextBrowser(Dialog)
        self.textBrowser_2.setObjectName("textBrowser_2")
        self.gridLayout.addWidget(self.textBrowser_2, 3, 1, 1, 1)
        self.pushButton_db = QtWidgets.QPushButton(Dialog)
        self.pushButton_db.setObjectName("pushButton_db")
        self.gridLayout.addWidget(self.pushButton_db, 2, 0, 1, 1)
        self.textBrowser = QtWidgets.QTextBrowser(Dialog)
        self.textBrowser.setObjectName("textBrowser")
        self.gridLayout.addWidget(self.textBrowser, 3, 0, 1, 1)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.pushButton_db2.setText(_translate("Dialog", "数据库平台"))
        self.pushButton_ml.setText(_translate("Dialog", "机器学习平台"))
        self.label.setText(_translate("Dialog", "TextLabel"))
        self.pushButton_db.setText(_translate("Dialog", "数据处理系统"))
