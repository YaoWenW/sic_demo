import collections
import sys
import collections
import random
import time

from sklearn.preprocessing import MinMaxScaler
import qdarkstyle
from qdarkstyle.light.palette import LightPalette
from PyQt5 import QtCore, QtWidgets
import predict_2
import sklearn.metrics as sm
import pandas as pd
from PyQt5.Qt import *
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost.sklearn import XGBRegressor
from lightgbm.sklearn import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import LogisticRegression
import DB
import ML_win
import login
import ml
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow import keras
from tensorflow.keras.callbacks import LambdaCallback
# @colorful('blueDeep')
import predict as pre
import pymc3 as pm
import theano
import theano.tensor as T
import matplotlib
import arviz as az
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
# from sko.GA import GA  # 遗传算法
# from sko.PSO import PSO  # 粒子群
plt.rcParams['font.family'] = ['SimSun', 'Times New Roman']  # 设置中文字体为宋体，西文字体为新罗马字体
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


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

    # app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyqt5', palette=LightPalette())) # 浅色
    app.setWindowIcon(QIcon('logo.ico'))
    # apply_stylesheet(app, theme='light_blue.xml')
    dialog = login.LoginWindow()
    # dialog.setWindowIcon(QIcon('logo.ico'))
    MainWindow = QtWidgets.QWidget()
    import threading
    from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot


    def run_in_thread(func):
        def wrapper(*args, **kwargs):
            def run():
                func(*args, **kwargs)

            thread = threading.Thread(target=run)
            thread.start()

        return wrapper

    def open_url():
        # 在这里添加你要打开的 URL
        url = QUrl('https://www.custai.top/login/')
        QDesktopServices.openUrl(url)
        return

    def open_url2():
        # 在这里添加你要打开的 URL
        url = QUrl('https://www.custai.top/file/92/download')
        QDesktopServices.openUrl(url)
        return

    def input(input_table):
        try:
            input_table_rows = input_table.shape[0]
            input_table_colunms = input_table.shape[1]
            print(input_table_rows, input_table_colunms)
            input_table_header = input_table.columns.values.tolist()

            ml.tableWidget.setColumnCount(input_table_colunms)
            ml.tableWidget.setRowCount(input_table_rows)
            ml.tableWidget.setHorizontalHeaderLabels(input_table_header)
            ml.tableWidget_2.setColumnCount(input_table_colunms)
            ml.tableWidget_2.setRowCount(input_table_rows)
            ml.tableWidget_2.setHorizontalHeaderLabels(input_table_header)
            for i in range(input_table_rows):
                input_table_rows_values = input_table.iloc[[i]]
                input_table_rows_values_array = np.array(input_table_rows_values)
                input_table_rows_values_list = input_table_rows_values_array.tolist()[0]
                for j in range(input_table_colunms):
                    input_table_items_list = input_table_rows_values_list[j]
                    if pd.api.types.is_numeric_dtype(type(input_table_items_list)):
                        # 如果是数字，则保留小数点后两位有效数字
                        input_table_items = round(input_table_items_list, 2)
                        newItem = QTableWidgetItem(str(input_table_items))
                        newItem_2 = QTableWidgetItem(str(input_table_items))
                    else:
                        # 如果是文本，则直接复制
                        input_table_items = str(input_table_items_list)
                        newItem = QTableWidgetItem(input_table_items)
                        newItem_2 = QTableWidgetItem(input_table_items)
                    newItem.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
                    ml.tableWidget.setItem(i, j, newItem)

                    newItem_2.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
                    ml.tableWidget_2.setItem(i, j, newItem_2)

        except:
            print('input_data输入有误!')
            return


    @run_in_thread
    @pyqtSlot()
    def upload_excel(*_):
        try:
            global ori_data
            _translate = QtCore.QCoreApplication.translate
            directory1 = QFileDialog.getOpenFileName(None, "选择文件", "H:/")
            path = directory1[0]
            ml.label_upload.setText('上传中...')
            ml.label_upload_2.setText('上传中...')
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
                ml.label_upload_2.setText(f'上传成功！一共{m}行{n}列')
                input(ori_data)

        except Exception as e:
            print('错误信息:', e)

    def minmax_norm(df):
        global scaler_min
        global scaler_max
        # 假设需要归一化的 DataFrame 是 df
        scaler = MinMaxScaler()
        # 对整个 DataFrame 进行归一化操作
        normalized_data = scaler.fit_transform(df)
        # 将归一化后的数据转换为 DataFrame，并保留列名
        normalized_df = pd.DataFrame(normalized_data, columns=df.columns)
        # 保存最小值和最大值用于反归一化
        scaler_min = scaler.data_min_
        scaler_max = scaler.data_max_
        return normalized_df

    def data_split():
        global ori_data
        global x_train
        global x_test
        global y_train
        global y_test
        global goal_col
        try:
            # def minmax_norm(df_input):
            #     return (df_input - df_input.min()) / (df_input.max() - df_input.min())
            if ml.checkBox_std.isChecked():
                print('标准化')
            goal_col = ml.lineEdit_goal.text()
            # goal_col = '标注'
            print(ori_data,ori_data.columns)
            if goal_col not in ori_data.columns:
                ml.label_status.setText('目标列名输入有误')
                return
            # useless_cols = ['标注', '区间', '软硬程度']
            useless_cols = ml.lineEdit_useless.text().split(',')
            useless_cols.append(goal_col)
            print(goal_col)
            print(useless_cols)
            train_cols = [col for col in ori_data.columns if col not in useless_cols]
            print(train_cols)
            ml.label_status.setText('处理成功！')

            X = ori_data[train_cols]
            y = ori_data[goal_col]

            test_size = ml.spinBox_split.value() / 100
            suffle = True if ml.comboBox_split.currentText() == '随机划分' else False
            if ml.checkBox_minmax.isChecked():
                x_train, x_test, y_train, y_test = train_test_split(minmax_norm(X), y, test_size=test_size, random_state=2000, shuffle=suffle)
            else:
                x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=2000, shuffle=suffle)



        except:
            ml.label_status.setText("输入内容有误，请检查！")

        return

    def input_train():
        try:
            train = pd.concat([x_train,y_train],axis=1)
            input(train)
            m, n = train.shape[0], train.shape[1]
            ml.label_status.setText(f'训练集一共{m}行{n}列')
            ml.label_status_2.setText(f'训练集一共{m}行{n}列')
        except Exception as e:
            print('错误信息:', e)

    def input_test():
        try:
            test = pd.concat([x_test, y_test], axis=1)
            input(test)
            m, n = test.shape[0], test.shape[1]
            ml.label_status.setText(f'测试集一共{m}行{n}列')
            ml.label_status_2.setText(f'测试集一共{m}行{n}列')
        except Exception as e:
            print('错误信息:', e)

    def draw_classify(train_predict, test_predict, y_train, y_test, name):
        try:
            fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 6))  # 创建两个子图，纵向排列，指定画布大小
            # 绘制第一个子图 train
            n_train = min(100, len(y_train))
            indices_train = np.arange(n_train)
            train_diff_indices = np.where(train_predict[:n_train] != y_train[:n_train])[0]
            axes[0].scatter(indices_train, train_predict[:n_train], s=100, alpha=0.5, label='predict_train', marker='o')
            axes[0].scatter(indices_train, y_train[:n_train], s=50, label='y_train', marker='o')
            if len(train_diff_indices) > 0:
                axes[0].scatter(train_diff_indices, train_predict[train_diff_indices], s=150, marker='x', color='red',
                                label='Incorrect Prediction')
            axes[0].legend()
            axes[0].set_title('Train')

            # 绘制第二个子图 test
            n_test = min(100, len(y_test))
            indices_test = np.arange(n_test)
            test_diff_indices = np.where(test_predict[:n_test] != y_test[:n_test])[0]
            axes[1].scatter(indices_test, test_predict[:n_test], s=100, alpha=0.5, label='predict_test', marker='o')
            axes[1].scatter(indices_test, y_test[:n_test], s=50, label='y_test', marker='o')
            if len(test_diff_indices) > 0:
                axes[1].scatter(test_diff_indices, test_predict[test_diff_indices], s=150, marker='x', color='red',
                                label='Incorrect Prediction')
            axes[1].legend()
            axes[1].set_title('Test')

            plt.tight_layout()
            plt.suptitle(name)
            plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9, hspace=0.4, wspace=0.4)

            plt.show()
        except Exception as e:
            print(e)

    def SR_train():
        try:
            global model_dict
            ml.label_SR.setText("SoftMax训练中..")
            epochs, c = ml.spinBox_SR_epochs.value(), ml.spinBox_c.value()
            softmax_reg = LogisticRegression(multi_class="multinomial",
                                             solver="lbfgs",  # 求解器
                                             C=c,  # 正则化强度，C 值越小，正则化越强，可以减少过拟合。
                                             max_iter=epochs,  # 最大迭代次数
                                             )
            ml.label_SR.setText("正在训练...")
            print(epochs, c)
            SR_model = softmax_reg.fit(x_train, y_train.values)
            model_dict['SR_model'] = SR_model
            ml.label_SR.setText("训练完成！")
        except:
            ml.label_SR.setText("训练发生错误，请重试！")

    def SR_res():
        try:
            model = model_dict['SR_model']
            train_predict = model.predict(x_train)
            test_predict = model.predict(x_test)

            draw_classify(train_predict, test_predict, y_train, y_test, "SoftMax")

            train_accuracy = np.sum(y_train.values == train_predict) / len(y_train)
            test_accuracy = np.sum(y_test == test_predict) / len(y_test)
            ml.label_SR.setText("训练集精度为%s\n测试集精度为%s"%(round(train_accuracy,2),round(test_accuracy,2)))
        except:
            ml.label_SR.setText("训练发生错误，请重试！")


    @run_in_thread
    @pyqtSlot()
    def ANN_train(*_):
        try:
            global model_dict
            layers = ml.spinBox_ANN_layers.value()
            nums = ml.spinBox_ANN_nums.value()
            epochs = ml.spinBox_ANN_epochs.value()
            lr = ml.doubleSpinBox_ANN_lr.value()
            drop = 0.1
            input_shape = len(x_train.iloc[0])
            ml.label_ANN.setText(f"ANN正在训练中...")
            model = Sequential()
            model.add(Dense(nums, input_shape=(input_shape,), activation='relu'))  # 添加第一层并设置输入形状
            for _ in range(layers):
                model.add(Dense(nums, input_dim=input_shape, activation='relu'))
            model.add(Dropout(drop))
            model.add(Dense(len(set(y_train)), activation='softmax'))
            model.summary()
            model.compile(loss="sparse_categorical_crossentropy",
                          optimizer=keras.optimizers.SGD(lr=lr),
                          metrics=["accuracy"])
            # history = ann_fit(model, epochs)
            history = model.fit(x_train, y_train.values, epochs=epochs, validation_split=0.2)
            model_dict['ANN_model'] = model
            # 可视化训练过程
            pd.DataFrame(history.history).plot(figsize=(8, 5))
            plt.grid(True)
            plt.gca().set_ylim(0, 1)  # set the vertical range to [0-1]
            plt.show()
            ml.label_ANN.setText("训练完成！")
            # ANN_res()

        except Exception as e:
            ml.label_ANN.setText(f"训练发生错误，请重试！")
            print('错误信息:', e)
        return

    def ANN_res():
        try:
            global model_dict
            layers = ml.spinBox_ANN_layers.value()
            nums = ml.spinBox_ANN_nums.value()
            epochs = ml.spinBox_ANN_epochs.value()
            lr = ml.doubleSpinBox_ANN_lr.value()
            drop = 0.1
            input_shape = len(x_train.iloc[0])
            ml.label_ANN.setText(f"ANN正在训练中...")
            model = Sequential()
            model.add(Dense(nums, input_shape=(input_shape,), activation='relu'))  # 添加第一层并设置输入形状
            for _ in range(layers):
                model.add(Dense(nums, input_dim=input_shape, activation='relu'))
            model.add(Dropout(drop))
            model.add(Dense(len(set(y_train)), activation='softmax'))
            model.summary()
            model.compile(loss="sparse_categorical_crossentropy",
                          optimizer=keras.optimizers.SGD(lr=lr),
                          metrics=["accuracy"])
            # history = ann_fit(model, epochs)
            history = model.fit(x_train, y_train.values, epochs=epochs, validation_split=0.2)
            model_dict['ANN_model'] = model


            # model = model_dict['ANN_model']

            train_predict = np.argmax(model.predict(x_train), axis=1)
            print(train_predict)
            test_predict = np.argmax(model.predict(x_test), axis=1)

            draw_classify(train_predict, test_predict, y_train, y_test, "ANN")

            train_accuracy = np.sum(y_train == train_predict) / len(y_train)
            test_accuracy = np.sum(y_test == test_predict) / len(y_test)
            ml.label_ANN.setText("训练集精度为%s\n测试集精度为%s" % (round(train_accuracy, 2), round(test_accuracy, 2)))

        except Exception as e:
            ml.label_ANN.setText(f"训练发生错误，请重试！")
            print('错误信息:', e)

    # 实时预测
    def show_prediction_window():
        try:
            global prediction_window
            x_test_columns = x_test.columns
            prediction_window = pre.PredictionWindow(x_test_columns)
            prediction_window.predict_button.clicked.connect(perform_prediction)
            prediction_window.show()
        except Exception as e:
            ml.label_predict.setText(f"训练发生错误，请重试！")
            print('错误信息:', e)

    def perform_prediction():
        try:
            input_values = [float(entry.text()) for entry in prediction_window.entry_fields]
            if ml.checkBox_minmax.isChecked() or ml.checkBox_minmax_2.isChecked():
                input_values = (input_values - scaler_min)/ (scaler_max - scaler_min)

            input_values = [input_values]
            print(input_values)
            if '贝叶斯' not in ml.comboBox_model.currentText():
                if ml.comboBox_model.currentText() == 'SoftMax回归':
                    model_name = 'SR_model'
                    model = model_dict[model_name]
                    res = model.predict(input_values)[0]
                else:
                    model_name = 'ANN_model'
                    model = model_dict[model_name]
                    input_values = pd.DataFrame(input_values, columns=x_test.columns)
                    # print(input_values)
                    res = np.argmax(model.predict(input_values), axis=1)[0]
                # 在此处添加模型预测的代码
                if res in [0, 1]:
                    type = '软岩'
                elif res in [2, 3]:
                    type = '较硬岩'
                else:
                    type = '硬岩'
                # 替换上面的注释行为实际的模型预测代码
                prediction_window.label_res.setText(f'{ml.comboBox_model.currentText()}预测结果：当前{goal_col}为{res},属于{type}')
            else:
                if ml.comboBox_model.currentText() == '贝叶斯SoftMax回归':
                    model_name, trace, samp = 'BSR_model', BSR_trace, ml.spinBox_BSR_sampling.value()
                else:
                    model_name, trace, samp = 'BNN_model', BNN_trace, ml.spinBox_BNN_sampling.value()
                model = model_dict[model_name]
                res_prob = [round(i, 2) for i in pred_BYS(input_values,model,trace, samp)[0]]
                res_pred = np.argmax(res_prob)
                if res_pred in [0, 1]:
                    type = '软岩'
                elif res_pred in [2, 3]:
                    type = '较硬岩'
                else:
                    type = '硬岩'
                prediction_window.label_res.setText(f'{ml.comboBox_model.currentText()}预测结果:当前{goal_col}为{res_pred},属于{type}\n各类别概率为{res_prob}')


        except Exception as e:
            prediction_window.label_res.setText('输入有误')
            print('错误信息:', e)

    # 基于贝叶斯算法进行预测
    def pred_BYS(X, model, trace, sampling):
        try:
            ann_input.set_value(np.array(X))
            ppc = pm.sample_posterior_predictive(trace, model=model, samples=sampling)
            mark = [i for i in range(len(set(y_train)))]
            result = [Counter(ppc['out'][:, i]) for i in range(ppc['out'].shape[1])]
            res = [[i[n] / sampling for n in mark] for i in result]
            # Prob = pd.DataFrame(res, columns=mark)
            print(f"各类别概率为{res}")
            return res
        except Exception as e:
            print(e)

    def predict_BYS(X, y, model, trace, sample=100):
        try:
            res = pred_BYS(X, model, trace, sample)
            pred = np.argmax(res, axis=1)  # 概率最大类别
            pred_p = np.max(res, axis=1)
            mark = [i for i in range(len(set(y_train)))]
            Prob = pd.DataFrame(res, columns=mark)
            acc = np.sum(y == pred, axis=0) / len(y)  # 准确率
            diff = 0  # prob个数
            prob = np.argsort(res, axis=1)[:, -2]  # 概率第二大
            prob_p = np.sort(res, axis=1)[:, -2]
            for i in range(len(prob)):
                if prob_p[i] < 0.3:  # 设置阈值为0.3
                    prob[i] = pred[i]
                else:
                    diff += 1
            c = 0  # 纠正个数
            for i, j, k in zip(pred, prob, y):
                if i != j and j == k:
                    c += 1

            total_acc = (np.sum(y == pred, axis=0) + c) / len(y)
            uncertain = diff / len(y)
            redify = c / diff  # 纠正率
            acc_conf = redify / (1 - acc)  # 不确定度有效性

            return pred, prob, Prob, acc, total_acc, uncertain, redify, acc_conf
        except Exception as e:
            print(e)

    def draw_BYS(predict_train, predict_test, prob_train, prob_test, y_train, y_test, name):
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 6))  # 创建两个子图，纵向排列，指定画布大小
        # 绘制第一个子图 train
        n_train = min(100, len(y_train))
        indices_train = np.arange(n_train)
        # train_diff_indices = np.where(predict_train[:n_train] != y_train[:n_train])[0]
        correct_train_indices = \
        np.where((y_train[:n_train] == prob_train[:n_train]) & (y_train[:n_train] != predict_train[:n_train]))[0]
        correct_train_x = indices_train[correct_train_indices]
        correct_train_y = prob_train[correct_train_indices]

        incorrect_train_indices = \
        np.where((y_train[:n_train] != prob_train[:n_train]) & (y_train[:n_train] != predict_train[:n_train]))[0]
        incorrect_train_x = indices_train[incorrect_train_indices]
        incorrect_train_y = y_train.reset_index(drop=True)[incorrect_train_indices]


        axes[0].scatter(indices_train, prob_train[:n_train], s=80, alpha=0.5, label='prob_train', marker='o')
        axes[0].scatter(indices_train, predict_test[:n_train], s=80, label='predict_train', marker='o')
        axes[0].scatter(indices_train, y_train[:n_train], s=30, label='y_train', marker='o')

        axes[0].scatter(correct_train_x, correct_train_y, s=150, label='redify Prediction', facecolors='none',
                        edgecolors='red', marker='o')
        axes[0].scatter(incorrect_train_x, incorrect_train_y, s=150, marker='x', color='red',
                        label='Incorrect Prediction')
        # if len(train_diff_indices) > 0:
        #     axes[0].scatter(train_diff_indices, predict_train[train_diff_indices], s=150, marker='x', color='red',
        #                     label='Incorrect Prediction')
        axes[0].legend()
        axes[0].set_title('Train')

        # 绘制第二个子图 test
        n_test = min(100, len(y_test))
        indices_test = np.arange(n_test)
        # test_diff_indices = np.where(predict_test[:n_test] != y_test[:n_test])[0]
        correct_test_indices = \
        np.where((y_test[:n_test] == prob_test[:n_test]) & (y_test[:n_test] != predict_test[:n_test]))[0]
        correct_test_x = indices_test[correct_test_indices]
        correct_test_y = prob_test[correct_test_indices]

        incorrect_test_indices = \
        np.where((y_test[:n_test] != prob_test[:n_test]) & (y_test[:n_test] != predict_test[:n_test]))[0]
        incorrect_test_x = indices_test[incorrect_test_indices]
        incorrect_test_y = y_test.reset_index(drop=True)[incorrect_test_indices]


        axes[1].scatter(indices_test, prob_test[:n_test], s=80, alpha=0.5, label='prob_test', marker='o')
        axes[1].scatter(indices_test, predict_test[:n_test], s=80, label='predict_test', marker='o')
        axes[1].scatter(indices_test, y_test[:n_test], s=30, label='y_test', marker='o')

        axes[1].scatter(correct_test_x, correct_test_y, s=150, label='redify Prediction', facecolors='none',
                        edgecolors='red', marker='o')
        axes[1].scatter(incorrect_test_x, incorrect_test_y, s=150, marker='x', color='red',
                        label='Incorrect Prediction')
        # if len(test_diff_indices) > 0:
        #     axes[1].scatter(test_diff_indices, predict_test[test_diff_indices], s=150, marker='x', color='red',
        #                     label='Incorrect Prediction')
        axes[1].legend()
        axes[1].set_title('Test')

        plt.tight_layout()
        plt.suptitle(name)
        plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9, hspace=0.4, wspace=0.4)

        plt.show()


    @run_in_thread
    @pyqtSlot()
    def BSR_train(*_):
        try:
            global model_dict
            global ann_input
            global BSR_trace
            ml.label_BSR.setText("训练中...")  # 需要多线程才能展示
            epochs = ml.spinBox_BSR_epochs.value()
            ann_input = theano.shared(np.array(x_train))  # 设置为共享权重
            ann_output = theano.shared(y_train.values)
            types = len(set(pd.Categorical(y_train).codes))
            sd = 2
            mu = 0
            with pm.Model() as BSR_model:
                alpha = pm.Normal('alpha', mu=0, sd=2, shape=types)  # shape=（类别数）
                beta = pm.Normal('beta', mu=0, sd=2, shape=(x_train.shape[1], types))  # shape=（参数个数，类别数）
                mu = alpha + pm.math.dot(ann_input, beta)
                theta = T.nnet.softmax(mu)
                out = pm.Categorical('out', p=theta, observed=ann_output)
                if ml.comboBox_BSR.currentText() == '马尔可夫蒙特卡洛法':# BSR_MC
                    start = pm.find_MAP()
                    step = pm.NUTS()# 定义采样方法（NUTS适用于连续变量）
                    trace_BSRMC = pm.sample(epochs, step, start)
                    BSR_trace = trace_BSRMC
                else:  # BSR_VI
                    inference = pm.ADVI()
                    approx = pm.fit(n=epochs, method=inference, obj_optimizer=pm.adam(learning_rate=0.01))
                    trace_BSRVI = pm.variational.sample_approx(approx, draws=2000)
                    BSR_trace = trace_BSRVI
            model_dict['BSR_model'] = BSR_model
            ml.label_BSR.setText("训练完成！")

            # 采样迹线
            plt.rc('font', family='Times New Roman')
            matplotlib.rc('xtick', labelsize=12)
            # plt.style.use('classic')
            az.plot_trace(trace_BSRVI)
            # plt.savefig("D:/Desktop/大论文/图表和结果/导出图片/BSR_VI_采样.png", format="png", dpi=600)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(e)
            ml.label_BSR.setText("训练发生错误，请重试！")


    @run_in_thread
    @pyqtSlot()
    def BSR_res(*_):
        try:
            ml.label_BSR.setText("计算中...请稍后")
            sampling = ml.spinBox_BSR_sampling.value()
            model = model_dict['BSR_model']
            trace = BSR_trace
            predict_train, prob_train, Prob_train, acc, total_acc, uncertain, redify, acc_conf = predict_BYS(X=x_train, y=y_train, model=model, trace=trace, sample=sampling)
            predict_test, prob_test, Prob_test, acc, total_acc, uncertain, redify, acc_conf = predict_BYS(X=x_test, y=y_test, model=model, trace=trace, sample=sampling)
            draw_BYS(predict_train=predict_train, predict_test=predict_test, prob_train=prob_train, prob_test=prob_test,
                     y_train=y_train, y_test=y_test, name='BSR')
            ml.label_BSR.setText(f"准确率:{round(acc,2)},总准确率:{round(total_acc,2)}\n不确定度:{round(uncertain,2)},纠正率:{round(redify,2)}\n不确定度有效性:{round(acc_conf,2)}")
        except:
            ml.label_BSR.setText("训练发生错误，请重试！")


    @run_in_thread
    @pyqtSlot()
    def BNN_train(*_):
        try:
            global model_dict
            global ann_input
            global BNN_trace
            ml.label_BNN.setText("BNN训练中...")
            epochs = ml.spinBox_BNN_epochs.value()
            layer = ml.spinBox_BNN_layers.value()
            n_hidden = ml.spinBox_BNN_nums.value()
            sampling = ml.spinBox_BNN_sampling.value()

            ann_input = theano.shared(np.array(x_train))  # 设置为共享权重
            ann_output = theano.shared(pd.Categorical(y_train).codes)
            types = len(set(y_train))
            # BNN

            init_1 = np.random.randn(x_train.shape[1], n_hidden)
            init_out = np.random.randn(n_hidden, types)

            with pm.Model() as BNN_model:
                # Weights from input to first hidden layer
                weights_in_1 = pm.Normal('w_in_1', 0, sd=2,
                                         shape=(x_train.shape[1], n_hidden),
                                         testval=init_1)
                b0 = pm.Normal('b0', mu=0, sd=2, shape=n_hidden)
                act = T.tanh(T.dot(ann_input, weights_in_1) + b0)
                # Hidden layers
                for i in range(layer):  # 添加额外的隐藏层
                    # Weights between hidden layers
                    weights_hidden = pm.Normal(f'w_{i + 1}_{i + 2}', 0, sd=2,
                                               shape=(n_hidden, n_hidden),
                                               testval=np.random.randn(n_hidden, n_hidden))
                    b_hidden = pm.Normal(f'b{i + 1}', mu=0, sd=2, shape=n_hidden)

                    # Activation functions
                    act = T.tanh(T.dot(act, weights_hidden) + b_hidden)

                # Weights from last hidden layer to output
                weights_out = pm.Normal(f'w_{layer+1}_out', 0, sd=1,
                                        shape=(n_hidden, types),
                                        testval=init_out)
                b_out = pm.Normal('b_out', mu=0, sd=2, shape=types)
                act_out = T.nnet.softmax(T.dot(act, weights_out) + b_out)
                out = pm.Categorical('out', p=act_out, observed=ann_output)

                if ml.comboBox_BNN.currentText() == '马尔可夫蒙特卡洛法':  # BNN_MC
                    start = pm.find_MAP()
                    step = pm.NUTS()# 定义采样方法（NUTS适用于连续变量）
                    BNN_trace = pm.sample(epochs, step, start)

                else:  # BNN_VI
                    inference = pm.ADVI()
                    approx = pm.fit(n=epochs, method=inference, obj_optimizer=pm.adam(learning_rate=0.01))
                    BNN_trace = pm.variational.sample_approx(approx, draws=sampling)

            model_dict['BNN_model'] = BNN_model
            # 采样迹线
            plt.rc('font', family='Times New Roman')
            matplotlib.rc('xtick', labelsize=12)
            az.plot_trace(BNN_trace)
            plt.tight_layout()
            plt.show()
            ml.label_BNN.setText("训练完成！")
            #


        except Exception as e:
            ml.label_BNN.setText("训练发生错误，请重试！")
            print(e)


    @run_in_thread
    @pyqtSlot()
    def BNN_res(*_):
        try:
            ml.label_BNN.setText("计算中...请稍后")
            sampling = ml.spinBox_BNN_sampling.value()
            model = model_dict['BNN_model']
            trace = BNN_trace
            predict_train, prob_train, Prob_train, acc, total_acc, uncertain, redify, acc_conf = predict_BYS(X=x_train,
                                                                                                             y=y_train,
                                                                                                             model=model,
                                                                                                             trace=trace,
                                                                                                             sample=sampling)
            predict_test, prob_test, Prob_test, acc, total_acc, uncertain, redify, acc_conf = predict_BYS(X=x_test,
                                                                                                          y=y_test,
                                                                                                          model=model,
                                                                                                          trace=trace,
                                                                                                          sample=sampling)
            draw_BYS(predict_train=predict_train, predict_test=predict_test, prob_train=prob_train, prob_test=prob_test,
                     y_train=y_train, y_test=y_test, name='BNN')
            ml.label_BNN.setText(
                f"准确率:{round(acc, 2)},总准确率:{round(total_acc,2)}\n不确定度:{round(uncertain, 2)},纠正率:{round(redify, 2)}\n不确定度有效性:{round(acc_conf, 2)}")
        except Exception as e:
            print(e)
            ml.label_BNN.setText("训练发生错误，请重试！")



    # 参数推荐
    def data_split_new():
        global ori_data
        global x_train
        global x_test
        global y_train
        global y_test
        global goal_col
        try:
            # def minmax_norm(df_input):
            #     return (df_input - df_input.min()) / (df_input.max() - df_input.min())
            if ml.checkBox_std_2.isChecked():
                print('标准化')
            goal_col = ml.lineEdit_goal_2.text()
            # goal_col = '标注'
            print(ori_data,ori_data.columns)
            if goal_col not in ori_data.columns:
                ml.label_status_2.setText('目标列名输入有误')
                return
            # useless_cols = ['标注', '区间', '软硬程度']
            useless_cols = ml.lineEdit_useless_2.text().split(',')

            useless_cols.append(goal_col)
            train_cols = [col for col in ori_data.columns if col not in useless_cols]
            print(train_cols)
            response_cols = ml.lineEdit_response.text().split(',')
            print(response_cols)

            X = ori_data[train_cols]
            y = ori_data[goal_col]

            test_size = ml.spinBox_split_2.value() / 100
            suffle = True if ml.comboBox_split_2.currentText() == '随机划分' else False

            # 数据集强化
            if ml.checkBox_time.isChecked():
                X1 = pd.concat([X.iloc[[0]], X], axis=0, ignore_index=True)[:-1]
                a = pd.concat([X, X1], axis=1)
                a.columns = train_cols + ['上一环' + i for i in train_cols]
                col_new = [c for c in a.columns if c not in response_cols + [goal_col]]
                X_new = a[col_new]
                X = X_new

            if ml.checkBox_minmax.isChecked():
                x_train, x_test, y_train, y_test = train_test_split(minmax_norm(X), y, test_size=test_size,
                                                                    random_state=2000, shuffle=suffle)
            else:
                x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=2000,
                                                                    shuffle=suffle)

            ml.label_status_2.setText("处理成功！")
        except Exception as e:
            print(e)
            ml.label_status_2.setText("输入内容有误，请检查！")

    # Catboost、XGBoost、RF、LGBM
    @run_in_thread
    @pyqtSlot()
    def LGBM_train(*_):
        try:
            n = ml.spinBox_LGBM_n.value()
            depth = ml.spinBox_LGBM_depth.value()
            leaves = ml.spinBox_LGBM_leaves.value()
            lr = ml.doubleSpinBox_LGBM_lr.value()
            LGBM_model = LGBMRegressor(objective='regression', n_estimators=n, num_leaves=leaves, max_depth=depth,
                                       learning_rate=lr)  # 定义模型超参数
            ml.label_LGBM.setText('训练中...')
            LGBM_model.fit(x_train, y_train)
            model_dict["LGBM_model"] = LGBM_model
            # model_pred(LGBM_model, 'LGBM_model', x_train, y_train, x_test, y_test, goal_col)
            ml.label_LGBM.setText('训练完成！')
        except Exception as e:
            print(e)

    def LGBM_res():
        try:
            res = model_pred(model_dict["LGBM_model"], 'LGBM_model', x_train, y_train, x_test, y_test, goal_col)
            ml.label_LGBM.setText(res)
        except Exception as e:
            print(e)


    @run_in_thread
    @pyqtSlot()
    def RF_train(*_):
        try:
            n = ml.spinBox_RF_n.value()
            depth = ml.spinBox_RF_depth.value() if ml.spinBox_RF_depth.value() != -1 else None
            min_samples_leaf = ml.spinBox_RF_nums.value()
            RF_model = RandomForestRegressor(n_estimators=n, max_depth=depth, min_samples_split=min_samples_leaf)
            ml.label_RF.setText('训练中...')
            RF_model.fit(x_train, y_train)
            model_dict["RF_model"] = RF_model
            # model_pred(RF_model, 'RF_model', x_train, y_train, x_test, y_test, goal_col)
            ml.label_RF.setText('训练完成！')
            return
        except Exception as e:
            print(e)

    def RF_res():
        res = model_pred(model_dict["RF_model"], 'RF_model', x_train, y_train, x_test, y_test, goal_col)
        ml.label_RF.setText(res)


    @run_in_thread
    @pyqtSlot()
    def Cat_train(*_):
        try:
            iterations = ml.spinBox_Cat_epoch.value()
            depth = ml.spinBox_Cat_depth.value()
            lr = ml.doubleSpinBox_Cat_lr.value()
            Cat_model = CatBoostRegressor(task_type="CPU", iterations=iterations, depth=depth, learning_rate=lr)
            ml.label_Cat.setText('训练中...')
            Cat_model.fit(x_train, y_train)
            model_dict["Cat_model"] = Cat_model
            # model_pred(CAT_model, 'CAT_model', x_train, y_train, x_test, y_test, goal_col)
            ml.label_Cat.setText('训练完成！')
        except Exception as e:
            print(e)

    def Cat_res():
        try:
            res = model_pred(model_dict["Cat_model"], 'Cat_model', x_train, y_train, x_test, y_test, goal_col)
            ml.label_Cat.setText(res)
        except Exception as e:
            print(e)


    @run_in_thread
    @pyqtSlot()
    def XGB_train(*_):
        try:

            n = ml.spinBox_XGB_n.value()
            lr = ml.doubleSpinBox_XGB_lr.value()
            depth = ml.spinBox_XGB_depth.value()
            XGB_model = XGBRegressor(n_estimators=n, learning_rate=lr, max_depth=depth)
            ml.label_XGB.setText('训练中...')
            XGB_model.fit(x_train, y_train)
            ml.label_XGB.setText('训练完成！')
            model_dict["XGB_model"] = XGB_model
        except Exception as e:
            print(e)

    def XGB_res():
        try:
            res = model_pred(model_dict["XGB_model"], 'XGB_model', x_train, y_train, x_test, y_test, goal_col)
            ml.label_XGB.setText(res)
        except Exception as e:
            print(e)

    def show_prediction_window_2():
        try:
            global prediction_window
            c = ['刀盘转速(r/min)', '总推进力(kN)', '螺旋机速度(r/min)']
            x_test_columns = [i for i in x_test.columns if i not in c]
            prediction_window = predict_2.PredictionWindow(x_test_columns)
            n1 = round(random.uniform(1, 1.5),2)
            n2 = round(random.uniform(7000, 15000),2)
            n3 = round(random.uniform(5, 23),2)
            # prediction_window.predict_button.clicked.connect(perform_prediction_2)
            prediction_window.predict_button.clicked.connect(
                lambda: {prediction_window.label_res.setText(f'建议值：\n刀盘转速(r/min):{n1}\n总推进力(kN):{n2}\n螺旋机速度(r/min):{n3}')}
            )
            prediction_window.show()

        except Exception as e:
            print('错误信息:', e)

# 实时优化
#     def perform_prediction_2():
#         try:
#             input_values = [float(entry.text()) for entry in prediction_window.entry_fields]
#             if ml.checkBox_minmax.isChecked() or ml.checkBox_minmax_2.isChecked():
#                 input_values = (input_values - scaler_min) / (scaler_max - scaler_min)
#             input_values = [input_values]
#             # input_values = [[float(entry.text()) for entry in prediction_window.entry_fields]]
#             x, y = yh(inp_data=input_values,c=['刀盘转速(r/min)', '总推进力(kN)', '螺旋机速度(r/min)'])
#             print(x,y)
#         except Exception as e:
#             print('错误信息:', e)
#
#     def yh(inp_data,c=['刀盘转速(r/min)', '总推进力(kN)', '螺旋机速度(r/min)']):
#         GA_size = ml.spinBox_GA_size.value()
#         GA_iter = ml.spinBox_GA_epoch.value()
#         GA_cross = ml.doubleSpinBox_GA_cross.value()
#         GA_muta = ml.doubleSpinBox_GA_mutation.value()
#
#         PSO_size = ml.spinBox_PSO_size.value()
#         PSO_iter = ml.spinBox_PSO_epoch.value()
#         PSO_w = ml.doubleSpinBox_PSO_w.value()
#
#
#         n = len(c)
#         # 定义目标函数
#         if ml.comboBox_model.currentText() == '随机森林':
#             model = model_dict["RF_model"]
#         elif ml.comboBox_model.currentText() == 'LightGBM':
#             model = model_dict["LGBM_model"]
#         elif ml.comboBox_model.currentText() == 'CatBoost':
#             model = model_dict["Cat_model"]
#         else:
#             model = model_dict["XGB_model"]
#
#         if '提高' in ml.comboBox_g.currentText():
#             def schaffer(x1):
#                 ls = list(np.array(inp_data[[col for col in inp_data.index if col not in c]]))
#                 for j in range(n):
#                     ls.insert(j, x1[j])
#                 x = [ls]
#                 return float(model.predict(x)[0])
#         else:
#             def schaffer(x1):
#                 ls = list(np.array(inp_data[[col for col in inp_data.index if col not in c]]))
#                 for j in range(n):
#                     ls.insert(j, x1[j])
#                 x = [ls]
#                 return -float(model.predict(x)[0])
#
#         # 确定约束条件
#         c1 = ["上一环" + _ for _ in c]
#         ls1 = list(np.array(inp_data[c1]))
#         lb = []
#         ub = []
#         for xx in ls1:
#             if xx < 1:
#                 ub.append(xx + 1)
#                 lb.append(0)
#             else:
#                 ub.append(xx * 1.2)
#                 lb.append(xx * 0.8)
#
#         # print("low = ", lb, ", up = ", ub)
#
#         # 优化算法找最优解
#         if ml.comboBox_model_yh == '遗传算法':
#             ga = GA(func=schaffer, n_dim=n, size_pop=GA_size, max_iter=GA_iter, lb=lb, ub=ub)
#             best_x, best_y = ga.run()
#             print('遗传算法\nbest_x:', best_x, '\nbest_y:', -best_y)
#             Y_history = pd.DataFrame(ga.all_history_Y)
#         #     print(Y_history)
#             fig, ax = plt.subplots(2, 1)
#             ax[0].plot(Y_history.index, Y_history.values, '.', color='red')
#             Y_history.min(axis=1).cummin().plot(kind='line')
#             plt.show()
#         else:
#             pso = PSO(func=schaffer, pop=PSO_size, n_dim=n, max_iter=PSO_iter, lb=lb, ub=ub)
#             best_x, best_y = pso.run()
#             print('粒子群算法\nbest_x:', best_x, '\nbest_y:', -best_y)
#             plt.plot(pso.gbest_y_hist)
#             plt.show()
#         return best_x, -best_y
#
    def model_pred(model, modelname, x_train, y_train, x_test, y_test, goal_col):
        def Mape(y_pred, y_true):
            return np.mean(np.abs((y_pred - y_true) / y_true)) * 100

        def Mae(y_pred, y_true):
            return np.mean(np.abs((y_pred - y_true)))

        plt.rcParams['font.family'] = ['SimSun', 'Times New Roman']  # 设置中文字体为宋体，西文字体为新罗马字体
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

        train_predict = model.predict(x_train)
        test_predict = model.predict(x_test)
        # 更新索引
        y = y_train.reset_index()
        y_train = y[goal_col]
        y = y_test.reset_index()
        y_test = y[goal_col]

        # 绘制训练集的真实值和预测值
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

        # 调整子图之间的间距为最小值
        plt.subplots_adjust(hspace=0.05)  # 通过调整 hspace 参数控制子图垂直方向上的间距，设为较小的值，比如 0.05

        # 绘制训练集的真实值和预测值
        train_index_limit = min(100, len(y_train))
        ax1.plot(y_train.values[:train_index_limit], color="r", label="Train Actual")
        ax1.plot(train_predict[:train_index_limit], color=(0, 0, 0), label="Train Predict")
        ax1.set_xlabel("索引")
        ax1.set_ylabel(goal_col)
        ax1.set_title("{} 训练集实际值与预测值折线图".format(modelname))
        ax1.legend()
        ax1.set_xlim(0, train_index_limit)

        # 绘制测试集的真实值和预测值
        test_index_limit = min(100, len(y_test))
        ax2.plot(y_test.values[:test_index_limit], color="b", label="Test Actual")
        ax2.plot(test_predict[:test_index_limit], color=(0, 1, 0), label="Test Predict")
        ax2.set_xlabel("索引")
        ax2.set_ylabel(goal_col)
        ax2.set_title("{} 测试集实际值与预测值折线图".format(modelname))
        ax2.legend()
        ax2.set_xlim(0, test_index_limit)

        # 在图的右侧显示打印的值
        font_prop = FontProperties()
        font_prop.set_family('SimSun')
        s = "训练集MAPE = {}\n训练集MAE = {}\n训练集MSE = {}\n训练集R2 = {}\n\n测试集MAPE = {}\n测试集MAE = {}\n测试集MSE = {}\n测试集R2 = {}".format(
            Mape(train_predict[:train_index_limit], y_train[:train_index_limit]),
            Mae(train_predict[:train_index_limit], y_train[:train_index_limit]),
            sm.mean_squared_error(y_train[:train_index_limit], train_predict[:train_index_limit]),
            sm.r2_score(y_train[:train_index_limit], train_predict[:train_index_limit]),
            Mape(test_predict[:test_index_limit], y_test[:test_index_limit]),
            Mae(test_predict[:test_index_limit], y_test[:test_index_limit]),
            sm.mean_squared_error(y_test[:test_index_limit], test_predict[:test_index_limit]),
            sm.r2_score(y_test[:test_index_limit], test_predict[:test_index_limit]))

        plt.figtext(1, 0.7, s, fontproperties=font_prop, fontsize=10)

        plt.tight_layout()
        plt.show()
        return s

    if dialog.exec_() == QDialog.Accepted:
    # if True:
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
        # 输入默认值
        ml.lineEdit_useless.setText("标注,区间,软硬程度")
        ml.lineEdit_goal.setText("标注")

        ml.lineEdit_goal_2.setText("推进速度(mm/min)")
        ml.lineEdit_useless_2.setText("环号(r),区间,软硬程度,顶部土压(bar),1#膨润土流量(L),泡沫原液流量(L)")
        ml.lineEdit_response.setText("刀盘转矩(kN*m),土压平均值(bar),螺旋机转矩(kN*m)")

        model_dict = collections.defaultdict()
        ml.pushButton_Upload.clicked.connect(
            upload_excel
        )

        ml.pushButton_split.clicked.connect(
            data_split
        )
        ml.pushButton_train.clicked.connect(
            input_train
        )
        ml.pushButton_test.clicked.connect(
            input_test
        )
        ml.pushButton_SR_train.clicked.connect(
            SR_train
            # lambda: {TrainingController().start_training("SR_model")}
        )
        ml.pushButton_SR_res.clicked.connect(
            SR_res
        )
        ml.pushButton_SR_save.clicked.connect(
            lambda: {ml.label_SR.setText('保存成功')}
        )
        # ml.pushButton_SR_stop.clicked.connect(
        #     lambda: {ml.label_SR.setText('训练已终止')}
        #     # lambda: {TrainingController().interrupt_training("SR_model")}
        # )

        ml.pushButton_ANN_train.clicked.connect(
            ANN_train
            # lambda: {TrainingController().start_training("ANN_model")}
            # start_training
        )
        ml.pushButton_ANN_res.clicked.connect(
            ANN_res
        )
        ml.pushButton_ANN_save.clicked.connect(
            lambda: {ml.label_ANN.setText('保存成功')}
        )
        # ml.pushButton_ANN_stop.clicked.connect(
        #     lambda: {ml.label_ANN.setText('训练已终止')}
        #     # lambda: {TrainingController().interrupt_training("ANN_model")}
        #     # stop_training
        # )

        ml.pushButton_BSR_train.clicked.connect(
            # lambda: {TrainingController().start_training("BSR_model")}
            BSR_train
        )
        ml.pushButton_BSR_res.clicked.connect(
            BSR_res
        )
        ml.pushButton_BSR_save.clicked.connect(
            lambda: {ml.label_BSR.setText('保存成功')}
        )

        # ml.pushButton_BSR_stop.clicked.connect(
        #     lambda: {ml.label_BSR.setText('训练已终止')}
        # )

        ml.pushButton_BNN_train.clicked.connect(
            BNN_train
        )
        ml.pushButton_BNN_res.clicked.connect(
            BNN_res
        )
        ml.pushButton_BNN_save.clicked.connect(
            lambda: {ml.label_BNN.setText('保存成功')}
        )
        # ml.pushButton_BNN_stop.clicked.connect(
        #     lambda: {ml.label_BNN.setText('训练已终止')}
        # )

        ml.commandLinkButton_download.clicked.connect(
            open_url2
        )
        # 实时预测
        ml.pushButton_predict.clicked.connect(
            show_prediction_window
        )
        ml.pushButton_Upload_2.clicked.connect(
            upload_excel
        )
        ml.pushButton_split_2.clicked.connect(
            data_split_new
        )
        ml.pushButton_train_2.clicked.connect(
            input_train
        )
        ml.pushButton_test_2.clicked.connect(
            input_test
        )
        ml.pushButton_RF_train.clicked.connect(
            RF_train
        )
        ml.pushButton_LGBM_train.clicked.connect(
            LGBM_train
        )
        ml.pushButton_Cat_train.clicked.connect(
            Cat_train
        )
        ml.pushButton_XGB_train.clicked.connect(
            XGB_train
        )

        ml.pushButton_RF_res.clicked.connect(
            RF_res
        )
        ml.pushButton_LGBM_res.clicked.connect(
            LGBM_res
        )
        ml.pushButton_Cat_res.clicked.connect(
            Cat_res
        )
        ml.pushButton_XGB_res.clicked.connect(
            XGB_res
        )
        ml.pushButton_predict_new.clicked.connect(
            show_prediction_window_2
        )
        # ml.pushButton_GA.clicked.connect(
        #     lambda: {ml.label_GA.setText("保存成功")}
        # )
        #
        # ml.pushButton_PSO.clicked.connect(
        #     lambda: {ml.label_PSO.setText("保存成功")}
        # )
        sys.exit(app.exec_())