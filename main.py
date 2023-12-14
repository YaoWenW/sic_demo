import collections
import sys
import collections
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
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import DB
import ML_win
import login
import ml
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
# import ML_demo
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
        try:
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
                    if pd.api.types.is_numeric_dtype(type(input_table_items_list)):
                        # 如果是数字，则保留小数点后两位有效数字
                        input_table_items = round(input_table_items_list, 2)
                        newItem = QTableWidgetItem(str(input_table_items))
                    else:
                        # 如果是文本，则直接复制
                        input_table_items = str(input_table_items_list)
                        newItem = QTableWidgetItem(input_table_items)
                    newItem.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
                    ml.tableWidget.setItem(i, j, newItem)
        except:
            print('input_data输入有误!')
            return

    def upload_excel():
        try:
            global ori_data
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

    def data_split():
        global ori_data
        global x_train
        global x_test
        global y_train
        global y_test
        global goal_col
        try:
            def minmax_norm(df_input):
                return (df_input - df_input.min()) / (df_input.max() - df_input.min())
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

            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=2000, shuffle=suffle)

            if ml.checkBox_minmax.isChecked():
                x_train, x_test = minmax_norm(x_train), minmax_norm(x_test)

        except:
            ml.label_status.setText("输入内容有误，请检查！")

        return

    def input_train():
        try:
            train = pd.concat([x_train,y_train],axis=1)
            input(train)
            m, n = train.shape[0], train.shape[1]
            ml.label_status.setText(f'训练集一共{m}行{n}列')
        except Exception as e:
            print('错误信息:', e)

    def input_test():
        try:
            test = pd.concat([x_test, y_test], axis=1)
            input(test)
            m, n = test.shape[0], test.shape[1]
            ml.label_status.setText(f'测试集一共{m}行{n}列')
        except Exception as e:
            print('错误信息:', e)

    def draw_classify(train_predict, test_predict, y_train, y_test, name):
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


    def SR_train():
        try:
            global model_dict
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
            ml.label_SR.setText("训练集精度为%s,测试集精度为%s"%(round(train_accuracy,2),round(test_accuracy,2)))
        except:
            ml.label_SR.setText("训练发生错误，请重试！")

    def ANN_train():
        try:
            global model_dict
            layers = ml.spinBox_ANN_layers.value()
            nums = ml.spinBox_ANN_nums.value()
            epochs = ml.spinBox_ANN_epochs.value()
            lr = ml.doubleSpinBox_ANN_lr.value()
            drop = 0.1
            ml.label_ANN.setText(f"正在训练中...")
            model = Sequential()
            for _ in range(layers):
                model.add(Dense(nums, input_dim=len(x_train.iloc[0]), activation='relu'))
            model.add(Dropout(drop))
            model.add(Dense(len(set(y_train)), activation='softmax'))
            model.summary()
            model.compile(loss="sparse_categorical_crossentropy",
                          optimizer=keras.optimizers.SGD(lr=lr),
                          metrics=["accuracy"])
            history = model.fit(x_train, y_train.values, epochs=epochs, validation_split=0.2)
            # 可视化训练过程
            pd.DataFrame(history.history).plot(figsize=(8, 5))
            plt.grid(True)
            plt.gca().set_ylim(0, 1)  # set the vertical range to [0-1]
            plt.show()
            ml.label_ANN.setText("训练完成！")
            model_dict['ANN_model'] = model
        except Exception as e:
            ml.label_ANN.setText(f"训练发生错误，请重试！")
            print('错误信息:', e)
        return

    def ANN_res():
        try:
            model = model_dict['ANN_model']

            train_predict = np.argmax(model.predict(x_train), axis=1)
            test_predict = np.argmax(model.predict(x_test), axis=1)

            draw_classify(train_predict, test_predict, y_train, y_test, "ANN")

            train_accuracy = np.sum(y_train == train_predict) / len(y_train)
            test_accuracy = np.sum(y_test == test_predict) / len(y_test)
            ml.label_ANN.setText("训练集精度为%s,测试集精度为%s" % (round(train_accuracy, 2), round(test_accuracy, 2)))

        except Exception as e:
            ml.label_ANN.setText(f"训练发生错误，请重试！")
            print('错误信息:', e)

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
            if ml.comboBox_model.currentText() == 'SoftMax回归':
                model_name = 'SR_model'
            elif ml.comboBox_model.currentText() == '深度神经网络':
                model_name = 'ANN_model'
            elif ml.comboBox_model.currentText() == '贝叶斯SoftMax回归':
                model_name = 'ANN_model'
            else:
                model_name = 'BNN_model'
            input_values = [[float(entry.text()) for entry in prediction_window.entry_fields]]
            # 在此处添加模型预测的代码
            model = model_dict[model_name]
            res = model.predict(input_values)[0]
            if res in [0, 1]:
                type = '软岩'
            elif res in [2, 3]:
                type = '较硬岩'
            else:
                type = '硬岩'
            # 替换上面的注释行为实际的模型预测代码
            prediction_window.label_res.setText(f'{ml.comboBox_model.currentText()}预测结果：当前{goal_col}为{res},属于{type}')
            print(res)
        except Exception as e:
            prediction_window.label_res.setText('输入有误')
            print('错误信息:', e)

    def predict_BYS(X, y, model, trace, sample=100):
        #     y = pd.Categorical(y).codes
        ann_input.set_value(np.array(X))
        ppc = pm.sample_posterior_predictive(trace, model=model, samples=sample)
        mark = [0, 1, 2, 3, 4, 5]
        result = [Counter(ppc['out'][:, i]) for i in range(ppc['out'].shape[1])]
        # pred = stats.mode(ppc['out'], axis=0)[0][0]
        res = []
        for i in result:
            ls = []
            for n in mark:
                ls.append(i[n] / sample)
            res.append(ls)

        Prob = pd.DataFrame(res, columns=mark)  # 概率表
        pred = np.argmax(res, axis=1)  # 概率最大类别
        # pred_p = np.max(res, axis=1)
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

        print("acc=", acc)
        print("total_acc=", total_acc)
        print("uncertain=", uncertain)
        print("redify=", redify)
        print("acc_conf=", acc_conf)
        return pred, prob, Prob, acc, total_acc, uncertain, redify, acc_conf

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

    def BSR_train():
        try:
            global model_dict
            global ann_input
            global BSR_trace
            ml.label_BSR.setText("训练中...")  # 需要多线程才能展示
            epochs = ml.spinBox_BSR_epochs.value()
            ann_input = theano.shared(np.array(x_train))  # 设置为共享权重
            ann_output = theano.shared(y_train.values)
            types = len(set(pd.Categorical(y_train).codes))
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


        except:
            ml.label_BSR.setText("训练发生错误，请重试！")

    def BSR_res():
        try:
            sampling = ml.spinBox_BSR_sampling.value()
            model = model_dict['BSR_model']
            trace = BSR_trace
            predict_train, prob_train, Prob_train, acc, total_acc, uncertain, redify, acc_conf = predict_BYS(X=x_train, y=y_train, model=model, trace=trace, sample=sampling)
            predict_test, prob_test, Prob_test, acc, total_acc, uncertain, redify, acc_conf = predict_BYS(X=x_test, y=y_test, model=model, trace=trace, sample=sampling)
            draw_BYS(predict_train=predict_train, predict_test=predict_test, prob_train=prob_train, prob_test=prob_test,
                     y_train=y_train, y_test=y_test, name='BSR')
            ml.label_BSR.setText(f"准确率:{round(acc,2)},总准确率:{round(total_acc)}\n不确定度:{round(uncertain,2)},纠正率:{round(redify,2)}\n不确定度有效性:{round(acc_conf,2)}")
        except:
            ml.label_BSR.setText("训练发生错误，请重试！")

    def BNN_train():
        try:
            global model_dict
            global ann_input
            global BNN_trace
            epochs = ml.spinBox_BNN_epochs.value()
            layer = ml.spinBox_BNN_layers.value()
            n_hidden = ml.spinBox_BNN_nums.value()

            ann_input = theano.shared(np.array(x_train))  # 设置为共享权重
            ann_output = theano.shared(pd.Categorical(y_train).codes)
            types = len(set(pd.Categorical(y_train).codes))
            # BNN
            n_hidden = 30
            init_1 = np.random.randn(x_train.shape[1], n_hidden)
            init_2 = np.random.randn(n_hidden, n_hidden)
            init_3 = np.random.randn(n_hidden, n_hidden)
            init_out = np.random.randn(n_hidden, types)

            with pm.Model() as BNN_model:
                # Weights from input to hidden layer
                weights_in_1 = pm.Normal('w_in_1', 0, sd=2,
                                         shape=(x_train.shape[1], n_hidden),
                                         testval=init_1)
                b0 = pm.Normal('b0', mu=0, sd=2, shape=n_hidden)

                # Weights from 1st to 2nd layer
                weights_1_2 = pm.Normal('w_1_2', 0, sd=2,
                                        shape=(n_hidden, n_hidden),
                                        testval=init_2)
                b1 = pm.Normal('b1', mu=0, sd=2, shape=n_hidden)

                # Weights from hidden layer to output
                weights_2_out = pm.Normal('w_2_out', 0, sd=1,
                                          shape=(n_hidden, types),
                                          testval=init_out)
                b = pm.Normal('b', mu=0, sd=2, shape=types)

                # Build neural-network using tanh activation function
                act_1 = T.tanh(T.dot(ann_input, weights_in_1) + b0)
                act_2 = T.tanh(T.dot(act_1, weights_1_2) + b1)
                act_out = T.nnet.softmax(T.dot(act_2, weights_2_out) + b)

                # Binary classification -> Bernoulli likelihood
                out = pm.Categorical('out', p=act_out, observed=ann_output)

                if ml.comboBox_BSR.currentText() == '马尔可夫蒙特卡洛法':  # BNN_MC
                    start = pm.find_MAP()
                    step = pm.NUTS()# 定义采样方法（NUTS适用于连续变量）
                    trace_BNN = pm.sample(2000, step, start)

                else:  # BNN_VI
                    inference = pm.ADVI()
                    approx = pm.fit(n=2000, method=inference, obj_optimizer=pm.adam(learning_rate=0.01))
                    trace_BNN = pm.variational.sample_approx(approx, draws=2000)

            model_dict['BNN_model'] = BNN_model
            ml.label_BNN.setText("训练完成！")

            # 采样迹线
            plt.rc('font', family='Times New Roman')
            matplotlib.rc('xtick', labelsize=12)
            az.plot_trace(trace_BNN)
            # plt.savefig("D:/Desktop/大论文/图表和结果/导出图片/BSR_VI_采样.png", format="png", dpi=600)
            plt.tight_layout()
            plt.show()


        except:
            ml.label_BSR.setText("训练发生错误，请重试！")

    def BNN_res():
        try:
            sampling = ml.spinBox_BSR_sampling.value()
            model = model_dict['BSR_model']
            trace = BSR_trace
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
                     y_train=y_train, y_test=y_test, name='BSR')
            ml.label_BSR.setText(
                f"准确率:{round(acc, 2)},总准确率:{round(total_acc)}\n不确定度:{round(uncertain, 2)},纠正率:{round(redify, 2)}\n不确定度有效性:{round(acc_conf, 2)}")
        except:
            ml.label_BSR.setText("训练发生错误，请重试！")

    # if dialog.exec_() == QDialog.Accepted:
    if True:
        ui = main_win_Ui()
        ui.setupUi(MainWindow)
        MainWindow.show()

        db = db_Ui() # 数据处理软件
        ml = ml_Ui()
        # x_train = pd.DataFrame()
        ui.pushButton_db.clicked.connect(
            lambda: {db.show()}
        )
        ui.pushButton_db2.clicked.connect(
            open_url
        )
        ui.pushButton_ml.clicked.connect(
            lambda: {ml.show()}
        )

        ml.lineEdit_useless.setText("标注,区间,软硬程度")
        ml.lineEdit_goal.setText("标注")
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
        )
        ml.pushButton_SR_res.clicked.connect(
            SR_res
        )
        ml.pushButton_SR_save.clicked.connect(
            lambda: {ml.label_SR.setText('保存成功')}
        )

        ml.pushButton_ANN_train.clicked.connect(
            ANN_train
        )
        ml.pushButton_ANN_res.clicked.connect(
            ANN_res
        )
        ml.pushButton_ANN_save.clicked.connect(
            lambda: {ml.label_ANN.setText('保存成功')}
        )

        ml.pushButton_BSR_train.clicked.connect(
            BSR_train
        )
        ml.pushButton_BSR_res.clicked.connect(
            BSR_res
        )
        # ml.pushButton_SR_save.clicked.connect(
        #     lambda: {ml.label_BSR.setText('保存成功')}
        # )
        #
        # ml.pushButton_ANN_train.clicked.connect(
        #     ANN_train
        # )
        # ml.pushButton_ANN_res.clicked.connect(
        #     ANN_res
        # )
        # ml.pushButton_SR_save.clicked.connect(
        #     lambda: {ml.label_ANN.setText('保存成功')}
        # )


        ml.pushButton_predict.clicked.connect(
            show_prediction_window
        )


        sys.exit(app.exec_())