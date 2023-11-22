from collections import Counter
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from lightgbm.sklearn import LGBMClassifier
import pymc3 as pm
import theano
import theano.tensor as T
import arviz as az


import warnings
warnings.filterwarnings('ignore')
palette = 'muted'
matplotlib.rcParams['font.family']='SimHei'
plt.rcParams['axes.unicode_minus'] = False


data_ring = pd.read_excel('D:/Desktop/Database/5标注后参数/每环标注-钻孔.xlsx')

def process(data):
    DATA = data.copy(deep=True)
    DATA = DATA.rename(columns={"埋深": "埋深(m)"})  # 重命名参数
    # 删除列
    if '环号(r)' in DATA.columns:
        del DATA['环号(r)']
    # 按照条件删除行
    DATA = DATA.drop(DATA[DATA['标注'] == '复合'].index).reset_index(drop=True)
    DATA = DATA.drop(DATA[DATA['标注'] == '溶洞'].index).reset_index(drop=True)
    DATA['软硬程度'] = DATA['标注']
    DATA['软硬程度'].loc[DATA['标注'] == '1类'] = 0
    DATA['软硬程度'].loc[DATA['标注'] == '2类'] = 0
    DATA['软硬程度'].loc[DATA['标注'] == '3类'] = 1
    DATA['软硬程度'].loc[DATA['标注'] == '4类'] = 1
    DATA['软硬程度'].loc[DATA['标注'] == '5类'] = 2
    DATA['软硬程度'].loc[DATA['标注'] == '6类'] = 2

    DATA = DATA[DATA['区间'] != 'mwz']
    DATA = DATA[DATA['区间'] != 'mwy']
    DATA = DATA[DATA['区间'] != 'xsz']
    DATA = DATA[DATA['区间'] != 'xsy']
    DATA = DATA[DATA['区间'] != 'exz']
    DATA = DATA[DATA['区间'] != 'exy']

    DATA['标注'] = pd.Categorical(DATA['标注']).codes
    DATA['软硬程度'] = pd.Categorical(DATA['软硬程度']).codes
    return DATA

data = process(data_ring)
print(data)

#  划分测试集
operation_cols = ['刀盘转速(r/min)','推进速度(mm/min)','总推进力(KN)','螺旋机速度(r/min)','1#膨润土流量(L)','泡沫原液流量(L)']
response_cols = ['刀盘转矩(kN*m)','顶部土压(bar)','土压平均值(bar)','螺旋机转矩(kN*m)']
other_cols = ['标注','埋深(m)']
train_cols = [col for col in data.columns if col not in ['标注','区间','软硬程度']]
# goal_col = '软硬程度'
goal_col = '标注'

# 归一化参数
def minmax_norm(df_input):
    return (df_input- df_input.min()) / ( df_input.max() - df_input.min())

#  地层识别
#  逻辑回归
def LR_model(x_train,y_train,x_test,y_test):
    softmax_reg = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=10)
    model = softmax_reg.fit(x_train, y_train.values)
    train_predict = model.predict(x_train)
    print("train_accuracy =", np.sum(y_train.values == train_predict) / len(pd.Categorical(y_train).codes))

    test_predict = model.predict(x_test)
    print("test_accuracy =", np.sum(y_test == test_predict) / len(y_test))
    return

#  ANN
def ANN_model(x_train,y_train,x_test,y_test):
    # 数据输入 → 全连层*3 → Dropout层（防止过拟合） → 分类输出
    model = Sequential()
    model.add(Dense(32, input_dim=10, activation='relu'))
    # model.add(Dense(512, activation='relu'))
    # model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(6, activation='softmax'))

    model.summary()  # 使用属性，获取神经层很容易，可以通过索引或名称获取对应的层
    # 创建好模型之后，必须调用compile()方法，设置损失函数和优化器。另外，还可以指定训练和评估过程中要计算的额外指标的列表

    model.compile(loss="sparse_categorical_crossentropy",  # 等同于loss=keras.losses.sparse_categorical_crossentropy
                  optimizer="sgd",  # 等同于optimizer=keras.optimizers.SGD() -- "sgd"表示使用随机梯度下降训练模型
                  # 调整学习率很重要，必须要手动设置好，optimizer=keras.optimizers.SGD(lr=???)。optimizer="sgd"不同，它的学习率默认为lr=0.01
                  metrics=["accuracy"])  # 等同于metrics=[keras.metrics.sparse_categorical_accuracy]
    # 因为是个分类器，最好在训练和评估时测量"accuracy"
    history = model.fit(x_train, y_train.values, epochs=2000,
                        validation_split=0.02)

    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)  # set the vertical range to [0-1]
    plt.show()
    print("test_predict =", np.array(np.argmax(model.predict(x_test), axis=1)))  # 概率最大的类别
    print("train_accuracy =", np.sum(y_train == np.argmax(model.predict(x_train), axis=1)) / len(y_train))
    print("test_accuracy =", np.sum(y_test == np.argmax(model.predict(x_test), axis=1)) / len(y_test))
    return

#  LGBM
def LGBM_C_model():
    # LGBM

    ## 定义 LightGBM 模型
    clf = LGBMClassifier()
    # 在训练集上训练LightGBM模型
    clf.fit(x_train, y_train)
    # 在训练集上训练LightGBM模型
    train_predict = clf.predict(x_train)
    test_predict = clf.predict(x_test)
    from sklearn import metrics

    ## 利用accuracy（准确度）【预测正确的样本数目占总预测样本数目的比例】评估模型效果
    print('The accuracy of the LGBM is:', metrics.accuracy_score(y_train, train_predict))
    print('The accuracy of the LGBM is:', metrics.accuracy_score(y_test, test_predict))

    ## 查看混淆矩阵 (预测值和真实值的各类情况统计矩阵)
    confusion_matrix_result = metrics.confusion_matrix(test_predict, y_test)
    print('The confusion matrix result:\n', confusion_matrix_result)

    # 利用热力图对于结果进行可视化
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix_result, annot=True, cmap='Blues')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    # plt.savefig("D:/Desktop/大论文/图表和结果/导出图片/混淆矩阵.png", format="png", dpi=600)
    plt.show()
    return

#  贝叶斯
def BSR():
    # 贝叶斯神经网络--小论文代码
    global ann_input
    ann_input = theano.shared(np.array(x_train))  # 设置为共享权重
    ann_output = theano.shared(y_train.values)
    types = len(set(pd.Categorical(y_train).codes))
    with pm.Model() as model_BSR:
        alpha = pm.Normal('alpha', mu=0, sd=2, shape=types)  # shape=（类别数）
        beta = pm.Normal('beta', mu=0, sd=2, shape=(x_train.shape[1], types))  # shape=（参数个数，类别数）
        mu = alpha + pm.math.dot(ann_input, beta)
        theta = T.nnet.softmax(mu)
        out = pm.Categorical('out', p=theta, observed=ann_output)

        #     # BSR_MC
        #     start = pm.find_MAP()
        #     step = pm.NUTS()# 定义采样方法（NUTS适用于连续变量）
        #     trace_BSRMC = pm.sample(2000, step, start)
        #     BSR_VI
        inference = pm.ADVI()
        approx = pm.fit(n=2000, method=inference, obj_optimizer=pm.adam(learning_rate=0.01))
        trace_BSRVI = pm.variational.sample_approx(approx, draws=2000)

    # 采样迹线
    plt.rc('font', family='Times New Roman')
    matplotlib.rc('xtick', labelsize=12)
    # plt.style.use('classic')
    az.plot_trace(trace_BSRVI)
    # plt.savefig("D:/Desktop/大论文/图表和结果/导出图片/BSR_VI_采样.png", format="png", dpi=600)
    plt.tight_layout()
    plt.show()
    trace = trace_BSRVI
    model = model_BSR
    predict(X=x_train, y=y_train, model=model, trace=trace, sample=10)
    draw(real=y_train, pred=pred, prob=prob, name='train')

    predict(X=x_test, y=y_test, model=model, trace=trace, sample=10)
    draw(real=y_test, pred=pred, prob=prob, name='test')
    return

def BNN():
    ann_input = theano.shared(np.array(x_train))  # 设置为共享权重
    ann_output = theano.shared(pd.Categorical(y_train).codes)
    types = len(set(pd.Categorical(y_train).codes))
    # BNN
    n_hidden = 30
    # Initialize random weights between each layer
    init_1 = np.random.randn(x_train.shape[1], n_hidden)
    init_2 = np.random.randn(n_hidden, n_hidden)
    init_3 = np.random.randn(n_hidden, n_hidden)
    init_out = np.random.randn(n_hidden, types)

    with pm.Model() as model_BNN:
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


        inference = pm.ADVI()
        approx = pm.fit(n=2000, method=inference, obj_optimizer=pm.adam(learning_rate=0.01))
        trace_BNNVI = pm.variational.sample_approx(approx, draws=2000)
    # #     MC
    #     start = pm.find_MAP()
    #     step = pm.NUTS()# 定义采样方法（NUTS适用于连续变量）
    #     trace_mc = pm.sample(2000, step, start)
    trace = trace_BNNVI
    model = model_BNN

    X = x_train
    y = y_train
    predict(X=X, y=y, model=model, trace=trace, sample=100)
    # pd.concat([pd.Series(prob), pd.Series(pred)], axis=1).to_csv('D:/Desktop/大论文/图表和结果/excel/BNNVI_train.csv',
    #                                                              index=False, header=0)
    draw(real=y, pred=pred, prob=prob, name='train')
    predict(X=x_test, y=y_test, model=model, trace=trace, sample=100)
    # pd.concat([pd.Series(prob), pd.Series(pred)], axis=1).to_csv('D:/Desktop/大论文/图表和结果/excel/BNNVI_test.csv',
    #                                                              index=False, header=0)
    draw(real=y_test, pred=pred, prob=prob, name='test')
    return

def predict(X, y, model, trace, sample=100):
    global pred
    global prob
    global Prob
    #     y = pd.Categorical(y).codes
    ann_input.set_value(np.array(X))
    ppc = pm.sample_posterior_predictive(trace, model=model, samples=sample)
    #     print(ppc)
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
    pred_p = np.max(res, axis=1)
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
    redify = c / diff  # 纠正率
    acc_conf = redify / (1 - acc)  # 不确定度有效性

    print("acc=", acc)

    print("total_acc=", (np.sum(y == pred, axis=0) + c) / len(y))
    print("uncertain=", diff / len(y))
    print("redify=", redify)
    print("acc_conf=", acc_conf)

def draw(real, pred, prob, name='title'):
    font1 = {'family': 'Times New Roman', 'weight': 'light', 'size': 22}  # 设置字体模板，
    font2 = {'family': 'Times New Roman', 'weight': 'light', 'size': 28}  # 设置字体模板，

    plt.rc('figure', figsize=(18, 6))
    plt.rcParams['axes.unicode_minus'] = False  # 使用上标小标小一字号
    plt.rcParams['font.sans-serif'] = ['Times New Roman']

    n = len(real) if len(real) < 1000 else 1000
    x = [i for i in range(n)]

    plt.plot(x, real[:n], '--', label='Real Line')
    plt.plot(x, prob[:n], 'o', label='prob')
    plt.plot(x, pred[:n], 'o', label='Predict')

    plt.ylabel("mark", fontdict=font1)
    plt.xlabel("index", fontdict=font1)
    plt.title(name, fontdict=font2)  # 标题

    plt.tick_params(
        axis='x',  # 设置x轴
        direction='in',  # 小坐标方向，in、out
        which='both',  # 主标尺和小标尺一起显示，major、minor、both
        bottom=True,  # 底部标尺打开
        top=False,  # 上部标尺关闭
        labelbottom=True,  # x轴标签打开
        labelsize=16)  # x轴标签大小
    plt.tick_params(
        axis='y',
        direction='in',
        which='both',
        left=True,
        right=False,
        labelbottom=True,
        labelsize=16)
    plt.show()

#  数据集强化

#  参数预测
def regress(x_train, y_train):
    from catboost import CatBoostRegressor as cat
    cat_model = cat(task_type="CPU")
    cat_model.fit(x_train, y_train)

    from lightgbm.sklearn import LGBMRegressor
    LGBM_model = LGBMRegressor(objective='regression', num_leaves=55, max_depth=15)
    LGBM_model.fit(x_train, y_train)

if __name__ == '__main__':
    ## 测试集大小为20%， 80%/20%分
    X = data[train_cols]
    y = data[goal_col]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2000)
    x_train, x_test = minmax_norm(x_train), minmax_norm(x_test)
    # LR_model(x_train,y_train,x_test,y_test)
    # ANN_model(x_train,y_train,x_test,y_test)
    # LGBM_C_model()
    # BSR()
    # BNN()
#  参数预测
#
#  参数优化
