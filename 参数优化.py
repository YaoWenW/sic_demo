# 数据处理库
import pandas as pd
pd.set_option('max_columns',200)
import numpy as np
# 科学绘图库
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from pylab import mpl
# 其他库
import warnings
from sko.GA import GA
from lightgbm.sklearn import LGBMRegressor
import sklearn.metrics as sm

# 设置默认字体表格默认配置
pd.set_option('max_columns',200)
warnings.filterwarnings('ignore')
palette = 'muted'
sns.set_palette(palette); sns.set_color_codes(palette)
np.set_printoptions(precision=2)
pd.set_option('display.precision', 2)
matplotlib.rcParams['font.family']='SimHei'
plt.rcParams['axes.unicode_minus'] = False


# 随机划分测试集
from sklearn.model_selection import train_test_split

def process(data):
    DATA = data.copy(deep=True)
    DATA = DATA.rename(columns={"岩土分级标注": "标注"})  # 重命名参数
    # 删除列
    if '环号' in DATA.columns:
        del DATA['环号']
    # 按照条件删除行
    #     DATA = DATA.drop(DATA[DATA['标注'] =='复合地层'].index).reset_index(drop=True)

    DATA['软硬程度'] = DATA['标注']
    DATA['软硬程度'].loc[DATA['标注'] == '1'] = '软'
    DATA['软硬程度'].loc[DATA['标注'] == '2'] = '软'
    DATA['软硬程度'].loc[DATA['标注'] == '3'] = '较硬'
    DATA['软硬程度'].loc[DATA['标注'] == '4'] = '较硬'
    DATA['软硬程度'].loc[DATA['标注'] == '5'] = '硬'
    DATA['软硬程度'].loc[DATA['标注'] == '6'] = '硬'
    DATA['软硬程度'].loc[DATA['标注'] == '复合地层'] = '复合'

    DATA = DATA[DATA['区间'] != 'mwz']
    DATA = DATA[DATA['区间'] != 'mwy']
    #     DATA = DATA[DATA['区间'] != 'xsz']
    DATA = DATA[DATA['区间'] != 'xsy']
    DATA = DATA[DATA['区间'] != 'exz']
    DATA = DATA[DATA['区间'] != 'exy']
    DATA = DATA[DATA['区间'] != 'skz']
    DATA = DATA[DATA['区间'] != 'sky']

    DATA = DATA[DATA['区间'] != 'lwz']
    DATA = DATA[DATA['区间'] != 'wyy']
    DATA = DATA[DATA['区间'] != 'wyz']
    DATA = DATA[DATA['区间'] != 'yhy']
    DATA = DATA[DATA['区间'] != 'yhz']

    DATA['标注'] = pd.Categorical(DATA['标注']).codes
    #     DATA['软硬程度']= pd.Categorical(DATA['软硬程度']).codes
    return DATA

# 归一化参数
def minmax_norm(df_input):
    return (df_input - df_input.min()) / (df_input.max() - df_input.min())

def Mape(y_pred, y_true):
    return np.mean(np.abs((y_pred - y_true) / y_true)) * 100


def Mae(y_pred, y_true):
    return np.mean(np.abs((y_pred - y_true)))


def model_pred(model, modelname, x_train, y_train, x_test, y_test, goal_col):
    train_predict = model.predict(x_train)
    test_predict = model.predict(x_test)
    # 更新索引
    y = y_train.reset_index()
    y_train = y[goal_col]
    y = y_test.reset_index()
    y_test = y[goal_col]

    # 训练集
    plt.rc('figure', figsize=(5, 4))
    a = np.arange(0, 500, 5)
    plt.plot(a, y_train[a], color="r", label="真实值")  #颜色表示
    plt.plot(a, train_predict[a], color=(0, 0, 0), label="预测集")
    plt.xlabel("索引")  #x轴命名表示
    plt.ylabel(goal_col)  #y轴命名表示
    plt.title("%s训练集实际值与预测值折线图" % modelname)
    plt.legend()  #增加图例
    # plt.savefig('D:/桌面/论文资料/论文图片/{}训练集.png'.format(goal_col),dpi=300 , figsize=(5, 4))
    plt.show()  #显示图片

    print("MAPE = ", Mape(train_predict, y_train))
    print("MAE = ", Mae(train_predict, y_train))
    print("MSE=", sm.mean_squared_error(y_train, train_predict))
    print("R2=", sm.r2_score(y_train, train_predict))

    # 测试集
    a = np.arange(0, 50, 1)
    plt.plot(a, y_test[a], color="r", label="真实值")  #颜色表示
    plt.plot(a, test_predict[a], color=(0, 0, 0), label="预测集")
    plt.xlabel("索引")  #x轴命名表示
    plt.ylabel(goal_col)  #y轴命名表示
    plt.title("%s测试集实际值与预测值折线图" % modelname)
    plt.legend(loc='upper right')  #增加图例
    # plt.savefig('D:/桌面/论文资料/论文图片/{}测试集.png'.format(goal_col),dpi=300 , figsize=(5, 4))
    plt.show()  #显示图片

    print("MAPE = ", Mape(test_predict, y_test))
    print("MAE = ", Mae(test_predict, y_test))
    print("MSE=", sm.mean_squared_error(y_test, test_predict))
    print("R2=", sm.r2_score(y_test, test_predict))

    # 误差图

    a = np.arange(len(y_test))
    b1 = y_test[a]
    b2 = test_predict[a]
    b = np.arange(len(y_train))
    c1 = y_train[b]
    c2 = train_predict[b]
    plt.style.use('seaborn-whitegrid')
    mpl.rcParams['font.sans-serif'] = ['SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    x = np.linspace(0, 3000, 100)

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    ## 绘制1:1对角线
    ax.plot((0, 1), (0, 1),
            transform=ax.transAxes,
            ls='--',
            c='k',
            label="1:1 line")

    plt.plot(b1,
             b2,
             'o',
             c='red',
             label="测试集MAE={:.3f},MAPE={:.3f}".format(
                 Mae(test_predict, y_test), Mape(test_predict, y_test)),
             alpha=0.6)
    plt.plot(c1,
             c2,
             'o',
             c='blue',
             label='训练集MAE={:.3f},MAPE={:.3f}'.format(
                 Mae(train_predict, y_train), Mape(train_predict, y_train)),
             alpha=0.6)

    plt.legend()
    ax.tick_params(labelsize=12)
    plt.xlabel(goal_col + "预测值", fontsize=15)  #x轴命名表示
    plt.ylabel(goal_col + "实际值", fontsize=15)  #y轴命名表示
    # plt.savefig('D:/桌面/科研材料/北延工程论文/论文图片/降噪前{}误差.png'.format(goal_col),dpi=300 , figsize=(3, 2))
    plt.title('误差图')
    plt.show()

def GA(data):
    X = []
    Y = []
    # data = x_test
    for i in range(len(data)):
        c = ['刀盘转速(r/min)', '总推进力(kN)', '螺旋机速度(r/min)']

        def schaffer(x1):
            ls = list(np.array(data[[col for col in data.columns if col not in c]])[i])
            ls.insert(0, x1[0])
            ls.insert(1, x1[1])
            ls.insert(2, x1[2])
            #         ls.insert(3,x1[3])
            #         ls.insert(4,x1[4])
            #         ls.insert(5,x1[5])
            x = [ls]
            return -float(LGBM_model.predict(x)[0])


        ls1 = list(np.array(x_train[[col for col in x_train.columns if col in c]])[i])
        lb = []
        ub = []
        for n in ls1:
            if n < 0.9:
                ub.append(n + 0.2)
            else:
                ub.append(0.95)
            if n > 0.2:
                lb.append(n - 0.2)
            else:
                lb.append(0.05)
        #     print(i,lb, ub)
        ga = GA(func=schaffer, n_dim=3, size_pop=100, max_iter=100, lb=lb, ub=ub, precision=1e-2)
        best_x, best_y = ga.run()
        X.append(best_x)
        Y.append(-best_y)
        print('best_x:', best_x, '\n', 'best_y:', -best_y)

if __name__ == '__main__':
    data_ring = pd.read_excel('D:/Desktop/Database/大论文数据集/每环标注-插值.xlsx')
    data = process(data_ring).reset_index(drop=True)
    operation_cols = ['刀盘转速(r/min)', '推进速度(mm/min)', '总推进力(kN)', '螺旋机速度(r/min)']
    response_cols = ['刀盘转矩(kN*m)', '顶部土压(bar)', '土压平均值(bar)', '螺旋机转矩(kN*m)']

    other_cols = ['标注', '埋深(m)']
    useless_cols = ['环号', '区间', '软硬程度', '顶部土压(bar)', '1#膨润土流量(L)', '泡沫原液流量(L)']
    train_cols = [col for col in data.columns if col not in useless_cols]
    # train_cols = operation_cols + other_cols
    # train_cols = operation_cols

    goal_col = '推进速度(mm/min)'
    X = data[train_cols]
    y = data[goal_col]

    # 加入上一行的时序信息()
    X1 = pd.concat([X.iloc[[0]], X], axis=0, ignore_index=True)[:-1]
    a = pd.concat([X, X1], axis=1)
    a.columns = train_cols + ['上一环' + i for i in train_cols]

    col_new = [c for c in a.columns if c not in response_cols + [goal_col]]
    X_new = a[col_new]

    # 测试集大小为20%， 80%/20%分
    x_train, x_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=2000)
    x_train, x_test = minmax_norm(x_train), minmax_norm(x_test)

    op_train = [i for i in operation_cols if i != goal_col]
    x_train1, x_test1 = x_train[op_train], x_test[op_train]

    # 建立预测模型（非强化数据集）
    LGBM_model = LGBMRegressor(objective='regression',num_leaves=55,max_depth=10) # 定义模型超参数
    LGBM_model.fit(x_train1,y_train)
    model_pred(LGBM_model, 'LGBM', x_train1, y_train, x_test1, y_test, goal_col)

    # 建立预测模型（强化数据集）
    LGBM_model = LGBMRegressor(objective='regression',num_leaves=55,max_depth=10) # 定义模型超参数
    LGBM_model.fit(x_train,y_train)
    model_pred(LGBM_model, 'LGBM', x_train, y_train, x_test, y_test, goal_col)