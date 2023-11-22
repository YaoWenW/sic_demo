def process(data):
    DATA = data.copy(deep=True)
    DATA = DATA.rename(columns={"埋深": "埋深(m)"})  # 重命名参数
    # 删除列
    if '环号(r)' in DATA.columns:
        del DATA['环号(r)']
    # 按照条件删除行
    DATA = DATA.drop(DATA[DATA['标注'] == '复合'].index).reset_index(drop=True)
    DATA = DATA.drop(DATA[DATA['标注'] == '溶洞'].index).reset_index(drop=True)
    #     DATA = DATA.drop(DATA[DATA['标注'] =='1类'].index).reset_index(drop=True)
    #     DATA = DATA.drop(DATA[DATA['标注'] =='2类'].index).reset_index(drop=True)
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

    #     DATA = DATA[DATA['区间'] == 'xsz']
    #     DATA = DATA[DATA['区间'] == 'xsy']
    #     DATA = DATA[DATA['区间'] == 'exz']
    #     DATA = DATA[DATA['区间'] == 'exy']

    DATA['标注'] = pd.Categorical(DATA['标注']).codes
    DATA['软硬程度'] = pd.Categorical(DATA['软硬程度']).codes
    return DATA