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
