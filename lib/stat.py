import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.stattools import adfuller


# 平稳性检验
def check_stationarity(series):
    result = adfuller(series)
    return result[1]  # 返回p值


def adf_check(df):
    adf_no_list = []
    for col in (df.columns):
        p_value = check_stationarity(df[col])
        if p_value > 0.05:            # 如果p值大于0.05，进行差分
            #print(col)
            adf_no_list.append(col)
    df = df.drop(adf_no_list, axis =1)
    return df
#volatilities = volatilities.drop(adf_no_list, axis=1)



def descriptive_statistics(df, precision=3):
    # 计算描述性统计量
    desc_stats = pd.DataFrame({
        'SampleSize': df.count(),
        'Mean': df.mean().round(precision),
        'Median': df.median().round(precision),
        'StdDev': df.std().round(precision),
        'Min': df.min().round(precision),
        'Max': df.max().round(precision),
        'Skewness': df.skew().round(precision),
        'Kurtosis': df.kurtosis().round(precision)
    })

    # Jarque-Bera检验
    jb_stat = []
    jb_pvalue = []
    for column in df.columns:
        jb_stat_value, jb_pvalue_value = stats.jarque_bera(df[column])
        jb_stat.append(round(jb_stat_value, precision))
        jb_pvalue.append(round(jb_pvalue_value, precision))

    desc_stats['Jarque-Bera Stat'] = jb_stat

    # ADF检验
    adf_stat = []
    adf_pvalue = []
    for column in df.columns:
        adf_stat_value, adf_pvalue_value, _, _, _, _ = adfuller(df[column])
        adf_stat.append(round(adf_stat_value, precision))
        adf_pvalue.append(round(adf_pvalue_value, precision))

    # 添加ADF Stat和P-Value到描述性统计结果
    desc_stats['ADF Stat'] = adf_stat

    # 格式化ADF Stat列，添加显著性标记
    def format_adf_stat(stat, p_value):
        if p_value < 0.01:
            return f"{stat} ***"
        elif p_value < 0.05:
            return f"{stat} **"
        elif p_value < 0.1:
            return f"{stat} *"
        else:
            return str(stat)

    desc_stats['ADF Stat'] = [
        format_adf_stat(stat, p_value) for stat, p_value in zip(adf_stat, adf_pvalue)
    ]

    return desc_stats