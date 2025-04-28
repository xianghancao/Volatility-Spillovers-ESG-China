import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from tqdm.notebook import tqdm
from statsmodels.tsa.api import VAR
import networkx as nx
import matplotlib.pyplot as plt
import os,sys
import seaborn as sns
#import igraph as ig
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib_inline import backend_inline
backend_inline.set_matplotlib_formats('svg')  
from tqdm.notebook import tqdm
import warnings
warnings.filterwarnings("ignore")
import datetime

def scale_one(df):
    # j 归一化
    s1 = df.values
    s2 = df.abs().values.sum(axis=1)
    s3 = (s1.T/s2).T
    return pd.DataFrame(s3, index=df.index, columns=df.index)

#  对于dataframe 如果里面有list 把它转成均值
def list_to_mean_ori(x):
    if isinstance(x, list):
        return sum(x) / len(x)
    else:
        return x


def cal_mean_max(x):
    return np.mean(eval(x)) / np.max(eval(x))

    
    
def list_to_mean(x):
    try:
        if isinstance(x, str) and isinstance(eval(x), list):
            return sum(eval(x)) / len(eval(x))
        elif isinstance(x, list):
            return sum(x) / len(x)
        else:
            return x
    except (NameError, SyntaxError, TypeError):
        return np.nan
    
    
#补全股票代码
def pad_stock_code(code):
    return str(code).zfill(6)

    
def normalize_rows_with_non_negative(df):
    # 将负数替换为0
    df_non_negative = df.where(df >= 0, 0)
    
    # 对每一行进行归一化
    normalized_df = df_non_negative.div(df_non_negative.sum(axis=1), axis=0)
    
    return normalized_df


def scale_one(df):
    # j 归一化
    s1 = df.values
    s2 = df.abs().values.sum(axis=1)
    s3 = (s1.T/s2).T
    return pd.DataFrame(s3, index=df.index, columns=df.index)


def generate_dates(start_year, end_year, day_month_pairs):
    dates = []
    for year in range(start_year, end_year + 1):
        for month, day in day_month_pairs:
            try:
                new_date = datetime.date(year, month, day)
                dates.append(new_date)
            except ValueError:
                # This is to handle invalid dates like 2023-07-31
                pass
    return dates



# 季度数据转换
def get_quarter_dates(date):
    year = date.year
    if date.month in [1, 2, 3]:
        start_date = datetime.date(year, 1, 1)
        end_date = datetime.date(year, 3, 31)
    elif date.month in [4, 5, 6]:
        start_date = datetime.date(year, 4, 1)
        end_date = datetime.date(year, 6, 30)
    elif date.month in [7, 8, 9]:
        start_date = datetime.date(year, 7, 1)
        end_date = datetime.date(year, 9, 30)
    else:
        start_date = datetime.date(year, 10, 1)
        end_date = datetime.date(year, 12, 31)
    return start_date, end_date



def dataframe_to_latex_table(df_result_mean, output_path):
    latex_output = r"""
    \begin{table*}[h]
    \caption{Summary Statistics of Volatility}
    \label{tbl1}
    \centering
    \begin{tabular*}{\linewidth}{@{} lrrrrrrrr@{}}
    \toprule
     节点数 & 边数 & 是否有向 & 是否加权 & 节点度数 & 最大度 & 网络直径 & 平均路径长度 & 度中心性\\
    \midrule
    """

    # 将 DataFrame 中的前半部分列数据写入 LaTeX 表格
    for index, row in df_result_mean.iterrows():
        row_data = " & ".join(map(str, row[:9]))
        latex_output += f"{row_data} \\\\ \n"

    latex_output += r"""
    \end{tabular*}
    \begin{tabular*}{\linewidth}{@{} lrrrrrrr@{}}
    \toprule
     接近中心性 & 中介中心性 & 特征向量中心性 & 平均介数中心性betweenness & 平均接近中心性closeness \\
    \midrule
    """

    # 将 DataFrame 中的后半部分列数据写入 LaTeX 表格
    for index, row in df_result_mean.iterrows():
        row_data = " & ".join(map(str, row[9:]))
        latex_output += f"{row_data} \\\\ \n"

    latex_output += r"""
    \bottomrule
    \end{tabular*}
    \end{table*}
    """
    # 保存为 .tex 文件
    output_path = 'test_result.tex'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(latex_output)