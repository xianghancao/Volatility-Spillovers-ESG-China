%load_ext autoreload
%autoreload 2
import sys
from lib.base import * 
from lib.network_spill import complex_network
from lib.network_topology import MyGraph

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import matplotlib.colors as mcolors
import matplotlib.cm as cm


# 描述性统计


def get_describe():
    df_num = pd.DataFrame()
    year_list = range(2013,2024)    
    if time_type == 'year':
        time_list = list(range(2013, 2024))
        df_hz_esg = CN.get_hz_esg_data(time_type)
    df_num = pd.DataFrame()
    time_list = list(range(2013, 2024))
    for i in time_list:
        df_esg_i = df_hz_esg[df_hz_esg['年份'] == i]
        df_t =  df_esg_i.groupby('综合评级').count()
        #print (df_t)
        df_num = pd.concat([df_num, df_t['年份']], axis =1 )
    df_num.columns = year_list
    df_num = df_num.fillna(0)

    df_num.loc['A'] += df_num.loc['AA']
    df_num = df_num.drop('AA')

    # 合并 "C" 和 "CC"
    df_num.loc['C'] += df_num.loc['CC']
    df_num = df_num.drop('CC')

    df_num.index = ['AAA,AA,A','B','BB','BBB', 'CC,C', 'CCC']

    new_index = ['AAA,AA,A','BBB','BB','B', 'CCC','CC,C',]
    df_num = df_num.reindex(new_index)

    df_num.T.astype(int).to_csv('output/describe/describe.csv')
    df_num.T.astype(int).to_latex('output/describe/describe.tex')