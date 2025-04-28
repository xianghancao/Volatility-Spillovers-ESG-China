from .base import *
import pandas as pd
import os
from .utils import pad_stock_code

def load_price():

    close_df = pd.read_feather(os.path.join(current_path, 'datasets/stock_price/Close.feather'))
    close_df.index = close_df['time']
    close_df = close_df.iloc[:, 1:]

    open_df = pd.read_feather(os.path.join(current_path, 'datasets/stock_price/Open.feather'))
    open_df.index = open_df['time']
    open_df = open_df.iloc[:, 1:]

    high_df = pd.read_feather(os.path.join(current_path, 'datasets/stock_price/High.feather'))
    high_df.index = high_df['time']
    high_df = high_df.iloc[:, 1:]

    low_df = pd.read_feather(os.path.join(current_path, 'datasets/stock_price/Low.feather'))
    low_df.index = low_df['time']
    low_df = low_df.iloc[:, 1:]

    #hz_path = r'D:\BaiduSyncdisk\papers\ESG\ESG评级大合集\华证2009-2023年（含细分项+季度)）\3-华证ESG评级和得分（2009-2023）\华证ESG评级-得分2009-2023.xlsx'
    #hz_path = 'D:\BaiduSyncdisk\papers\ESG\ESG评级大合集\华证2009-2023年（含细分项+季度)）\华证esg评级2009-2023（细分项）\华证esg评级含细分项（年度）2009-2023.xlsx'
    # df_hz_esg = pd.read_excel(hz_path)
    # #df_hz_esg['年份'].unique()

    # #df_hz_esg[df_hz_esg['年份'] == 2023].to_csv('test.csv')


    # # 应用函数到 '股票代码' 列
    # df_hz_esg['股票代码'] = df_hz_esg['股票代码'].apply(pad_stock_code)

    # df_hz_esg = df_hz_esg.set_index('股票代码')
    
    return open_df, high_df, low_df, close_df#, df_hz_esg


def load_vol():
    # 读取波动率数据
    vol_interday = pd.read_csv('datasets/processing/vol_df.csv',
                              index_col=0)
    vol_interday.head(1)
    return vol_interday




def get_hz_esg_data(time_type = 'year'):

    if time_type == 'year':
        hz_path = os.path.join(current_path, 'datasets/华证2009-2023年（含细分项+季度)）/华证esg评级2009-2023（细分项）/华证esg评级含细分项（年度）2009-2023.xlsx')
        df_hz_esg = pd.read_excel(hz_path)
        df_hz_esg['股票代码'] = df_hz_esg['股票代码'].apply(pad_stock_code)
        df_hz_esg = df_hz_esg.set_index('股票代码')
    if time_type == 'season':
        print(current_path)
        hz_path = os.path.join(current_path, 'datasets/华证2009-2023年（含细分项+季度)）/华证esg评级2009-2023（细分项）/华证esg评级含细分项（季度）2009-2023.xlsx')
        print(hz_path)
        df_hz_esg = pd.read_excel(hz_path)
        df_hz_esg['股票代码'] = df_hz_esg['证券代码'].apply(pad_stock_code)
        df_hz_esg = df_hz_esg.set_index('股票代码')
        df_hz_esg = df_hz_esg.drop('证券代码', axis =1 )

    return df_hz_esg