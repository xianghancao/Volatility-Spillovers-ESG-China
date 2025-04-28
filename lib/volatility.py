# 计算波动率的库

import numpy as np
import pandas as pd
#波动率={0.511*（H-L）2-0.019*[（C-O）（H+L-2*O）-2*（H-O）*(L-O)]-0.383*(C-O)2}1/2
#其中H、L、O和C分别表示最高价、最低价、开盘价和收盘价。从计算的波动率时间序列来看，与VIX指数走势基本一致。
def calculate_volatility(open_, high, low, close):
    term1 = 0.511 * (high - low) ** 2
    term2 = -0.019 * ((close - open_) * (high + low - 2 * open_) - 2 * (high - open_) * (low - open_))
    term3 = -0.383 * (close - open_) ** 2
    #print (term1 + term2)
    volatility = np.sqrt(term1 + term2 + term3) / open_
    volatility.index = pd.to_datetime(volatility.index)
    return volatility

