from pathlib import Path
from shutil import rmtree
from time import perf_counter, sleep
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
# import yfinance as yf
from loguru import logger
from tqdm import tqdm
from dataloading import MovingWindow, DataParser
pd.options.mode.chained_assignment = None
from experts import ByBitExpert, PyConfig
from utils import Broker
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from easydict import EasyDict
from copy import deepcopy

# Если проблемы с отрисовкой графиков
# export QT_QPA_PLATFORM=offscreen
 
def get_bybit_hist(mresult, size):
    data = EasyDict(Date=np.empty(size, dtype=np.datetime64),
        Id=np.zeros(size, dtype=np.int64),
        Open=np.zeros(size, dtype=np.float32),
        Close=np.zeros(size, dtype=np.float32),
        High=np.zeros(size, dtype=np.float32),
        Low=np.zeros(size, dtype=np.float32),
        Volume=np.zeros(size, dtype=np.int64)
        )    

    input = np.array(mresult["list"], dtype=np.float32)[::-1]
    data.Date = input[:, 0].astype(int)*1000000
    data.Id = input[:, 0].astype(int)
    data.Open = input[:, 1]
    data.High = input[:, 2]
    data.Low  = input[:, 3]
    data.Close= input[:, 4]
    data.Volume = input[:, 5]
    return data
    
    
if __name__ == "__main__":
    import sys
    from pybit.unified_trading import HTTP
    
    
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")
    cfg = PyConfig().test()
    cfg.ticker = sys.argv[1]
    cfg.save_plots = True
    cfg.period = "M5"
    cfg.trailing_stop_rate = 0.01
    cfg.lot = 0.01 if cfg.ticker == "ETHUSDT" else 0.001
    

    session = HTTP(
        testnet=False,
        api_key="aA2DKjelcik0WbJyxI",
        api_secret="hIhnPUEBVmDII1FfeYEicTljZjwrUHW8pTm8",
    )
    
    exp = ByBitExpert(cfg, session)
    if cfg.save_plots:
        save_path = Path("real_trading") / f"{cfg.ticker}-{cfg.period}"
        if save_path.exists():
            rmtree(save_path)
        save_path.mkdir()
    
    get_rounded_time = lambda tmessage: int(int(tmessage["timeSecond"])/60/int(cfg.period[1:]))
    while True:
        t = t0 = get_rounded_time(session.get_server_time()["result"])
        while t == t0:
            try:
                tmessage = session.get_server_time()["result"]
                t = get_rounded_time(tmessage)
                print(tmessage["timeSecond"], t0, t)
            except Exception as ex:
                print(ex)
            sleep(1)
        
        message = session.get_kline(category="linear",
                        symbol=cfg.ticker,
                        interval=cfg.period[1:],
                        start=0,
                        end=int(tmessage["timeSecond"])*1000,
                        limit=cfg.hist_buffer_size)
        
        h = get_bybit_hist(message["result"], cfg.hist_buffer_size)
        open_orders = session.get_open_orders(category="linear", symbol=cfg.ticker)["result"]["list"]
        positions = session.get_positions(category="linear", symbol=cfg.ticker)["result"]["list"]
        open_positions = [pos for pos in positions if float(pos["size"])]
        
        for pos in open_positions:
            pos_side = 1 if pos["side"] == "Buy" else -1
            sl = float(pos["stopLoss"])
            sl = sl + pos_side*cfg.trailing_stop_rate*abs(h.Open[-1] - sl)
            resp = session.set_trading_stop(
                category="linear",
                symbol=cfg.ticker,
                stopLoss=sl,
                slTriggerB="IndexPrice",
                positionIdx=0,
            )
            logger.debug(resp)
        
        if len(open_orders) == 0 and len(open_positions) == 0:
            texp = exp.update(h)
            if cfg.save_plots:
                try:
                    hist2plot = pd.DataFrame(h)
                    hist2plot.index = pd.to_datetime(hist2plot.Date)
                    lines2plot = deepcopy(exp.lines)
                    for line in lines2plot:
                        for i, point in enumerate(line):
                            y = point[1]
                            try:
                                y = y.item()
                            except:
                                pass
                            line[i] = (hist2plot.index[hist2plot.Id==point[0]][0], y)
                            
                    fig = mpf.plot(hist2plot, 
                        type='candle', 
                        block=False,
                        alines=dict(alines=lines2plot),
                        savefig=save_path / f"fig-{str(t).split('.')[0]}.png")
                    del fig        
                except Exception as ex:
                    print(ex)
                    print(hist2plot.Id)  
                    print(point[0])  
        else:
            exp.reset_state()
            

        
    
    