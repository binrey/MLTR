from pathlib import Path
from shutil import rmtree
from time import perf_counter, sleep
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
from utils import Position
from loguru import logger
from tqdm import tqdm
from dataloading import MovingWindow, DataParser
pd.options.mode.chained_assignment = None
from experts import ByBitExpert, PyConfig
from datetime import datetime
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
    
    
def trailing_sl(cfg, pos):
    sl = float(pos["stopLoss"])
    try:
        sl = sl + cfg.trailing_stop_rate*(h.Open[-1] - sl)
        resp = session.set_trading_stop(
            category="linear",
            symbol=cfg.ticker,
            stopLoss=sl,
            slTriggerB="IndexPrice",
            positionIdx=0,
        )
        logger.debug(resp)
    except Exception as ex:
        print(ex)
    return sl
                
                
if __name__ == "__main__":
    import sys
    from pybit.unified_trading import HTTP
    
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")
    cfg = PyConfig().test()
    cfg.ticker = sys.argv[1]
    cfg.save_plots = True
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
    hist2plot = None  
    while True:
        t = t0 = get_rounded_time(session.get_server_time()["result"])
        while t == t0:
            sleep(1)
            try:
                tmessage = session.get_server_time()["result"]
                t = get_rounded_time(tmessage)
                print(datetime.fromtimestamp(int(session.get_server_time()["result"]["timeSecond"])), t0, t)
            except Exception as ex:
                print(ex)
        
        open_orders = session.get_open_orders(category="linear", symbol=cfg.ticker)["result"]["list"]
        positions = session.get_positions(category="linear", symbol=cfg.ticker)["result"]["list"]
        open_position = None
        for pos in positions :
            if float(pos["size"]):
                open_position = pos
        
        
            
        if cfg.save_plots:
            if hist2plot is not None:
                h = pd.DataFrame(h).iloc[-2:-1]
                h.index = pd.to_datetime(h.Date)
                hist2plot = pd.concat([hist2plot, h])
                lines2plot[-1].append((pd.to_datetime(hist2plot.iloc[-1].Date), sl))
                            
            if open_position is not None and hist2plot is None:
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
                open_time = pd.to_datetime(hist2plot.iloc[-1].Date)
                lines2plot.append([(open_time, float(open_position["avgPrice"])), (None, float(open_position["avgPrice"]))])
                lines2plot.append([(open_time, float(open_position["stopLoss"]))])
                hist2plot = hist2plot.iloc[:-1]
                
        if open_position is not None:
            sl = trailing_sl(cfg, open_position)
            
        message = session.get_kline(category="linear",
                        symbol=cfg.ticker,
                        interval=cfg.period[1:],
                        start=0,
                        end=int(tmessage["timeSecond"])*1000,
                        limit=cfg.hist_buffer_size)
        h = get_bybit_hist(message["result"], cfg.hist_buffer_size)
        
        if cfg.save_plots:
            if open_position is None and exp.order_sent:
                try:     
                    last_row = pd.DataFrame(h).iloc[-2:]
                    last_row.index = pd.to_datetime(last_row.Date)
                    hist2plot = pd.concat([hist2plot, last_row])                    
                    lines2plot[-2][-1] = (pd.to_datetime(hist2plot.iloc[-1].Date), lines2plot[-2][-1][-1])
                    lines2plot[-1].append((pd.to_datetime(hist2plot.iloc[-1].Date), sl))
                    fig = mpf.plot(hist2plot, 
                        type='candle', 
                        block=False,
                        alines=dict(alines=lines2plot),
                        savefig=save_path / f"fig-{str(t).split('.')[0]}.png")
                    del fig    
                    hist2plot = None    
                except Exception as ex:
                    print(ex)
                    print(hist2plot.Id)  
                    print(point[0])        
        
        texp = exp.update(h, open_position)

        
    
    