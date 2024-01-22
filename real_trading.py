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
from experts import ExpertFormation, PyConfig
from utils import Broker
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from easydict import EasyDict

# Если проблемы с отрисовкой графиков
# export QT_QPA_PLATFORM=offscreen

def fight(cfg):
    exp = ExpertFormation(cfg)
    broker = Broker(cfg)
    hist_pd, hist = DataParser(cfg).load()
    mw = MovingWindow(hist, cfg.hist_buffer_size)
    if cfg.save_plots:
        save_path = Path("backtests") / f"{cfg.ticker}-{cfg.period}"
        if save_path.exists():
            rmtree(save_path)
        save_path.mkdir()
    tstart = max(cfg.hist_buffer_size+1, cfg.tstart)
    tend = cfg.tend if cfg.tend is not None else hist.Id.shape[0]
    t0, texp, tbrok, tdata = perf_counter(), 0, 0, 0
    for t in tqdm(range(tstart, tend), "back test"):
        h, dt = mw(t)
        tdata += dt
        if t < tstart or len(broker.active_orders) == 0:
            texp += exp.update(h)
            broker.active_orders = exp.orders
        
        pos, dt = broker.update(h)
        tbrok += dt
        if pos is not None:
            logger.debug(f"t = {t} -> postprocess closed position")
            broker.close_orders(h.Id[-2])
            if cfg.save_plots:
                ords_lines = [order.lines for order in broker.orders if order.open_indx >= pos.open_indx]
                lines2plot = exp.lines + ords_lines + [pos.lines]
                colors = ["blue"]*(len(lines2plot)-1) + ["green" if pos.profit > 0 else "red"]
                widths = [1]*(len(lines2plot)-1) + [2]
                
                hist2plot = hist_pd.iloc[lines2plot[0][0][0]:lines2plot[-1][-1][0]+1]
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
                            alines=dict(alines=lines2plot, colors=colors, linewidths=widths),
                            savefig=save_path / f"fig-{str(pos.open_date).split('.')[0]}.png")
                del fig
            exp.reset_state()
    
    ttotal = perf_counter() - t0
    
    sformat = "{:>30}: {:>3.0f} %"
    logger.info(f"{cfg.ticker}-{cfg.period}: {cfg.body_classifier.func.name}, sl={cfg.stops_processor.func.name}, sl-rate={cfg.trailing_stop_rate}")
    logger.info("{:>30}: {:.1f} sec".format("total backtest", ttotal))
    logger.info(sformat.format("expert updates", texp/ttotal*100))
    logger.info(sformat.format("broker updates", tbrok/ttotal*100))
    logger.info(sformat.format("data loadings", tdata/ttotal*100))
    logger.info("-"*30)
    logger.info(sformat.format("FINAL PROFIT", broker.profits.sum()) + f" ({len(broker.positions)} deals)") 
    
    import pickle
    pickle.dump((cfg, broker), open(str(Path("backtests") / f"btest{0:003.0f}.pickle"), "wb"))
    return broker
    
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
    logger.add(sys.stderr, level="INFO")
    cfg = PyConfig().test()
    

    session = HTTP(
        testnet=False,
        api_key="aA2DKjelcik0WbJyxI",
        api_secret="hIhnPUEBVmDII1FfeYEicTljZjwrUHW8pTm8",
    )

    print(session.get_wallet_balance(
        accountType="UNIFIED",
        coin="BTC",
    ))
    
    exp = ExpertFormation(cfg)
    broker = Broker(cfg)
    if cfg.save_plots:
        save_path = Path("real_trading") / f"{cfg.ticker}-{cfg.period}"
        if save_path.exists():
            rmtree(save_path)
        save_path.mkdir()
    
    
    get_rounded_time = lambda tmessage: int(int(tmessage["timeSecond"])/10)
    t0 = get_rounded_time(session.get_server_time()["result"])
    t = t0
    while t == t0:
        tmessage = session.get_server_time()["result"]
        t = get_rounded_time(tmessage)
        print(t0, t)
        message = session.get_kline(category="spot",
                        symbol=cfg.ticker,
                        interval=cfg.period[1:],
                        start=0,
                        end=int(tmessage["timeSecond"])*1000,
                        limit=cfg.hist_buffer_size)
        
        h = get_bybit_hist(message["result"], cfg.hist_buffer_size)
        open_orders = session.get_open_orders(category="spot")["result"]["list"]
        if len(open_orders) == 0:
            texp = exp.update(h)
            broker.active_orders = exp.orders
        
        if cfg.save_plots:
            hist2plot = pd.DataFrame(h)
            hist2plot.index = pd.to_datetime(hist2plot.Date)
            fig = mpf.plot(hist2plot, 
                type='candle', 
                block=False,
                savefig=save_path / f"fig-{str(t).split('.')[0]}.png")
            del fig
        sleep(1)
    
    