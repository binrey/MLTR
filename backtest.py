from pathlib import Path
from shutil import rmtree
from time import perf_counter
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

# Если проблемы с отрисовкой графиков
# export QT_QPA_PLATFORM=offscreen

def backtest(cfg):
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
    logger.info(sformat.format("FINAL PROFIT", broker.profits.sum())) 
    
    import pickle
    pickle.dump((cfg, broker), open(str(Path("backtests") / f"btest{0:003.0f}.pickle"), "wb"))
    return broker
    
    
if __name__ == "__main__":
    import sys
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    cfg = PyConfig().test()
    cfg.run_model_device = None
    brok_results = backtest(cfg)
    plt.subplots(figsize=(20, 10))
    plt.plot([pos.close_date for pos in brok_results.positions], brok_results.profits.cumsum(), linewidth=2, alpha=0.6)
    
    plt.savefig("backtest.png")