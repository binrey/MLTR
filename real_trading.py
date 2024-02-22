from pathlib import Path
from shutil import rmtree
from time import sleep
import mplfinance as mpf
import pandas as pd
from loguru import logger
pd.options.mode.chained_assignment = None
from experts import ByBitExpert, PyConfig
from datetime import datetime
import numpy as np
import pandas as pd
from easydict import EasyDict
from copy import deepcopy
import telebot
from PIL import Image
import pickle



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

    input = np.array(mresult["list"], dtype=np.float64)[::-1]
    data.Id = input[:, 0].astype(np.int64)
    data.Date = data.Id*1000000
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
                
             
class Telebot:
    def __init__(self, token) -> None:
        self.bot = telebot.TeleBot(token)
        self.chat_id = 480902846
    def send_image(self, img_path, caption=None):
        try:
            img = Image.open(img_path)
            if caption is not None:
                self.bot.send_message(self.chat_id, caption)
            self.bot.send_photo(self.chat_id, img)
        except Exception as ex:
            self.bot.send_message(self.chat_id, ex)
                
                
def plot_fig(hist2plot, lines2plot, save_path=None, prefix=None, t=None, side=None, ticker="X", send2telegram=False):
    # try:
    if True:
        global my_telebot
        mystyle=mpf.make_mpf_style(base_mpf_style='yahoo',rc={'axes.labelsize':'small'})
        kwargs = dict(
            type='candle', 
            block=False,
            alines=dict(alines=lines2plot, linewidths=[1]*len(lines2plot)),
            volume=True,
            figscale=1.5,
            style=mystyle,
            datetime_format='%m-%d %H:%M',
            title=f"{t}-{ticker}-{side}",
            returnfig=True
        )

        fig, axlist = mpf.plot(data=hist2plot, **kwargs)
        
        if side in ["Buy", "Sell"]:
            side_int = 1 if side == "Buy" else -1
            x = hist2plot.index.get_loc(t)
            # if type(x) is slice:
            #     x = x.start
            y = hist2plot.loc[t].Open
            axlist[0].annotate("", (x, y*(1+0.001*side_int)), fontsize=20, xytext=(x, y),
                        color="black", 
                        arrowprops=dict(
                            arrowstyle='->',
                            facecolor='b',
                            edgecolor='b'))
            
        if save_path is not None:
            save_path = save_path / f"{prefix}-{str(t).split('.')[0]}.png".replace(":", "-")    
            fig.savefig(save_path, bbox_inches='tight', pad_inches=0.2)  
        if send2telegram:
            my_telebot.send_image(save_path)
    # except Exception as ex:
    #     logger.error(ex)
    # return fig
      
      
def log_position(t, hist2plot, lines2plot, save_path):
    d = {"time": t,
         "hist": hist2plot,
         "lines": lines2plot
         }
    save_path = save_path / f"{str(t).split('.')[0]}.pkl".replace(":", "-")
    pickle.dump(d, open(save_path, "wb"))
    logger.info(f"save logs to {save_path}")
    
    
if __name__ == "__main__":
    import sys
    from pybit.unified_trading import HTTP
    
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")
    cfg = PyConfig().test()
    cfg.ticker = sys.argv[1]
    cfg.save_plots = True
    cfg.lot = 0.02 if cfg.ticker == "ETHUSDT" else 0.001
    
    api_key, api_secret, bot_token = Path("./configs/api.yaml").read_text().splitlines()
    
    my_telebot = Telebot(bot_token)

    session = HTTP(
        testnet=False,
        api_key=api_key,
        api_secret=api_secret,
    )
    
    exp = ByBitExpert(cfg, session)
    if cfg.save_plots:
        save_path = Path("real_trading") / f"{cfg.ticker}-{cfg.period}"
        if save_path.exists():
            rmtree(save_path)
        save_path.mkdir()
    
    get_rounded_time = lambda tmessage: int(int(tmessage["timeSecond"])/60/int(cfg.period[1:]))
    hist2plot, lines2plot, open_time, side = None, None, None, None
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
        # try:
        if True:
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
                    log_position(open_time, hist2plot, lines2plot, save_path)
                                
                if open_position is not None and hist2plot is None:
                    hist2plot = pd.DataFrame(h)
                    hist2plot.index = pd.to_datetime(hist2plot.Date)
                    lines2plot = deepcopy(exp.lines)
                    for line in lines2plot:
                        for i, point in enumerate(line):
                            y = point[1]
                            try:
                                y = y.item() #  If y is 1D numpy array
                            except:
                                pass
                            x = point[0]
                            x = max(hist2plot.Id[0], x)
                            x = min(hist2plot.Id[-1], x)
                            line[i] = (hist2plot.index[hist2plot.Id==x][0], y)    
                    open_time = pd.to_datetime(hist2plot.iloc[-1].Date)
                    side = open_position["side"]
                    lines2plot.append([(open_time, float(open_position["avgPrice"])), (open_time, float(open_position["avgPrice"]))])
                    lines2plot.append([(open_time, float(open_position["stopLoss"])), (open_time, float(open_position["stopLoss"]))])
                    log_position(open_time, hist2plot, lines2plot, save_path)
                    plot_fig(hist2plot, lines2plot, 
                             save_path, 
                             "open", 
                             open_time, 
                             side=side, 
                             ticker=cfg.ticker,
                             send2telegram=True)
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
                if open_position is None and exp.order_sent and lines2plot: 
                    last_row = pd.DataFrame(h).iloc[-2:]
                    last_row.index = pd.to_datetime(last_row.Date)
                    hist2plot = pd.concat([hist2plot, last_row])                    
                    lines2plot[-2][-1] = (pd.to_datetime(hist2plot.iloc[-1].Date), lines2plot[-2][-1][-1])
                    lines2plot[-1].append((pd.to_datetime(hist2plot.iloc[-1].Date), sl))
                    log_position(open_time, hist2plot, lines2plot, save_path)
                    plot_fig(hist2plot, lines2plot, save_path, "close", open_time, 
                                side=side,
                                ticker=cfg.ticker,
                                send2telegram=True)
                    hist2plot = None    
            
            texp = exp.update(h, open_position)

        # except Exception as ex:
        #     logger.error(ex)
        #     continue
    
    