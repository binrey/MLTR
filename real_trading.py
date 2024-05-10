from pathlib import Path
from shutil import rmtree
from time import sleep
import mplfinance as mpf
import pandas as pd
from loguru import logger
pd.options.mode.chained_assignment = None
from experts import ByBitExpert
from utils import PyConfig
from datetime import datetime
import numpy as np
import pandas as pd
from easydict import EasyDict
from copy import deepcopy
import telebot
from PIL import Image
import pickle
from multiprocessing import Process



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
                
                
def date2save_format(date, prefix=None):
    s = str(np.array(date).astype('datetime64[s]')).split('.')[0].replace(":", "-") 
    if prefix is not None and len(prefix) > 0:
        s += f"-{prefix}"
    return s + ".png"

     
def plot_fig(hist2plot, lines2plot, save_path=None, prefix=None, t=None, side=None, ticker="X"):
    for line in lines2plot:
        assert len(line) >= 2, "line must have more than 1 point"
        for point in line:
            assert len(point) == 2
            assert type(point[0]) is pd.Timestamp
            assert type(point[1]) is float
            assert point[0] >= hist2plot.index[0]
            assert point[0] <= hist2plot.index[-1]
    mystyle=mpf.make_mpf_style(base_mpf_style='yahoo',rc={'axes.labelsize':'small'})
    kwargs = dict(
        type='candle', 
        block=False,
        alines=dict(alines=lines2plot, linewidths=[1]*len(lines2plot)),
        volume=True,
        figscale=1.5,
        style=mystyle,
        datetime_format='%m-%d %H:%M',
        title=f"{np.array(t).astype('datetime64[s]')}-{ticker}-{side}",
        returnfig=True
    )

    fig, axlist = mpf.plot(data=hist2plot, **kwargs)
    
    if side in ["Buy", "Sell"]:
        side_int = 1 if side == "Buy" else -1
        x = hist2plot.index.get_loc(t)
        if type(x) is slice:
            x = x.start
        y = hist2plot.loc[t].Open
        if y.ndim > 0:
            y = y.iloc[0]
        arrow_size = (hist2plot.iloc[-10:].High - hist2plot.iloc[-10:].Low).mean()
        axlist[0].annotate("", (x, y + arrow_size*side_int), fontsize=20, xytext=(x, y),
                    color="black", 
                    arrowprops=dict(
                        arrowstyle='->',
                        facecolor='b',
                        edgecolor='b'))
        
    if save_path is not None:
        save_path = save_path / date2save_format(t, prefix)  
        fig.savefig(save_path, bbox_inches='tight', pad_inches=0.2)  
    return save_path
      
      
def log_position(t, hist2plot, lines2plot, save_path):
    d = {"time": t,
         "hist": hist2plot,
         "lines": lines2plot
         }
    save_path = save_path / f"{str(t).split('.')[0]}.pkl".replace(":", "-")
    pickle.dump(d, open(save_path, "wb"))
    logger.info(f"save logs to {save_path}")
    

class BybitTrading:
    def __init__(self, cfg, credentials) -> None:
        self.cfg = cfg
        api_key, api_secret, bot_token = Path(credentials).read_text().splitlines()[:3]
        self.t0 = 0
        self.my_telebot = Telebot(bot_token)
        self.h = None
        self.open_position = None
        self.session = HTTP(
            testnet=False,
            api_key=api_key,
            api_secret=api_secret,
        )
        self.exp = ByBitExpert(cfg, self.session)
        if cfg.save_plots:
            self.save_path = Path("real_trading") / f"{cfg.ticker}-{cfg.period}"
            if self.save_path.exists():
                rmtree(self.save_path)
            self.save_path.mkdir()
            
        self.hist2plot, self.lines2plot, self.open_time, self.side, self.sl = None, None, None, None, None
        
    def test_connection(self):
        self.get_open_orders_positions()        
        if self.open_position is not None:
            logger.error("Есть открытые позиции! Сначала надо всех их закрыть :(")
            raise ConnectionError()
            
    def handle_trade_message(self, message):
        try:
            data = message.get('data')
            self.time = int(data[0].get("T"))#/60/int(cfg.period[1:]))
            time_rounded = int(int(data[0].get("T"))/1000/60/int(cfg.period[1:]))
            # logger.debug(f"{datetime.fromtimestamp(int(self.time/1000))} {time_rounded}")
        except (ValueError, AttributeError):
            pass            
        if time_rounded > self.t0:
            if self.t0:
                self.update()
                logger.info(f"update for {self.cfg.ticker} {datetime.fromtimestamp(int(self.time/1000))} finished!")
            self.t0 = time_rounded

        
    def trailing_sl(self, pos):
        sl = float(pos["stopLoss"])
        try:
            sl = sl + self.cfg.trailing_stop_rate*(self.h.Open.iloc[-1] - sl)
            resp = self.session.set_trading_stop(
                category="linear",
                symbol=self.cfg.ticker,
                stopLoss=sl,
                slTriggerB="IndexPrice",
                positionIdx=0,
            )
            logger.debug(resp)
        except Exception as ex:
            print(ex)
        return sl        

    def get_open_orders_positions(self):
            self.open_orders = []
            self.open_position = None
            self.open_orders = self.session.get_open_orders(category="linear", symbol=cfg.ticker)["result"]["list"]
            positions = self.session.get_positions(category="linear", symbol=cfg.ticker)["result"]["list"]
            for pos in positions :
                if float(pos["size"]):
                    self.open_position = pos        

    def update(self):
        try:
            self.get_open_orders_positions()    
            if cfg.save_plots:
                if self.hist2plot is not None:
                    self.h = pd.DataFrame(self.h).iloc[-2:-1]
                    self.h.index = pd.to_datetime(self.h.Date)
                    self.hist2plot = pd.concat([self.hist2plot, self.h])
                    self.lines2plot[-1].append((pd.to_datetime(self.hist2plot.iloc[-1].Date), self.sl))
                    log_position(self.open_time, self.hist2plot, self.lines2plot, self.save_path)
                                
                if self.open_position is not None and self.hist2plot is None:
                    self.hist2plot = pd.DataFrame(self.h)
                    self.hist2plot.index = pd.to_datetime(self.hist2plot.Date)
                    self.lines2plot = deepcopy(self.exp.lines)
                    for line in self.lines2plot:
                        for i, point in enumerate(line):
                            y = point[1]
                            try:
                                y = y.item() #  If y is 1D numpy array
                            except:
                                pass
                            x = point[0]
                            x = max(self.hist2plot.Id[0], x)
                            x = min(self.hist2plot.Id[-1], x)
                            line[i] = (self.hist2plot.index[self.hist2plot.Id==x][0], y)    
                    self.open_time = pd.to_datetime(self.hist2plot.iloc[-1].Date)
                    self.side = self.open_position["side"]
                    self.lines2plot.append([(self.open_time, float(self.open_position["avgPrice"])), (self.open_time, float(self.open_position["avgPrice"]))])
                    self.lines2plot.append([(self.open_time, float(self.open_position["stopLoss"])), (self.open_time, float(self.open_position["stopLoss"]))])
                    log_position(self.open_time, self.hist2plot, self.lines2plot, self.save_path)
                    p = Process(target=plot_fig, args=(self.hist2plot, self.lines2plot, self.save_path, None, self.open_time, self.side, cfg.ticker))
                    p.start()
                    p.join()
                    self.my_telebot.send_image(self.save_path / date2save_format(self.open_time))
                    self.hist2plot = self.hist2plot.iloc[:-1]
                    
                    
            if self.open_position is not None:
                self.sl = self.trailing_sl(self.open_position)
                
            message = self.session.get_kline(category="linear",
                            symbol=cfg.ticker,
                            interval=cfg.period[1:],
                            start=0,
                            end=self.time,
                            limit=cfg.hist_buffer_size)
            self.h = get_bybit_hist(message["result"], cfg.hist_buffer_size)
                
            if cfg.save_plots:
                if self.open_position is None and self.exp.order_sent and self.lines2plot: 
                    last_row = pd.DataFrame(self.h).iloc[-2:]
                    last_row.index = pd.to_datetime(last_row.Date)
                    self.hist2plot = pd.concat([self.hist2plot, last_row])                    
                    self.lines2plot[-2][-1] = (pd.to_datetime(self.hist2plot.iloc[-1].Date), self.lines2plot[-2][-1][-1])
                    self.lines2plot[-1].append((pd.to_datetime(self.hist2plot.iloc[-1].Date), self.sl))
                    log_position(self.open_time, self.hist2plot, self.lines2plot, self.save_path)
                    p = Process(target=plot_fig, args=(self.hist2plot, self.lines2plot, self.save_path, None, self.open_time, self.side, cfg.ticker))
                    p.start()
                    p.join()
                    self.my_telebot.send_image(self.save_path / date2save_format(self.open_time))
                    self.hist2plot = None    
            
            texp = self.exp.update(self.h, self.open_position)
        except Exception as ex:
            logger.error(ex)
        

    
if __name__ == "__main__":
    import sys
    from pybit.unified_trading import HTTP, WebSocket
    
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")
    cfg = PyConfig(sys.argv[1]).test()
    cfg.save_plots = True
    
    
    
    public = WebSocket(channel_type='linear', testnet=False)
    # private = WebSocket(channel_type='private',
    #                     api_key=api_key,
    #                     api_secret=api_secret, 
    #                     testnet=False) 
    bybit_trading = BybitTrading(cfg, "./configs/api.yaml")
    bybit_trading.test_connection()
    public.trade_stream(symbol=cfg.ticker, callback=bybit_trading.handle_trade_message)
    while True:
        sleep(1)
    

    
    