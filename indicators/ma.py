import numpy as np
import pandas as pd
from common.type import Line
from copy import deepcopy


class MovingAverage:
    def __init__(self, period=20, ma_type="sma"):
        """
        Initialize Moving Average indicator
        Args:
            period (int): Period for moving average calculation
            ma_type (str): Type of moving average - 'sma' or 'ema'
        """
        self.period = period
        self.ma_type = ma_type.lower()
        self.values = np.empty(period)
        self.current_index = 0
        self.line2draw: Line = Line()
        
    def update(self, h) -> np.ndarray:
        """
        Update moving average values based on new data
        """        
        if self.ma_type == 'sma':
            # append point to Line opject
            ma_current_value = h["Close"][-self.period-1:-2].mean()
            # shift all values to the left
            self.values[:-1] = self.values[1:]
            # append new value to the right
            self.values[-1] = ma_current_value
            # update Line object
            self.line2draw.points.append((h["Date"][-2], ma_current_value))
        elif self.ma_type == 'ema':
            raise NotImplementedError
        else:
            raise ValueError(f"Unknown MA type: {self.ma_type}")
        
        return self.values

    def get_vis_objects(self):
        return [self.line2draw]