import numpy as np
import pandas as pd
from common.type import Line
from copy import deepcopy
from typing import List


class MovingAverage:
    def __init__(self, period=20, ma_type="sma", levels_count=0, levels_step=1):
        """
        Initialize Moving Average indicator
        Args:
            period (int): Period for moving average calculation
            ma_type (str): Type of moving average - 'sma' or 'ema'
        """
        self.period = period
        self.ma_type = ma_type.lower()
        self.main_ma_values = -np.ones(period)
        self.current_index = 0
        self.main_line: Line = Line()
        self.levels_count = levels_count*2
        self.levels_step = levels_step
        
        if levels_count == 0:
            self.levels = [0]
        else:
            self.levels = np.arange(self.levels_count//2, -self.levels_count//2-1, -1)
        self.last_ma_values = {level: None for level in self.levels}
        self.last_ma_directions = {level: None for level in self.levels}
        self.levels_lines: List[Line] = {level: Line(width=((level==0)+1)*3) for level in self.levels}

    def update(self, h) -> float:
        """
        Update moving average values based on new data
        """        
        # append point to Line object
        ma_current_value = h["Close"][-self.period-1:-2].mean()
        # Calculate standard deviation
        std_dev = h["Close"][-self.period-1:-2].std()
        level_step_value = std_dev * self.levels_step
        # Calculate levels
        for level in self.levels:
            level_cur = ma_current_value + level * level_step_value
            self.levels_lines[level].points.append((h["Date"][-2], level_cur))
            # Determine direction: 1 for growing, 0 for flat, -1 for downgrading
            if self.last_ma_values[level] is not None:
                if level_cur > self.last_ma_values[level]:
                    self.last_ma_directions[level] = 1
                elif level_cur < self.last_ma_values[level]:
                    self.last_ma_directions[level] = -1
                else:
                    self.last_ma_directions[level] = 0
            else:
                self.last_ma_directions[level] = 0  # Default to flat if no previous value
            self.last_ma_values[level] = level_cur
        # shift all values to the left
        self.main_ma_values[:-1] = self.main_ma_values[1:]
        # append new value to the right
        self.main_ma_values[-1] = ma_current_value
        
        return ma_current_value

    def get_vis_objects(self) -> List[Line]:
        return list(self.levels_lines.values())