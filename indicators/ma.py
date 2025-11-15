from copy import deepcopy
from math import atan
from typing import List

import numpy as np
import pandas as pd

from common.type import Line


class MovingAverage:
    def __init__(self, period=20, upper_levels=0, lower_levels=0, min_step=1, speed=1.0):
        """
        Initialize Moving Average indicator
        Args:
            period (int): Period for moving average calculation
            upper_levels (int): Number of levels above the main MA
            lower_levels (int): Number of levels below the main MA
            min_step (float): Minimum step size for the first level
            speed (float): Growth rate of step size between levels
        """
        self.period = period
        self.main_ma_values = -np.ones(period)
        self.current_index = 0
        self.main_line: Line = Line()
        self.min_step = min_step
        self.speed = speed
        self.upper_levels = upper_levels
        self.lower_levels = lower_levels
        
        if self.upper_levels == 0 and self.lower_levels == 0:
            self.levels = [0]
        else:
            upper = np.arange(self.upper_levels, 0, -1) if self.upper_levels > 0 else []
            lower = np.arange(-1, -(self.lower_levels + 1), -1) if self.lower_levels > 0 else []
            self.levels = np.concatenate(([0], upper, lower)) if self.upper_levels > 0 or lower_levels > 0 else [0]
        
        self.current_ma_values = {level: None for level in self.levels}
        self.previous_ma_values = {level: None for level in self.levels}
        self.last_ma_directions = {level: None for level in self.levels}
        self.levels_lines: List[Line] = {level: Line(width=((level==0)+1)*3) for level in self.levels}

    def update(self, h) -> float:
        """
        Update moving average values based on new data
        """        
        ma_current_value = h["Close"][-self.period-1:-2].mean()
        # Calculate standard deviation as base for level steps
        base_step = h["Close"][-self.period-1:-2].std() * self.min_step
        # Calculate slope of the moving average
        if self.main_ma_values[-1] > 0:
            ma_slope = (ma_current_value - self.main_ma_values[-1]) / self.main_ma_values[-1] * 100
        else:
            ma_slope = 0
        # Calculate levels
        for level in self.levels:
            # Calculate dynamic step size that grows with level number
            level_multiplier = pow(abs(level), self.speed) if level != 0 else 0
            level_cur = ma_current_value + np.sign(level) * level_multiplier * base_step
            self.levels_lines[level].points.append((h["Date"][-2], level_cur))
            # Determine direction: 1 for growing, 0 for flat, -1 for downgrading
            if self.current_ma_values[level] is not None:
                if level_cur > self.current_ma_values[level]:
                    self.last_ma_directions[level] = 1
                elif level_cur < self.current_ma_values[level]:
                    self.last_ma_directions[level] = -1
                else:
                    self.last_ma_directions[level] = 0
            else:
                self.last_ma_directions[level] = 0  # Default to flat if no previous value
            self.previous_ma_values[level] = self.current_ma_values[level]
            self.current_ma_values[level] = level_cur
        # shift all values to the left
        self.main_ma_values[:-1] = self.main_ma_values[1:]
        # append new value to the right
        self.main_ma_values[-1] = ma_current_value

        self.ma_slope = ma_slope
        return ma_current_value

    @property
    def vis_objects(self) -> List[Line]:
        return list(self.levels_lines.values())