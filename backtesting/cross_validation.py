"""Cross-validation utilities for backtesting."""
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from backtesting.optimization import Optimizer
from backtesting.utils import BackTestResults
from common.type import Symbol


class CrossValidation:
    def __init__(self, 
                 data: pd.DataFrame,
                 n_splits: int = 5,
                 test_size: Optional[timedelta] = None,
                 gap: Optional[timedelta] = None):
        """
        Initialize cross-validation for backtesting.
        
        Args:
            data: Historical price data
            n_splits: Number of splits for cross-validation
            test_size: Size of each test period. If None, will be calculated as data_size / n_splits
            gap: Gap between train and test periods. If None, no gap will be used
        """
        self.data = data
        self.n_splits = n_splits
        self.test_size = test_size or (data.index[-1] - data.index[0]) / n_splits
        self.gap = gap or timedelta(0)
        self.optimizer = Optimizer()
        
    def _get_split_bounds(self) -> List[tuple]:
        """Calculate bounds for each split."""
        splits = []
        data_start = self.data.index[0]
        data_end = self.data.index[-1]
        
        for i in range(self.n_splits):
            test_start = data_start + i * (self.test_size + self.gap)
            test_end = test_start + self.test_size
            
            if test_end > data_end:
                break
                
            splits.append((test_start, test_end))
            
        return splits
    
    def cross_validate(self, 
                      config: Dict,
                      metric: str = "APR",
                      n_jobs: int = -1) -> Dict:
        """
        Perform cross-validation using the Optimizer.
        
        Args:
            config: Configuration dictionary for backtesting
            metric: Metric to optimize (default: "APR")
            n_jobs: Number of jobs for parallel processing (-1 for all available)
            
        Returns:
            Dictionary containing:
            - best_params: Best parameters found across all splits
            - split_results: Results for each split
            - mean_score: Mean score across all splits
            - std_score: Standard deviation of scores
        """
        splits = self._get_split_bounds()
        split_results = []
        
        for i, (test_start, test_end) in enumerate(splits):
            logger.info(f"Processing split {i+1}/{len(splits)}")
            
            # Create split-specific config
            split_config = config.copy()
            split_config["date_start"] = test_start
            split_config["date_end"] = test_end
            
            # Run optimization for this split
            self.optimizer.optimize(split_config)
            
            # Get best results for this split
            best_result = self._get_best_result(metric)
            split_results.append({
                "split": i,
                "test_period": (test_start, test_end),
                "best_params": best_result["params"],
                "score": best_result["score"]
            })
        
        # Calculate aggregate metrics
        scores = [r["score"] for r in split_results]
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        # Find best parameters across all splits
        best_split = max(split_results, key=lambda x: x["score"])
        
        return {
            "best_params": best_split["best_params"],
            "split_results": split_results,
            "mean_score": mean_score,
            "std_score": std_score
        }
    
    def _get_best_result(self, metric: str) -> Dict:
        """Get best result from the optimizer based on specified metric."""
        # This is a placeholder - actual implementation would depend on how
        # the Optimizer class stores and exposes its results
        # You would need to modify this based on the actual Optimizer implementation
        results = self.optimizer.get_results()  # This method would need to be added to Optimizer
        best_result = max(results, key=lambda x: x[metric])
        return {
            "params": best_result["params"],
            "score": best_result[metric]
        }