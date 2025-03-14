"""Cross-validation utilities for backtesting."""
import pickle
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger
from tabulate import tabulate

from backtesting.optimization import Optimizer
from backtesting.utils import BackTestResults
from common.type import Symbol, to_datetime
from trade.backtest import launch as backtest_launch

logger.remove()
logger.add(sys.stderr, level="INFO")
logger = logger.bind(module="data_processing.dataloading")
logger.disable("data_processing.dataloading")
logger = logger.bind(module="backtesting.optimization")
logger.disable("backtesting.optimization")


class CrossValidation:
    @dataclass
    class CrossValidationResults:
        """
        Results of the cross-validation process.
        """
        ids: Optional[List[str]] = field(default_factory=list)
        train_scores: Optional[List[float]] = field(default_factory=list)
        test_scores: Optional[List[float]] = field(default_factory=list)

        def add_train_test_scores(self, fold_id: str, train_score: float, test_score: float):
            self.ids.append(fold_id)
            self.train_scores.append(train_score)
            self.test_scores.append(test_score)

        def as_averall_table(self):
            """Print a formatted table of cross-validation results."""
            data = []
            for fold, (fold_id, train_score, test_score) in enumerate(zip(self.ids, self.train_scores, self.test_scores), 1):
                data.append(
                    [f"Fold {fold}", f"{fold_id}", f"{train_score:.4f}", f"{test_score:.4f}"])

            return tabulate(data,
                            headers=["Fold", "ID", "Train Score", "Test Score"],
                            tablefmt="grid")

        def as_summary_table(self):
            # Print summary statistics
            avg_train = np.mean(self.train_scores)
            avg_test = np.mean(self.test_scores)
            std_train = np.std(self.train_scores)
            std_test = np.std(self.test_scores)

            summary_data = [
                ["Average", f"{avg_train:.4f}", f"{avg_test:.4f}"],
                ["Std Dev", f"{std_train:.4f}", f"{std_test:.4f}"]
            ]

            return tabulate(summary_data,
                            headers=["Fold", "Train Score", "Test Score"],
                            tablefmt="grid")

    class Strategy(Enum):
        EQUAL_TEST_CUMULATIVE_TRAIN = "equal_test_cumulative_train"
        EQUAL_TEST_EQUAL_TRAIN = "equal_test_equal_train"

    def __init__(self, optim_cfg: Dict[str, Any], n_splits: int = 5,
                 strategy: Strategy = Strategy.EQUAL_TEST_CUMULATIVE_TRAIN):
        """
        Initialize cross-validation for backtesting.

        Args:
            optim_cfg: Configuration dictionary for optimization
            n_splits: Number of splits for cross-validation
            test_size: Size of each test period. If None, will be calculated based on data range
        """
        self.optim_cfg = optim_cfg
        self.n_splits = n_splits
        self.max_ndeals_per_month = 10
        self.min_ndeals_per_month = 1
        self.strategy = strategy
        
        self.cache_dir = Path(".cache/cross_validation")
        self.cache_dir.mkdir(exist_ok=True, parents=True)

        # Get data range from configuration
        assert "date_start" in optim_cfg and "date_end" in optim_cfg, "date_start and date_end must be provided in optim_cfg"
        self.date_start: np.datetime64 = np.datetime64(optim_cfg["date_start"])
        self.date_end: np.datetime64 = np.datetime64(optim_cfg["date_end"])

        # Set test_size based on data range if not provided
        # Calculate total period in days
        total_days = (self.date_end -
                      self.date_start).astype('timedelta64[D]').astype(int)
        # Convert numpy.int64 to Python int
        days_per_split = int(total_days // self.n_splits)
        self.test_size = timedelta(days=days_per_split)

    def _get_cumulative_split_bounds(self) -> List[tuple]:
        """
        Iterate over the data range and calculate split bounds for cumulative splits.
        """

        splits = []
        test_size_days = np.timedelta64(self.test_size.days, 'D')

        # Calculate splits from the end of the data range
        for i in range(self.n_splits - 1):
            # Calculate test period for this split
            test_end = self.date_end - (self.n_splits - i - 2) * test_size_days
            test_start = test_end - test_size_days
            train_start = self.date_start
            splits.append((train_start, test_start, test_start, test_end))

        return splits

    def _get_equal_split_bounds(self) -> List[tuple]:
        raise NotImplementedError("Equal split bounds not implemented")

    def _get_split_bounds(self) -> List[tuple]:
        if self.strategy == self.Strategy.EQUAL_TEST_CUMULATIVE_TRAIN:
            return self._get_cumulative_split_bounds()
        elif self.strategy == self.Strategy.EQUAL_TEST_EQUAL_TRAIN:
            return self._get_equal_split_bounds()
        else:
            raise ValueError(f"Invalid strategy: {self.strategy}")

    def cross_validate(self,
                       config: Dict,
                       metric: str = "recovery") -> Dict:
        """
        Perform cross-validation using the Optimizer.

        Args:
            config: Configuration dictionary for backtesting
            metric: Metric to optimize (default: "APR")
        """
        cval_results = self.CrossValidationResults()

        for i, (train_start, train_end, test_start, test_end) in enumerate(self._get_split_bounds()):
            logger.info(f"Processing test split {i + 1}/{self.n_splits - 1}")
            logger.info(f"Train period: {train_start} to {train_end}")
            logger.info(f"Test period: {test_start} to {test_end}")

            # Create train config for optimization
            train_config = config.copy()
            train_config["date_start"] = train_start
            train_config["date_end"] = train_end

            # Run optimization on training data
            optimizer = Optimizer(train_config, sortby=metric)
            optimizer.clear_cache()
            optimizer.run_backtests()
            optimization_results: Optimizer.OptimizationResults = optimizer.optimize(
                train_config)

            # Extract best result from training
            optimization_results.sort_by(score_name=metric)
            optimization_results.apply_filters(max_ndeals_per_month=self.max_ndeals_per_month,
                                               min_ndeals_per_month=self.min_ndeals_per_month)
            assert len(
                optimization_results.opt_summary) > 0, "No valid results found"
            best_params = optimization_results.best_config
            train_score = optimization_results.opt_summary.loc[optimization_results.top_run_id]["APR"]
            train_id = optimization_results.top_configuration_id

            # Create test config with optimized parameters
            test_config = config.copy()
            test_config.update(best_params)
            test_config["date_start"] = test_start
            test_config["date_end"] = test_end

            # Run backtest on test data with optimized parameters
            test_results = backtest_launch(test_config)
            test_score = test_results.APR

            # Save results to cache
            cache_file = self.cache_dir / f"split_{i}.pickle"
            with open(cache_file, "wb") as f:
                pickle.dump((train_config, test_config,
                            best_params, test_score), f)

            cval_results.add_train_test_scores(train_id, train_score, test_score)

        logger.info(
            f"Cross-validation results:\n{cval_results.as_averall_table()}")
        logger.info(
            f"Cross-validation summary:\n{cval_results.as_summary_table()}")
        return cval_results
