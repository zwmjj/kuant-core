"""Strategy base class - every strategy implements this interface."""
import pandas as pd
from abc import ABC, abstractmethod


class BaseStrategy(ABC):
    """
    Strategy plugin interface. A strategy only needs to implement generate_signal().

    Usage:
        class MyStrategy(BaseStrategy):
            name = "My Strategy"
            def generate_signal(self, data):
                # data is the dict returned by prepare_data()
                return signal_dataframe  # (date x permno), higher = more bullish
    """
    name: str = "未命名策略"
    description: str = ""

    # 默认参数 (子类可覆盖)
    long_n: int = 20
    short_n: int = 20
    long_pct: float = 1.15
    short_pct: float = 0.15
    turnover_penalty: float = 0.25
    weight_mode: str = 'inv_vol'

    @abstractmethod
    def generate_signal(self, data: dict) -> pd.DataFrame:
        """
        Generate a signal DataFrame (date x permno).
        Higher values = more bullish, lower values = more bearish.
        """
        raise NotImplementedError

    def get_params(self) -> dict:
        """Return the strategy parameters (used for logging and reporting)."""
        return {
            'name': self.name,
            'long_n': self.long_n,
            'short_n': self.short_n,
            'long_pct': self.long_pct,
            'short_pct': self.short_pct,
            'turnover_penalty': self.turnover_penalty,
            'weight_mode': self.weight_mode,
        }
