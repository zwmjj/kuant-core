"""策略基类 — 所有策略实现这个接口"""
import pandas as pd
from abc import ABC, abstractmethod


class BaseStrategy(ABC):
    """
    策略插件接口。任何策略只需实现 generate_signal()。

    使用:
        class MyStrategy(BaseStrategy):
            name = "我的策略"
            def generate_signal(self, data):
                # data 是 prepare_data() 返回的字典
                return signal_dataframe  # (date x permno), 越高越看多
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
        生成信号DataFrame (date x permno)。
        值越高 = 越看多, 值越低 = 越看空。
        """
        raise NotImplementedError

    def get_params(self) -> dict:
        """返回策略参数 (用于日志和报告)"""
        return {
            'name': self.name,
            'long_n': self.long_n,
            'short_n': self.short_n,
            'long_pct': self.long_pct,
            'short_pct': self.short_pct,
            'turnover_penalty': self.turnover_penalty,
            'weight_mode': self.weight_mode,
        }
