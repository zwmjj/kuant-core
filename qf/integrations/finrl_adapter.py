"""FinRL 深度强化学习适配器
FinRL: https://github.com/AI4Finance-Foundation/FinRL
封装 Stable-Baselines3 的 PPO/A2C/DDPG/SAC/TD3 算法,
构建股票交易 RL 环境并训练/评估 agent。
"""
import warnings
import numpy as np
import pandas as pd
from typing import Any, Optional

warnings.filterwarnings("ignore")

try:
    import gymnasium as gym
    from gymnasium import spaces

    _HAS_GYM = True
except ImportError:
    try:
        import gym
        from gym import spaces

        _HAS_GYM = True
    except ImportError:
        _HAS_GYM = False

try:
    from stable_baselines3 import PPO, A2C, DDPG, SAC, TD3

    _HAS_SB3 = True
    _ALGO_MAP = {
        "PPO": PPO,
        "A2C": A2C,
        "DDPG": DDPG,
        "SAC": SAC,
        "TD3": TD3,
    }
except ImportError:
    _HAS_SB3 = False
    _ALGO_MAP = {}


def _check_sb3():
    if not _HAS_SB3:
        raise ImportError(
            "stable-baselines3 未安装。请运行: "
            "pip install stable-baselines3[extra]"
        )
    if not _HAS_GYM:
        raise ImportError(
            "gymnasium 未安装。请运行: pip install gymnasium"
        )


# ---------------------------------------------------------------------------
# RL Environment (FinRL-style stock trading env)
# ---------------------------------------------------------------------------


class StockTradingEnv(gym.Env):
    """简化版股票交易 RL 环境 (FinRL StockTradingEnv 风格)。

    Observation: [cash_balance, stock_holdings..., stock_prices..., features...]
    Action:      连续值 [-1, 1] 对应 [卖出, 买入] 每只股票

    Parameters
    ----------
    prices : np.ndarray
        形状 (n_steps, n_stocks), 收盘价
    features : np.ndarray
        形状 (n_steps, n_stocks * n_features), 额外特征
    initial_cash : float
        初始资金
    transaction_cost : float
        交易成本比例
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        prices: np.ndarray,
        features: np.ndarray = None,
        initial_cash: float = 1_000_000.0,
        transaction_cost: float = 0.001,
        max_shares: int = 100,
    ):
        super().__init__()
        self.prices = prices
        self.features = features if features is not None else np.zeros((len(prices), 0))
        self.n_steps, self.n_stocks = prices.shape
        self.initial_cash = initial_cash
        self.transaction_cost = transaction_cost
        self.max_shares = max_shares

        # State: cash + holdings + prices + features
        n_obs = 1 + self.n_stocks + self.n_stocks + self.features.shape[1]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_obs,), dtype=np.float32
        )
        # Action: buy/sell proportion for each stock
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_stocks,), dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.cash = self.initial_cash
        self.holdings = np.zeros(self.n_stocks, dtype=np.float32)
        self.portfolio_values = [self.initial_cash]
        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        obs = np.concatenate([
            [self.cash / self.initial_cash],  # normalized cash
            self.holdings / self.max_shares,  # normalized holdings
            self.prices[self.current_step] / self.prices[0],  # normalized prices
            self.features[self.current_step] if self.features.shape[1] > 0 else [],
        ]).astype(np.float32)
        return obs

    def step(self, action: np.ndarray):
        current_prices = self.prices[self.current_step]

        # Execute trades
        for i in range(self.n_stocks):
            act = np.clip(action[i], -1.0, 1.0)
            price = current_prices[i]
            if price <= 0:
                continue

            if act > 0:  # buy
                shares_to_buy = int(act * self.max_shares)
                cost = shares_to_buy * price * (1 + self.transaction_cost)
                if cost <= self.cash:
                    self.holdings[i] += shares_to_buy
                    self.cash -= cost
            elif act < 0:  # sell
                shares_to_sell = min(
                    int(abs(act) * self.max_shares),
                    int(self.holdings[i]),
                )
                revenue = shares_to_sell * price * (1 - self.transaction_cost)
                self.holdings[i] -= shares_to_sell
                self.cash += revenue

        # Move to next step
        self.current_step += 1
        done = self.current_step >= self.n_steps - 1

        # Calculate portfolio value
        if not done:
            next_prices = self.prices[self.current_step]
        else:
            next_prices = current_prices
        portfolio_value = self.cash + np.sum(self.holdings * next_prices)
        self.portfolio_values.append(portfolio_value)

        # Reward: log return of portfolio
        prev_value = self.portfolio_values[-2]
        reward = float(np.log(portfolio_value / prev_value)) if prev_value > 0 else 0.0

        info = {
            "portfolio_value": portfolio_value,
            "cash": self.cash,
            "holdings_value": float(np.sum(self.holdings * next_prices)),
        }

        return self._get_obs(), reward, done, False, info


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def prepare_rl_env(
    prices: pd.DataFrame,
    features: pd.DataFrame = None,
    initial_cash: float = 1_000_000.0,
    transaction_cost: float = 0.001,
    max_shares: int = 100,
) -> "StockTradingEnv":
    """构建股票交易 RL 环境。

    Parameters
    ----------
    prices : pd.DataFrame
        价格矩阵 (date x stock), 日频
    features : pd.DataFrame, optional
        特征矩阵 (date x n), 如技术指标。
        若为 None 则使用价格的简单技术特征。
    initial_cash : float
        初始资金
    transaction_cost : float
        交易成本
    max_shares : int
        每只股票最大持仓股数

    Returns
    -------
    StockTradingEnv
    """
    if not _HAS_GYM:
        raise ImportError("gymnasium 未安装。请运行: pip install gymnasium")

    prices_arr = prices.ffill().bfill().values.astype(np.float32)

    if features is not None:
        feat_arr = features.fillna(0).values.astype(np.float32)
    else:
        # Auto-generate simple features: returns, volatility, momentum
        ret = prices.pct_change().fillna(0)
        vol = ret.rolling(20).std().fillna(0)
        mom = prices.pct_change(20).fillna(0)
        feat_arr = pd.concat([ret, vol, mom], axis=1).values.astype(np.float32)

    env = StockTradingEnv(
        prices=prices_arr,
        features=feat_arr,
        initial_cash=initial_cash,
        transaction_cost=transaction_cost,
        max_shares=max_shares,
    )
    return env


def train_drl_agent(
    env: "StockTradingEnv",
    algorithm: str = "PPO",
    timesteps: int = 100_000,
    **kwargs,
) -> Any:
    """训练深度 RL agent。

    Parameters
    ----------
    env : StockTradingEnv
        交易环境
    algorithm : str
        'PPO', 'A2C', 'DDPG', 'SAC', 'TD3'
    timesteps : int
        训练步数
    **kwargs
        传给 SB3 算法的额外参数

    Returns
    -------
    stable_baselines3 model
    """
    _check_sb3()

    algo = algorithm.upper()
    if algo not in _ALGO_MAP:
        raise ValueError(
            f"不支持的算法: {algorithm}。支持: {list(_ALGO_MAP.keys())}"
        )

    AlgoClass = _ALGO_MAP[algo]
    policy = "MlpPolicy"
    model = AlgoClass(policy, env, verbose=0, **kwargs)
    model.learn(total_timesteps=timesteps)
    return model


def evaluate_agent(
    agent: Any,
    env: "StockTradingEnv",
    n_episodes: int = 1,
) -> dict:
    """评估 RL agent 表现。

    Parameters
    ----------
    agent : SB3 model
        训练好的模型
    env : StockTradingEnv
        评估环境 (应使用测试集构建)
    n_episodes : int
        评估次数

    Returns
    -------
    dict
        {
            'total_return': float,
            'sharpe_ratio': float,
            'max_drawdown': float,
            'portfolio_values': list,
        }
    """
    _check_sb3()

    all_values = []
    all_returns_list = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            if done or truncated:
                break

        values = env.portfolio_values
        all_values.append(values)
        rets = np.diff(values) / np.array(values[:-1])
        all_returns_list.append(rets)

    # Aggregate
    avg_values = np.mean(all_values, axis=0) if len(all_values) > 1 else all_values[0]
    avg_rets = np.mean(all_returns_list, axis=0) if len(all_returns_list) > 1 else all_returns_list[0]

    total_return = float(avg_values[-1] / avg_values[0] - 1) if len(avg_values) > 1 else 0.0
    sharpe = float(np.mean(avg_rets) / np.std(avg_rets) * np.sqrt(252)) if np.std(avg_rets) > 0 else 0.0

    cummax = np.maximum.accumulate(avg_values)
    drawdown = (avg_values - cummax) / np.where(cummax > 0, cummax, 1)
    max_dd = float(np.min(drawdown))

    return {
        "total_return": total_return,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "final_value": float(avg_values[-1]),
        "portfolio_values": [float(v) for v in avg_values],
    }


def get_rl_actions(
    agent: Any,
    observation: np.ndarray,
    deterministic: bool = True,
) -> np.ndarray:
    """获取 RL agent 对单个 observation 的动作。

    Parameters
    ----------
    agent : SB3 model
    observation : np.ndarray
        环境状态向量
    deterministic : bool
        是否使用确定性策略

    Returns
    -------
    np.ndarray
        动作向量, shape=(n_stocks,), 值域 [-1, 1]
    """
    _check_sb3()
    action, _ = agent.predict(observation, deterministic=deterministic)
    return action
