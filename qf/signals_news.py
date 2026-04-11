"""新闻情绪因子 — 基于 Alpaca News 数据的情绪信号生成器"""

import re
import math
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


class NewsSentimentGenerator:
    """基于关键词的新闻情绪因子生成器

    Alpaca news 数据格式: list[dict]
        每条新闻包含: headline, created_at, source, symbols, url
    """

    # ── 关键词词典 ──

    POSITIVE_WORDS = frozenset({
        "beat", "surge", "rally", "upgrade", "growth", "profit", "record",
        "breakthrough", "soar", "bullish", "outperform", "raise", "strong",
        "positive", "approve", "gain", "rise", "expand", "exceed", "boost",
        "buy", "recommend", "innovation", "deal", "partnership", "dividend",
        "buyback", "acquisition", "upbeat", "optimistic", "recover", "rebound",
        "accelerate", "impressive", "robust", "solid", "beat", "top", "jump",
        "advance", "improve", "upside", "momentum", "confident", "success",
        "win", "opportunity", "breakout", "upgrade", "high", "peak",
    })

    NEGATIVE_WORDS = frozenset({
        "miss", "crash", "decline", "downgrade", "loss", "warning", "cut",
        "plunge", "bearish", "underperform", "lower", "weak", "negative",
        "reject", "fall", "drop", "shrink", "below", "sell", "layoff",
        "lawsuit", "recall", "investigation", "debt", "default", "bankruptcy",
        "fraud", "slump", "tumble", "disappoint", "concern", "risk", "fear",
        "volatility", "uncertainty", "struggle", "delay", "suspend", "halt",
        "probe", "fine", "penalty", "crisis", "downturn", "recession",
        "inflation", "overvalued", "shortfall", "writedown", "impairment",
    })

    AMPLIFIER_WORDS = frozenset({
        "very", "significantly", "sharply", "dramatically", "massive", "huge",
        "record", "unprecedented", "extremely", "remarkably", "substantially",
        "considerably", "exceptionally", "extraordinary", "intensely",
    })

    NEGATION_WORDS = frozenset({"not", "no", "never", "neither", "nor", "cannot"})

    _WORD_RE = re.compile(r"[a-zA-Z]+")

    # ── 初始化 ──

    def __init__(self):
        """加载关键词情绪打分器（无需外部 NLP 模型）"""
        # 预编译查找集合，加速运行时匹配
        self._pos = self.POSITIVE_WORDS
        self._neg = self.NEGATIVE_WORDS
        self._amp = self.AMPLIFIER_WORDS
        self._negation = self.NEGATION_WORDS

    # ── 单标题评分 ──

    def score_headline(self, headline: str) -> float:
        """对单条新闻标题打情绪分，返回 [-1, 1]

        算法:
            1. 分词并转小写
            2. 统计正面/负面词数量
            3. 放大词将相邻情绪词权重 ×1.5
            4. 否定词翻转下一个情绪词的符号
            5. score = (pos_count - neg_count) / total_words, clip 到 [-1, 1]
        """
        tokens = self._WORD_RE.findall(headline.lower())
        if not tokens:
            return 0.0

        total_words = len(tokens)
        pos_score = 0.0
        neg_score = 0.0

        negate_next = False
        amplify_next = False

        for token in tokens:
            # 否定词: 翻转下一个情绪词
            if token in self._negation:
                negate_next = True
                continue

            # 放大词: 加强下一个情绪词
            if token in self._amp:
                amplify_next = True
                continue

            weight = 1.5 if amplify_next else 1.0

            if token in self._pos:
                if negate_next:
                    neg_score += weight
                else:
                    pos_score += weight
                negate_next = False
                amplify_next = False

            elif token in self._neg:
                if negate_next:
                    pos_score += weight
                else:
                    neg_score += weight
                negate_next = False
                amplify_next = False

            else:
                # 非情绪词重置否定/放大状态
                negate_next = False
                amplify_next = False

        raw = (pos_score - neg_score) / total_words
        return float(np.clip(raw, -1.0, 1.0))

    # ── 聚合情绪 ──

    def aggregate_sentiment(
        self,
        news_list: List[dict],
        symbol: str,
        hours: int = 24,
    ) -> float:
        """聚合特定标的在时间窗口内的情绪得分

        Parameters
        ----------
        news_list : list[dict]
            Alpaca 格式的新闻列表
        symbol : str
            标的代码，如 'AAPL'
        hours : int
            回看时间窗口（小时）

        Returns
        -------
        float
            加权平均情绪得分，时间越近权重越高（指数衰减）
        """
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=hours)

        scores = []
        weights = []

        for item in news_list:
            # 过滤标的
            syms = item.get("symbols", [])
            if symbol not in syms:
                continue

            # 过滤时间
            created = item.get("created_at")
            if created is None:
                continue
            if isinstance(created, str):
                created = datetime.fromisoformat(created.replace("Z", "+00:00"))
            if created < cutoff:
                continue

            # 情绪打分
            headline = item.get("headline", "")
            score = self.score_headline(headline)
            scores.append(score)

            # 指数衰减权重: 越近权重越高
            age_hours = (now - created).total_seconds() / 3600.0
            decay = math.exp(-0.1 * age_hours)
            weights.append(decay)

        if not scores:
            return 0.0

        weights = np.array(weights)
        scores = np.array(scores)
        return float(np.average(scores, weights=weights))

    # ── 截面信号构建 ──

    @staticmethod
    def _cross_sectional_rank(series: pd.Series) -> pd.Series:
        """截面排名映射到 [-1, 1]"""
        ranked = series.rank(pct=True)
        return ranked * 2 - 1

    def build_sentiment_signal(self, news_by_symbol: Dict[str, list]) -> pd.Series:
        """构建截面情绪信号

        Parameters
        ----------
        news_by_symbol : dict[str, list]
            键为标的代码，值为该标的相关 Alpaca 新闻列表

        Returns
        -------
        pd.Series
            索引为标的代码，值为 [-1, 1] 的截面排名情绪信号
        """
        sentiments = {}
        for symbol, news_list in news_by_symbol.items():
            sentiments[symbol] = self.aggregate_sentiment(news_list, symbol)

        series = pd.Series(sentiments, dtype=float)
        if series.empty:
            return series
        return self._cross_sectional_rank(series)

    def news_volume_signal(self, news_by_symbol: Dict[str, list]) -> pd.Series:
        """新闻数量信号 — 关注度因子

        更多新闻 = 更多市场关注，截面排名后输出

        Parameters
        ----------
        news_by_symbol : dict[str, list]
            键为标的代码，值为该标的新闻列表

        Returns
        -------
        pd.Series
            索引为标的代码，值为 [-1, 1] 的截面排名
        """
        counts = {sym: len(news) for sym, news in news_by_symbol.items()}
        series = pd.Series(counts, dtype=float)
        if series.empty:
            return series
        return self._cross_sectional_rank(series)

    def news_breadth_signal(self, news_by_symbol: Dict[str, list]) -> pd.Series:
        """新闻广度信号 — 来源多样性因子

        覆盖来源越多，信号越强

        Parameters
        ----------
        news_by_symbol : dict[str, list]
            键为标的代码，值为该标的新闻列表

        Returns
        -------
        pd.Series
            索引为标的代码，值为 [-1, 1] 的截面排名
        """
        breadth = {}
        for sym, news_list in news_by_symbol.items():
            sources = {item.get("source", "") for item in news_list}
            sources.discard("")
            breadth[sym] = len(sources)

        series = pd.Series(breadth, dtype=float)
        if series.empty:
            return series
        return self._cross_sectional_rank(series)

    def build_composite_news_signal(
        self,
        news_by_symbol: Dict[str, list],
        w_sentiment: float = 0.5,
        w_volume: float = 0.3,
        w_breadth: float = 0.2,
    ) -> pd.Series:
        """复合新闻信号 — 情绪 + 关注度 + 广度加权组合

        Parameters
        ----------
        news_by_symbol : dict[str, list]
            键为标的代码，值为该标的新闻列表
        w_sentiment : float
            情绪信号权重，默认 0.5
        w_volume : float
            新闻数量信号权重，默认 0.3
        w_breadth : float
            来源广度信号权重，默认 0.2

        Returns
        -------
        pd.Series
            索引为标的代码，值为 [-1, 1] 的复合信号（截面排名后输出）
        """
        sig_sent = self.build_sentiment_signal(news_by_symbol)
        sig_vol = self.news_volume_signal(news_by_symbol)
        sig_brd = self.news_breadth_signal(news_by_symbol)

        # 对齐索引
        symbols = sig_sent.index.union(sig_vol.index).union(sig_brd.index)
        sig_sent = sig_sent.reindex(symbols, fill_value=0.0)
        sig_vol = sig_vol.reindex(symbols, fill_value=0.0)
        sig_brd = sig_brd.reindex(symbols, fill_value=0.0)

        composite = w_sentiment * sig_sent + w_volume * sig_vol + w_breadth * sig_brd

        if composite.empty:
            return composite
        return self._cross_sectional_rank(composite)
