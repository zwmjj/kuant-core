"""News sentiment factor - sentiment signal generator built on Alpaca News data"""

import re
import math
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


class NewsSentimentGenerator:
    """Keyword-based news sentiment factor generator.

    Alpaca news format: list[dict]
        Each item carries: headline, created_at, source, symbols, url
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
        """Score a single headline for sentiment, returning a value in [-1, 1].

        Algorithm:
            1. Tokenize and lowercase
            2. Count positive and negative words
            3. Intensifiers scale the weight of an adjacent sentiment word by 1.5x
            4. Negations flip the sign of the following sentiment word
            5. score = (pos_count - neg_count) / total_words, clipped to [-1, 1]
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
        """Aggregate sentiment scores for one instrument over a lookback window.

        Parameters
        ----------
        news_list : list[dict]
            News items in Alpaca format
        symbol : str
            Instrument ticker, e.g. 'AAPL'
        hours : int
            Lookback window in hours

        Returns
        -------
        float
            Weighted-average sentiment score; more recent items carry more
            weight via exponential decay
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
        """Build a cross-sectional sentiment signal.

        Parameters
        ----------
        news_by_symbol : dict[str, list]
            Keyed by ticker; each value is that instrument's Alpaca news list

        Returns
        -------
        pd.Series
            Indexed by ticker; cross-sectionally ranked sentiment in [-1, 1]
        """
        sentiments = {}
        for symbol, news_list in news_by_symbol.items():
            sentiments[symbol] = self.aggregate_sentiment(news_list, symbol)

        series = pd.Series(sentiments, dtype=float)
        if series.empty:
            return series
        return self._cross_sectional_rank(series)

    def news_volume_signal(self, news_by_symbol: Dict[str, list]) -> pd.Series:
        """News-count signal - an attention factor.

        More news items means more market attention; the count is
        cross-sectionally ranked before being returned.

        Parameters
        ----------
        news_by_symbol : dict[str, list]
            Keyed by ticker; each value is that instrument's news list

        Returns
        -------
        pd.Series
            Indexed by ticker; cross-sectional rank in [-1, 1]
        """
        counts = {sym: len(news) for sym, news in news_by_symbol.items()}
        series = pd.Series(counts, dtype=float)
        if series.empty:
            return series
        return self._cross_sectional_rank(series)

    def news_breadth_signal(self, news_by_symbol: Dict[str, list]) -> pd.Series:
        """News-breadth signal - a source-diversity factor.

        The more distinct sources cover an instrument, the stronger the signal.

        Parameters
        ----------
        news_by_symbol : dict[str, list]
            Keyed by ticker; each value is that instrument's news list

        Returns
        -------
        pd.Series
            Indexed by ticker; cross-sectional rank in [-1, 1]
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
        """Composite news signal - a weighted blend of sentiment, attention and breadth.

        Parameters
        ----------
        news_by_symbol : dict[str, list]
            Keyed by ticker; each value is that instrument's news list
        w_sentiment : float
            Weight on the sentiment signal, default 0.5
        w_volume : float
            Weight on the news-count signal, default 0.3
        w_breadth : float
            Weight on the source-breadth signal, default 0.2

        Returns
        -------
        pd.Series
            Indexed by ticker; composite signal in [-1, 1], cross-sectionally
            ranked before being returned
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
