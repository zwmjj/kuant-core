"""
Alpaca WebSocket 实时行情流模块

通过 alpaca-py 的 StockDataStream / CryptoDataStream 接收实时
trades、quotes、bars 数据，并将其缓存在内存 deque 中供实时因子计算使用。

使用示例
--------
>>> import asyncio
>>> from qf.data_stream import AlpacaStreamManager
>>>
>>> mgr = AlpacaStreamManager(buffer_size=500)
>>>
>>> # 注册回调（可选）
>>> mgr.on_bar = lambda data: print(f"[BAR] {data}")
>>> mgr.on_trade = lambda data: print(f"[TRADE] {data}")
>>> mgr.on_quote = lambda data: print(f"[QUOTE] {data}")
>>>
>>> # 订阅股票 bars + trades
>>> mgr.subscribe(["AAPL", "TSLA"], data_type="stock_bars")
>>> mgr.subscribe(["AAPL"], data_type="stock_trades")
>>>
>>> # 订阅加密货币
>>> mgr.subscribe(["BTC/USD", "ETH/USD"], data_type="crypto_bars")
>>>
>>> # 启动（阻塞），也可用 start_background() 在后台线程运行
>>> # mgr.run()            # 阻塞式
>>> mgr.start_background()  # 后台线程
>>>
>>> # 查询缓存
>>> latest = mgr.get_latest("AAPL", "bar")
>>> buf = mgr.get_buffer("AAPL", n=50, kind="bar")
>>>
>>> # 取消订阅 & 停止
>>> mgr.unsubscribe(["TSLA"], data_type="stock_bars")
>>> mgr.stop()
"""

import os
import asyncio
import threading
import logging
from collections import defaultdict, deque
from typing import List, Optional, Callable, Dict, Any

# ── 日志 ──────────────────────────────────────────────────────
logger = logging.getLogger("AlpacaStream")

# ── Alpaca 默认凭证（与 data_alpaca.py 保持一致） ─────────────
_DEFAULT_API_KEY = os.getenv("ALPACA_API_KEY", "")
_DEFAULT_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")

# ── 支持的数据类型 ────────────────────────────────────────────
STOCK_DATA_TYPES = {"stock_trades", "stock_quotes", "stock_bars"}
CRYPTO_DATA_TYPES = {"crypto_trades", "crypto_bars"}
ALL_DATA_TYPES = STOCK_DATA_TYPES | CRYPTO_DATA_TYPES

# 数据类型 → 缓存 key 的映射
_KIND_MAP = {
    "stock_trades": "trade",
    "stock_quotes": "quote",
    "stock_bars": "bar",
    "crypto_trades": "trade",
    "crypto_bars": "bar",
}


class AlpacaStreamManager:
    """
    Alpaca WebSocket 实时行情管理器

    功能:
      - 订阅 / 取消订阅 股票与加密货币的 trades / quotes / bars
      - 回调注册：on_bar, on_trade, on_quote
      - 数据缓存：每个 (symbol, kind) 保存最近 buffer_size 条记录
      - 断线自动重连（指数退避）

    参数:
      api_key      : Alpaca API Key，默认读环境变量或硬编码
      secret_key   : Alpaca Secret Key
      paper        : 是否使用 paper 环境（影响 base_url）
      buffer_size  : 每个 symbol/kind 缓存条数，默认 1000
      max_reconnect: 最大重连尝试次数，0 表示无限重试
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        paper: bool = True,
        buffer_size: int = 1000,
        max_reconnect: int = 0,
    ):
        self.api_key = api_key or _DEFAULT_API_KEY
        self.secret_key = secret_key or _DEFAULT_SECRET_KEY
        self.paper = paper
        self.buffer_size = buffer_size
        self.max_reconnect = max_reconnect

        # ── 订阅记录 ──────────────────────────────────────────
        # data_type → set of symbols
        self._subscriptions: Dict[str, set] = defaultdict(set)

        # ── 数据缓存 ──────────────────────────────────────────
        # (symbol, kind) → deque，kind ∈ {"trade", "quote", "bar"}
        self._buffers: Dict[tuple, deque] = defaultdict(
            lambda: deque(maxlen=self.buffer_size)
        )

        # ── 回调函数 ──────────────────────────────────────────
        self.on_bar: Optional[Callable[[Any], None]] = None
        self.on_trade: Optional[Callable[[Any], None]] = None
        self.on_quote: Optional[Callable[[Any], None]] = None

        # ── 内部状态 ──────────────────────────────────────────
        self._stock_stream = None
        self._crypto_stream = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

    # ══════════════════════════════════════════════════════════
    #  懒加载 WebSocket 流客户端
    # ══════════════════════════════════════════════════════════

    def _get_stock_stream(self):
        """懒加载股票实时流客户端"""
        if self._stock_stream is None:
            from alpaca.data.live import StockDataStream

            self._stock_stream = StockDataStream(
                self.api_key,
                self.secret_key,
            )
            logger.info("股票 WebSocket 流客户端已创建")
        return self._stock_stream

    def _get_crypto_stream(self):
        """懒加载加密货币实时流客户端"""
        if self._crypto_stream is None:
            from alpaca.data.live import CryptoDataStream

            self._crypto_stream = CryptoDataStream(
                self.api_key,
                self.secret_key,
            )
            logger.info("加密货币 WebSocket 流客户端已创建")
        return self._crypto_stream

    # ══════════════════════════════════════════════════════════
    #  内部回调处理器
    # ══════════════════════════════════════════════════════════

    async def _handle_stock_trade(self, data):
        """股票 trade 回调"""
        symbol = data.symbol
        record = {
            "symbol": symbol,
            "price": float(data.price),
            "size": int(data.size),
            "timestamp": data.timestamp,
            "kind": "trade",
            "raw": data,
        }
        self._buffers[(symbol, "trade")].append(record)
        if self.on_trade:
            try:
                self.on_trade(record)
            except Exception as e:
                logger.error(f"on_trade 回调异常: {e}")

    async def _handle_stock_quote(self, data):
        """股票 quote 回调"""
        symbol = data.symbol
        record = {
            "symbol": symbol,
            "bid_price": float(data.bid_price),
            "ask_price": float(data.ask_price),
            "bid_size": int(data.bid_size),
            "ask_size": int(data.ask_size),
            "timestamp": data.timestamp,
            "kind": "quote",
            "raw": data,
        }
        self._buffers[(symbol, "quote")].append(record)
        if self.on_quote:
            try:
                self.on_quote(record)
            except Exception as e:
                logger.error(f"on_quote 回调异常: {e}")

    async def _handle_stock_bar(self, data):
        """股票 bar 回调"""
        symbol = data.symbol
        record = {
            "symbol": symbol,
            "open": float(data.open),
            "high": float(data.high),
            "low": float(data.low),
            "close": float(data.close),
            "volume": int(data.volume),
            "timestamp": data.timestamp,
            "kind": "bar",
            "raw": data,
        }
        self._buffers[(symbol, "bar")].append(record)
        if self.on_bar:
            try:
                self.on_bar(record)
            except Exception as e:
                logger.error(f"on_bar 回调异常: {e}")

    async def _handle_crypto_trade(self, data):
        """加密货币 trade 回调"""
        symbol = data.symbol
        record = {
            "symbol": symbol,
            "price": float(data.price),
            "size": float(data.size),
            "timestamp": data.timestamp,
            "kind": "trade",
            "raw": data,
        }
        self._buffers[(symbol, "trade")].append(record)
        if self.on_trade:
            try:
                self.on_trade(record)
            except Exception as e:
                logger.error(f"on_trade 回调异常: {e}")

    async def _handle_crypto_bar(self, data):
        """加密货币 bar 回调"""
        symbol = data.symbol
        record = {
            "symbol": symbol,
            "open": float(data.open),
            "high": float(data.high),
            "low": float(data.low),
            "close": float(data.close),
            "volume": float(data.volume),
            "timestamp": data.timestamp,
            "kind": "bar",
            "raw": data,
        }
        self._buffers[(symbol, "bar")].append(record)
        if self.on_bar:
            try:
                self.on_bar(record)
            except Exception as e:
                logger.error(f"on_bar 回调异常: {e}")

    # ══════════════════════════════════════════════════════════
    #  订阅 / 取消订阅
    # ══════════════════════════════════════════════════════════

    def subscribe(self, symbols: List[str], data_type: str = "stock_bars"):
        """
        订阅实时数据流

        参数:
          symbols   : 标的列表，如 ["AAPL", "TSLA"] 或 ["BTC/USD"]
          data_type : 数据类型，可选值:
                      "stock_trades", "stock_quotes", "stock_bars",
                      "crypto_trades", "crypto_bars"

        示例:
          >>> mgr.subscribe(["AAPL", "TSLA"], data_type="stock_bars")
          >>> mgr.subscribe(["BTC/USD"], data_type="crypto_trades")
        """
        if data_type not in ALL_DATA_TYPES:
            raise ValueError(
                f"不支持的 data_type: {data_type}，可选: {ALL_DATA_TYPES}"
            )

        self._subscriptions[data_type].update(symbols)

        # 如果流已经在运行，动态注册到对应的 stream 客户端
        if data_type in STOCK_DATA_TYPES:
            stream = self._get_stock_stream()
            if data_type == "stock_trades":
                stream.subscribe_trades(self._handle_stock_trade, *symbols)
            elif data_type == "stock_quotes":
                stream.subscribe_quotes(self._handle_stock_quote, *symbols)
            elif data_type == "stock_bars":
                stream.subscribe_bars(self._handle_stock_bar, *symbols)
        else:
            stream = self._get_crypto_stream()
            if data_type == "crypto_trades":
                stream.subscribe_trades(self._handle_crypto_trade, *symbols)
            elif data_type == "crypto_bars":
                stream.subscribe_bars(self._handle_crypto_bar, *symbols)

        logger.info(f"已订阅 {data_type}: {symbols}")

    def unsubscribe(self, symbols: List[str], data_type: str = "stock_bars"):
        """
        取消订阅

        参数:
          symbols   : 标的列表
          data_type : 数据类型（同 subscribe）

        示例:
          >>> mgr.unsubscribe(["TSLA"], data_type="stock_bars")
        """
        if data_type not in ALL_DATA_TYPES:
            raise ValueError(
                f"不支持的 data_type: {data_type}，可选: {ALL_DATA_TYPES}"
            )

        self._subscriptions[data_type].difference_update(symbols)

        if data_type in STOCK_DATA_TYPES:
            stream = self._get_stock_stream()
            if data_type == "stock_trades":
                stream.unsubscribe_trades(*symbols)
            elif data_type == "stock_quotes":
                stream.unsubscribe_quotes(*symbols)
            elif data_type == "stock_bars":
                stream.unsubscribe_bars(*symbols)
        else:
            stream = self._get_crypto_stream()
            if data_type == "crypto_trades":
                stream.unsubscribe_trades(*symbols)
            elif data_type == "crypto_bars":
                stream.unsubscribe_bars(*symbols)

        logger.info(f"已取消订阅 {data_type}: {symbols}")

    # ══════════════════════════════════════════════════════════
    #  数据查询
    # ══════════════════════════════════════════════════════════

    def get_latest(
        self, symbol: str, kind: str = "bar"
    ) -> Optional[Dict[str, Any]]:
        """
        获取某标的最新一条缓存数据

        参数:
          symbol : 标的代码，如 "AAPL" 或 "BTC/USD"
          kind   : "trade" / "quote" / "bar"

        返回:
          dict 或 None（无数据时）

        示例:
          >>> mgr.get_latest("AAPL", "bar")
          {'symbol': 'AAPL', 'open': 175.2, 'high': 175.5, ...}
        """
        buf = self._buffers.get((symbol, kind))
        if buf and len(buf) > 0:
            return buf[-1]
        return None

    def get_buffer(
        self, symbol: str, n: Optional[int] = None, kind: str = "bar"
    ) -> List[Dict[str, Any]]:
        """
        获取某标的最近 n 条缓存数据

        参数:
          symbol : 标的代码
          n      : 条数，None 返回全部缓存
          kind   : "trade" / "quote" / "bar"

        返回:
          list[dict]，按时间升序（最旧在前）

        示例:
          >>> bars = mgr.get_buffer("AAPL", n=50, kind="bar")
          >>> len(bars)
          50
        """
        buf = self._buffers.get((symbol, kind))
        if buf is None:
            return []
        if n is None:
            return list(buf)
        return list(buf)[-n:]

    def get_subscriptions(self) -> Dict[str, List[str]]:
        """返回当前所有订阅 {data_type: [symbols]}"""
        return {k: sorted(v) for k, v in self._subscriptions.items() if v}

    def buffer_stats(self) -> Dict[str, int]:
        """返回每个 (symbol, kind) 的缓存条数统计"""
        return {
            f"{sym}:{kind}": len(buf)
            for (sym, kind), buf in self._buffers.items()
            if len(buf) > 0
        }

    # ══════════════════════════════════════════════════════════
    #  启动 / 停止 / 断线重连
    # ══════════════════════════════════════════════════════════

    async def _run_stream_with_reconnect(self, stream, name: str):
        """
        带断线重连的流运行协程

        使用指数退避策略: 1s → 2s → 4s → ... → 最大 60s
        """
        attempt = 0
        base_delay = 1.0
        max_delay = 60.0

        while self._running:
            try:
                attempt += 1
                logger.info(f"{name} 流启动中 (第 {attempt} 次)...")
                await stream._run_forever()
            except Exception as e:
                if not self._running:
                    # 主动停止，不重连
                    logger.info(f"{name} 流已主动关闭")
                    break

                # 计算退避延迟
                delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                logger.warning(
                    f"{name} 流断开: {e}，{delay:.1f}s 后第 {attempt + 1} 次重连..."
                )

                # 检查最大重连次数
                if self.max_reconnect > 0 and attempt >= self.max_reconnect:
                    logger.error(
                        f"{name} 流已达最大重连次数 {self.max_reconnect}，停止重连"
                    )
                    break

                await asyncio.sleep(delay)
            else:
                # 正常退出（不常见），重置计数
                attempt = 0

    async def _run_all(self):
        """并发运行所有已订阅的流"""
        tasks = []

        # 判断是否有股票订阅
        has_stock = any(
            self._subscriptions.get(dt) for dt in STOCK_DATA_TYPES
        )
        # 判断是否有加密货币订阅
        has_crypto = any(
            self._subscriptions.get(dt) for dt in CRYPTO_DATA_TYPES
        )

        if has_stock and self._stock_stream:
            tasks.append(
                asyncio.create_task(
                    self._run_stream_with_reconnect(
                        self._stock_stream, "Stock"
                    )
                )
            )

        if has_crypto and self._crypto_stream:
            tasks.append(
                asyncio.create_task(
                    self._run_stream_with_reconnect(
                        self._crypto_stream, "Crypto"
                    )
                )
            )

        if not tasks:
            logger.warning("没有任何订阅，流未启动。请先调用 subscribe()")
            return

        logger.info(f"共启动 {len(tasks)} 个数据流")
        await asyncio.gather(*tasks, return_exceptions=True)

    def run(self):
        """
        阻塞式运行所有已订阅的实时流

        适合脚本主入口直接调用。Ctrl+C 可安全退出。

        示例:
          >>> mgr.subscribe(["AAPL"], data_type="stock_bars")
          >>> mgr.run()  # 阻塞直到手动停止
        """
        self._running = True
        try:
            asyncio.run(self._run_all())
        except KeyboardInterrupt:
            logger.info("收到 KeyboardInterrupt，正在关闭...")
        finally:
            self._running = False

    def start_background(self):
        """
        在后台线程中启动实时流（非阻塞）

        返回后主线程可继续执行其他逻辑（如策略计算、UI 交互等）。

        示例:
          >>> mgr.subscribe(["AAPL"], data_type="stock_bars")
          >>> mgr.start_background()
          >>> # 主线程继续做其他事...
          >>> latest = mgr.get_latest("AAPL", "bar")
        """
        if self._running:
            logger.warning("流已在运行中，请勿重复启动")
            return

        self._running = True

        def _target():
            # 在新线程中创建独立的事件循环
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            try:
                self._loop.run_until_complete(self._run_all())
            except Exception as e:
                logger.error(f"后台流异常: {e}")
            finally:
                self._loop.close()
                self._running = False

        self._thread = threading.Thread(
            target=_target, name="AlpacaStream", daemon=True
        )
        self._thread.start()
        logger.info("实时数据流已在后台线程启动")

    def stop(self):
        """
        停止所有实时流

        示例:
          >>> mgr.stop()
        """
        logger.info("正在停止实时数据流...")
        self._running = False

        # 关闭底层 WebSocket 连接
        try:
            if self._stock_stream:
                self._stock_stream.stop()
        except Exception as e:
            logger.debug(f"关闭股票流: {e}")

        try:
            if self._crypto_stream:
                self._crypto_stream.stop()
        except Exception as e:
            logger.debug(f"关闭加密货币流: {e}")

        # 等待后台线程结束
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)

        logger.info("实时数据流已停止")

    def clear_buffers(self):
        """清空所有缓存数据"""
        self._buffers.clear()
        logger.info("所有缓存已清空")

    @property
    def is_running(self) -> bool:
        """流是否正在运行"""
        return self._running

    def __repr__(self):
        subs = self.get_subscriptions()
        stats = self.buffer_stats()
        return (
            f"<AlpacaStreamManager running={self._running} "
            f"subscriptions={subs} buffer_stats={stats}>"
        )
