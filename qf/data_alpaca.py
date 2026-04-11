"""Alpaca数据集成 — alpaca-py SDK + 缓存"""
import os, pickle, warnings, time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')


# Alpaca API凭证 (从环境变量读取)
_DEFAULT_API_KEY = os.getenv("ALPACA_API_KEY", "")
_DEFAULT_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")


class AlpacaDataLoader:
    """Alpaca数据加载器 — 股票/加密/期权/新闻/筛选"""

    def __init__(self, api_key=None, secret_key=None, paper=True):
        self.api_key = api_key or _DEFAULT_API_KEY
        self.secret_key = secret_key or _DEFAULT_SECRET_KEY
        self.paper = paper
        self.cache_dir = "data_cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        # 懒加载客户端
        self._stock_client = None
        self._crypto_client = None
        self._option_client = None
        self._news_client = None
        self._screener_client = None
        self._trading_client = None

    # ── 客户端懒加载 ──────────────────────────────────────────

    @property
    def stock_client(self):
        if self._stock_client is None:
            from alpaca.data.historical import StockHistoricalDataClient
            self._stock_client = StockHistoricalDataClient(
                self.api_key, self.secret_key
            )
            print("  Alpaca股票数据客户端已创建")
        return self._stock_client

    @property
    def crypto_client(self):
        if self._crypto_client is None:
            from alpaca.data.historical import CryptoHistoricalDataClient
            self._crypto_client = CryptoHistoricalDataClient(
                self.api_key, self.secret_key
            )
            print("  Alpaca加密货币客户端已创建")
        return self._crypto_client

    @property
    def option_client(self):
        if self._option_client is None:
            from alpaca.data.historical import OptionHistoricalDataClient
            self._option_client = OptionHistoricalDataClient(
                self.api_key, self.secret_key
            )
            print("  Alpaca期权客户端已创建")
        return self._option_client

    @property
    def news_client(self):
        if self._news_client is None:
            from alpaca.data.historical import NewsClient
            self._news_client = NewsClient(
                self.api_key, self.secret_key
            )
            print("  Alpaca新闻客户端已创建")
        return self._news_client

    @property
    def screener_client(self):
        if self._screener_client is None:
            from alpaca.data.historical import ScreenerClient
            self._screener_client = ScreenerClient(
                self.api_key, self.secret_key
            )
            print("  Alpaca筛选客户端已创建")
        return self._screener_client

    @property
    def trading_client(self):
        """交易客户端 — 用于获取可交易资产列表"""
        if self._trading_client is None:
            from alpaca.trading.client import TradingClient
            self._trading_client = TradingClient(
                self.api_key, self.secret_key, paper=self.paper
            )
            print("  Alpaca交易客户端已创建")
        return self._trading_client

    # ── 缓存机制 ─────────────────────────────────────────────

    def _cache_path(self, name):
        # Windows文件名不允许冒号等特殊字符
        safe = str(name).replace(':', '-').replace(' ', '_').replace('/', '_')
        # 文件名过长则截断+哈希
        if len(safe) > 150:
            import hashlib
            h = hashlib.md5(safe.encode()).hexdigest()[:8]
            safe = safe[:120] + '_' + h
        return os.path.join(self.cache_dir, f"alpaca_{safe}.pkl")

    def _load_cache(self, name, max_age_seconds=86400):
        """加载缓存（默认1天过期）"""
        path = self._cache_path(name)
        if os.path.exists(path):
            age = datetime.now().timestamp() - os.path.getmtime(path)
            if age < max_age_seconds:
                print(f"  使用缓存: alpaca_{name}")
                with open(path, 'rb') as f:
                    return pickle.load(f)
        return None

    def _save_cache(self, name, data):
        with open(self._cache_path(name), 'wb') as f:
            pickle.dump(data, f)

    # ── 辅助方法 ─────────────────────────────────────────────

    @staticmethod
    def _ensure_list(symbols):
        """统一转为列表"""
        if isinstance(symbols, str):
            return [symbols]
        return list(symbols)

    @staticmethod
    def _to_datetime(dt):
        """转为datetime对象"""
        if isinstance(dt, str):
            return pd.Timestamp(dt).to_pydatetime()
        if isinstance(dt, pd.Timestamp):
            return dt.to_pydatetime()
        return dt

    def _bars_to_dataframe(self, bars_response):
        """将Alpaca bars响应转为DataFrame"""
        try:
            df = bars_response.df
            if df.empty:
                return df
            # bars_response.df 自带MultiIndex (symbol, timestamp)
            if isinstance(df.index, pd.MultiIndex):
                df = df.reset_index()
                if 'symbol' in df.columns and 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df = df.set_index(['symbol', 'timestamp'])
            return df
        except Exception as e:
            print(f"  bars转DataFrame失败: {e}")
            return pd.DataFrame()

    # ── 1. 日K线 ────────────────────────────────────────────

    def get_daily_bars(self, symbols, start, end):
        """获取日K线 — OHLCV+vwap+trade_count

        Parameters:
            symbols: str或list — 股票代码
            start: str或datetime — 开始日期
            end: str或datetime — 结束日期

        Returns:
            DataFrame with MultiIndex (symbol, timestamp)
        """
        symbols = self._ensure_list(symbols)
        cache_key = f"daily_{'_'.join(sorted(symbols))}_{start}_{end}"
        cached = self._load_cache(cache_key)
        if cached is not None:
            return cached

        try:
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame

            request = StockBarsRequest(
                symbol_or_symbols=symbols,
                timeframe=TimeFrame.Day,
                start=self._to_datetime(start),
                end=self._to_datetime(end),
            )
            print(f"下载日K线: {symbols} ({start} → {end})...")
            bars = self.stock_client.get_stock_bars(request)
            df = self._bars_to_dataframe(bars)
            if not df.empty:
                self._save_cache(cache_key, df)
                print(f"  完成: {len(df)} 条记录")
            return df
        except Exception as e:
            print(f"  获取日K线失败: {e}")
            return pd.DataFrame()

    # ── 2. 分钟K线 ──────────────────────────────────────────

    def get_minute_bars(self, symbols, start, end):
        """获取分钟K线"""
        symbols = self._ensure_list(symbols)
        cache_key = f"min_{'_'.join(sorted(symbols))}_{start}_{end}"
        cached = self._load_cache(cache_key, max_age_seconds=3600)
        if cached is not None:
            return cached

        try:
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame

            request = StockBarsRequest(
                symbol_or_symbols=symbols,
                timeframe=TimeFrame.Minute,
                start=self._to_datetime(start),
                end=self._to_datetime(end),
            )
            print(f"下载分钟K线: {symbols} ({start} → {end})...")
            bars = self.stock_client.get_stock_bars(request)
            df = self._bars_to_dataframe(bars)
            if not df.empty:
                self._save_cache(cache_key, df)
                print(f"  完成: {len(df)} 条记录")
            return df
        except Exception as e:
            print(f"  获取分钟K线失败: {e}")
            return pd.DataFrame()

    # ── 3. 小时K线 ──────────────────────────────────────────

    def get_hourly_bars(self, symbols, start, end):
        """获取小时K线"""
        symbols = self._ensure_list(symbols)
        cache_key = f"hour_{'_'.join(sorted(symbols))}_{start}_{end}"
        cached = self._load_cache(cache_key, max_age_seconds=3600)
        if cached is not None:
            return cached

        try:
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame

            request = StockBarsRequest(
                symbol_or_symbols=symbols,
                timeframe=TimeFrame.Hour,
                start=self._to_datetime(start),
                end=self._to_datetime(end),
            )
            print(f"下载小时K线: {symbols} ({start} → {end})...")
            bars = self.stock_client.get_stock_bars(request)
            df = self._bars_to_dataframe(bars)
            if not df.empty:
                self._save_cache(cache_key, df)
                print(f"  完成: {len(df)} 条记录")
            return df
        except Exception as e:
            print(f"  获取小时K线失败: {e}")
            return pd.DataFrame()

    # ── 4. 实时快照 ─────────────────────────────────────────

    def get_snapshots(self, symbols):
        """获取实时快照 — bid/ask/last/daily_bar

        Returns:
            dict: {symbol: {bid, ask, last, daily_bar: {...}}}
        """
        symbols = self._ensure_list(symbols)
        try:
            from alpaca.data.requests import StockSnapshotRequest

            request = StockSnapshotRequest(symbol_or_symbols=symbols)
            print(f"获取快照: {symbols}...")
            snapshots = self.stock_client.get_stock_snapshot(request)
            result = {}
            for sym, snap in snapshots.items():
                entry = {}
                # 最新报价
                if snap.latest_quote:
                    entry['bid'] = snap.latest_quote.bid_price
                    entry['ask'] = snap.latest_quote.ask_price
                    entry['bid_size'] = snap.latest_quote.bid_size
                    entry['ask_size'] = snap.latest_quote.ask_size
                # 最新成交
                if snap.latest_trade:
                    entry['last'] = snap.latest_trade.price
                    entry['last_size'] = snap.latest_trade.size
                    entry['last_time'] = str(snap.latest_trade.timestamp)
                # 日K线
                if snap.daily_bar:
                    entry['daily_bar'] = {
                        'open': snap.daily_bar.open,
                        'high': snap.daily_bar.high,
                        'low': snap.daily_bar.low,
                        'close': snap.daily_bar.close,
                        'volume': snap.daily_bar.volume,
                        'vwap': snap.daily_bar.vwap,
                        'trade_count': snap.daily_bar.trade_count,
                    }
                result[sym] = entry
            print(f"  完成: {len(result)} 只股票")
            return result
        except Exception as e:
            print(f"  获取快照失败: {e}")
            return {}

    # ── 5. 逐笔成交 ─────────────────────────────────────────

    def get_trades(self, symbol, start, end, limit=1000):
        """获取逐笔成交数据

        Returns:
            DataFrame with columns: price, size, exchange, conditions, timestamp
        """
        try:
            from alpaca.data.requests import StockTradesRequest

            request = StockTradesRequest(
                symbol_or_symbols=symbol,
                start=self._to_datetime(start),
                end=self._to_datetime(end),
                limit=limit,
            )
            print(f"下载逐笔成交: {symbol} (limit={limit})...")
            trades = self.stock_client.get_stock_trades(request)
            df = trades.df
            if not df.empty:
                print(f"  完成: {len(df)} 笔")
            return df
        except Exception as e:
            print(f"  获取逐笔成交失败: {e}")
            return pd.DataFrame()

    # ── 6. 新闻 ─────────────────────────────────────────────

    def get_news(self, symbols=None, limit=50):
        """获取新闻

        Returns:
            list of dicts: headline, created_at, source, symbols, url, summary
        """
        try:
            from alpaca.data.requests import NewsRequest

            params = {'limit': limit}
            if symbols is not None:
                if isinstance(symbols, str):
                    params['symbols'] = symbols
                else:
                    params['symbols'] = ','.join(symbols)

            request = NewsRequest(**params)
            print(f"获取新闻 (symbols={symbols}, limit={limit})...")
            news_response = self.news_client.get_news(request)

            # 解析新闻列表
            articles = []
            news_list = news_response.data.get('news', []) if hasattr(news_response, 'data') else []
            # 兼容不同版本的返回格式
            if not news_list and hasattr(news_response, 'news'):
                news_list = news_response.news

            for article in news_list:
                entry = {
                    'headline': getattr(article, 'headline', ''),
                    'created_at': str(getattr(article, 'created_at', '')),
                    'source': getattr(article, 'source', ''),
                    'symbols': getattr(article, 'symbols', []),
                    'url': getattr(article, 'url', ''),
                    'summary': getattr(article, 'summary', ''),
                }
                articles.append(entry)

            print(f"  完成: {len(articles)} 条新闻")
            return articles
        except Exception as e:
            print(f"  获取新闻失败: {e}")
            return []

    # ── 7. 期权链 ────────────────────────────────────────────

    def get_option_chain(self, underlying):
        """获取期权链

        Returns:
            DataFrame: strike, expiry, type (call/put), bid, ask, mid, iv_proxy
        """
        cache_key = f"optchain_{underlying}"
        cached = self._load_cache(cache_key, max_age_seconds=3600)
        if cached is not None:
            return cached

        try:
            from alpaca.data.requests import OptionChainRequest

            request = OptionChainRequest(underlying_symbol=underlying)
            print(f"获取期权链: {underlying}...")
            chain = self.option_client.get_option_chain(request)

            rows = []
            for contract_symbol, snapshot in chain.items():
                row = {'contract': contract_symbol}

                # 解析合约信息 — 标准OCC格式: AAPL240315C00170000
                sym_str = str(contract_symbol)
                try:
                    # 找到标的结束位置
                    idx = len(underlying)
                    date_str = sym_str[idx:idx + 6]
                    opt_type = sym_str[idx + 6]
                    strike_str = sym_str[idx + 7:]
                    row['expiry'] = pd.Timestamp(
                        '20' + date_str[:2] + '-' + date_str[2:4] + '-' + date_str[4:6]
                    )
                    row['type'] = 'call' if opt_type == 'C' else 'put'
                    row['strike'] = float(strike_str) / 1000.0
                except Exception:
                    row['expiry'] = None
                    row['type'] = None
                    row['strike'] = None

                # 最新报价
                if snapshot.latest_quote:
                    row['bid'] = snapshot.latest_quote.bid_price
                    row['ask'] = snapshot.latest_quote.ask_price
                    row['mid'] = (row['bid'] + row['ask']) / 2 if row['bid'] and row['ask'] else None
                else:
                    row['bid'] = None
                    row['ask'] = None
                    row['mid'] = None

                # IV代理 — 用bid-ask spread占比估算
                if row['mid'] and row['mid'] > 0 and row['bid'] is not None and row['ask'] is not None:
                    spread = row['ask'] - row['bid']
                    row['iv_proxy'] = spread / row['mid']
                else:
                    row['iv_proxy'] = None

                rows.append(row)

            df = pd.DataFrame(rows)
            if not df.empty:
                df = df.sort_values(['expiry', 'type', 'strike']).reset_index(drop=True)
                self._save_cache(cache_key, df)
                print(f"  完成: {len(df)} 个合约")
            return df
        except Exception as e:
            print(f"  获取期权链失败: {e}")
            return pd.DataFrame()

    # ── 8. 加密货币K线 ──────────────────────────────────────

    def get_crypto_bars(self, symbols, start, end, timeframe='day'):
        """获取加密货币K线

        Parameters:
            symbols: 如 'BTC/USD' 或 ['BTC/USD', 'ETH/USD']
            timeframe: 'day', 'hour', 'minute'
        """
        symbols = self._ensure_list(symbols)
        cache_key = f"crypto_{'_'.join(sorted(s.replace('/', '') for s in symbols))}_{timeframe}_{start}_{end}"
        cached = self._load_cache(cache_key)
        if cached is not None:
            return cached

        try:
            from alpaca.data.requests import CryptoBarsRequest
            from alpaca.data.timeframe import TimeFrame

            tf_map = {
                'day': TimeFrame.Day,
                'hour': TimeFrame.Hour,
                'minute': TimeFrame.Minute,
            }
            tf = tf_map.get(timeframe, TimeFrame.Day)

            request = CryptoBarsRequest(
                symbol_or_symbols=symbols,
                timeframe=tf,
                start=self._to_datetime(start),
                end=self._to_datetime(end),
            )
            print(f"下载加密货币K线: {symbols} ({timeframe}, {start} → {end})...")
            bars = self.crypto_client.get_crypto_bars(request)
            df = self._bars_to_dataframe(bars)
            if not df.empty:
                self._save_cache(cache_key, df)
                print(f"  完成: {len(df)} 条记录")
            return df
        except Exception as e:
            print(f"  获取加密货币K线失败: {e}")
            return pd.DataFrame()

    # ── 8b. 加密货币OHLCV（按币种拆分） ──────────────────────

    def get_crypto_ohlcv(self, symbols=None, start="2024-01-01", end="2026-03-31",
                         timeframe="1Day"):
        """获取加密货币日线OHLCV数据，按币种拆分返回

        与 get_crypto_bars 的区别：本方法返回 dict[str, DataFrame]，
        每个 symbol 一个独立的 DataFrame，列为标准 OHLCV，便于策略直接使用。

        Parameters:
            symbols : str, list 或 None
                加密货币交易对，如 'BTC/USD' 或 ['BTC/USD', 'ETH/USD', 'SOL/USD']。
                默认为 ['BTC/USD', 'ETH/USD', 'SOL/USD']。
            start : str 或 datetime — 开始日期
            end   : str 或 datetime — 结束日期
            timeframe : str — K线周期，支持 '1Day', '1Hour', '1Min'

        Returns:
            dict[str, pd.DataFrame]
                键为 symbol（如 'BTC/USD'），值为 DataFrame，
                列: open, high, low, close, volume, vwap, trade_count
                索引: DatetimeIndex (timestamp)
        """
        # 默认三大主流币
        if symbols is None:
            symbols = ["BTC/USD", "ETH/USD", "SOL/USD"]
        symbols = self._ensure_list(symbols)

        # 构建缓存键
        safe_syms = "_".join(sorted(s.replace("/", "") for s in symbols))
        cache_key = f"crypto_ohlcv_{safe_syms}_{timeframe}_{start}_{end}"
        cached = self._load_cache(cache_key)
        if cached is not None:
            return cached

        try:
            from alpaca.data.requests import CryptoBarsRequest
            from alpaca.data.timeframe import TimeFrame

            # 解析 timeframe 字符串
            tf_map = {
                "1Day": TimeFrame.Day,
                "1day": TimeFrame.Day,
                "day":  TimeFrame.Day,
                "1Hour": TimeFrame.Hour,
                "1hour": TimeFrame.Hour,
                "hour":  TimeFrame.Hour,
                "1Min":  TimeFrame.Minute,
                "1min":  TimeFrame.Minute,
                "minute": TimeFrame.Minute,
            }
            tf = tf_map.get(timeframe)
            if tf is None:
                print(f"  未知 timeframe '{timeframe}'，使用默认 Day")
                tf = TimeFrame.Day

            request = CryptoBarsRequest(
                symbol_or_symbols=symbols,
                timeframe=tf,
                start=self._to_datetime(start),
                end=self._to_datetime(end),
            )
            print(f"下载加密货币OHLCV: {symbols} ({timeframe}, {start} → {end})...")
            bars = self.crypto_client.get_crypto_bars(request)
            df = self._bars_to_dataframe(bars)

            if df.empty:
                print("  未获取到数据")
                return {s: pd.DataFrame() for s in symbols}

            # 拆分为每个 symbol 一个 DataFrame
            if isinstance(df.index, pd.MultiIndex):
                df = df.reset_index()

            result = {}
            for sym in symbols:
                sym_df = df[df["symbol"] == sym].copy()
                if sym_df.empty:
                    print(f"  {sym}: 无数据")
                    result[sym] = pd.DataFrame()
                    continue

                # 设置时间索引，仅保留 OHLCV 标准列
                sym_df["timestamp"] = pd.to_datetime(sym_df["timestamp"])
                sym_df = sym_df.set_index("timestamp").sort_index()

                # 保留标准列（存在则保留）
                keep_cols = ["open", "high", "low", "close", "volume",
                             "vwap", "trade_count"]
                sym_df = sym_df[[c for c in keep_cols if c in sym_df.columns]]

                result[sym] = sym_df
                print(f"  {sym}: {len(sym_df)} 条记录")

            self._save_cache(cache_key, result)
            print(f"  加密货币OHLCV下载完成，共 {len(result)} 个币种")
            return result

        except Exception as e:
            print(f"  获取加密货币OHLCV失败: {e}")
            # 返回空 DataFrame 字典，避免下游报错
            return {s: pd.DataFrame() for s in symbols}

    # ── 9. 最活跃股票 ───────────────────────────────────────

    def get_most_actives(self, top=20):
        """获取最活跃股票

        Returns:
            DataFrame: symbol, volume, trade_count
        """
        try:
            from alpaca.data.requests import MostActivesRequest

            request = MostActivesRequest(top=top, by='volume')
            print(f"获取最活跃股票 (top={top})...")
            response = self.screener_client.get_most_actives(request)

            rows = []
            actives = response.most_actives if hasattr(response, 'most_actives') else []
            for item in actives:
                rows.append({
                    'symbol': getattr(item, 'symbol', ''),
                    'volume': getattr(item, 'volume', 0),
                    'trade_count': getattr(item, 'trade_count', 0),
                })

            df = pd.DataFrame(rows)
            print(f"  完成: {len(df)} 只股票")
            return df
        except Exception as e:
            print(f"  获取最活跃股票失败: {e}")
            return pd.DataFrame()

    # ── 10. 涨跌幅排行 ──────────────────────────────────────

    def get_market_movers(self, top=10):
        """获取涨跌幅排行

        Returns:
            dict: {'gainers': DataFrame, 'losers': DataFrame}
        """
        try:
            from alpaca.data.requests import MarketMoversRequest

            request = MarketMoversRequest(top=top)
            print(f"获取涨跌幅排行 (top={top})...")
            response = self.screener_client.get_market_movers(request)

            def _parse_movers(mover_list):
                rows = []
                for item in mover_list:
                    rows.append({
                        'symbol': getattr(item, 'symbol', ''),
                        'price': getattr(item, 'price', 0),
                        'change': getattr(item, 'change', 0),
                        'percent_change': getattr(item, 'percent_change', 0),
                        'volume': getattr(item, 'volume', 0),
                    })
                return pd.DataFrame(rows)

            gainers = _parse_movers(
                response.gainers if hasattr(response, 'gainers') else []
            )
            losers = _parse_movers(
                response.losers if hasattr(response, 'losers') else []
            )
            print(f"  完成: {len(gainers)} 涨 / {len(losers)} 跌")
            return {'gainers': gainers, 'losers': losers}
        except Exception as e:
            print(f"  获取涨跌幅排行失败: {e}")
            return {'gainers': pd.DataFrame(), 'losers': pd.DataFrame()}

    # ── 11. 可交易宇宙 ──────────────────────────────────────

    def get_universe(self, min_price=5, min_volume=100000):
        """获取可交易股票宇宙 — 按最低价格和成交量过滤

        Returns:
            list of symbol strings
        """
        cache_key = f"universe_{min_price}_{min_volume}"
        cached = self._load_cache(cache_key, max_age_seconds=86400)
        if cached is not None:
            return cached

        try:
            from alpaca.trading.requests import GetAssetsRequest
            from alpaca.trading.enums import AssetClass, AssetStatus

            print(f"获取可交易宇宙 (价格>={min_price}, 成交量>={min_volume})...")
            request = GetAssetsRequest(
                asset_class=AssetClass.US_EQUITY,
                status=AssetStatus.ACTIVE,
            )
            assets = self.trading_client.get_all_assets(request)

            # 筛选可交易、可做空的普通股
            tradable = [
                a.symbol for a in assets
                if a.tradable and a.shortable and len(a.symbol) <= 5
                and not any(c in a.symbol for c in ['.', '-', '/'])
            ]
            print(f"  可交易股票: {len(tradable)} 只")

            # 用快照进一步按价格和成交量过滤
            # 分批获取快照（API限制）
            filtered = []
            batch_size = 100
            for i in range(0, len(tradable), batch_size):
                batch = tradable[i:i + batch_size]
                try:
                    snaps = self.get_snapshots(batch)
                    for sym, data in snaps.items():
                        price = data.get('last', 0) or 0
                        vol = data.get('daily_bar', {}).get('volume', 0) or 0
                        if price >= min_price and vol >= min_volume:
                            filtered.append(sym)
                except Exception:
                    continue
                # 避免API限流
                time.sleep(0.2)

            filtered = sorted(filtered)
            self._save_cache(cache_key, filtered)
            print(f"  过滤后: {len(filtered)} 只股票")
            return filtered
        except Exception as e:
            print(f"  获取宇宙失败: {e}")
            return []

    # ── 12. prepare_alpaca_data — 类似WRDS的prepare_data ────

    def prepare_alpaca_data(self, symbols=None, lookback_days=252):
        """准备Alpaca数据集 — 类似WRDS prepare_data

        Parameters:
            symbols: list或None — 指定股票列表，None则使用最活跃股票
            lookback_days: int — 回溯天数

        Returns:
            dict: prices, returns, volume, vwap, trade_count, high, low
        """
        end = datetime.now()
        start = end - timedelta(days=int(lookback_days * 1.5))  # 多取一些以确保交易日数量
        start_str = start.strftime('%Y-%m-%d')
        end_str = end.strftime('%Y-%m-%d')

        cache_key = f"prepared_{'_'.join(sorted(symbols)) if symbols else 'auto'}_{lookback_days}"
        cached = self._load_cache(cache_key)
        if cached is not None:
            return cached

        # 如果没有指定股票，获取最活跃的
        if symbols is None:
            print("未指定股票，获取最活跃股票作为宇宙...")
            try:
                actives_df = self.get_most_actives(top=100)
                if not actives_df.empty:
                    symbols = actives_df['symbol'].tolist()
                else:
                    # 默认列表
                    symbols = [
                        'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'TSLA',
                        'JPM', 'V', 'JNJ', 'WMT', 'PG', 'MA', 'UNH', 'HD',
                        'DIS', 'BAC', 'ADBE', 'CRM', 'NFLX', 'XOM', 'CSCO',
                        'PFE', 'ABT', 'KO', 'PEP', 'TMO', 'COST', 'AVGO', 'MRK',
                    ]
            except Exception:
                symbols = [
                    'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'TSLA',
                    'JPM', 'V', 'JNJ', 'WMT', 'PG', 'MA', 'UNH', 'HD',
                ]

        print(f"准备Alpaca数据: {len(symbols)} 只股票, {lookback_days} 交易日...")

        # 获取日K线
        bars = self.get_daily_bars(symbols, start_str, end_str)
        if bars.empty:
            print("  无数据可用")
            return {}

        # 构建宽表
        if isinstance(bars.index, pd.MultiIndex):
            bars = bars.reset_index()

        # 标准化列名
        col_map = {}
        for col in bars.columns:
            col_lower = col.lower()
            if col_lower in ['close', 'open', 'high', 'low', 'volume',
                             'vwap', 'trade_count', 'symbol', 'timestamp']:
                col_map[col] = col_lower
        bars = bars.rename(columns=col_map)

        # 确保时间列存在
        if 'timestamp' in bars.columns:
            bars['timestamp'] = pd.to_datetime(bars['timestamp'])
        elif 'date' in bars.columns:
            bars['timestamp'] = pd.to_datetime(bars['date'])

        # 构建各指标宽表
        prices = bars.pivot_table(index='timestamp', columns='symbol', values='close')
        volume = bars.pivot_table(index='timestamp', columns='symbol', values='volume')
        vwap = bars.pivot_table(index='timestamp', columns='symbol', values='vwap')
        high = bars.pivot_table(index='timestamp', columns='symbol', values='high')
        low = bars.pivot_table(index='timestamp', columns='symbol', values='low')

        trade_count = None
        if 'trade_count' in bars.columns:
            trade_count = bars.pivot_table(
                index='timestamp', columns='symbol', values='trade_count'
            )

        # 截取最近lookback_days个交易日
        if len(prices) > lookback_days:
            prices = prices.iloc[-lookback_days:]
            volume = volume.iloc[-lookback_days:]
            vwap = vwap.iloc[-lookback_days:]
            high = high.iloc[-lookback_days:]
            low = low.iloc[-lookback_days:]
            if trade_count is not None:
                trade_count = trade_count.iloc[-lookback_days:]

        # 计算收益率
        returns = prices.pct_change()

        result = {
            'prices': prices,
            'returns': returns,
            'volume': volume,
            'vwap': vwap,
            'trade_count': trade_count,
            'high': high,
            'low': low,
            'symbols': list(prices.columns),
        }
        self._save_cache(cache_key, result)
        print(f"  完成: {prices.shape[1]} 只股票, {prices.shape[0]} 交易日")
        return result


    # ══════════════════════════════════════════════════════════════
    #  新增数据端点 (2026-03-28)
    # ══════════════════════════════════════════════════════════════

    # ── 13. 逐笔报价历史 ────────────────────────────────────────

    def get_stock_quotes(self, symbols, start, end, limit=10000):
        """获取历史逐笔报价 (bid/ask)

        Returns:
            DataFrame: bid_price, bid_size, bid_exchange, ask_price, ask_size, ask_exchange, conditions, tape
        """
        symbols = self._ensure_list(symbols)
        try:
            from alpaca.data.requests import StockQuotesRequest
            request = StockQuotesRequest(
                symbol_or_symbols=symbols,
                start=self._to_datetime(start),
                end=self._to_datetime(end),
                limit=limit,
            )
            print(f"下载逐笔报价: {symbols} ({start} → {end})...")
            quotes = self.stock_client.get_stock_quotes(request)
            df = quotes.df
            print(f"  完成: {len(df)} 条报价")
            return df
        except Exception as e:
            print(f"  获取逐笔报价失败: {e}")
            return pd.DataFrame()

    # ── 14. 最新单条数据 ────────────────────────────────────────

    def get_latest_bars(self, symbols):
        """获取最新单条K线"""
        symbols = self._ensure_list(symbols)
        try:
            from alpaca.data.requests import StockLatestBarRequest
            request = StockLatestBarRequest(symbol_or_symbols=symbols)
            result = self.stock_client.get_stock_latest_bar(request)
            out = {}
            for sym, bar in result.items():
                out[sym] = {
                    'open': bar.open, 'high': bar.high, 'low': bar.low,
                    'close': bar.close, 'volume': bar.volume,
                    'vwap': bar.vwap, 'trade_count': bar.trade_count,
                    'timestamp': str(bar.timestamp),
                }
            return out
        except Exception as e:
            print(f"  获取最新K线失败: {e}")
            return {}

    def get_latest_quotes(self, symbols):
        """获取最新报价 (bid/ask)"""
        symbols = self._ensure_list(symbols)
        try:
            from alpaca.data.requests import StockLatestQuoteRequest
            result = self.stock_client.get_stock_latest_quote(
                StockLatestQuoteRequest(symbol_or_symbols=symbols)
            )
            out = {}
            for sym, q in result.items():
                out[sym] = {
                    'bid_price': q.bid_price, 'bid_size': q.bid_size,
                    'ask_price': q.ask_price, 'ask_size': q.ask_size,
                    'bid_exchange': getattr(q, 'bid_exchange', ''),
                    'ask_exchange': getattr(q, 'ask_exchange', ''),
                    'timestamp': str(q.timestamp),
                }
            return out
        except Exception as e:
            print(f"  获取最新报价失败: {e}")
            return {}

    def get_latest_trades(self, symbols):
        """获取最新成交"""
        symbols = self._ensure_list(symbols)
        try:
            from alpaca.data.requests import StockLatestTradeRequest
            result = self.stock_client.get_stock_latest_trade(
                StockLatestTradeRequest(symbol_or_symbols=symbols)
            )
            out = {}
            for sym, t in result.items():
                out[sym] = {
                    'price': t.price, 'size': t.size,
                    'exchange': getattr(t, 'exchange', ''),
                    'timestamp': str(t.timestamp),
                }
            return out
        except Exception as e:
            print(f"  获取最新成交失败: {e}")
            return {}

    # ── 15. 加密货币订单簿 (L2) ────────────────────────────────

    def get_crypto_orderbook(self, symbols):
        """获取加密货币L2订单簿

        Returns:
            dict: {symbol: {'bids': [(price, size), ...], 'asks': [(price, size), ...]}}
        """
        symbols = self._ensure_list(symbols)
        try:
            from alpaca.data.requests import CryptoLatestOrderbookRequest
            result = self.crypto_client.get_crypto_latest_orderbook(
                CryptoLatestOrderbookRequest(symbol_or_symbols=symbols)
            )
            out = {}
            for sym, book in result.items():
                out[sym] = {
                    'bids': [(b.price, b.size) for b in book.bids],
                    'asks': [(a.price, a.size) for a in book.asks],
                }
            print(f"  订单簿: {list(out.keys())}")
            return out
        except Exception as e:
            print(f"  获取订单簿失败: {e}")
            return {}

    # ── 16. 加密货币报价历史 ───────────────────────────────────

    def get_crypto_quotes(self, symbols, start, end, limit=10000):
        """获取加密货币逐笔报价"""
        symbols = self._ensure_list(symbols)
        try:
            from alpaca.data.requests import CryptoQuoteRequest
            request = CryptoQuoteRequest(
                symbol_or_symbols=symbols,
                start=self._to_datetime(start),
                end=self._to_datetime(end),
                limit=limit,
            )
            quotes = self.crypto_client.get_crypto_quotes(request)
            df = quotes.df
            print(f"  加密报价: {len(df)} 条")
            return df
        except Exception as e:
            print(f"  获取加密报价失败: {e}")
            return pd.DataFrame()

    # ── 17. 加密货币最新数据 ───────────────────────────────────

    def get_crypto_latest(self, symbols):
        """获取加密货币最新bar/quote/trade"""
        symbols = self._ensure_list(symbols)
        out = {}
        try:
            from alpaca.data.requests import (CryptoLatestBarRequest,
                CryptoLatestQuoteRequest, CryptoLatestTradeRequest)

            bars = self.crypto_client.get_crypto_latest_bar(
                CryptoLatestBarRequest(symbol_or_symbols=symbols))
            quotes = self.crypto_client.get_crypto_latest_quote(
                CryptoLatestQuoteRequest(symbol_or_symbols=symbols))
            trades = self.crypto_client.get_crypto_latest_trade(
                CryptoLatestTradeRequest(symbol_or_symbols=symbols))

            for sym in symbols:
                entry = {}
                if sym in bars:
                    b = bars[sym]
                    entry['bar'] = {'open': b.open, 'high': b.high, 'low': b.low,
                                    'close': b.close, 'volume': b.volume, 'vwap': b.vwap}
                if sym in quotes:
                    q = quotes[sym]
                    entry['quote'] = {'bid': q.bid_price, 'ask': q.ask_price,
                                      'bid_size': q.bid_size, 'ask_size': q.ask_size}
                if sym in trades:
                    t = trades[sym]
                    entry['trade'] = {'price': t.price, 'size': t.size,
                                      'timestamp': str(t.timestamp)}
                out[sym] = entry
            return out
        except Exception as e:
            print(f"  获取加密最新数据失败: {e}")
            return {}

    # ── 18. 加密货币逐笔成交 ──────────────────────────────────

    def get_crypto_trades(self, symbols, start, end, limit=10000):
        """获取加密货币逐笔成交"""
        symbols = self._ensure_list(symbols)
        try:
            from alpaca.data.requests import CryptoTradesRequest
            request = CryptoTradesRequest(
                symbol_or_symbols=symbols,
                start=self._to_datetime(start),
                end=self._to_datetime(end),
                limit=limit,
            )
            trades = self.crypto_client.get_crypto_trades(request)
            df = trades.df
            print(f"  加密成交: {len(df)} 笔")
            return df
        except Exception as e:
            print(f"  获取加密成交失败: {e}")
            return pd.DataFrame()

    # ── 19. 加密货币快照 ──────────────────────────────────────

    def get_crypto_snapshots(self, symbols):
        """获取加密货币实时快照"""
        symbols = self._ensure_list(symbols)
        try:
            from alpaca.data.requests import CryptoSnapshotRequest
            result = self.crypto_client.get_crypto_snapshot(
                CryptoSnapshotRequest(symbol_or_symbols=symbols))
            out = {}
            for sym, snap in result.items():
                entry = {}
                if snap.latest_quote:
                    entry['bid'] = snap.latest_quote.bid_price
                    entry['ask'] = snap.latest_quote.ask_price
                if snap.latest_trade:
                    entry['last'] = snap.latest_trade.price
                if snap.daily_bar:
                    entry['daily_bar'] = {
                        'open': snap.daily_bar.open, 'high': snap.daily_bar.high,
                        'low': snap.daily_bar.low, 'close': snap.daily_bar.close,
                        'volume': snap.daily_bar.volume, 'vwap': snap.daily_bar.vwap,
                    }
                out[sym] = entry
            return out
        except Exception as e:
            print(f"  获取加密快照失败: {e}")
            return {}

    # ── 20. 期权K线 ──────────────────────────────────────────

    def get_option_bars(self, symbols, start, end, timeframe='day'):
        """获取期权合约K线

        Parameters:
            symbols: 期权合约代码, 如 'AAPL260424C00250000'
        """
        symbols = self._ensure_list(symbols)
        try:
            from alpaca.data.requests import OptionBarsRequest
            from alpaca.data.timeframe import TimeFrame
            tf_map = {'day': TimeFrame.Day, 'hour': TimeFrame.Hour, 'minute': TimeFrame.Minute}
            request = OptionBarsRequest(
                symbol_or_symbols=symbols,
                timeframe=tf_map.get(timeframe, TimeFrame.Day),
                start=self._to_datetime(start),
                end=self._to_datetime(end),
            )
            bars = self.option_client.get_option_bars(request)
            df = self._bars_to_dataframe(bars)
            print(f"  期权K线: {len(df)} 条")
            return df
        except Exception as e:
            print(f"  获取期权K线失败: {e}")
            return pd.DataFrame()

    # ── 21. 期权逐笔成交 ────────────────────────────────────

    def get_option_trades(self, symbols, start, end, limit=10000):
        """获取期权逐笔成交"""
        symbols = self._ensure_list(symbols)
        try:
            from alpaca.data.requests import OptionTradesRequest
            request = OptionTradesRequest(
                symbol_or_symbols=symbols,
                start=self._to_datetime(start),
                end=self._to_datetime(end),
                limit=limit,
            )
            trades = self.option_client.get_option_trades(request)
            df = trades.df
            print(f"  期权成交: {len(df)} 笔")
            return df
        except Exception as e:
            print(f"  获取期权成交失败: {e}")
            return pd.DataFrame()

    # ── 22. 期权最新报价/成交/快照 ──────────────────────────

    def get_option_latest_quote(self, symbols):
        """获取期权最新报价"""
        symbols = self._ensure_list(symbols)
        try:
            from alpaca.data.requests import OptionLatestQuoteRequest
            result = self.option_client.get_option_latest_quote(
                OptionLatestQuoteRequest(symbol_or_symbols=symbols))
            out = {}
            for sym, q in result.items():
                out[sym] = {'bid': q.bid_price, 'ask': q.ask_price,
                            'bid_size': q.bid_size, 'ask_size': q.ask_size}
            return out
        except Exception as e:
            print(f"  获取期权最新报价失败: {e}")
            return {}

    def get_option_latest_trade(self, symbols):
        """获取期权最新成交"""
        symbols = self._ensure_list(symbols)
        try:
            from alpaca.data.requests import OptionLatestTradeRequest
            result = self.option_client.get_option_latest_trade(
                OptionLatestTradeRequest(symbol_or_symbols=symbols))
            out = {}
            for sym, t in result.items():
                out[sym] = {'price': t.price, 'size': t.size,
                            'exchange': getattr(t, 'exchange', ''),
                            'timestamp': str(t.timestamp)}
            return out
        except Exception as e:
            print(f"  获取期权最新成交失败: {e}")
            return {}

    def get_option_snapshot(self, symbols):
        """获取期权合约快照"""
        symbols = self._ensure_list(symbols)
        try:
            from alpaca.data.requests import OptionSnapshotRequest
            result = self.option_client.get_option_snapshot(
                OptionSnapshotRequest(symbol_or_symbols=symbols))
            out = {}
            for sym, snap in result.items():
                entry = {}
                if snap.latest_quote:
                    entry['bid'] = snap.latest_quote.bid_price
                    entry['ask'] = snap.latest_quote.ask_price
                if snap.latest_trade:
                    entry['last'] = snap.latest_trade.price
                if hasattr(snap, 'greeks') and snap.greeks:
                    entry['greeks'] = {
                        'delta': getattr(snap.greeks, 'delta', None),
                        'gamma': getattr(snap.greeks, 'gamma', None),
                        'theta': getattr(snap.greeks, 'theta', None),
                        'vega': getattr(snap.greeks, 'vega', None),
                        'rho': getattr(snap.greeks, 'rho', None),
                    }
                if hasattr(snap, 'implied_volatility'):
                    entry['iv'] = snap.implied_volatility
                out[sym] = entry
            return out
        except Exception as e:
            print(f"  获取期权快照失败: {e}")
            return {}

    # ── 23. 期权交易所代码 ──────────────────────────────────

    def get_option_exchanges(self):
        """获取期权交易所代码映射"""
        try:
            codes = self.option_client.get_option_exchange_codes()
            return dict(codes)
        except Exception as e:
            print(f"  获取期权交易所失败: {e}")
            return {}

    # ── 24. 公司行动 (股息/拆股/合并) ──────────────────────

    def get_corporate_actions(self, ca_type='dividend', since=None, until=None):
        """获取公司行动数据

        Parameters:
            ca_type: 'dividend'|'split'|'merger'|'spinoff'|'rights_distribution'
            since/until: date对象

        Returns:
            DataFrame: target_symbol, ca_type, cash, new_rate, old_rate,
                       ex_date, record_date, payable_date, declaration_date
        """
        from datetime import date as date_type
        if since is None:
            since = date_type.today() - timedelta(days=30)
        if until is None:
            until = date_type.today()

        cache_key = f"corpaction_{ca_type}_{since}_{until}"
        cached = self._load_cache(cache_key, max_age_seconds=86400)
        if cached is not None:
            return cached

        try:
            from alpaca.trading.requests import GetCorporateAnnouncementsRequest
            from alpaca.trading.enums import CorporateActionType, CorporateActionDateType

            type_map = {
                'dividend': CorporateActionType.DIVIDEND,
                'split': CorporateActionType.SPLIT,
                'merger': CorporateActionType.MERGER,
            }
            # 部分枚举在某些SDK版本中不可用
            for extra in ['SPINOFF', 'RIGHTS_DISTRIBUTION']:
                if hasattr(CorporateActionType, extra):
                    type_map[extra.lower()] = getattr(CorporateActionType, extra)
            ca_enum = type_map.get(ca_type, CorporateActionType.DIVIDEND)

            print(f"获取公司行动: {ca_type} ({since} → {until})...")
            announcements = self.trading_client.get_corporate_announcements(
                GetCorporateAnnouncementsRequest(
                    ca_types=[ca_enum],
                    since=since, until=until,
                    date_type=CorporateActionDateType.DECLARATION_DATE,
                )
            )

            rows = []
            for a in announcements:
                rows.append({
                    'target_symbol': getattr(a, 'target_symbol', ''),
                    'initiating_symbol': getattr(a, 'initiating_symbol', ''),
                    'ca_type': ca_type,
                    'ca_sub_type': str(getattr(a, 'ca_sub_type', '')),
                    'cash': getattr(a, 'cash', 0),
                    'new_rate': getattr(a, 'new_rate', 0),
                    'old_rate': getattr(a, 'old_rate', 0),
                    'ex_date': getattr(a, 'ex_date', None),
                    'record_date': getattr(a, 'record_date', None),
                    'payable_date': getattr(a, 'payable_date', None),
                    'declaration_date': getattr(a, 'declaration_date', None),
                })

            df = pd.DataFrame(rows)
            print(f"  完成: {len(df)} 条公司行动")
            self._save_cache(cache_key, df)
            return df
        except Exception as e:
            print(f"  获取公司行动失败: {e}")
            return pd.DataFrame()

    # ── 25. 期权合约详情 ────────────────────────────────────

    def get_option_contracts(self, underlying, expiration_gte=None, expiration_lte=None):
        """获取期权合约元数据 (strike, OI, style, status)

        Returns:
            DataFrame: symbol, name, strike_price, expiration_date, type, style,
                       open_interest, status, tradable, size
        """
        try:
            from alpaca.trading.requests import GetOptionContractsRequest
            from datetime import date as date_type

            params = {'underlying_symbols': [underlying]}
            if expiration_gte:
                params['expiration_date_gte'] = expiration_gte
            if expiration_lte:
                params['expiration_date_lte'] = expiration_lte

            print(f"获取期权合约: {underlying}...")
            result = self.trading_client.get_option_contracts(
                GetOptionContractsRequest(**params))
            contracts = result.option_contracts if hasattr(result, 'option_contracts') else result

            rows = []
            for c in contracts:
                rows.append({
                    'symbol': c.symbol,
                    'name': c.name,
                    'underlying': c.underlying_symbol,
                    'strike_price': float(c.strike_price),
                    'expiration_date': c.expiration_date,
                    'type': str(c.type),
                    'style': str(c.style),
                    'open_interest': c.open_interest,
                    'close_price': c.close_price,
                    'size': c.size,
                    'status': str(c.status),
                    'tradable': c.tradable,
                })

            df = pd.DataFrame(rows)
            print(f"  完成: {len(df)} 个合约")
            return df
        except Exception as e:
            print(f"  获取期权合约失败: {e}")
            return pd.DataFrame()

    # ── 26. 投资组合历史 ────────────────────────────────────

    def get_portfolio_history(self, period='1M', timeframe='1D'):
        """获取投资组合历史 (equity/P&L曲线)

        Parameters:
            period: '1D','1W','1M','3M','6M','1A','all'
            timeframe: '1Min','5Min','15Min','1H','1D'

        Returns:
            DataFrame: timestamp, equity, profit_loss, profit_loss_pct
        """
        try:
            from alpaca.trading.requests import GetPortfolioHistoryRequest
            ph = self.trading_client.get_portfolio_history(
                GetPortfolioHistoryRequest(period=period, timeframe=timeframe))

            df = pd.DataFrame({
                'timestamp': [datetime.fromtimestamp(t) for t in ph.timestamp],
                'equity': ph.equity,
                'profit_loss': ph.profit_loss,
                'profit_loss_pct': ph.profit_loss_pct,
            })
            df = df.set_index('timestamp')
            print(f"  组合历史: {len(df)} 期, period={period}")
            return df
        except Exception as e:
            print(f"  获取组合历史失败: {e}")
            return pd.DataFrame()

    # ── 27. 市场时钟 ────────────────────────────────────────

    def get_clock(self):
        """获取市场时钟

        Returns:
            dict: is_open, timestamp, next_open, next_close
        """
        try:
            clock = self.trading_client.get_clock()
            return {
                'is_open': clock.is_open,
                'timestamp': str(clock.timestamp),
                'next_open': str(clock.next_open),
                'next_close': str(clock.next_close),
            }
        except Exception as e:
            print(f"  获取时钟失败: {e}")
            return {}

    # ── 28. 交易日历 ────────────────────────────────────────

    def get_calendar(self, start, end):
        """获取交易日历

        Returns:
            DataFrame: date, open, close, settlement_date
        """
        try:
            from alpaca.trading.requests import GetCalendarRequest
            from datetime import date as date_type
            if isinstance(start, str):
                start = datetime.strptime(start, '%Y-%m-%d').date()
            if isinstance(end, str):
                end = datetime.strptime(end, '%Y-%m-%d').date()

            cal = self.trading_client.get_calendar(
                GetCalendarRequest(start=start, end=end))

            rows = []
            for day in cal:
                rows.append({
                    'date': day.date,
                    'open': str(day.open),
                    'close': str(day.close),
                    'settlement_date': getattr(day, 'settlement_date', None),
                })
            df = pd.DataFrame(rows)
            print(f"  交易日历: {len(df)} 天")
            return df
        except Exception as e:
            print(f"  获取日历失败: {e}")
            return pd.DataFrame()

    # ── 29. 完整资产数据库 ──────────────────────────────────

    def get_all_assets(self, asset_class='us_equity'):
        """获取完整资产数据库

        Returns:
            DataFrame: symbol, name, exchange, asset_class, status,
                       tradable, shortable, fractionable, marginable
        """
        cache_key = f"all_assets_{asset_class}"
        cached = self._load_cache(cache_key, max_age_seconds=86400)
        if cached is not None:
            return cached

        try:
            print(f"获取资产数据库: {asset_class}...")
            assets = self.trading_client.get_all_assets()

            rows = []
            for a in assets:
                if asset_class and str(a.asset_class).lower() != f'assetclass.{asset_class}':
                    if asset_class not in str(a.asset_class).lower():
                        continue
                rows.append({
                    'symbol': a.symbol,
                    'name': a.name,
                    'exchange': str(a.exchange),
                    'asset_class': str(a.asset_class),
                    'status': str(a.status),
                    'tradable': a.tradable,
                    'shortable': a.shortable,
                    'fractionable': a.fractionable,
                    'marginable': a.marginable,
                    'easy_to_borrow': getattr(a, 'easy_to_borrow', None),
                    'maintenance_margin_requirement': getattr(a, 'maintenance_margin_requirement', None),
                })

            df = pd.DataFrame(rows)
            self._save_cache(cache_key, df)
            print(f"  完成: {len(df)} 资产")
            return df
        except Exception as e:
            print(f"  获取资产数据库失败: {e}")
            return pd.DataFrame()

    # ── 30. 账户详情 ────────────────────────────────────────

    def get_account(self):
        """获取账户完整信息

        Returns:
            dict: equity, cash, buying_power, portfolio_value, 等
        """
        try:
            acct = self.trading_client.get_account()
            return {
                'id': str(acct.id),
                'status': str(acct.status),
                'equity': float(acct.equity),
                'cash': float(acct.cash),
                'buying_power': float(acct.buying_power),
                'portfolio_value': float(acct.portfolio_value),
                'long_market_value': float(acct.long_market_value),
                'short_market_value': float(acct.short_market_value),
                'initial_margin': float(acct.initial_margin),
                'maintenance_margin': float(acct.maintenance_margin),
                'last_equity': float(acct.last_equity),
                'daytrade_count': acct.daytrade_count,
                'sma': float(acct.sma),
                'multiplier': float(acct.multiplier),
                'currency': str(acct.currency),
            }
        except Exception as e:
            print(f"  获取账户信息失败: {e}")
            return {}

    # ── 31. 持仓查询 ───────────────────────────────────────

    def get_positions(self):
        """获取当前所有持仓

        Returns:
            DataFrame: symbol, qty, side, market_value, cost_basis,
                       unrealized_pl, unrealized_plpc, current_price
        """
        try:
            positions = self.trading_client.get_all_positions()
            rows = []
            for p in positions:
                rows.append({
                    'symbol': p.symbol,
                    'qty': float(p.qty),
                    'side': str(p.side),
                    'market_value': float(p.market_value),
                    'cost_basis': float(p.cost_basis),
                    'unrealized_pl': float(p.unrealized_pl),
                    'unrealized_plpc': float(p.unrealized_plpc),
                    'current_price': float(p.current_price),
                    'avg_entry_price': float(p.avg_entry_price),
                    'asset_class': str(p.asset_class),
                })
            return pd.DataFrame(rows)
        except Exception as e:
            print(f"  获取持仓失败: {e}")
            return pd.DataFrame()

    # ── 32. 订单查询 ───────────────────────────────────────

    def get_orders(self, status='all', limit=100):
        """获取历史订单

        Returns:
            DataFrame: symbol, side, qty, type, status, filled_qty,
                       filled_avg_price, submitted_at, filled_at
        """
        try:
            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import QueryOrderStatus
            status_map = {
                'all': QueryOrderStatus.ALL,
                'open': QueryOrderStatus.OPEN,
                'closed': QueryOrderStatus.CLOSED,
            }
            orders = self.trading_client.get_orders(
                GetOrdersRequest(status=status_map.get(status, QueryOrderStatus.ALL), limit=limit))
            rows = []
            for o in orders:
                rows.append({
                    'symbol': o.symbol,
                    'side': str(o.side),
                    'qty': o.qty,
                    'type': str(o.type),
                    'status': str(o.status),
                    'filled_qty': o.filled_qty,
                    'filled_avg_price': o.filled_avg_price,
                    'submitted_at': str(o.submitted_at),
                    'filled_at': str(o.filled_at) if o.filled_at else None,
                    'created_at': str(o.created_at),
                })
            return pd.DataFrame(rows)
        except Exception as e:
            print(f"  获取订单失败: {e}")
            return pd.DataFrame()


# ── 便捷函数 ──────────────────────────────────────────────────

def prepare_alpaca_data(symbols=None, lookback_days=252):
    """模块级便捷函数 — 直接调用"""
    loader = AlpacaDataLoader()
    return loader.prepare_alpaca_data(symbols=symbols, lookback_days=lookback_days)


# ── 独立运行测试 ─────────────────────────────────────────────

if __name__ == '__main__':
    loader = AlpacaDataLoader()

    # 测试日K线
    print("\n=== 测试日K线 ===")
    bars = loader.get_daily_bars(['AAPL', 'MSFT'], '2024-01-01', '2024-03-01')
    if not bars.empty:
        print(bars.head())

    # 测试快照
    print("\n=== 测试快照 ===")
    snaps = loader.get_snapshots(['AAPL', 'MSFT'])
    for sym, data in snaps.items():
        print(f"  {sym}: last={data.get('last')}, bid={data.get('bid')}, ask={data.get('ask')}")

    # 测试新闻
    print("\n=== 测试新闻 ===")
    news = loader.get_news('AAPL', limit=3)
    for n in news:
        print(f"  {n['created_at']}: {n['headline'][:80]}")

    # 测试最活跃
    print("\n=== 测试最活跃 ===")
    actives = loader.get_most_actives(top=5)
    print(actives)

    # 测试prepare
    print("\n=== 测试prepare_alpaca_data ===")
    data = loader.prepare_alpaca_data(
        symbols=['AAPL', 'MSFT', 'GOOGL'], lookback_days=20
    )
    if data:
        print(f"  prices shape: {data['prices'].shape}")
        print(f"  symbols: {data['symbols']}")
