"""A-share intraday and alternative data loading — akshare (minute bars / dragon-tiger list / block trades / northbound flow / margin trading)"""
import os, pickle, warnings, time
import akshare as ak
import pandas as pd
from datetime import datetime
warnings.filterwarnings('ignore')

CACHE_DIR = "data_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


# ── 缓存工具 ─────────────────────────────────────────────
def _cache_path(name):
    return os.path.join(CACHE_DIR, f"cn_intra_{name}.pkl")


def _load_cache(name, max_age_days=1):
    """读取缓存，日内数据默认缓存1天"""
    path = _cache_path(name)
    if os.path.exists(path):
        age = datetime.now().timestamp() - os.path.getmtime(path)
        if age < 86400 * max_age_days:
            print(f"  使用缓存: cn_intra_{name}")
            with open(path, 'rb') as f:
                return pickle.load(f)
    return None


def _save_cache(name, data):
    with open(_cache_path(name), 'wb') as f:
        pickle.dump(data, f)


# ── 主类 ─────────────────────────────────────────────────
class ChinaIntradayLoader:
    """A-share intraday and alternative data loader, built on akshare"""

    # ── 分钟线 ────────────────────────────────────────────
    @staticmethod
    def get_minute_bars(
        symbols: list[str],
        period: str = "1",
        start_date: str = "",
        end_date: str = "",
    ) -> dict[str, pd.DataFrame]:
        """Fetch A-share minute bar data

        Args:
            symbols: list of stock codes, e.g. ["000001", "600519"]
            period: bar frequency "1"/"5"/"15"/"30"/"60" (minutes)
            start_date: start datetime "YYYY-MM-DD HH:MM:SS" (optional)
            end_date: end datetime "YYYY-MM-DD HH:MM:SS" (optional)

        Returns:
            dict[str, pd.DataFrame], one DataFrame per symbol
            Columns: time, open, close, high, low, volume, turnover
        """
        assert period in ("1", "5", "15", "30", "60"), \
            f"period 须为 1/5/15/30/60，收到: {period}"

        cache_key = f"min_{period}_{','.join(sorted(symbols))}_{start_date}_{end_date}"
        cached = _load_cache(cache_key, max_age_days=0.5)
        if cached is not None:
            return cached

        result = {}
        errors = []
        for sym in symbols:
            try:
                print(f"  下载分钟线: {sym} ({period}分钟)...")
                df = ak.stock_zh_a_hist_min_em(
                    symbol=sym,
                    period=period,
                    start_date=start_date,
                    end_date=end_date,
                    adjust="qfq",  # 前复权
                )
                if df is not None and not df.empty:
                    result[sym] = df
                    print(f"    {sym}: {len(df)} 条")
                else:
                    print(f"    {sym}: 无数据")
                time.sleep(0.3)  # 限速，避免被封
            except Exception as e:
                errors.append((sym, str(e)))
                print(f"    {sym} 出错: {e}")

        if errors:
            print(f"  分钟线下载完成，{len(errors)} 个错误: "
                  f"{[e[0] for e in errors]}")

        if result:
            _save_cache(cache_key, result)
        return result

    # ── 龙虎榜 ────────────────────────────────────────────
    @staticmethod
    def get_dragon_tiger(date: str) -> pd.DataFrame:
        """Fetch dragon-tiger list data

        Args:
            date: date "YYYYMMDD", e.g. "20260330"

        Returns:
            DataFrame with: code, name, listing reason, buy amount, sell amount, net buy amount, etc.
        """
        cache_key = f"lhb_{date}"
        cached = _load_cache(cache_key, max_age_days=30)
        if cached is not None:
            return cached

        print(f"  下载龙虎榜: {date}...")
        try:
            df = ak.stock_lhb_detail_em(
                start_date=date,
                end_date=date,
            )
            if df is not None and not df.empty:
                print(f"    龙虎榜: {len(df)} 条记录")
                _save_cache(cache_key, df)
                return df
            else:
                print(f"    {date} 无龙虎榜数据")
                return pd.DataFrame()
        except Exception as e:
            print(f"    龙虎榜出错: {e}")
            return pd.DataFrame()

    # ── 大宗交易 ──────────────────────────────────────────
    @staticmethod
    def get_block_trades(date: str) -> pd.DataFrame:
        """Fetch block trade data

        Args:
            date: date "YYYYMMDD"

        Returns:
            DataFrame with: code, name, trade price, volume, turnover, premium/discount rate, etc.
        """
        cache_key = f"dzjy_{date}"
        cached = _load_cache(cache_key, max_age_days=30)
        if cached is not None:
            return cached

        print(f"  下载大宗交易: {date}...")
        try:
            df = ak.stock_dzjy_sctj(start_date=date, end_date=date)
            if df is not None and not df.empty:
                print(f"    大宗交易: {len(df)} 条")
                _save_cache(cache_key, df)
                return df
            else:
                print(f"    {date} 无大宗交易数据")
                return pd.DataFrame()
        except Exception as e:
            print(f"    大宗交易出错: {e}")
            return pd.DataFrame()

    # ── 北向资金 ──────────────────────────────────────────
    @staticmethod
    def get_northbound_flow(
        start_date: str = "",
        end_date: str = "",
    ) -> pd.DataFrame:
        """Fetch northbound capital net inflow data

        Args:
            start_date: start date (optional)
            end_date: end date (optional)

        Returns:
            DataFrame with: date, Shanghai Connect net inflow, Shenzhen Connect net inflow, total northbound net inflow
        """
        cache_key = f"hsgt_north_{start_date}_{end_date}"
        cached = _load_cache(cache_key, max_age_days=1)
        if cached is not None:
            return cached

        print(f"  下载北向资金流数据...")
        try:
            df = ak.stock_hsgt_north_net_flow_in_em(symbol="北上")
            if df is not None and not df.empty:
                # 按日期筛选
                if "日期" in df.columns:
                    df["日期"] = pd.to_datetime(df["日期"])
                    if start_date:
                        df = df[df["日期"] >= pd.to_datetime(start_date)]
                    if end_date:
                        df = df[df["日期"] <= pd.to_datetime(end_date)]
                print(f"    北向资金: {len(df)} 条")
                _save_cache(cache_key, df)
                return df
            else:
                print("    无北向资金数据")
                return pd.DataFrame()
        except Exception as e:
            print(f"    北向资金出错: {e}")
            return pd.DataFrame()

    # ── 融资融券 ──────────────────────────────────────────
    @staticmethod
    def get_margin_trading(
        symbol: str,
        start_date: str = "",
        end_date: str = "",
    ) -> pd.DataFrame:
        """Fetch per-stock margin trading data

        Args:
            symbol: stock code, e.g. "000001"
            start_date: start date "YYYYMMDD"
            end_date: end date "YYYYMMDD"

        Returns:
            DataFrame with: date, margin balance, short balance, margin buy amount, short sell volume, etc.
        """
        cache_key = f"margin_{symbol}_{start_date}_{end_date}"
        cached = _load_cache(cache_key, max_age_days=1)
        if cached is not None:
            return cached

        print(f"  下载融资融券: {symbol}...")
        try:
            df = ak.stock_margin_detail_sse(symbol=symbol)
            if df is not None and not df.empty:
                # 按日期筛选
                date_col = None
                for col in df.columns:
                    if "日期" in str(col) or "date" in str(col).lower():
                        date_col = col
                        break
                if date_col:
                    df[date_col] = pd.to_datetime(df[date_col])
                    if start_date:
                        df = df[df[date_col] >= pd.to_datetime(start_date)]
                    if end_date:
                        df = df[df[date_col] <= pd.to_datetime(end_date)]
                print(f"    融资融券 {symbol}: {len(df)} 条")
                _save_cache(cache_key, df)
                return df
            else:
                print(f"    {symbol} 无融资融券数据")
                return pd.DataFrame()
        except Exception as e:
            print(f"    融资融券出错: {e}")
            return pd.DataFrame()


# ── 示例 ─────────────────────────────────────────────────
if __name__ == "__main__":
    loader = ChinaIntradayLoader()

    # 1. 分钟线示例
    print("=" * 50)
    print("1. 分钟线数据")
    bars = loader.get_minute_bars(["000001", "600519"], period="5")
    for sym, df in bars.items():
        print(f"  {sym}: {len(df)} 条, 列={list(df.columns)}")
        print(df.tail(3).to_string(index=False))

    # 2. 龙虎榜示例
    print("\n" + "=" * 50)
    print("2. 龙虎榜数据")
    today = datetime.now().strftime("%Y%m%d")
    lhb = loader.get_dragon_tiger(today)
    if not lhb.empty:
        print(f"  {len(lhb)} 条, 列={list(lhb.columns)}")
        print(lhb.head(3).to_string(index=False))

    # 3. 北向资金示例
    print("\n" + "=" * 50)
    print("3. 北向资金")
    north = loader.get_northbound_flow()
    if not north.empty:
        print(f"  {len(north)} 条, 列={list(north.columns)}")
        print(north.tail(5).to_string(index=False))

    # 4. 大宗交易示例
    print("\n" + "=" * 50)
    print("4. 大宗交易")
    dzjy = loader.get_block_trades(today)
    if not dzjy.empty:
        print(f"  {len(dzjy)} 条")

    # 5. 融资融券示例
    print("\n" + "=" * 50)
    print("5. 融资融券")
    margin = loader.get_margin_trading("000001")
    if not margin.empty:
        print(f"  {len(margin)} 条, 列={list(margin.columns)}")
