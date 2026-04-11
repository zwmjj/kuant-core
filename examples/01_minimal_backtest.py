"""Minimal runnable example — 12-month momentum on a yfinance-sourced
universe of 10 sector ETFs, no WRDS required.

Run:
    python examples/01_minimal_backtest.py
"""
import numpy as np
import pandas as pd
import yfinance as yf


def main():
    tickers = ["XLK", "XLF", "XLE", "XLV", "XLY",
               "XLP", "XLI", "XLB", "XLU", "XLRE"]
    print(f"downloading {len(tickers)} sector ETFs from yfinance ...")
    raw = yf.download(tickers, start="2016-01-01", end="2025-06-30",
                      auto_adjust=True, progress=False)
    close = raw["Close"]
    monthly = close.resample("ME").last().pct_change().dropna()
    print(f"monthly panel: {monthly.shape}")

    # 12-month momentum signal
    mom12 = (1 + monthly).rolling(12).apply(lambda x: np.prod(x) - 1, raw=True).shift(1)

    # Quintile-spread long-short book (equal weight inside each sleeve)
    rets = []
    for t in monthly.index:
        s = mom12.loc[t].dropna()
        r = monthly.loc[t].dropna()
        common = s.index.intersection(r.index)
        if len(common) < 6:
            continue
        ranked = s.loc[common].sort_values()
        long_names = ranked.tail(2).index
        short_names = ranked.head(2).index
        rets.append({
            "date": t,
            "long":  float(r.loc[long_names].mean()),
            "short": float(r.loc[short_names].mean()),
        })

    df = pd.DataFrame(rets).set_index("date")
    df["ls"] = df["long"] - df["short"]
    sharpe = df["ls"].mean() / df["ls"].std() * (12 ** 0.5)
    cum = (1 + df["ls"]).cumprod()
    print(f"\n12m-momentum L/S quintile over {len(df)} months:")
    print(f"  Sharpe  = {sharpe:+.3f}")
    print(f"  CAGR    = {(cum.iloc[-1] ** (12 / len(df)) - 1):+.2%}")
    print(f"  Max DD  = {((cum / cum.cummax() - 1).min()):+.2%}")
    print(f"  (install kuant-core[us-data] + pass a WRDS CRSP data dict")
    print(f"   to qf.run_event_driven for the full library version)")


if __name__ == "__main__":
    main()
