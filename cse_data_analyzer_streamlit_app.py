# CSE Data Analyzer – Streamlit App
# --------------------------------------------------------------
# A self-contained Streamlit app to analyze Colombo Stock Exchange data.
# Features
# - Upload one or more CSVs (Date, Open, High, Low, Close, Volume)
# - Candlestick + SMA(20/50/200) overlays
# - RSI(14), MACD(12,26,9)
# - Return metrics: YTD, 1Y, CAGR, Volatility, Sharpe (user RF), Max Drawdown
# - Multi-symbol screener & quick filters
# - Simple portfolio backtest with custom weights, rebalanced monthly
# - Watchlist & price alerts (saved locally as JSON)
# --------------------------------------------------------------

import io
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

APP_STATE_FILE = "cse_app_state.json"

# --------------------------- Utilities ---------------------------

def load_app_state() -> dict:
    if os.path.exists(APP_STATE_FILE):
        try:
            with open(APP_STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_app_state(state: dict):
    try:
        with open(APP_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.warning(f"Couldn't save state: {e}")

@st.cache_data(show_spinner=False)
def read_csv_file(file_or_bytes, symbol_hint: str = None) -> pd.DataFrame:
    """Read CSV into OHLCV DataFrame. Expects Date, Open, High, Low, Close, Volume.
    Tolerant to extra columns; auto-detects date column.
    """
    if isinstance(file_or_bytes, (bytes, bytearray)):
        buf = io.BytesIO(file_or_bytes)
    else:
        buf = file_or_bytes

    df = pd.read_csv(buf)
    # Normalize columns
    cols = {c.lower().strip(): c for c in df.columns}
    # Find date col
    date_col = None
    for key in ["date", "timestamp", "trade_date"]:
        if key in cols:
            date_col = cols[key]
            break
    if date_col is None:
        # Try first column as date
        date_col = df.columns[0]
    df["Date"] = pd.to_datetime(df[date_col], errors="coerce")
    # Map common price cols
    def pick(*names):
        for n in names:
            if n in cols:
                return cols[n]
        return None

    map_open = pick("open", "o")
    map_high = pick("high", "h")
    map_low = pick("low", "l")
    map_close = pick("close", "c", "adj close", "adj_close", "closing price")
    map_vol = pick("volume", "vol", "qty")

    # Build standardized frame
    out = pd.DataFrame({
        "Date": df["Date"],
        "Open": df[map_open] if map_open else np.nan,
        "High": df[map_high] if map_high else np.nan,
        "Low": df[map_low] if map_low else np.nan,
        "Close": df[map_close] if map_close else np.nan,
        "Volume": df[map_vol] if map_vol else np.nan,
    })

    out = out.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    # Forward fill remaining NaNs sensibly
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if out[c].isna().all():
            out[c] = np.nan
        else:
            out[c] = pd.to_numeric(out[c], errors="coerce")
            out[c] = out[c].ffill()

    if symbol_hint:
        out["Symbol"] = symbol_hint
    return out

# --------------------------- Indicators ---------------------------

def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

# --------------------------- Metrics ---------------------------

def perf_metrics(df: pd.DataFrame, rf_annual: float = 0.0) -> dict:
    px = df["Close"].dropna()
    if px.empty:
        return {}
    ret = px.pct_change().dropna()
    if ret.empty:
        return {}
    ann_factor = 252  # trading days approximation
    total_return = px.iloc[-1] / px.iloc[0] - 1

    # CAGR
    days = (df["Date"].iloc[-1] - df["Date"].iloc[0]).days
    years = max(days / 365.25, 1e-9)
    cagr = (px.iloc[-1] / px.iloc[0]) ** (1 / years) - 1

    vol = ret.std() * math.sqrt(ann_factor)
    rf_daily = (1 + rf_annual) ** (1 / ann_factor) - 1
    sharpe = ((ret - rf_daily).mean() * ann_factor) / (ret.std() + 1e-12)

    # Max Drawdown
    cum = (1 + ret).cumprod()
    peak = cum.cummax()
    dd = (cum / peak - 1)
    max_dd = dd.min()

    # Periodic returns
    ytd = _period_return(px, period="ytd")
    one_y = _period_return(px, period="1y")
    six_m = _period_return(px, period="6m")
    one_m = _period_return(px, period="1m")

    return {
        "Start": df["Date"].iloc[0].date().isoformat(),
        "End": df["Date"].iloc[-1].date().isoformat(),
        "Total Return": total_return,
        "CAGR": cagr,
        "Volatility": vol,
        "Sharpe": sharpe,
        "Max Drawdown": max_dd,
        "YTD": ytd,
        "1Y": one_y,
        "6M": six_m,
        "1M": one_m,
    }

def _period_return(px: pd.Series, period: str = "ytd") -> float:
    idx = px.index
    dates = px.index
    if not isinstance(dates, pd.DatetimeIndex):
        # attempt to infer
        dates = pd.to_datetime(dates)
        px = pd.Series(px.values, index=dates)

    last = px.index.max()
    if period == "ytd":
        start = pd.Timestamp(year=last.year, month=1, day=1)
    elif period == "1y":
        start = last - pd.DateOffset(years=1)
    elif period == "6m":
        start = last - pd.DateOffset(months=6)
    elif period == "1m":
        start = last - pd.DateOffset(months=1)
    else:
        return np.nan

    px_period = px[px.index >= start]
    if len(px_period) < 2:
        return np.nan
    return px_period.iloc[-1] / px_period.iloc[0] - 1

# --------------------------- Charting ---------------------------

def plot_candles(df: pd.DataFrame, title: str = ""):
    fig = go.Figure(data=[go.Candlestick(
        x=df["Date"], open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price"
    )])
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=520, title=title)
    return fig


def overlay_sma(fig: go.Figure, df: pd.DataFrame, windows=(20, 50, 200)):
    for w in windows:
        fig.add_trace(go.Scatter(x=df["Date"], y=sma(df["Close"], w), mode="lines", name=f"SMA {w}"))
    return fig


def plot_rsi(df: pd.DataFrame, period: int = 14):
    r = rsi(df["Close"], period)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=r, mode="lines", name=f"RSI {period}"))
    fig.add_hrect(y0=30, y1=70, fillcolor="LightGray", opacity=0.3, line_width=0)
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=250, title="RSI")
    return fig


def plot_macd(df: pd.DataFrame):
    m, s, h = macd(df["Close"]) 
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=m, mode="lines", name="MACD"))
    fig.add_trace(go.Scatter(x=df["Date"], y=s, mode="lines", name="Signal"))
    fig.add_trace(go.Bar(x=df["Date"], y=h, name="Hist"))
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=250, title="MACD")
    return fig

# --------------------------- Portfolio ---------------------------

def monthly_rebalance(returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """Rebalance to target weights at each month start. Daily returns input."""
    weights = weights / weights.sum()
    eq_curve = []
    cur_weights = weights.copy()
    cum = 1.0
    last_month = None
    for dt, row in returns.iterrows():
        if last_month is None or dt.month != last_month:
            cur_weights = weights.copy()
            last_month = dt.month
        daily_ret = float((row * cur_weights).sum())
        cum *= (1 + daily_ret)
        eq_curve.append((dt, cum))
    return pd.Series({d: v for d, v in eq_curve})

# --------------------------- Layout ---------------------------

st.set_page_config(page_title="CSE Data Analyzer", layout="wide")

st.title("📊 CSE Data Analyzer")

with st.sidebar:
    st.header("Data Source")
    st.caption("Upload one CSV per symbol or a single multi-symbol CSV with a 'Symbol' column.")
    uploaded = st.file_uploader("Upload CSV files", type=["csv"], accept_multiple_files=True)

    st.divider()
    st.header("Risk-Free Rate")
    rf = st.number_input("Annual risk-free rate (e.g., T-bill)", min_value=-0.5, max_value=1.0, value=0.05, step=0.005, format="%.3f")

    st.divider()
    st.header("Watchlist & Alerts")
    app_state = load_app_state()
    watchlist = app_state.get("watchlist", [])
    new_symbol = st.text_input("Add symbol (e.g., JKH.N0000)")
    if st.button("➕ Add to watchlist") and new_symbol:
        if new_symbol not in watchlist:
            watchlist.append(new_symbol)
            app_state["watchlist"] = watchlist
            save_app_state(app_state)
            st.success(f"Added {new_symbol}")
    if watchlist:
        sym_to_remove = st.selectbox("Remove from watchlist", watchlist)
        if st.button("🗑️ Remove"):
            watchlist = [s for s in watchlist if s != sym_to_remove]
            app_state["watchlist"] = watchlist
            save_app_state(app_state)
            st.info(f"Removed {sym_to_remove}")

# Load dataframes per symbol
symbol_frames: Dict[str, pd.DataFrame] = {}

if uploaded:
    for file in uploaded:
        sym_hint = os.path.splitext(file.name)[0].upper()
        try:
            df = read_csv_file(file, symbol_hint=sym_hint)
            # Ensure Date index
            df = df.dropna(subset=["Date"]).sort_values("Date")
            df_indexed = df.set_index("Date")
            symbol = df.get("Symbol", pd.Series([sym_hint]*len(df))).iloc[0]
            symbol_frames[symbol] = df_indexed
        except Exception as e:
            st.error(f"Failed to read {file.name}: {e}")

# Tabs
main_tabs = st.tabs(["📈 Chart", "🧮 Metrics", "🧹 Screener", "📦 Portfolio", "⚙️ How to Use"])

# --------------------------- Chart Tab ---------------------------
with main_tabs[0]:
    st.subheader("Price & Indicators")
    if not symbol_frames:
        st.info("Upload CSVs to begin. Expected columns: Date, Open, High, Low, Close, Volume. Optional: Symbol.")
    else:
        symbols = list(symbol_frames.keys())
        sym = st.selectbox("Select symbol", symbols)
        df = symbol_frames[sym].copy()
        df = df.reset_index()

        col1, col2 = st.columns([3, 1])
        with col2:
            w20 = st.checkbox("SMA 20", value=True)
            w50 = st.checkbox("SMA 50", value=True)
            w200 = st.checkbox("SMA 200", value=False)
            show_rsi = st.checkbox("Show RSI", value=True)
            show_macd = st.checkbox("Show MACD", value=True)
        with col1:
            fig = plot_candles(df, title=f"{sym} – OHLC")
            wins = []
            if w20: wins.append(20)
            if w50: wins.append(50)
            if w200: wins.append(200)
            if wins:
                fig = overlay_sma(fig, df, windows=tuple(wins))
            st.plotly_chart(fig, use_container_width=True)

        if show_rsi:
            st.plotly_chart(plot_rsi(df), use_container_width=True)
        if show_macd:
            st.plotly_chart(plot_macd(df), use_container_width=True)

# --------------------------- Metrics Tab ---------------------------
with main_tabs[1]:
    st.subheader("Return & Risk Metrics")
    if not symbol_frames:
        st.info("Upload data to see metrics.")
    else:
        data = []
        for sym, df in symbol_frames.items():
            m = perf_metrics(df.reset_index(), rf_annual=rf)
            if m:
                row = {"Symbol": sym}
                row.update({k: v for k, v in m.items()})
                data.append(row)
        if data:
            metrics_df = pd.DataFrame(data)
            fmt_cols = ["Total Return", "CAGR", "Volatility", "Sharpe", "Max Drawdown", "YTD", "1Y", "6M", "1M"]
            st.dataframe(metrics_df.style.format({c: "{:.2%}" for c in fmt_cols if c != "Sharpe"}).format({"Sharpe": "{:.2f}"}), use_container_width=True)
        else:
            st.warning("No metrics computed. Check data columns.")

# --------------------------- Screener Tab ---------------------------
with main_tabs[2]:
    st.subheader("Multi-Symbol Screener")
    if len(symbol_frames) < 2:
        st.info("Upload 2 or more symbols to screen.")
    else:
        # Compute last close and momentum
        rows = []
        for sym, df in symbol_frames.items():
            dfr = df.reset_index()
            px = dfr["Close"].dropna()
            last = px.iloc[-1] if not px.empty else np.nan
            r1m = _period_return(px, "1m")
            r6m = _period_return(px, "6m")
            r1y = _period_return(px, "1y")
            rows.append({"Symbol": sym, "Last Close": last, "1M": r1m, "6M": r6m, "1Y": r1y})
        screen_df = pd.DataFrame(rows)

        c1, c2, c3 = st.columns(3)
        with c1:
            min_6m = st.number_input("Min 6M return", value=-1.0, step=0.05, format="%.2f")
        with c2:
            min_1y = st.number_input("Min 1Y return", value=-1.0, step=0.05, format="%.2f")
        with c3:
            min_price = st.number_input("Min Last Close (LKR)", value=0.0, step=1.0)

        filt = (screen_df["6M"].fillna(-9e9) >= min_6m) & (screen_df["1Y"].fillna(-9e9) >= min_1y) & (screen_df["Last Close"].fillna(-9e9) >= min_price)
        st.dataframe(screen_df[filt].sort_values(["6M", "1Y"], ascending=False), use_container_width=True)

# --------------------------- Portfolio Tab ---------------------------
with main_tabs[3]:
    st.subheader("Backtest a Portfolio (Monthly Rebalance)")
    if len(symbol_frames) < 2:
        st.info("Upload 2 or more symbols to backtest.")
    else:
        symbols = list(symbol_frames.keys())
        pick = st.multiselect("Select symbols", symbols, default=symbols[: min(4, len(symbols))])
        weight_inputs = {}
        cols = st.columns(len(pick)) if pick else []
        for i, sym in enumerate(pick):
            with cols[i]:
                weight_inputs[sym] = st.number_input(f"Weight {sym}", min_value=0.0, max_value=1.0, value=round(1.0/len(pick), 2), step=0.01)
        if pick:
            wsum = sum(weight_inputs.values())
            if not math.isclose(wsum, 1.0, rel_tol=1e-3):
                st.warning(f"Weights sum to {wsum:.2f}. They will be normalized to 1.0 for the backtest.")

            # Build aligned daily returns
            close_table = pd.DataFrame({sym: symbol_frames[sym]["Close"] for sym in pick}).dropna()
            rets = close_table.pct_change().dropna()
            eq = monthly_rebalance(rets, pd.Series(weight_inputs))
            eq = eq.rename("Equity").to_frame()
            ret = eq["Equity"].pct_change().fillna(0)

            # Plot equity curve
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=eq.index, y=eq["Equity"], mode="lines", name="Equity"))
            fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=420, title="Portfolio Equity Curve")
            st.plotly_chart(fig, use_container_width=True)

            # Show metrics
            df_bt = pd.DataFrame({"Date": eq.index, "Close": eq["Equity"]}).reset_index(drop=True)
            m = perf_metrics(df_bt, rf_annual=rf)
            if m:
                st.write({k: (f"{v:.2%}" if isinstance(v, (float, int)) and k not in ["Sharpe"] else (f"{v:.2f}" if k == "Sharpe" else v)) for k, v in m.items()})

# --------------------------- Help Tab ---------------------------
with main_tabs[4]:
    st.markdown(
        """
        ### How to Use
        1. **Prepare CSVs** for each CSE symbol you want to analyze. Required columns: `Date, Open, High, Low, Close, Volume`. Optional: `Symbol`.
        2. **Upload** one or more files from the sidebar. The app will auto-detect columns and parse dates.
        3. Explore **Chart** for candlesticks & indicators, **Metrics** for risk/return, **Screener** to filter by momentum, and **Portfolio** for a simple backtest.

        **Tips**
        - File name (e.g., `JKH.N0000.csv`) is used as a symbol if no `Symbol` column exists.
        - This app runs fully offline. To connect live CSE data, replace the upload step with a function that fetches OHLCV and returns the same DataFrame schema.
        - Risk-free rate is used for Sharpe calculation.
        - App settings and watchlist are saved to `cse_app_state.json` in the working folder.
        """
    )

st.caption("Made for the Colombo Stock Exchange. Bring your own data sources.")
import streamlit as st
import pandas as pd

@st.cache_data
def load_data():
    df = pd.read_csv("your_dataset.csv")
    return df

data = load_data()
st.dataframe(data)
if st.button("Load Analysis"):
    # heavy computation here
    result = some_expensive_function()
    st.write(result)
