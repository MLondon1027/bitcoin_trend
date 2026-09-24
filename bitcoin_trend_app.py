from __future__ import annotations

from datetime import date

import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf

from bitcoin_trend_engine import (
    run_trend_comparison,
    strict_trend_parameter_stability,
)


st.set_page_config(page_title="Bitcoin Trend & Cash Lab", page_icon="₿", layout="wide")


@st.cache_data(ttl=21_600, show_spinner=False)
def download_close(ticker: str, start: date, end: date) -> pd.Series:
    data = yf.download(
        ticker,
        start=pd.Timestamp(start).strftime("%Y-%m-%d"),
        end=(pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
        threads=False,
        timeout=30,
    )
    if data.empty:
        raise ValueError(f"No market data were returned for {ticker}.")
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"].iloc[:, 0]
    else:
        close = data["Close"]
    close.name = ticker
    return close.dropna().sort_index()


def money(value: float) -> str:
    return f"${value:,.0f}"


def percent(value: float) -> str:
    return "—" if pd.isna(value) else f"{value:.1%}"


st.title("Bitcoin Trend & Cash Lab")
st.write(
    "Compare fixed Bitcoin trend-following rules with buy-and-hold. Strategies "
    "move into a short-term Treasury ETF when their Bitcoin signal is off."
)

with st.sidebar:
    st.header("Backtest settings")
    start_date = st.date_input(
        "Start date", value=date(2015, 1, 1), min_value=date(2011, 1, 1)
    )
    end_date = st.date_input("End date", value=date.today(), max_value=date.today())
    starting_capital = st.number_input(
        "Starting investment", min_value=100.0, value=10_000.0, step=1_000.0
    )
    cash_ticker = st.text_input(
        "Risk-off Treasury ticker",
        value="BIL",
        help="BIL is a 1–3 month U.S. Treasury bill ETF.",
    ).strip().upper()
    transaction_cost_bps = st.number_input(
        "Cost per one-way allocation change (basis points)",
        min_value=0.0,
        max_value=100.0,
        value=5.0,
        step=1.0,
    )

    st.header("Trend rules")
    fast_ema_days = st.slider("Fast EMA", 20, 100, 50, 5)
    slow_ema_days = st.slider("Slow EMA", 120, 300, 200, 5)
    slope_lookback_days = st.slider("Slow-EMA slope lookback", 5, 60, 20, 5)

    st.header("Volatility targeting")
    volatility_lookback_days = st.slider("Realized-volatility lookback", 10, 90, 30, 5)
    target_volatility_pct = st.slider("Target annual volatility (%)", 10, 80, 30, 5)
    maximum_btc_allocation_pct = st.slider("Maximum BTC allocation (%)", 10, 100, 100, 5)
    run_stability = st.checkbox("Run nearby-parameter stability test", value=True)
    run = st.button("Run comparison", type="primary", use_container_width=True)

st.info(
    "No strategy is guaranteed to produce a positive return every year. This test "
    "is designed to expose losing years, drawdowns, missed upside, and parameter fragility."
)

if run:
    if start_date >= end_date:
        st.error("The start date must be before the end date.")
        st.stop()
    if fast_ema_days >= slow_ema_days:
        st.error("Fast EMA must be shorter than slow EMA.")
        st.stop()

    try:
        with st.spinner("Downloading Bitcoin and Treasury prices…"):
            warmup_start = (pd.Timestamp(start_date) - pd.Timedelta(days=500)).date()
            btc = download_close("BTC-USD", warmup_start, end_date)
            cash = download_close(cash_ticker, warmup_start, end_date)
            result = run_trend_comparison(
                btc,
                cash,
                pd.Timestamp(start_date),
                pd.Timestamp(end_date),
                starting_capital=starting_capital,
                transaction_cost_bps=transaction_cost_bps,
                fast_ema_days=fast_ema_days,
                slow_ema_days=slow_ema_days,
                slope_lookback_days=slope_lookback_days,
                volatility_lookback_days=volatility_lookback_days,
                target_volatility_pct=target_volatility_pct,
                maximum_btc_allocation_pct=maximum_btc_allocation_pct,
            )

        comparison = result.metrics.sort_values(
            ["Losing Complete Years", "Maximum Drawdown", "CAGR"],
            ascending=[True, False, False],
        )
        leader_name = comparison.index[0]
        leader = comparison.iloc[0]
        buy_hold = result.metrics.loc["Buy & Hold BTC"]

        summary = st.columns(5)
        summary[0].metric("Fewest-losing-years rule", leader_name)
        summary[1].metric("Rule ending value", money(leader["Ending Value"]))
        summary[2].metric("Rule CAGR", percent(leader["CAGR"]))
        summary[3].metric("Rule maximum drawdown", percent(leader["Maximum Drawdown"]))
        summary[4].metric(
            "Losing complete years", f"{int(leader['Losing Complete Years'])}"
        )

        if leader["Losing Complete Years"] == 0:
            st.warning(
                "This rule had no losing complete calendar years in this sample. "
                "That is a historical observation, not a guarantee or a selection criterion for live trading."
            )

        st.subheader("Growth of the starting investment")
        growth = result.equity.rename_axis("Date").reset_index().melt(
            "Date", var_name="Strategy", value_name="Value"
        )
        growth_chart = px.line(growth, x="Date", y="Value", color="Strategy")
        growth_chart.update_layout(
            yaxis_tickprefix="$", hovermode="x unified", legend_title_text=""
        )
        st.plotly_chart(growth_chart, use_container_width=True)

        st.subheader("Drawdowns")
        drawdowns = result.drawdowns.rename_axis("Date").reset_index().melt(
            "Date", var_name="Strategy", value_name="Drawdown"
        )
        drawdown_chart = px.line(
            drawdowns, x="Date", y="Drawdown", color="Strategy"
        )
        drawdown_chart.update_layout(
            yaxis_tickformat=".0%", hovermode="x unified", legend_title_text=""
        )
        st.plotly_chart(drawdown_chart, use_container_width=True)

        st.subheader("Performance statistics")
        metrics = result.metrics.copy()
        metrics["Ending Value"] = metrics["Ending Value"].map(money)
        for column in [
            "Total Return",
            "CAGR",
            "Annualized Volatility",
            "Maximum Drawdown",
            "Worst Complete Year",
            "Time With BTC Exposure",
            "Average BTC Allocation",
        ]:
            metrics[column] = metrics[column].map(percent)
        metrics["Sharpe (0% risk-free)"] = metrics["Sharpe (0% risk-free)"].map(
            lambda value: "—" if pd.isna(value) else f"{value:.2f}"
        )
        for column in ["Losing Complete Years", "Positive Complete Years", "BTC Entries"]:
            metrics[column] = metrics[column].astype(int)
        st.dataframe(metrics.T, use_container_width=True)

        st.subheader("Calendar-year returns")
        annual = result.annual_returns.copy()
        if len(annual):
            final_year = int(annual.index[-1])
            if pd.Timestamp(end_date) < pd.Timestamp(final_year, 12, 25):
                annual = annual.rename(index={final_year: f"{final_year} YTD"})
        st.dataframe(annual.style.format("{:.1%}"), use_container_width=True)
        st.caption(
            "Losing-year statistics use only years with at least 350 daily observations; "
            "partial first and current years remain visible in the table."
        )

        st.subheader("Bitcoin allocation over time")
        allocation = result.weights.rename_axis("Date").reset_index().melt(
            "Date", var_name="Strategy", value_name="BTC Weight"
        )
        allocation_chart = px.line(
            allocation, x="Date", y="BTC Weight", color="Strategy"
        )
        allocation_chart.update_layout(
            yaxis_tickformat=".0%", hovermode="x unified", legend_title_text=""
        )
        st.plotly_chart(allocation_chart, use_container_width=True)

        if run_stability:
            with st.spinner("Testing nearby EMA settings…"):
                stability = strict_trend_parameter_stability(
                    btc,
                    cash,
                    pd.Timestamp(start_date),
                    pd.Timestamp(end_date),
                    starting_capital,
                    transaction_cost_bps,
                )
            st.subheader("Strict-trend parameter stability")
            formatted_stability = stability.copy()
            for column in ["CAGR", "Maximum Drawdown", "Worst Complete Year"]:
                formatted_stability[column] = formatted_stability[column].map(percent)
            for column in ["Fast EMA", "Slow EMA", "Losing Complete Years", "BTC Entries"]:
                formatted_stability[column] = formatted_stability[column].astype(int)
            st.dataframe(formatted_stability, hide_index=True, use_container_width=True)
            st.caption(
                "A credible rule should remain broadly useful across nearby settings. "
                "One isolated winning parameter combination is evidence of fragility."
            )

        st.download_button(
            "Download daily strategy values",
            result.equity.rename_axis("Date").reset_index().to_csv(index=False).encode("utf-8"),
            "bitcoin_trend_daily_values.csv",
            "text/csv",
        )
        st.download_button(
            "Download calendar-year returns",
            result.annual_returns.reset_index().to_csv(index=False).encode("utf-8"),
            "bitcoin_trend_annual_returns.csv",
            "text/csv",
        )

        st.caption(
            "Signals use information available at each close and are applied to the next "
            "daily return. Transaction costs are charged when the BTC allocation changes. "
            f"The risk-off allocation uses adjusted {cash_ticker} prices. Taxes are excluded."
        )
    except Exception as exc:
        st.error(f"Backtest could not run: {exc}")
        st.caption("Yahoo Finance can throttle downloads temporarily. Retry later if both tickers are valid.")
else:
    st.subheader("Strategies compared")
    st.markdown(
        """
        - **Buy & Hold BTC:** continuously holds Bitcoin.
        - **Price above slow SMA:** holds Bitcoin only above its long moving average.
        - **Fast/slow EMA:** holds Bitcoin while the fast trend exceeds the slow trend.
        - **Strict Trend:** also requires price above the slow EMA and a rising slow EMA.
        - **Vol-Targeted Strict Trend:** uses the strict signal but reduces Bitcoin exposure when realized volatility is high.
        """
    )
