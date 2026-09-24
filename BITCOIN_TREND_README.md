# Bitcoin Trend & Cash Lab

This standalone Streamlit application compares fixed Bitcoin trend-following
rules with Bitcoin buy-and-hold. When a trend rule is risk-off, its unallocated
capital earns the adjusted return of a short-term Treasury ETF such as `BIL`.

## Run

```bash
python -m pip install -r requirements.txt
streamlit run bitcoin_trend_app.py
```

Required files:

- `bitcoin_trend_app.py`
- `bitcoin_trend_engine.py`
- `requirements.txt`

## Method

- Downloads adjusted daily prices through Yahoo Finance.
- Computes every signal from information available at that day's close.
- Applies the resulting allocation beginning with the next daily return.
- Charges transaction costs when the Bitcoin allocation changes.
- Compares buy-and-hold, a price/slow-SMA rule, a fast/slow-EMA rule, a strict
  trend regime, and a volatility-targeted strict regime.
- Reports all calendar-year returns, losing complete years, maximum drawdown,
  Bitcoin entries, time exposed, and average allocation.
- Optionally tests 15 nearby EMA combinations to reveal parameter fragility.

The app does not optimize a rule to eliminate historical losing years. A rule
that happened to remain positive in every tested year is not guaranteed to do
so in the future.
