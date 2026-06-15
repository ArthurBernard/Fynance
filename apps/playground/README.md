# fynance backtest playground

A thin Streamlit app over `fynance.plot.tearsheet`: write a
`signal(prices) -> positions` function on the left, see its tearsheet on the
right.

## Run

```bash
pip install -e ".[ui]"
streamlit run apps/playground/app.py
```

Upload a CSV/Parquet price file (or use the built-in demo series), optionally
set a fee, edit the `signal` function, and click **Run backtest**.

> **Security note**: the app executes the code you type via `exec`. It is a
> *local-only* research tool — do not expose it as a hosted/multi-user service.

The signal-running logic lives in `runner.py` (Streamlit-free, unit-tested);
`app.py` is only the UI shell.
