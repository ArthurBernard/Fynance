#!/usr/bin/env python3
# coding: utf-8

""" Streamlit backtest playground.

Code a ``signal(prices) -> positions`` function on the left; see its tearsheet
on the right. Run with::

    streamlit run apps/playground/app.py

Requires the ``ui`` extra: ``pip install -e ".[ui]"``.

**Security**: this executes user-entered Python via ``exec`` — it is a
*local-only* research tool, not a hosted/multi-user surface.

"""

from __future__ import annotations

# Built-in packages
import io

# Third-party packages
import numpy as np

# Local packages
from apps.playground.runner import TEMPLATE, compile_signal, run_signal
from fynance.data import load
from fynance.plot import tearsheet


def _demo_prices(n: int = 500) -> np.ndarray:
    rng = np.random.default_rng(0)

    return 100.0 * np.cumprod(1.0 + rng.normal(0.0003, 0.01, n))


def main() -> None:
    """ Render the playground app. """
    import streamlit as st

    st.set_page_config(page_title="fynance playground", layout="wide")
    st.title("fynance — backtest playground")

    left, right = st.columns(2)

    with left:
        st.subheader("Signal function")
        upload = st.file_uploader("Price data (CSV/Parquet)", type=["csv", "parquet"])
        fee = st.number_input("Fee (per unit traded)", value=0.0, step=0.0005,
                              format="%.4f")
        code = st.text_area("def signal(prices): ...", value=TEMPLATE, height=320)
        run = st.button("Run backtest")

    with right:
        st.subheader("Performance")

        if run:
            try:
                if upload is not None:
                    suffix = "csv" if upload.name.endswith("csv") else "parquet"
                    ps = load(io.BytesIO(upload.getvalue()), source=suffix)
                    prices = np.asarray(getattr(ps, "values", ps), dtype=float)

                else:
                    prices = _demo_prices()

                signal = compile_signal(code)
                result = run_signal(prices, signal, fee=fee)
                st.pyplot(tearsheet(result))
                st.json(result.summary())

            except Exception as exc:  # surface user-code errors in the UI
                st.error(f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
