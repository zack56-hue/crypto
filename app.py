
# app.py

import datetime as dt

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go

# ----------------------------------------------------
# Page config
# ----------------------------------------------------
st.set_page_config(
    page_title="Crypto Price & Volatility Explorer",
    page_icon="💹",
    layout="wide"
)

# ----------------------------------------------------
# Helper functions
# ----------------------------------------------------
def load_price_data(tickers, start_date, end_date):
    """
    Download daily closing price data for the given tickers and date range.
    Returns a DataFrame with Date index and one column per ticker.
    """
    if not tickers:
        return pd.DataFrame()

    data = yf.download(
        tickers,
        start=start_date,
        end=end_date
    )

    # yfinance returns different shapes depending on number of tickers.
    if "Close" in data.columns:
        close = data["Close"]
    else:
        # Fallback, just in case
        close = data

    # If only one ticker, convert Series -> DataFrame
    if isinstance(close, pd.Series):
        close = close.to_frame()

    # Ensure columns are ticker symbols (not a MultiIndex)
    if isinstance(close.columns, pd.MultiIndex):
        close.columns = close.columns.get_level_values(0)

    return close.dropna(how="all")


def compute_returns(price_df):
    """Compute simple daily returns from price data."""
    return price_df.pct_change().dropna(how="all")


def compute_rolling_volatility(returns_df, window):
    """
    Compute rolling standard deviation of returns over given window (in days).
    Returns a DataFrame with same columns as returns_df.
    """
    return returns_df.rolling(window=window).std().dropna(how="all")


def compute_summary_stats(price_df, returns_df):
    """
    Produce basic summary stats: total return, average daily return, daily volatility.
    """
    if price_df.empty or returns_df.empty:
        return pd.DataFrame()

    total_return = (price_df.iloc[-1] / price_df.iloc[0] - 1) * 100
    avg_daily_return = returns_df.mean() * 100
    daily_vol = returns_df.std() * 100

    summary = pd.DataFrame({
        "Total return (%)": total_return,
        "Average daily return (%)": avg_daily_return,
        "Daily volatility (%)": daily_vol
    })

    return summary.round(3)


def normalise_weights(weight_dict):
    """
    Take a dict of {ticker: weight_value} and normalise so they sum to 1.
    If all weights are zero, returns equal weights.
    """
    tickers = list(weight_dict.keys())
    weights = np.array(list(weight_dict.values()), dtype=float)

    total = weights.sum()
    if total > 0:
        norm = weights / total
    else:
        # If everything is 0, use equal weights
        n = len(tickers)
        norm = np.ones(n) / n

    return dict(zip(tickers, norm))


def compute_portfolio_returns(returns_df, weights_dict):
    """
    Given a returns DataFrame and a dict of weights (that sum to 1),
    compute a portfolio returns Series.
    """
    if returns_df.empty:
        return pd.Series(dtype=float)

    # Align weights with columns
    weights = np.array([weights_dict.get(col, 0.0) for col in returns_df.columns])
    port_ret = (returns_df * weights).sum(axis=1)
    return port_ret


def make_cumulative_returns(returns_df):
    """
    Turn daily returns into cumulative return series: (1 + r).cumprod() - 1
    Works for DataFrame or Series.
    """
    return (1 + returns_df).cumprod() - 1


# ----------------------------------------------------
# Sidebar controls
# ----------------------------------------------------
st.sidebar.title("⚙️ Settings")

# Available cryptos (you can extend this list)
available_coins = {
    "Bitcoin (BTC)": "BTC-USD",
    "Ethereum (ETH)": "ETH-USD",
    "Solana (SOL)": "SOL-USD",
    "Cardano (ADA)": "ADA-USD",
    "XRP (XRP)": "XRP-USD"
}

default_selection = ["BTC-USD", "ETH-USD"]

selected_labels = st.sidebar.multiselect(
    "Select cryptocurrencies",
    options=list(available_coins.keys()),
    default=[k for k, v in available_coins.items() if v in default_selection]
)

selected_coins = [available_coins[label] for label in selected_labels]

# Date range
today = dt.date.today()
default_start = today - dt.timedelta(days=365)

start_date = st.sidebar.date_input(
    "Start date",
    value=default_start
)

end_date = st.sidebar.date_input(
    "End date",
    value=today
)

if start_date > end_date:
    st.sidebar.error("Start date must be before end date.")

# Volatility window
vol_window = st.sidebar.slider(
    "Volatility window (days)",
    min_value=5,
    max_value=60,
    value=30,
    step=1
)

# Price scale
price_scale = st.sidebar.radio(
    "Price scale",
    options=["Linear", "Logarithmic"],
    index=0
)

# Portfolio section
st.sidebar.markdown("---")
st.sidebar.subheader("💼 Portfolio weights")

use_custom_weights = st.sidebar.checkbox(
    "Use custom weights (otherwise equal-weighted portfolio)",
    value=False
)

weights_input = {}
if selected_coins:
    if use_custom_weights:
        st.sidebar.caption("Enter raw weights; they will be normalised to sum to 1.")
        for ticker in selected_coins:
            weights_input[ticker] = st.sidebar.number_input(
                f"{ticker} weight",
                min_value=0.0,
                max_value=100.0,
                value=1.0,
                step=0.5,
                key=f"weight_{ticker}"
            )
    else:
        # Equal weights placeholder (normalised later)
        for ticker in selected_coins:
            weights_input[ticker] = 1.0

    portfolio_weights = normalise_weights(weights_input)
else:
    portfolio_weights = {}

# ----------------------------------------------------
# Main page
# ----------------------------------------------------
st.title("💹 Crypto Price & Volatility Explorer")
st.markdown(
    """
Explore the behaviour of major cryptocurrencies over time:  
**prices**, **daily returns**, **rolling volatility**, **correlations**, and a simple **portfolio view**.
"""
)

if not selected_coins:
    st.warning("Please select at least one cryptocurrency from the sidebar.")
elif start_date > end_date:
    st.warning("Fix the date range in the sidebar to continue.")
else:
    # ------------------------------------------------
    # Load data
    # ------------------------------------------------
    price_df = load_price_data(selected_coins, start_date, end_date)

    if price_df.empty or len(price_df) < 2:
        st.error("Not enough data for the chosen settings. Try a different date range or coins.")
    else:
        returns_df = compute_returns(price_df)
        rolling_vol_df = compute_rolling_volatility(returns_df, vol_window)
        summary_stats = compute_summary_stats(price_df, returns_df)

        # ------------------------------------------------
        # Layout: tabs
        # ------------------------------------------------
        tab_prices, tab_returns, tab_corr, tab_portfolio = st.tabs(
            ["📈 Prices", "📉 Returns & Volatility", "🧩 Correlations", "💼 Portfolio"]
        )

        # ------------------------------------------------
        # Tab 1: Prices
        # ------------------------------------------------
        with tab_prices:
            st.subheader("Closing Prices")

            # Plotly line chart for prices
            fig_price = px.line(
                price_df,
                x=price_df.index,
                y=price_df.columns,
                labels={"value": "Price (USD)", "variable": "Ticker", "x": "Date"},
                template="plotly_white"
            )
            if price_scale == "Logarithmic":
                fig_price.update_yaxes(type="log")

            fig_price.update_layout(
                legend_title_text="Cryptocurrency",
                hovermode="x unified"
            )

            st.plotly_chart(fig_price, use_container_width=True)

            st.markdown("### Summary statistics")
            st.dataframe(summary_stats.style.format("{:.3f}"))

        # ------------------------------------------------
        # Tab 2: Returns & Volatility
        # ------------------------------------------------
        with tab_returns:
            st.subheader("Daily Returns")

            fig_ret = px.line(
                returns_df,
                x=returns_df.index,
                y=returns_df.columns,
                labels={"value": "Daily return", "variable": "Ticker", "x": "Date"},
                template="plotly_white"
            )
            fig_ret.update_layout(
                legend_title_text="Cryptocurrency",
                hovermode="x unified"
            )
            st.plotly_chart(fig_ret, use_container_width=True)

            st.subheader(f"Rolling Volatility ({vol_window}-day window)")
            fig_vol = px.line(
                rolling_vol_df,
                x=rolling_vol_df.index,
                y=rolling_vol_df.columns,
                labels={"value": "Rolling std of return", "variable": "Ticker", "x": "Date"},
                template="plotly_white"
            )
            fig_vol.update_layout(
                legend_title_text="Cryptocurrency",
                hovermode="x unified"
            )
            st.plotly_chart(fig_vol, use_container_width=True)

            st.caption(
                "Note: Volatility here is the standard deviation of daily returns over a rolling window."
            )

        # ------------------------------------------------
        # Tab 3: Correlations
        # ------------------------------------------------
        with tab_corr:
            st.subheader("Correlation of Daily Returns")

            corr_matrix = returns_df.corr()

            fig_corr = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                color_continuous_scale="RdBu",
                zmin=-1,
                zmax=1,
                labels=dict(color="Correlation"),
            )
            fig_corr.update_layout(
                template="plotly_white",
                xaxis_title="Ticker",
                yaxis_title="Ticker"
            )

            st.plotly_chart(fig_corr, use_container_width=True)

            st.markdown(
                """
A value close to **1** = move together,  
close to **-1** = move in opposite directions,  
around **0** = weak or no linear relationship.
"""
            )

        # ------------------------------------------------
        # Tab 4: Portfolio
        # ------------------------------------------------
        with tab_portfolio:
            st.subheader("Portfolio Overview")

            if not portfolio_weights:
                st.info("Select some coins in the sidebar to build a portfolio.")
            else:
                # Show normalised weights
                weight_df = pd.DataFrame.from_dict(
                    portfolio_weights,
                    orient="index",
                    columns=["Normalised weight"]
                )
                st.markdown("#### Portfolio weights")
                st.dataframe(weight_df.style.format("{:.3f}"))

                # Compute portfolio returns
                port_returns = compute_portfolio_returns(returns_df, portfolio_weights)

                # Combine into cumulative returns
                cum_individual = make_cumulative_returns(returns_df)
                cum_portfolio = make_cumulative_returns(port_returns.to_frame("Portfolio"))

                combined_cum = pd.concat([cum_individual, cum_portfolio], axis=1)

                st.markdown("#### Cumulative returns (individual coins vs portfolio)")
                fig_cum = px.line(
                    combined_cum,
                    x=combined_cum.index,
                    y=combined_cum.columns,
                    labels={"value": "Cumulative return", "variable": "Asset", "x": "Date"},
                    template="plotly_white"
                )
                fig_cum.update_layout(
                    legend_title_text="Asset",
                    hovermode="x unified"
                )
                st.plotly_chart(fig_cum, use_container_width=True)

                # Portfolio stats
                st.markdown("#### Portfolio statistics")
                port_total_ret = (cum_portfolio.iloc[-1, 0]) * 100
                port_avg_daily = port_returns.mean() * 100
                port_daily_vol = port_returns.std() * 100

                col1, col2, col3 = st.columns(3)
                col1.metric("Total return (%)", f"{port_total_ret:.2f}")
                col2.metric("Average daily return (%)", f"{port_avg_daily:.3f}")
                col3.metric("Daily volatility (%)", f"{port_daily_vol:.3f}")

                st.caption(
                    "Portfolio returns are computed as the weighted sum of individual coin returns."
                )
