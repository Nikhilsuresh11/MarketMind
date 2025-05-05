import streamlit as st
st.set_page_config(page_title="Stock Market Analysis Platform", layout="wide")

import os
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.together import Together
from agno.tools.yfinance import YFinanceTools
from agno.tools.duckduckgo import DuckDuckGoTools
import markdown

# Load environment variables
load_dotenv()

# Setup Together API key
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
os.environ["TOGETHER_API_KEY"] = TOGETHER_API_KEY or ""

# Initialize agent
@st.cache_resource
def initialize_agent():
    return Agent(
        name="Stock Advisor",
        role="Analyze Indian stock market trends and individual stocks",
        model=Together(id="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"),
        tools=[YFinanceTools(), DuckDuckGoTools()],
        instructions="""
        - Provide fundamental + technical analysis
        - Summarize recent news
        - Include clear Buy/Hold/Sell recommendation, target price, expected return, and risk level
        - Format as markdown with headers and bullet points
        """,
        show_tool_calls=False,
        markdown=True
    )

advisor = initialize_agent()

# Title
st.title("📈 Real-time Stock Analysis & Portfolio Optimizer")

# Sidebar
st.sidebar.header("Stock Settings")
ticker = st.sidebar.text_input("Enter NSE/BSE Stock Symbol", value="RELIANCE.NS")
period = st.sidebar.selectbox("Select Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=2)

# Fetch stock data
@st.cache_data(ttl=300)
def get_stock_data(ticker, period):
    try:
        if ticker.upper() == "NIFTY50":
            ticker = "^NSEI"
        elif ticker.upper() == "SENSEX":
            ticker = "^BSESN"
        elif "." not in ticker:
            ticker += ".NS"

        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        return hist, stock.info
    except Exception as e:
        st.error(f"Error: {e}")
        return pd.DataFrame(), {}

# Display data
hist, info = get_stock_data(ticker, period)

if not hist.empty:
    st.subheader(f"📊 {info.get('shortName', ticker)}")
    st.write(f"**Sector:** {info.get('sector', 'N/A')}, **Market Cap:** ₹{info.get('marketCap', 'N/A'):,}")
    st.line_chart(hist['Close'], use_container_width=True)

    with st.expander("📃 Fundamentals"):
        st.json({
            "P/E Ratio": info.get("trailingPE"),
            "EPS": info.get("trailingEps"),
            "Dividend Yield": info.get("dividendYield"),
            "Book Value": info.get("bookValue"),
            "Debt to Equity": info.get("debtToEquity")
        })

    with st.expander("📉 Technical Overview"):
        hist['MA20'] = hist['Close'].rolling(20).mean()
        hist['MA50'] = hist['Close'].rolling(50).mean()
        st.line_chart(hist[['Close', 'MA20', 'MA50']], use_container_width=True)

    with st.expander("🤖 AI-powered Insights"):
        if st.button("Generate Insights"):
            with st.spinner("Analyzing..."):
                query = f"""Analyze {ticker} stock. Include fundamentals, technicals, recent news, target price, expected return %, and risk level. Recommend Buy/Hold/Sell."""
                try:
                    response = advisor.run(query)
                    st.markdown(response.content)
                except Exception as e:
                    st.error(f"Error generating analysis: {e}")
else:
    st.warning("No data found. Please check the symbol.")

# Portfolio Optimization
st.header("💼 Portfolio Optimization")
with st.form("portfolio_form"):
    tickers = st.text_area("Enter tickers (comma separated, NSE format)", value="RELIANCE.NS, TCS.NS, INFY.NS")
    weights = st.text_input("Enter weights (%) (comma separated)", value="40,30,30")
    submitted = st.form_submit_button("Optimize")

if submitted:
    try:
        ticker_list = [t.strip() for t in tickers.split(",")]
        weight_list = [float(w.strip())/100 for w in weights.split(",")]
        prices = {}
        for t in ticker_list:
            prices[t], _ = get_stock_data(t, "1y")

        returns = pd.DataFrame({t: prices[t]['Close'].pct_change().dropna() for t in ticker_list})
        expected_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252

        vol = np.sqrt(np.diag(cov_matrix))
        inv_vol = 1 / vol
        opt_weights = inv_vol / sum(inv_vol)

        st.subheader("📈 Suggested Weights")
        df = pd.DataFrame({"Ticker": ticker_list, "User Weight %": np.array(weight_list)*100, "Optimized %": opt_weights*100})
        st.dataframe(df)

        current_return = np.dot(expected_returns, weight_list)
        opt_return = np.dot(expected_returns, opt_weights)
        current_risk = np.sqrt(np.dot(weight_list, np.dot(cov_matrix, weight_list)))
        opt_risk = np.sqrt(np.dot(opt_weights, np.dot(cov_matrix, opt_weights)))

        st.metric("Expected Return", f"{current_return:.2%} → {opt_return:.2%}")
        st.metric("Expected Risk (Volatility)", f"{current_risk:.2%} → {opt_risk:.2%}")

    except Exception as e:
        st.error(f"Error in optimization: {e}")

# Tax Calculator
st.header("🧾 Tax Calculator")
with st.form("tax_form"):
    buy_price = st.number_input("Buy Price", value=100.0)
    sell_price = st.number_input("Sell Price", value=120.0)
    qty = st.number_input("Quantity", value=10)
    holding_days = st.number_input("Holding Period (in days)", value=180)
    tax_submit = st.form_submit_button("Calculate")

if tax_submit:
    profit = (sell_price - buy_price) * qty
    if profit <= 0:
        tax = 0
        tax_type = "No tax on capital loss"
    elif holding_days > 365:
        tax = max(0, (profit - 100000) * 0.10)
        tax_type = "LTCG @ 10%"
    else:
        tax = profit * 0.15
        tax_type = "STCG @ 15%"

    st.success(f"💰 Net Profit: ₹{profit-tax:.2f} ({tax_type}, Tax: ₹{tax:.2f})")
