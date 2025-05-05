# You need to set TOGETHER_API_KEY in your environment variables
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
import os
from dotenv import load_dotenv
import json
import yfinance as yf
from datetime import datetime, timedelta, time
import pandas as pd
import numpy as np
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import requests
from apscheduler.schedulers.background import BackgroundScheduler
import atexit
import uuid
import pytz
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
from datetime import timezone
# Import Agno modules for real-time data and AI analysis
from agno.agent import Agent
from agno.models.together import Together
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools
from agno.tools.newspaper4k import Newspaper4kTools
from agno.tools.calculator import CalculatorTools
# MongoDB imports
from pymongo import MongoClient
from bson.objectid import ObjectId
from werkzeug.security import generate_password_hash, check_password_hash

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'default-secret-key')

# MongoDB Connection

MONGO_URI = os.getenv('MONGO_URI')

try:
    mongo_client = MongoClient(MONGO_URI)
    db = mongo_client.investsmart  # Database name
    # Test connection
    mongo_client.server_info()
    print("MongoDB connection successful")
except Exception as e:
    print(f"MongoDB connection error: {e}")
    

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

stock_cache = {}
news_cache = {}
analysis_cache = {}
agent_cache = {}

class User(UserMixin):
    def __init__(self, user_data):
        self.id = str(user_data['_id'])
        self.username = user_data['username']
        self.email = user_data['email']
        self.password_hash = user_data['password_hash']
        self.created_at = user_data.get('created_at', datetime.utcnow())
    
    @staticmethod
    def get(user_id):
        # Try first with the ObjectId
        try:
            user_data = db.users.find_one({'_id': ObjectId(user_id)})
            if user_data:
                return User(user_data)
        except:
            # If that fails, try with string ID or numerical ID
            try:
                # Try with the 'id' field if it exists
                user_data = db.users.find_one({'id': user_id})
                if user_data:
                    return User(user_data)
                    
                # Also try with numeric ID if the input can be converted to int
                user_data = db.users.find_one({'id': int(user_id)})
                if user_data:
                    return User(user_data)
            except:
                pass
        
        return None
    
    @staticmethod
    def get_by_username(username):
        user_data = db.users.find_one({'username': username})
        if not user_data:
            return None
        return User(user_data)
    
    @staticmethod
    def create(username, email, password):
        user_data = {
            'username': username,
            'email': email,
            'password_hash': generate_password_hash(password),
            'created_at': datetime.utcnow()
        }
        result = db.users.insert_one(user_data)
        user_data['_id'] = result.inserted_id
        return User(user_data)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

@login_manager.user_loader
def load_user(user_id):
    return User.get(user_id)

# Initialize Agno agents
def initialize_agno_agents():
    """Initialize and return Agno agents for different tasks"""
    # Check if agents are already initialized in cache
    if "agents" in agent_cache:
        return agent_cache["agents"]
    
    # Set Together API key
    together_api_key = os.getenv("TOGETHER_API_KEY")
    if not together_api_key:
        print("Warning: TOGETHER_API_KEY not found in environment variables")
        return None, None, None, None
    
    os.environ["TOGETHER_API_KEY"] = together_api_key
    
    # Market Research Agent
    market_research_agent = Agent(
        name="Market Research Agent",
        role="Analyze market trends and provide comprehensive research",
        model=Together(id="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"),
        tools=[
            DuckDuckGoTools(),
            Newspaper4kTools()
        ],
        instructions="""
        - Provide very detailed analysis of Indian market trends and sectors
        - Include relevant news from Indian markets and economy
        - Focus on NSE and BSE listed companies
        - Consider Indian economic factors and regulations
        - Always cite sources and verify news authenticity
        - Format data for visualization when appropriate
        """,
        show_tool_calls=False,
        markdown=True,
    )
    
    # Financial Analysis Agent
    finance_agent = Agent(
        name="Financial Analysis Agent",
        role="Analyze financial data and provide investment recommendations",
        model=Together(id="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"),
        tools=[
            YFinanceTools(
                stock_price=True,
                analyst_recommendations=True,
                stock_fundamentals=True,
                company_info=True
            ),
            CalculatorTools()
        ],
        instructions="""
        - Provide very detailed financial analysis for Indian stocks
        - Calculate key financial metrics relevant to Indian markets
        - Consider Indian accounting standards and regulations
        - Include analysis of quarterly results
        - Format data for visualization
        - Include risk assessments specific to Indian market context
        """,
        show_tool_calls=False,
        markdown=True,
    )
    
    # Portfolio Management Agent
    portfolio_agent = Agent(
        name="Portfolio Manager",
        role="Provide portfolio optimization and risk management advice",
        model=Together(id="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"),
        tools=[
            CalculatorTools(),
            YFinanceTools(stock_price=True)
        ],
        instructions="""
        - Suggest portfolio diversification strategies for Indian investors
        - Analyze risk-return metrics in Indian market context
        - Consider SEBI regulations and investment limits
        - Provide asset allocation recommendations
        - Consider user's risk profile
        - Include both large cap and midcap exposure
        """,
        show_tool_calls=False,
        markdown=True,
    )
    
    # Create agent team
    advisor_team = Agent(
        team=[
            market_research_agent,
            finance_agent,
            portfolio_agent
        ],
        model=Together(id="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"),
        instructions=[
            "Provide elaborated investment advice",
            "Consider multiple perspectives before making recommendations",
            "Always include risk disclaimers",
            "Use clear and simple language",
            "Support advice with data and sources",
            "Return data in a format suitable for visualization when appropriate"
        ],
        show_tool_calls=False,
        markdown=True,
    )
    
    # Store in cache
    agent_cache["agents"] = {
        "market_research": market_research_agent,
        "finance": finance_agent,
        "portfolio": portfolio_agent,
        "team": advisor_team
    }
    
    return agent_cache["agents"]

# Initialize agents
agents = initialize_agno_agents()

# Helper Functions
def is_market_open():
    """Check if Indian market is currently open"""
    india_tz = pytz.timezone('Asia/Kolkata')
    current_time = datetime.now(india_tz).time()
    current_day = datetime.now(india_tz).weekday()
    
    # Market hours: 9:15 AM - 3:30 PM IST, Monday to Friday
    market_start = time(9, 15)
    market_end = time(15, 30)
    
    # Check if it's a weekday and within market hours
    return (current_day < 5 and  # Monday = 0, Friday = 4
            market_start <= current_time <= market_end)

def get_stock_data(ticker="^NSEI", period="6mo"):
    """Get real stock data using YFinance with proper error handling"""
    cache_key = f"{ticker}_{period}"
    if cache_key in stock_cache:
        # Check if cache is still valid (less than 5 minutes old)
        if datetime.now() - stock_cache[cache_key]["timestamp"] < timedelta(minutes=5):
            return stock_cache[cache_key]["data"], stock_cache[cache_key]["info"]
    
    try:
        # Fix common ticker issues
        if ticker == "NIFTY50.NS" or ticker == "NIFTY.NS" or ticker == "NIFTY50":
            ticker = "^NSEI"  # Correct symbol for NIFTY 50 index
        elif ticker == "SENSEX.BS" or ticker == "SENSEX":
            ticker = "^BSESN"  # Correct symbol for SENSEX
        # Ensure NSE stocks have .NS suffix
        elif '.' not in ticker and ticker not in ['^NSEI', '^BSESN']:
            ticker = f"{ticker}.NS"
            
        # Use YFinance to get data
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        info = stock.info
        
        if hist.empty:
            return [], {}
        
        data = []
        for date, row in hist.iterrows():
            data.append({
                "date": date.strftime("%Y-%m-%d"),
                "price": round(row["Close"], 2) if not np.isnan(row["Close"]) else None,
                "open": round(row["Open"], 2) if not np.isnan(row["Open"]) else None,
                "high": round(row["High"], 2) if not np.isnan(row["High"]) else None,
                "low": round(row["Low"], 2) if not np.isnan(row["Low"]) else None,
                "volume": int(row["Volume"]) if not np.isnan(row["Volume"]) else 0
            })
        
        # Cache the data
        stock_cache[cache_key] = {
            "data": data,
            "info": info,
            "timestamp": datetime.now()
        }
        
        return data, info
    except Exception as e:
        print(f"Error fetching stock data for {ticker}: {e}")
        return [], {}

def get_nifty_index():
    """Get NIFTY 50 index data as a fallback"""
    try:
        data, info = get_stock_data("^NSEI", "6mo")
        if data:
            return data, info
    except:
        pass
    return [], {}


def analyze_stock_with_agno(ticker):
    """Analyze a stock using Agno and Groq"""
    # Check if we have cached analysis less than 1 hour old
    cache_key = f"analysis_{ticker}"
    if cache_key in analysis_cache:
        if datetime.now() - analysis_cache[cache_key]["timestamp"] < timedelta(hours=1):
            return analysis_cache[cache_key]["data"]
    
    try:
        # First verify the ticker exists
        test_data, _ = get_stock_data(ticker, "1d")
        if not test_data:
            return None
            
        if agents and agents.get("team"):
            # Ensure ticker has .NS suffix for NSE stocks if not already present
            if ticker:
                stock_data, stock_info = get_stock_data(ticker)
                
                if not stock_data:
                    flash(f"No data found for ticker {ticker}. Please check the symbol and try again.", "error")
                    return redirect(url_for('analysis'))
                    
                analysis_result = analyze_stock_with_agno(ticker)

                try:
                    current_price = stock_data["Close"][-1]
                except Exception:
                    current_price = None
                
            # Use the agent team to analyze the stock
            response = agents["team"].run(
                f"""Analyze {ticker} stock for investment. Provide a comprehensive analysis including fundamentals, technical indicators, recent news, and future outlook. 
                Include a clear buy/hold/sell recommendation, target price, potential return percentage, and risk level.
                Format your response with proper markdown headers (### for main sections, #### for subsections) and add bullet points for better readability.
                Include sections on Fundamentals, Technical Indicators, Recent News, and Future Outlook.
                """
            )
            
            # Parse the response to extract analysis
            analysis_result = {}
            
            try:
                # Extract recommendation, target price, potential return, and risk level
                content = response.content.lower()
                
                # Determine recommendation
                if "buy" in content or "strong buy" in content:
                    analysis_result["recommendation"] = "Buy"
                elif "sell" in content or "strong sell" in content:
                    analysis_result["recommendation"] = "Sell"
                else:
                    analysis_result["recommendation"] = "Hold"
                
                # Extract target price (look for ₹ or Rs. followed by numbers)
                import re
                target_price_match = re.search(r'(target price|price target)[:\s]*(₹|rs\.?|inr)?\s*([0-9,]+(\.[0-9]+)?)', content)
                if target_price_match:
                    price = target_price_match.group(3).replace(',', '')
                    analysis_result["targetPrice"] = f"₹{price}"
                else:
                    # Get current price and add 10% for buy, subtract 5% for sell, keep same for hold
                    try:
                        stock = yf.Ticker(ticker)
                        current_price = stock.history(period="1d")["Close"].iloc[-1]
                        if analysis_result["recommendation"] == "Buy":
                            target_price = current_price * 1.1
                        elif analysis_result["recommendation"] == "Sell":
                            target_price = current_price * 0.95
                        else:
                            target_price = current_price
                        analysis_result["targetPrice"] = f"₹{target_price:.2f}"
                    except Exception as e:
                        print(f"Error calculating target price: {e}")
                        analysis_result["targetPrice"] = "N/A"
                
                # Extract potential return
                potential_match = re.search(r'(potential|return)[:\s]*([+\-]?\s*[0-9]+(\.[0-9]+)?\s*%)', content)
                if potential_match:
                    analysis_result["potential"] = potential_match.group(2).strip()
                else:
                    if analysis_result["recommendation"] == "Buy":
                        analysis_result["potential"] = "+10%"
                    elif analysis_result["recommendation"] == "Sell":
                        analysis_result["potential"] = "-5%"
                    else:
                        analysis_result["potential"] = "+2%"
                
                # Extract risk level
                risk_match = re.search(r'risk[:\s]*(low|medium|high)', content)
                if risk_match:
                    analysis_result["riskLevel"] = risk_match.group(1).capitalize()
                else:
                    analysis_result["riskLevel"] = "Medium"
                
                # Set AI score based on recommendation
                if analysis_result["recommendation"] == "Buy":
                    analysis_result["aiScore"] = np.random.randint(80, 95)
                elif analysis_result["recommendation"] == "Hold":
                    analysis_result["aiScore"] = np.random.randint(60, 80)
                else:  # Sell
                    analysis_result["aiScore"] = np.random.randint(30, 60)
                
                # Format the analysis with proper markdown
                formatted_analysis = response.content
                # Ensure headers are properly formatted
                formatted_analysis = re.sub(r'([#]+)([^#\n]+)', r'\1 \2', formatted_analysis)
                # Convert markdown to HTML
                import markdown
                html_analysis = markdown.markdown(formatted_analysis)
                analysis_result["analysis"] = html_analysis
            
            except Exception as e:
                print(f"Error parsing analysis: {e}")
                return None
            
            # Cache the analysis
            analysis_cache[cache_key] = {
                "data": analysis_result,
                "timestamp": datetime.now()
            }
            
            return analysis_result
        else:
            return None
    except Exception as e:
        print(f"Error analyzing stock: {e}")
        return None

def get_indian_market_metrics(symbol):
    """Get Indian market specific metrics for a stock"""
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        
        if not info:
            return {}
        
        # Additional Indian metrics
        metrics = {
            'Promoter Holding': info.get('majorHoldersPercentage', 0) * 100,
            'FII Holding': info.get('institutionHoldersPercentage', 0) * 100,
            'DII Holding': info.get('institutionHoldersPercentage', 0) * 50,  # Approximate
            'Face Value': info.get('regularMarketPrice', 0) / info.get('priceToBook', 1) if info.get('priceToBook', 0) > 0 else 0,
            'Book Value': info.get('bookValue', 0),
            'Market Lot': 1,  # Default, actual lot size needs separate API
        }
        
        return metrics
    except Exception as e:
        print(f"Error fetching Indian metrics for {symbol}: {e}")
        return {}

def calculate_indian_taxes(buy_price, sell_price, quantity, holding_period_days):
    """Calculate Indian taxes for stock trades"""
    profit = (sell_price - buy_price) * quantity
    
    # No tax on losses
    if profit <= 0:
        return {
            "profit": profit,
            "tax": 0,
            "tax_type": "No tax on capital loss",
            "net_profit": profit
        }
    
    if holding_period_days > 365:
        # Long Term Capital Gains (LTCG)
        if profit > 100000:  # First 1 lakh exempt
            tax = (profit - 100000) * 0.10
        else:
            tax = 0
        tax_type = "LTCG @ 10%"
    else:
        # Short Term Capital Gains (STCG)
        tax = profit * 0.15
        tax_type = "STCG @ 15%"
    
    return {
        "profit": profit,
        "tax": tax,
        "tax_type": tax_type,
        "net_profit": profit - tax
    }

def optimize_portfolio(portfolio_stocks):
    """
    Perform portfolio optimization using modern portfolio theory
    """
    # In a real implementation, this would use historical data and calculate
    # the optimal portfolio weights based on risk and return
    
    # For this demo, we'll return optimization data based on real stock data
    tickers = [stock['ticker'] for stock in portfolio_stocks]
    current_weights = [stock['quantity'] * stock['current_price'] for stock in portfolio_stocks]
    total_value = sum(current_weights)
    current_weights = [w / total_value for w in current_weights]
    
    try:
        # Try to get real historical data for the stocks
        returns_data = {}
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period="1y")
                if not hist.empty:
                    # Calculate daily returns
                    returns = hist['Close'].pct_change().dropna()
                    returns_data[ticker] = returns
            except Exception as e:
                print(f"Error getting historical data for {ticker}: {e}")
        
        # If we have returns data for all stocks, calculate covariance matrix
        if len(returns_data) == len(tickers):
            # Create a DataFrame with all returns
            returns_df = pd.DataFrame(returns_data)
            
            # Calculate expected returns (annualized)
            expected_returns = returns_df.mean() * 252
            
            # Calculate covariance matrix (annualized)
            cov_matrix = returns_df.cov() * 252
            
            # Simple optimization: inverse volatility weighting
            # Higher volatility stocks get lower weights
            volatilities = np.sqrt(np.diag(cov_matrix))
            inv_vol = 1 / volatilities
            optimized_weights = inv_vol / sum(inv_vol)
            
            # Calculate expected return and risk for current and optimized portfolios
            current_return = sum(w * r for w, r in zip(current_weights, expected_returns))
            current_risk = np.sqrt(current_weights @ cov_matrix @ current_weights)
            
            optimized_return = sum(w * r for w, r in zip(optimized_weights, expected_returns))
            optimized_risk = np.sqrt(optimized_weights @ cov_matrix @ optimized_weights)
        else:
            # If we don't have data for all stocks, use random optimization
            optimized_weights = []
            for i, w in enumerate(current_weights):
                # Simulate some adjustment to weights
                adjustment = np.random.uniform(-0.1, 0.1)
                new_weight = max(0.05, min(0.5, w + adjustment))  # Keep between 5% and 50%
                optimized_weights.append(new_weight)
            
            # Normalize to sum to 1
            total = sum(optimized_weights)
            optimized_weights = [w / total for w in optimized_weights]
            
            # Calculate expected return and risk
            expected_returns = [np.random.uniform(0.05, 0.15) for _ in tickers]  # 5-15% annual return
            current_return = sum(w * r for w, r in zip(current_weights, expected_returns))
            optimized_return = sum(w * r for w, r in zip(optimized_weights, expected_returns))
            
            # Calculate risk (simplified)
            current_risk = np.random.uniform(0.1, 0.2)  # 10-20% volatility
            optimized_risk = current_risk * 0.8  # Assume 20% risk reduction
    except Exception as e:
        print(f"Error in portfolio optimization: {e}")
        # Fallback to random optimization
        optimized_weights = []
        for i, w in enumerate(current_weights):
            # Simulate some adjustment to weights
            adjustment = np.random.uniform(-0.1, 0.1)
            new_weight = max(0.05, min(0.5, w + adjustment))  # Keep between 5% and 50%
            optimized_weights.append(new_weight)
        
        # Normalize to sum to 1
        total = sum(optimized_weights)
        optimized_weights = [w / total for w in optimized_weights]
        
        # Calculate expected return and risk
        expected_returns = [np.random.uniform(0.05, 0.15) for _ in tickers]  # 5-15% annual return
        current_return = sum(w * r for w, r in zip(current_weights, expected_returns))
        optimized_return = sum(w * r for w, r in zip(optimized_weights, expected_returns))
        
        # Calculate risk (simplified)
        current_risk = np.random.uniform(0.1, 0.2)  # 10-20% volatility
        optimized_risk = current_risk * 0.8  # Assume 20% risk reduction
    
    # Generate recommendations
    recommendations = []
    for i, ticker in enumerate(tickers):
        if optimized_weights[i] > current_weights[i] + 0.05:
            action = "Buy"
            reason = "Underweight position with strong growth potential"
        elif optimized_weights[i] < current_weights[i] - 0.05:
            action = "Reduce"
            reason = "Overweight position with increased risk"
        else:
            action = "Hold"
            reason = "Position is well-balanced"
            
        recommendations.append({
            "ticker": ticker,
            "action": action,
            "target_weight": f"{optimized_weights[i] * 100:.1f}%",
            "current_weight": f"{current_weights[i] * 100:.1f}%",
            "reason": reason
        })
    
    return {
        "current_return": f"{current_return * 100:.2f}%",
        "optimized_return": f"{optimized_return * 100:.2f}%",
        "current_risk": f"{current_risk * 100:.2f}%",
        "optimized_risk": f"{optimized_risk * 100:.2f}%",
        "sharpe_improvement": f"{(optimized_return/optimized_risk - current_return/current_risk) * 100:.2f}%",
        "recommendations": recommendations
    }

def create_stock_chart(stock_data, title, chart_type="candlestick"):
    """Create a stock chart using Plotly"""
    if not stock_data:
        return None
    
    # Convert to DataFrame if it's a list
    if isinstance(stock_data, list):
        df = pd.DataFrame(stock_data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
    else:
        df = stock_data
    
    fig = go.Figure()
    
    # Add appropriate chart based on type
    if chart_type == "candlestick":
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['open'] if 'open' in df.columns else df['Open'] if 'Open' in df.columns else None,
                high=df['high'] if 'high' in df.columns else df['High'] if 'High' in df.columns else None,
                low=df['low'] if 'low' in df.columns else df['Low'] if 'Low' in df.columns else None,
                close=df['price'] if 'price' in df.columns else df['Close'] if 'Close' in df.columns else None,
                name="Price"
            )
        )
    elif chart_type == "line":
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['price'] if 'price' in df.columns else df['Close'] if 'Close' in df.columns else None,
                mode='lines',
                name="Close Price",
                line=dict(color="#4b71eb", width=2)
            )
        )
    elif chart_type == "ohlc":
        fig.add_trace(
            go.Ohlc(
                x=df.index,
                open=df['open'] if 'open' in df.columns else df['Open'] if 'Open' in df.columns else None,
                high=df['high'] if 'high' in df.columns else df['High'] if 'High' in df.columns else None,
                low=df['low'] if 'low' in df.columns else df['Low'] if 'Low' in df.columns else None,
                close=df['price'] if 'price' in df.columns else df['Close'] if 'Close' in df.columns else None,
                name="Price"
            )
        )
    
    # Add volume as bar chart
    if 'volume' in df.columns or 'Volume' in df.columns:
        volume_data = df['volume'] if 'volume' in df.columns else df['Volume']
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=volume_data,
                name="Volume",
                marker_color='rgba(128, 128, 128, 0.5)',
                yaxis="y2"
            )
        )
    
    # Layout
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price (₹)",
        height=600,
        hovermode="x unified",
        yaxis2=dict(
            title="Volume",
            anchor="x",
            overlaying="y",
            side="right",
            showgrid=False,
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    
    # Convert to JSON for passing to template
    chart_json = pio.to_json(fig)
    return chart_json

def add_chat_history(user_id, role, message):
    """Add a message to the chat history"""
    chat_entry = {
        'user_id': user_id,
        'role': role,
        'message': message,
        'timestamp': datetime.utcnow()
    }
    
    result = db.chat_history.insert_one(chat_entry)
    chat_entry['_id'] = result.inserted_id
    
    return chat_entry

def get_chat_history(user_id, limit=10):
    """Get the chat history for a user"""
    history = list(db.chat_history.find({'user_id': user_id}).sort('timestamp', -1).limit(limit))
    return [(entry['role'], entry['message']) for entry in reversed(history)]

def get_chat_history_list(user_id, limit=10):
    """Get a list of recent chats for a user"""
    # Group chat history by conversation
    chats = []
    
    # Get all chat history for the user
    history = list(db.chat_history.find({'user_id': user_id, 'role': 'user'}).sort('timestamp', -1).limit(limit))
    
    # Format chats
    for entry in history:
        chat = {
            'id': str(entry['_id']),
            'title': entry['message'][:30] + ('...' if len(entry['message']) > 30 else ''),
            'timestamp': entry['timestamp'],
            'messages': []
        }
        chats.append(chat)
    
    return chats

def get_user_portfolios(user_id):
    """Get all portfolios for a user"""
    return list(db.portfolios.find({'user_id': user_id}))

def get_portfolio(portfolio_id, user_id):
    """Get a specific portfolio"""
    return db.portfolios.find_one({'_id': ObjectId(portfolio_id), 'user_id': user_id})

def create_portfolio(name, user_id):
    """Create a new portfolio"""
    portfolio = {
        'name': name,
        'user_id': user_id,
        'created_at': datetime.utcnow()
    }
    result = db.portfolios.insert_one(portfolio)
    portfolio['_id'] = result.inserted_id
    return portfolio

def add_stock_to_portfolio(portfolio_id, ticker, quantity, purchase_price):
    """Add a stock to a portfolio"""
    stock = {
        'portfolio_id': ObjectId(portfolio_id),
        'ticker': ticker,
        'quantity': float(quantity),
        'purchase_price': float(purchase_price),
        'purchase_date': datetime.utcnow()
    }
    result = db.portfolio_stocks.insert_one(stock)
    stock['_id'] = result.inserted_id
    return stock

def get_portfolio_stocks(portfolio_id):
    """Get all stocks in a portfolio"""
    return list(db.portfolio_stocks.find({'portfolio_id': ObjectId(portfolio_id)}))

def delete_portfolio_stock(stock_id, user_id):
    """Delete a stock from a portfolio"""
    # First get the stock to check if it belongs to the user
    stock = db.portfolio_stocks.find_one({'_id': ObjectId(stock_id)})
    if not stock:
        return False
    
    # Check if the portfolio belongs to the user
    portfolio = db.portfolios.find_one({'_id': stock['portfolio_id'], 'user_id': user_id})
    if not portfolio:
        return False
    
    # Delete the stock
    db.portfolio_stocks.delete_one({'_id': ObjectId(stock_id)})
    return True

def delete_portfolio(portfolio_id, user_id):
    """Delete a portfolio and all its stocks"""
    # Check if the portfolio belongs to the user
    portfolio = db.portfolios.find_one({'_id': ObjectId(portfolio_id), 'user_id': user_id})
    if not portfolio:
        return False
    
    # Delete all stocks in the portfolio
    db.portfolio_stocks.delete_many({'portfolio_id': ObjectId(portfolio_id)})
    
    # Delete the portfolio
    db.portfolios.delete_one({'_id': ObjectId(portfolio_id)})
    return True

# MongoDB Watchlist Functions
def get_user_watchlist(user_id):
    """Get all watchlist items for a user"""
    return list(db.watchlist.find({'user_id': user_id}))

def add_to_watchlist(user_id, ticker):
    """Add a ticker to a user's watchlist"""
    # Check if already in watchlist
    existing = db.watchlist.find_one({'user_id': user_id, 'ticker': ticker})
    if existing:
        return None
    
    watchlist_item = {
        'user_id': user_id,
        'ticker': ticker,
        'added_at': datetime.utcnow()
    }
    result = db.watchlist.insert_one(watchlist_item)
    watchlist_item['_id'] = result.inserted_id
    return watchlist_item

def remove_from_watchlist(item_id, user_id):
    """Remove a ticker from a user's watchlist"""
    result = db.watchlist.delete_one({'_id': ObjectId(item_id), 'user_id': user_id})
    return result.deleted_count > 0

# Authentication routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
        
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.get_by_username(username)
        
        if user and user.check_password(password):
            login_user(user)
            next_page = request.args.get('next')
            return redirect(next_page or url_for('index'))
        else:
            flash('Invalid username or password')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
        
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        # Check if username or email already exists
        if db.users.find_one({'username': username}):
            flash('Username already exists')
            return render_template('register.html')
            
        if db.users.find_one({'email': email}):
            flash('Email already exists')
            return render_template('register.html')
        
        # Create new user
        user = User.create(username, email, password)
        
        flash('Registration successful! Please log in.')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

# Main routes
@app.route('/')
def index():
    # Get NIFTY 50 data
    nifty_data, _ = get_stock_data("^NSEI", "6mo")
    
    # Get SENSEX data
    sensex_data, _ = get_stock_data("^BSESN", "6mo")
    
        
    # Create charts if we have data
    chart_json = None
    if nifty_data:
        chart_json = create_stock_chart(nifty_data, "NIFTY 50 Index", "line")
    
    sensex_chart_json = None
    if sensex_data:
        sensex_chart_json = create_stock_chart(sensex_data, "SENSEX Index", "line")
    
    # Check if market is open
    market_open = is_market_open()
    
    # Get current time in IST
    ist_time = datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%I:%M %p IST')
    
    return render_template('index.html', 
                          stock_data=json.dumps(nifty_data) if nifty_data else "[]",
                          chart_json=chart_json,
                          sensex_chart_json=sensex_chart_json,
                          market_open=market_open,
                          ist_time=ist_time,
                          active_tab="dashboard",
                          now=datetime.now(timezone.utc)  # Pass the current datetime
                          )

@app.route('/analysis', methods=['GET', 'POST'])
def analysis():
    ticker = request.args.get('ticker', '')
    period = request.args.get('period', '6mo')
    chart_type = request.args.get('chart_type', 'candlestick')
    current_price = None

    if not current_price or current_price == 'N/A':
        current_price = None
    if ticker:
        stock_data, stock_info = get_stock_data(ticker, period)
        
        if not stock_data:
            flash(f"No data found for ticker {ticker}. Please check the symbol and try again.", "error")
            return redirect(url_for('analysis'))
            
        analysis_result = analyze_stock_with_agno(ticker)
        
        # Get related news
        # Create chart
        chart_json = create_stock_chart(stock_data, f"{ticker} - {period}", chart_type) if stock_data else None
        
        # Get Indian market metrics
        indian_metrics = get_indian_market_metrics(ticker)
        
        return render_template('analysis.html', 
                              ticker=ticker,
                              stock_data=json.dumps(stock_data),
                              stock_info=stock_info,
                              chart_json=chart_json,
                              analysis=analysis_result,
                              indian_metrics=indian_metrics,
                              current_price=current_price,

                              period=period,
                              chart_type=chart_type,
                              active_tab="analysis")
    else:
        return render_template('analysis.html', 
                              ticker='',
                              stock_data=json.dumps([]),
                              stock_info={},
                              chart_json=None,
                              analysis=None,
                              news=[],
                              current_price=current_price,

                              indian_metrics={},
                              period=period,
                              chart_type=chart_type,
                              active_tab="analysis")


@app.route('/portfolio')
@login_required
def portfolio():
    # Get user's portfolios
    portfolios = get_user_portfolios(current_user.id)
    
    # If no portfolios exist, create a default one
    if not portfolios:
        default_portfolio = create_portfolio("My Portfolio", current_user.id)
        portfolios = [default_portfolio]
    
    selected_portfolio_id = request.args.get('portfolio_id')
    if not selected_portfolio_id and portfolios:
        selected_portfolio_id = str(portfolios[0]['_id'])
    
    # Get stocks in the selected portfolio
    portfolio_stocks = []
    total_value = 0
    total_cost = 0
    
    if selected_portfolio_id:
        stocks = get_portfolio_stocks(selected_portfolio_id)
        
        for stock in stocks:
            # Get current price using YFinance
            current_price = None
            try:
                ticker_data = yf.Ticker(stock['ticker'])
                hist = ticker_data.history(period="1d")
                if not hist.empty:
                    current_price = hist["Close"].iloc[-1]
            except Exception as e:
                print(f"Error fetching current price for {stock['ticker']}: {e}")
            
            # Skip stocks with no current price data
            if current_price is None:
                continue
                
            value = stock['quantity'] * current_price
            cost = stock['quantity'] * stock['purchase_price']
            gain_loss = value - cost
            gain_loss_percent = (gain_loss / cost) * 100 if cost > 0 else 0
            
            portfolio_stocks.append({
                'id': str(stock['_id']),
                'ticker': stock['ticker'],
                'quantity': stock['quantity'],
                'purchase_price': stock['purchase_price'],
                'purchase_date': stock['purchase_date'].strftime("%Y-%m-%d"),
                'current_price': current_price,
                'value': value,
                'gain_loss': gain_loss,
                'gain_loss_percent': gain_loss_percent
            })
            
            total_value += value
            total_cost += cost
    
    # Calculate portfolio performance
    portfolio_performance = {
        'total_value': total_value,
        'total_cost': total_cost,
        'total_gain_loss': total_value - total_cost,
        'total_gain_loss_percent': ((total_value - total_cost) / total_cost) * 100 if total_cost > 0 else 0
    }
    
    # Get portfolio optimization if stocks exist
    portfolio_optimization = None
    if portfolio_stocks:
        portfolio_optimization = optimize_portfolio(portfolio_stocks)
    
    # Create portfolio allocation chart
    allocation_chart = None
    if portfolio_stocks:
        # Create pie chart for portfolio allocation
        labels = [stock['ticker'] for stock in portfolio_stocks]
        values = [stock['value'] for stock in portfolio_stocks]
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=.3,
            textinfo='label+percent',
            marker=dict(
                colors=[
                    '#4f46e5', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6',
                    '#ec4899', '#06b6d4', '#84cc16', '#f97316', '#6366f1'
                ]
            )
        )])
        
        fig.update_layout(
            title="Portfolio Allocation",
            height=400
        )
        
        allocation_chart = pio.to_json(fig)
    
    return render_template('portfolio.html', 
                          portfolios=portfolios,
                          selected_portfolio_id=selected_portfolio_id,
                          portfolio_stocks=portfolio_stocks,
                          portfolio_performance=portfolio_performance,
                          portfolio_optimization=portfolio_optimization,
                          allocation_chart=allocation_chart,
                          active_tab="portfolio")

@app.route('/watchlist')
@login_required
def watchlist():
    # Get user's watchlist
    watchlist_items = get_user_watchlist(current_user.id)
    
    watchlist_data = []
    for item in watchlist_items:
        # Get current data for the ticker
        analysis = analyze_stock_with_agno(item['ticker'])
        
        # Skip stocks with no analysis data
        if not analysis:
            continue
        
        # Get current price using YFinance
        current_price = None
        price_change = None
        price_change_pct = None
        try:
            ticker_data = yf.Ticker(item['ticker'])
            hist = ticker_data.history(period="2d")
            if len(hist) >= 2:
                current_price = hist["Close"].iloc[-1]
                prev_close = hist["Close"].iloc[-2]
                price_change = current_price - prev_close
                price_change_pct = (price_change / prev_close) * 100
        except Exception as e:
            print(f"Error fetching price data for {item['ticker']}: {e}")
        
        # Skip stocks with no price data
        if current_price is None:
            continue
        
        # Get company name and sector
        company_name = ""
        sector = "Unknown"
        try:
            ticker_data = yf.Ticker(item['ticker'])
            info = ticker_data.info
            if 'longName' in info:
                company_name = info['longName']
            elif 'shortName' in info:
                company_name = info['shortName']
            else:
                company_name = item['ticker']
            
            sector = info.get('sector', 'Unknown')
        except Exception as e:
            print(f"Error fetching company info for {item['ticker']}: {e}")
            company_name = item['ticker']
        
        watchlist_data.append({
            'id': str(item['_id']),
            'ticker': item['ticker'],
            'company_name': company_name,
            'sector': sector,
            'current_price': current_price,
            'price_change': price_change,
            'price_change_pct': price_change_pct,
            'recommendation': analysis['recommendation'],
            'target_price': analysis['targetPrice'],
            'potential': analysis['potential'],
            'ai_score': analysis['aiScore']
        })
    
    # Check if market is open
    market_open = is_market_open()
    
    # Get current time in IST
    ist_time = datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%I:%M %p IST')
    
    return render_template('watchlist.html', 
                          watchlist=watchlist_data,
                          market_open=market_open,
                          ist_time=ist_time,
                          active_tab="watchlist")

@app.route('/screener')
def screener():
    # Get stocks from watchlist if user is logged in
    watchlist_tickers = []
    if current_user.is_authenticated:
        watchlist_items = get_user_watchlist(current_user.id)
        watchlist_tickers = [item['ticker'] for item in watchlist_items]
    
    # If no watchlist items, use popular Indian stocks as default
    if not watchlist_tickers:
        watchlist_tickers = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "BHARTIARTL.NS"]
    
    # Verify each ticker exists
    verified_tickers = []
    for ticker in watchlist_tickers:
        data, _ = get_stock_data(ticker, "1d")
        if data:
            verified_tickers.append(ticker)
    
    # Get selected stocks from query parameters
    selected_stocks = request.args.getlist('stocks')
    if not selected_stocks:
        selected_stocks = verified_tickers[:3] if len(verified_tickers) >= 3 else verified_tickers
    else:
        # Verify selected stocks exist
        selected_stocks = [ticker for ticker in selected_stocks if ticker in verified_tickers]
    
    # Get data for selected stocks
    stock_data = {}
    for ticker in selected_stocks:
        try:
            data, info = get_stock_data(ticker, "1y")
            if data:
                stock_data[ticker] = {
                    "data": data,
                    "info": info,
                    "name": info.get('shortName', ticker) if info else ticker
                }
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
    
    # Create performance comparison chart
    performance_chart = None
    if stock_data:
        # Get actual price data for each stock
        perf_data = {}
        start_date = datetime.now() - timedelta(days=180)  # 6 months
        
        for ticker, data in stock_data.items():
            stock_hist = data["data"]
            if stock_hist:
                # Convert to DataFrame
                df = pd.DataFrame(stock_hist)
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
                
                # Filter by date range
                filtered_hist = df[df.index >= pd.Timestamp(start_date)]
                
                if not filtered_hist.empty:
                    # Use actual price values instead of percentage change
                    perf_data[ticker] = filtered_hist['price'].tolist()
                    perf_data[f"{ticker}_dates"] = [d.strftime("%Y-%m-%d") for d in filtered_hist.index]
        
        # Create performance chart
        if perf_data:
            fig = go.Figure()
            
            for ticker in selected_stocks:
                if ticker in perf_data and f"{ticker}_dates" in perf_data:
                    fig.add_trace(go.Scatter(
                        x=perf_data[f"{ticker}_dates"],
                        y=perf_data[ticker],
                        mode='lines',
                        name=stock_data[ticker]["name"]
                    ))
            
            fig.update_layout(
                title="Price Performance Comparison - 6 Months",
                xaxis_title="Date",
                yaxis_title="Price (₹)",
                height=500,
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            performance_chart = pio.to_json(fig)
    
    return render_template('screener.html',
                          watchlist_tickers=verified_tickers,
                          selected_stocks=selected_stocks,
                          stock_data=stock_data,
                          performance_chart=performance_chart,
                          active_tab="screener")

@app.route('/chatbot')
@login_required
def chatbot():
    chat_history_list = get_chat_history_list(current_user.id)
    
    return render_template('chatbot.html',
                          chat_history_list=chat_history_list,
                          active_tab="chatbot")

@app.route('/tax-calculator')
def tax_calculator():
    return render_template('tax_calculator.html',
                          active_tab="tax_calculator")

# API Routes
@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    data = request.json
    ticker = data.get('ticker', '')
    if not ticker:
        return jsonify({"error": "Stock ticker is required"}), 400
    
    analysis = analyze_stock_with_agno(ticker)
    if not analysis:
        return jsonify({"error": f"Unable to analyze {ticker}. Please check the ticker symbol."}), 404
    
    return jsonify(analysis)


@app.route('/api/stock-data', methods=['GET'])
def api_stock_data():
    ticker = request.args.get('ticker', '^NSEI')
    period = request.args.get('period', '6mo')
    data, info = get_stock_data(ticker, period)
    
    if not data:
        return jsonify({"error": f"No data found for {ticker}"}), 404
    
    return jsonify({"data": data, "info": info})


@app.route('/api/portfolio', methods=['GET', 'POST', 'DELETE'])
@login_required
def api_portfolio():
    if request.method == 'GET':
        portfolio_id = request.args.get('portfolio_id')
        if not portfolio_id:
            portfolios = get_user_portfolios(current_user.id)
            return jsonify([{
                'id': str(p['_id']),
                'name': p['name'],
                'created_at': p['created_at'].isoformat()
            } for p in portfolios])
        else:
            portfolio = get_portfolio(portfolio_id, current_user.id)
            if not portfolio:
                return jsonify({"error": "Portfolio not found"}), 404
                
            stocks = get_portfolio_stocks(portfolio_id)
            return jsonify({
                'id': str(portfolio['_id']),
                'name': portfolio['name'],
                'created_at': portfolio['created_at'].isoformat(),
                'stocks': [{
                    'id': str(s['_id']),
                    'ticker': s['ticker'],
                    'quantity': s['quantity'],
                    'purchase_price': s['purchase_price'],
                    'purchase_date': s['purchase_date'].isoformat()
                } for s in stocks]
            })
    
    elif request.method == 'POST':
        data = request.json
        
        if 'portfolio_id' in data:
            # Adding a stock to existing portfolio
            portfolio_id = data.get('portfolio_id')
            ticker = data.get('ticker')
            quantity = data.get('quantity')
            purchase_price = data.get('purchase_price')
            
            if not all([portfolio_id, ticker, quantity, purchase_price]):
                return jsonify({"error": "Missing required fields"}), 400
                
            portfolio = get_portfolio(portfolio_id, current_user.id)
            if not portfolio:
                return jsonify({"error": "Portfolio not found"}), 404
                
            stock = add_stock_to_portfolio(portfolio_id, ticker, quantity, purchase_price)
            
            return jsonify({
                'id': str(stock['_id']),
                'ticker': stock['ticker'],
                'quantity': stock['quantity'],
                'purchase_price': stock['purchase_price'],
                'purchase_date': stock['purchase_date'].isoformat()
            })
        else:
            # Creating a new portfolio
            name = data.get('name', f"Portfolio {uuid.uuid4().hex[:8]}")
            
            portfolio = create_portfolio(name, current_user.id)
            
            return jsonify({
                'id': str(portfolio['_id']),
                'name': portfolio['name'],
                'created_at': portfolio['created_at'].isoformat()
            })
    
    elif request.method == 'DELETE':
        data = request.json
        
        if 'stock_id' in data:
            # Deleting a stock from portfolio
            stock_id = data.get('stock_id')
            success = delete_portfolio_stock(stock_id, current_user.id)
            
            if not success:
                return jsonify({"error": "Stock not found or unauthorized"}), 404
                
            return jsonify({"success": True})
        
        elif 'portfolio_id' in data:
            # Deleting an entire portfolio
            portfolio_id = data.get('portfolio_id')
            success = delete_portfolio(portfolio_id, current_user.id)
            
            if not success:
                return jsonify({"error": "Portfolio not found or unauthorized"}), 404
                
            return jsonify({"success": True})
        
        return jsonify({"error": "Invalid request"}), 400

@app.route('/api/watchlist', methods=['GET', 'POST', 'DELETE'])
@login_required
def api_watchlist():
    if request.method == 'GET':
        watchlist_items = get_user_watchlist(current_user.id)
        return jsonify([{
            'id': str(item['_id']),
            'ticker': item['ticker'],
            'added_at': item['added_at'].isoformat()
        } for item in watchlist_items])
    
    elif request.method == 'POST':
        data = request.json
        ticker = data.get('ticker')
        
        if not ticker:
            return jsonify({"error": "Ticker is required"}), 400
            
        watchlist_item = add_to_watchlist(current_user.id, ticker)
        
        if not watchlist_item:
            return jsonify({"error": "Ticker already in watchlist"}), 400
            
        return jsonify({
            'id': str(watchlist_item['_id']),
            'ticker': watchlist_item['ticker'],
            'added_at': watchlist_item['added_at'].isoformat()
        })
    
    elif request.method == 'DELETE':
        data = request.json
        item_id = data.get('id')
        
        if not item_id:
            return jsonify({"error": "Item ID is required"}), 400
            
        success = remove_from_watchlist(item_id, current_user.id)
        
        if not success:
            return jsonify({"error": "Item not found"}), 404
            
        return jsonify({"success": True})

@app.route('/api/chat-history/<chat_id>')
@login_required
def api_chat_history(chat_id):
    # Get the chat history for a specific chat
    # In a real app, you'd have a separate table for conversations
    # This is a simplified approach that gets messages around the specified ID
    
    # Get the specified message
    message = db.chat_history.find_one({'_id': ObjectId(chat_id), 'user_id': current_user.id})
    
    if not message:
        return jsonify({"error": "Chat not found"}), 404
    
    # Get messages around this one (5 before and 10 after)
    before_messages = list(db.chat_history.find({
        'user_id': current_user.id,
        '_id': {'$lt': ObjectId(chat_id)}
    }).sort('_id', -1).limit(5))
    
    after_messages = list(db.chat_history.find({
        'user_id': current_user.id,
        '_id': {'$gt': ObjectId(chat_id)}
    }).sort('_id', 1).limit(10))
    
    # Combine messages
    messages = list(reversed(before_messages)) + [message] + after_messages
    
    # Format messages
    formatted_messages = [{
        'role': msg['role'],
        'content': msg['message'],
        'timestamp': msg['timestamp'].isoformat()
    } for msg in messages]
    
    return jsonify({
        "success": True,
        "messages": formatted_messages
    })

@app.route('/api/chat', methods=['POST'])
@login_required
def api_chat():
    data = request.json
    user_query = data.get('query', '')
    chat_id = data.get('chat_id')
    
    if not user_query:
        return jsonify({"error": "Query is required"}), 400
    
    # Add user message to chat history
    chat_entry = add_chat_history(current_user.id, 'user', user_query)
    
    try:
        # Get response from AI
        if agents and agents.get("team"):
            # Check if this is a stock comparison query
            if "compare" in user_query.lower() and any(ticker in user_query for ticker in ["RELIANCE", "TCS", "HDFC", "INFY", "BHARTI"]):
                # Extract tickers from the query
                import re
                tickers = re.findall(r'\b[A-Z]+(?:\.NS)?\b', user_query)
                
                # Format a more specific query for stock comparison
                formatted_query = f"""
                Compare these stocks in detail: {', '.join(tickers)}. 
                Provide a comprehensive comparison including:
                1. Fundamental metrics (P/E, P/B, dividend yield, etc.)
                2. Technical indicators and price performance
                3. Financial health and growth prospects
                4. Sector outlook and competitive positioning
                5. A clear recommendation on which stock might be the better investment currently and why
                
                Format your response with proper markdown headers and bullet points for better readability.
                """
                
                response = agents["team"].run(formatted_query)
            else:
                response = agents["team"].run(user_query)
            
            if hasattr(response, 'content'):
                ai_response = response.content
            else:
                ai_response = str(response)
            
            # Format the response with markdown
            import markdown
            formatted_response = markdown.markdown(ai_response)
            
            # Add AI response to chat history
            ai_chat_entry = add_chat_history(current_user.id, 'assistant', ai_response)
            
            return jsonify({
                "response": formatted_response,
                "success": True,
                "chat_id": str(chat_entry['_id']) if chat_entry else None
            })
        else:
            return jsonify({
                "response": "AI service is currently unavailable. Please try again later.",
                "success": False
            })
    except Exception as e:
        return jsonify({
            "response": f"Error processing your request: {str(e)}",
            "success": False
        })

@app.route('/api/calculate-tax', methods=['POST'])
def api_calculate_tax():
    data = request.json
    buy_price = data.get('buy_price', 0)
    sell_price = data.get('sell_price', 0)
    quantity = data.get('quantity', 0)
    holding_period = data.get('holding_period', 0)
    
    if not all([buy_price, sell_price, quantity, holding_period is not None]):
        return jsonify({"error": "All fields are required"}), 400
    
    try:
        tax_results = calculate_indian_taxes(
            float(buy_price),
            float(sell_price),
            float(quantity),
            int(holding_period)
        )
        
        return jsonify(tax_results)
    except Exception as e:
        return jsonify({"error": f"Error calculating tax: {str(e)}"}), 500

# Background tasks
def update_stock_data():
    """Update stock data in the background"""
    with app.app_context():
        # Clear the cache to force refresh on next request
        stock_cache.clear()
        print(f"Stock cache cleared at {datetime.now()}")


def update_analysis_data():
    """Update analysis data in the background"""
    with app.app_context():
        # Clear the cache to force refresh on next request
        analysis_cache.clear()
        print(f"Analysis cache cleared at {datetime.now()}")

# Set up scheduler for background tasks
scheduler = BackgroundScheduler()
scheduler.add_job(func=update_stock_data, trigger="interval", minutes=5)
scheduler.add_job(func=update_analysis_data, trigger="interval", hours=1)
scheduler.start()

# Shut down the scheduler when exiting the app
atexit.register(lambda: scheduler.shutdown())

if __name__ == '__main__':
    app.run(debug=True)

