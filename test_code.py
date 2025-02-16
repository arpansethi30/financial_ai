import streamlit as st

# Configure page - must be the first Streamlit command
st.set_page_config(
    page_title="Financial AI Agent",
    page_icon="📈",
    layout="wide"
)

import os
from dotenv import load_dotenv
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.llms.gemini import Gemini
import google.generativeai as genai
import yfinance as yf
from newsapi import NewsApiClient
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta
import time
from functools import wraps
import json
import hashlib
from collections import OrderedDict
import threading
import queue
import websocket
import json
from concurrent.futures import ThreadPoolExecutor

# Load environment variables
load_dotenv()

class LRUCache:
    """Least Recently Used (LRU) cache implementation"""
    def __init__(self, capacity):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key):
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

class StockAlert:
    def __init__(self):
        self.alerts = {}
        self.alert_history = []
    
    def add_alert(self, symbol, condition, value, message):
        """Add a new price alert"""
        if symbol not in self.alerts:
            self.alerts[symbol] = []
        self.alerts[symbol].append({
            'condition': condition,
            'value': float(value),
            'message': message,
            'created_at': datetime.now()
        })
    
    def check_alerts(self, symbol, current_price):
        """Check if any alerts have been triggered"""
        if symbol not in self.alerts:
            return []
        
        triggered = []
        remaining = []
        
        for alert in self.alerts[symbol]:
            if self._evaluate_condition(current_price, alert['condition'], alert['value']):
                triggered.append(alert)
                self.alert_history.append({
                    **alert,
                    'triggered_at': datetime.now(),
                    'trigger_price': current_price
                })
            else:
                remaining.append(alert)
        
        self.alerts[symbol] = remaining
        return triggered
    
    def _evaluate_condition(self, current_price, condition, target_value):
        """Evaluate if an alert condition has been met"""
        if condition == 'above':
            return current_price > target_value
        elif condition == 'below':
            return current_price < target_value
        return False
    
    def get_active_alerts(self, symbol):
        """Get all active alerts for a symbol"""
        return self.alerts.get(symbol, [])
    
    def get_alert_history(self):
        """Get history of triggered alerts"""
        return self.alert_history

class RealTimeDataManager:
    def __init__(self):
        self.data_queue = queue.Queue()
        self.subscribers = []
        self.running = False
        self.current_prices = {}
        self.alert_manager = StockAlert()
    
    def start_streaming(self, symbols):
        """Start streaming real-time data for given symbols"""
        if not self.running:
            self.running = True
            self.symbols = symbols
            self.thread = threading.Thread(target=self._stream_data)
            self.thread.daemon = True
            self.thread.start()
    
    def _stream_data(self):
        """Stream real-time data using yfinance"""
        while self.running:
            try:
                for symbol in self.symbols:
                    ticker = yf.Ticker(symbol)
                    data = ticker.history(period='1d', interval='1m').iloc[-1]
                    self.current_prices[symbol] = data['Close']
                    
                    # Check alerts
                    triggered_alerts = self.alert_manager.check_alerts(symbol, data['Close'])
                    for alert in triggered_alerts:
                        st.warning(f"Alert triggered for {symbol}: {alert['message']}")
                    
                    self.data_queue.put({
                        'symbol': symbol,
                        'price': data['Close'],
                        'volume': data['Volume'],
                        'timestamp': datetime.now(),
                        'change': data['Close'] - data['Open'],
                        'change_percent': ((data['Close'] - data['Open']) / data['Open']) * 100
                    })
                
                time.sleep(60)  # Update every minute
            except Exception as e:
                st.error(f"Error streaming data: {str(e)}")
                time.sleep(5)
    
    def stop_streaming(self):
        """Stop streaming data"""
        self.running = False
    
    def get_latest_data(self, symbol):
        """Get the latest data for a symbol"""
        try:
            while not self.data_queue.empty():
                data = self.data_queue.get()
                if data['symbol'] == symbol:
                    return data
        except Exception as e:
            st.error(f"Error getting latest data: {str(e)}")
        return None

class PredictiveAnalytics:
    def __init__(self, symbol):
        self.symbol = symbol
        self.ticker = yf.Ticker(symbol)
        self.predictions = {}
    
    def generate_predictions(self):
        """Generate price predictions using various methods"""
        try:
            # Get historical data
            hist = self.ticker.history(period='1y')
            
            # Calculate technical indicators
            hist['SMA20'] = ta.trend.sma_indicator(hist['Close'], window=20)
            hist['SMA50'] = ta.trend.sma_indicator(hist['Close'], window=50)
            hist['RSI'] = ta.momentum.rsi(hist['Close'], window=14)
            
            # Simple prediction based on trend
            last_price = hist['Close'].iloc[-1]
            trend = 'up' if hist['SMA20'].iloc[-1] > hist['SMA50'].iloc[-1] else 'down'
            
            # Calculate volatility
            hist['Returns'] = hist['Close'].pct_change()
            volatility = hist['Returns'].std() * np.sqrt(252)  # Annualized volatility
            
            # Generate price ranges
            self.predictions = {
                'current_price': last_price,
                'trend': trend,
                'volatility': volatility,
                'price_ranges': {
                    'day': {
                        'high': last_price * (1 + volatility/np.sqrt(252)),
                        'low': last_price * (1 - volatility/np.sqrt(252))
                    },
                    'week': {
                        'high': last_price * (1 + volatility/np.sqrt(52)),
                        'low': last_price * (1 - volatility/np.sqrt(52))
                    },
                    'month': {
                        'high': last_price * (1 + volatility/np.sqrt(12)),
                        'low': last_price * (1 - volatility/np.sqrt(12))
                    }
                }
            }
            
            return self.predictions
        except Exception as e:
            st.error(f"Error generating predictions: {str(e)}")
            return None

class AIService:
    def __init__(self):
        self.llm = None
        self.genai_model = None
        self.last_request_time = 0
        self.requests_this_minute = 0
        self.max_requests_per_minute = 60  # Adjust based on your API tier
        self.cache = LRUCache(capacity=100)
        self.backoff_time = 1  # Start with 1 second backoff
        self.initialize_llm()
        
    def initialize_llm(self):
        """Initialize the LLM with error handling"""
        try:
            api_key = os.getenv("GOOGLE_AI_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_AI_API_KEY not found in environment variables")
            
            # Configure Gemini
            genai.configure(api_key=api_key)
            self.genai_model = genai.GenerativeModel('gemini-pro')
            
            # Configure LlamaIndex Gemini integration
            self.llm = Gemini(api_key=api_key)
            Settings.llm = self.llm
            
            # Test the connection with a simple query
            test_response = self.genai_model.generate_content("Test connection")
            if test_response:
                st.sidebar.success("AI Service initialized successfully")
                return True
            else:
                raise Exception("Failed to get response from AI service")
        except Exception as e:
            st.error(f"Error initializing AI service: {str(e)}")
            return False
    
    def _get_cache_key(self, query, context):
        """Generate a cache key from query and context"""
        combined = f"{query}|{context}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _wait_for_rate_limit(self):
        """Implement exponential backoff for rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < 60:  # Within the same minute
            if self.requests_this_minute >= self.max_requests_per_minute:
                sleep_time = max(60 - time_since_last, self.backoff_time)
                st.warning(f"Rate limit reached. Waiting {int(sleep_time)} seconds...")
                time.sleep(sleep_time)
                self.backoff_time = min(self.backoff_time * 2, 30)  # Exponential backoff up to 30 seconds
                self.requests_this_minute = 0
                self.last_request_time = time.time()
            else:
                self.requests_this_minute += 1
        else:
            # Reset counters for new minute
            self.requests_this_minute = 1
            self.last_request_time = current_time
            self.backoff_time = 1  # Reset backoff time
    
    def get_ai_response(self, query, context, fallback_to_basic=True):
        """Get AI response with improved rate limiting and error handling"""
        try:
            # Check cache first
            cache_key = self._get_cache_key(query, context)
            cached_response = self.cache.get(cache_key)
            if cached_response:
                st.info("Retrieved from cache")
                return cached_response
            
            # Wait for rate limit if necessary
            self._wait_for_rate_limit()
            
            if not self.genai_model:
                if not self.initialize_llm():
                    raise Exception("AI service not available")
            
            # Try direct completion with Gemini
            try:
                response = self.genai_model.generate_content(self._create_prompt(query, context))
                if response and hasattr(response, 'text') and response.text.strip():
                    result = {"success": True, "response": response.text}
                    self.cache.put(cache_key, result)
                    return result
            except Exception as e:
                if "429" in str(e):
                    st.warning("Gemini API rate limit reached. Trying alternative method...")
                    # Try LlamaIndex as fallback
                    try:
                        documents = [Document(text=context)]
                        index = VectorStoreIndex.from_documents(documents)
                        query_engine = index.as_query_engine()
                        response = query_engine.query(query)
                        
                        if response and response.response.strip():
                            result = {"success": True, "response": response.response}
                            self.cache.put(cache_key, result)
                            return result
                    except Exception as llama_error:
                        if fallback_to_basic:
                            st.info("Using basic analysis as fallback...")
                            return self.get_basic_analysis(context)
                        raise llama_error
                else:
                    raise e
            
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg:
                if fallback_to_basic:
                    st.info("API rate limit reached. Using basic analysis...")
                    return self.get_basic_analysis(context)
            return {
                "success": False,
                "error": error_msg,
                "response": "Unable to generate AI analysis at this time. Please try again in a few minutes."
            }
    
    def _create_prompt(self, query, context):
        """Create a focused prompt for the AI"""
        return f"""
        You are a professional financial analyst. Based on the following information:

        {context}

        Please answer this specific question: {query}

        Provide a detailed analysis that:
        1. Uses specific data points from the provided information
        2. Includes relevant financial metrics and their interpretation
        3. Cites sources for all information
        4. Provides clear reasoning
        5. Ends with a specific conclusion

        Format your response with clear sections and bullet points where appropriate.
        If certain information is not available, acknowledge this and explain its impact on the analysis.
        """
    
    def get_basic_analysis(self, context):
        """Provide basic analysis when AI is unavailable"""
        try:
            # Parse the context into a structured format
            lines = context.split('\n')
            data = {}
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    data[key.strip()] = value.strip()
            
            # Generate basic analysis
            analysis = []
            
            # Company Overview
            if 'Company Name' in data or 'Company' in data:
                company = data.get('Company Name', data.get('Company', ''))
                industry = data.get('Industry', 'N/A')
                sector = data.get('Sector', 'N/A')
                analysis.append(f"Company Overview:\n{company} operates in the {industry} industry within the {sector} sector.")
            
            # Market Position
            if 'Market Cap' in data:
                analysis.append(f"Market Position:\nThe company has a market capitalization of {data['Market Cap']}.")
            
            # Price Analysis
            if 'Current Price' in data and '52-Week Range' in data:
                analysis.append(f"Price Analysis:\nCurrent Price: {data['Current Price']}\n52-Week Range: {data['52-Week Range']}")
            
            # Technical Analysis
            if 'Technical Analysis' in context:
                tech_analysis = "Technical Analysis:\n"
                if "RSI" in context:
                    tech_analysis += f"- RSI indicates {data.get('RSI', 'N/A')}\n"
                if "Trend" in context:
                    tech_analysis += f"- Current trend is {data.get('Trend', 'N/A')}\n"
                if "MACD" in context:
                    tech_analysis += f"- MACD shows {data.get('MACD', 'N/A')}\n"
                analysis.append(tech_analysis)
            
            # Business Summary
            if 'Business Summary' in data:
                summary = data['Business Summary']
                if len(summary) > 200:
                    summary = summary[:200] + "..."
                analysis.append(f"Business Summary:\n{summary}")
            
            # Combine analysis
            response = "\n\n".join(analysis)
            
            return {
                "success": True,
                "response": response,
                "note": "(This is a basic analysis generated due to API rate limits. Try again in a minute for AI-powered analysis.)"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "response": "Unable to generate analysis. Please try again later."
            }

class SentimentAnalyzer:
    def __init__(self, ai_service):
        self.ai_service = ai_service
        self.sentiment_cache = {}
        self.sentiment_history = {}
    
    def analyze_text(self, text, source):
        """Analyze sentiment of text with source context"""
        cache_key = hashlib.md5(text.encode()).hexdigest()
        
        if cache_key in self.sentiment_cache:
            return self.sentiment_cache[cache_key]
        
        prompt = f"""
        Analyze the sentiment of this {source} text:
        {text}
        
        Provide:
        1. Sentiment (positive/negative/neutral)
        2. Confidence score (0-1)
        3. Key factors influencing the sentiment
        4. Potential market impact
        
        Format: JSON
        """
        
        result = self.ai_service.get_ai_response(prompt, text)
        if result["success"]:
            try:
                sentiment_data = json.loads(result["response"])
                self.sentiment_cache[cache_key] = sentiment_data
                return sentiment_data
            except:
                return {
                    "sentiment": "neutral",
                    "confidence": 0.5,
                    "factors": ["Unable to parse sentiment"],
                    "market_impact": "Unknown"
                }
        return None
    
    def track_sentiment(self, symbol, sentiment_data, timestamp=None):
        """Track sentiment history for a symbol"""
        if symbol not in self.sentiment_history:
            self.sentiment_history[symbol] = []
        
        self.sentiment_history[symbol].append({
            'timestamp': timestamp or datetime.now(),
            'data': sentiment_data
        })
    
    def get_sentiment_trend(self, symbol, days=30):
        """Get sentiment trend for a symbol"""
        if symbol not in self.sentiment_history:
            return None
        
        cutoff = datetime.now() - timedelta(days=days)
        recent_sentiment = [
            s for s in self.sentiment_history[symbol]
            if s['timestamp'] > cutoff
        ]
        
        return recent_sentiment

# Initialize AI service
ai_service = AIService()

# Initialize sentiment analyzer
sentiment_analyzer = SentimentAnalyzer(ai_service)

# Helper Functions
def calculate_technical_indicators(data):
    """Calculate technical indicators for the stock data"""
    # Moving averages
    data['SMA20'] = ta.trend.sma_indicator(data['Close'], window=20)
    data['SMA50'] = ta.trend.sma_indicator(data['Close'], window=50)
    data['SMA200'] = ta.trend.sma_indicator(data['Close'], window=200)
    
    # RSI
    data['RSI'] = ta.momentum.rsi(data['Close'], window=14)
    
    # MACD
    macd = ta.trend.MACD(data['Close'])
    data['MACD'] = macd.macd()
    data['MACD_Signal'] = macd.macd_signal()
    
    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(data['Close'])
    data['BB_Upper'] = bollinger.bollinger_hband()
    data['BB_Lower'] = bollinger.bollinger_lband()
    data['BB_Middle'] = bollinger.bollinger_mavg()
    
    return data

def plot_stock_data(data, symbol):
    """Create an interactive stock chart with technical indicators"""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                       vertical_spacing=0.03, 
                       row_heights=[0.7, 0.3])
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(x=data.index,
                                open=data['Open'],
                                high=data['High'],
                                low=data['Low'],
                                close=data['Close'],
                                name='OHLC'),
                 row=1, col=1)
    
    # Add Moving Averages
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA20'],
                            name='SMA20', line=dict(color='blue')),
                 row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA50'],
                            name='SMA50', line=dict(color='orange')),
                 row=1, col=1)
    
    # Add Bollinger Bands
    fig.add_trace(go.Scatter(x=data.index, y=data['BB_Upper'],
                            name='BB Upper', line=dict(color='gray', dash='dash')),
                 row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['BB_Lower'],
                            name='BB Lower', line=dict(color='gray', dash='dash')),
                 row=1, col=1)
    
    # Add RSI
    fig.add_trace(go.Scatter(x=data.index, y=data['RSI'],
                            name='RSI', line=dict(color='purple')),
                 row=2, col=1)
    
    # Add RSI overbought/oversold lines
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # Update layout
    fig.update_layout(
        title=f'{symbol} Stock Price and Technical Indicators',
        yaxis_title='Stock Price',
        yaxis2_title='RSI',
        xaxis_rangeslider_visible=False,
        height=800
    )
    
    return fig

def get_competitor_analysis(symbol):
    """Get competitor analysis for a given stock"""
    try:
        # Get base ticker info
        ticker = yf.Ticker(symbol)
        info = ticker.info
        industry = info.get('industry', '')
        
        if not industry:
            return []
            
        # Define major competitors based on industry
        industry_competitors = {
            'Technology': ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'AMD', 'INTC'],
            'Consumer Electronics': ['AAPL', 'SONY', 'SAMSUNG.KS', 'HPQ', 'DELL'],
            'Software': ['MSFT', 'ORCL', 'CRM', 'ADBE', 'INTU'],
            'Internet Content & Information': ['GOOGL', 'META', 'PINS', 'SNAP', 'TWTR'],
            'Semiconductors': ['NVDA', 'AMD', 'INTC', 'TSM', 'QCOM'],
            'E-Commerce': ['AMZN', 'BABA', 'SHOP', 'ETSY', 'MELI'],
            'Electric Vehicles': ['TSLA', 'F', 'GM', 'NIO', 'RIVN'],
            'Social Media': ['META', 'SNAP', 'PINS', 'TWTR', 'MTCH'],
        }
        
        # Get default competitors if industry not found
        competitor_symbols = industry_competitors.get(info.get('sector', ''), 
            ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'])
            
        # Remove the current symbol from competitors
        competitor_symbols = [s for s in competitor_symbols if s != symbol]
        
        competitors = []
        for comp_symbol in competitor_symbols[:5]:  # Limit to top 5 competitors
            try:
                comp_ticker = yf.Ticker(comp_symbol)
                comp_info = comp_ticker.info
                
                if comp_info and isinstance(comp_info, dict):
                    competitors.append({
                        'symbol': comp_symbol,
                        'name': comp_info.get('longName', comp_symbol),
                        'market_cap': comp_info.get('marketCap', 0),
                        'pe_ratio': comp_info.get('forwardPE', 0),
                        'price': comp_info.get('currentPrice', 0),
                        'revenue': comp_info.get('totalRevenue', 0),
                        'profit_margin': comp_info.get('profitMargins', 0)
                    })
            except Exception as e:
                st.warning(f"Could not fetch data for {comp_symbol}: {str(e)}")
                continue
                
        return sorted(competitors, key=lambda x: x['market_cap'] if x['market_cap'] else 0, reverse=True)
    except Exception as e:
        st.warning(f"Error in competitor analysis: {str(e)}")
        return []

def format_large_number(number):
    """Format large numbers into readable format with B/M suffix"""
    if number is None:
        return 'N/A'
    if number >= 1e9:
        return f'${number/1e9:.2f}B'
    elif number >= 1e6:
        return f'${number/1e6:.2f}M'
    else:
        return f'${number:,.0f}'

def get_financial_analysis(symbol):
    """Get detailed financial analysis including quarterly and annual results"""
    try:
        ticker = yf.Ticker(symbol)
        
        # Get quarterly and annual financials
        quarterly_financials = ticker.quarterly_financials
        annual_financials = ticker.financials
        quarterly_earnings = ticker.quarterly_earnings
        annual_earnings = ticker.earnings
        
        analysis = {
            'quarterly_financials': quarterly_financials,
            'annual_financials': annual_financials,
            'quarterly_earnings': quarterly_earnings,
            'annual_earnings': annual_earnings,
            'key_metrics': {}
        }
        
        # Calculate key financial metrics
        if not quarterly_financials.empty:
            latest_quarter = quarterly_financials.iloc[:, 0]
            prev_quarter = quarterly_financials.iloc[:, 1]
            
            # Revenue growth
            if 'Total Revenue' in latest_quarter and 'Total Revenue' in prev_quarter:
                qoq_growth = ((latest_quarter['Total Revenue'] - prev_quarter['Total Revenue']) / 
                             prev_quarter['Total Revenue']) * 100
                analysis['key_metrics']['revenue_growth_qoq'] = qoq_growth
            
            # Profit margins
            if 'Net Income' in latest_quarter and 'Total Revenue' in latest_quarter:
                profit_margin = (latest_quarter['Net Income'] / latest_quarter['Total Revenue']) * 100
                analysis['key_metrics']['profit_margin'] = profit_margin
        
        return analysis
    except Exception as e:
        st.error(f"Error fetching financial analysis: {str(e)}")
        return None

def get_comprehensive_analysis(symbol, info, financial_analysis):
    """Generate comprehensive analysis including quarterly and annual insights"""
    analysis_sections = []
    
    # Business Overview
    if info.get('longBusinessSummary'):
        analysis_sections.append(f"""
        ## Business Overview
        {info.get('longBusinessSummary')}
        
        **Industry:** {info.get('industry', 'N/A')}
        **Sector:** {info.get('sector', 'N/A')}
        **Employees:** {format_large_number(info.get('fullTimeEmployees', 0))}
        """)
    
    # Financial Performance
    if financial_analysis and 'quarterly_financials' in financial_analysis:
        qf = financial_analysis['quarterly_financials']
        if not qf.empty:
            latest_quarter = qf.iloc[:, 0]
            prev_quarter = qf.iloc[:, 1]
            
            revenue_growth = financial_analysis['key_metrics'].get('revenue_growth_qoq', 'N/A')
            profit_margin = financial_analysis['key_metrics'].get('profit_margin', 'N/A')
            
            # Fix the f-string formatting
            growth_text = f"{revenue_growth:.1f}%" if isinstance(revenue_growth, float) else "N/A"
            margin_text = f"{profit_margin:.1f}%" if isinstance(profit_margin, float) else "N/A"
            
            analysis_sections.append(f"""
            ## Financial Performance
            
            ### Quarterly Results
            - Revenue Growth (QoQ): {growth_text}
            - Profit Margin: {margin_text}
            
            ### Key Financial Metrics
            - Market Cap: {format_large_number(info.get('marketCap', 0))}
            - P/E Ratio: {info.get('forwardPE', 'N/A')}
            - Price/Book: {info.get('priceToBook', 'N/A')}
            """)
    
    # Market Position
    analysis_sections.append(f"""
    ## Market Position
    - Current Price: ${info.get('currentPrice', 'N/A')}
    - 52-Week Range: ${info.get('fiftyTwoWeekLow', 'N/A')} - ${info.get('fiftyTwoWeekHigh', 'N/A')}
    - Average Volume: {format_large_number(info.get('averageVolume', 0))}
    """)
    
    # Risk Analysis
    beta = info.get('beta', 'N/A')
    debt_equity = info.get('debtToEquity', 'N/A')
    current_ratio = info.get('currentRatio', 'N/A')
    
    beta_text = f"{beta:.2f}" if isinstance(beta, float) else 'N/A'
    debt_equity_text = f"{debt_equity:.2f}" if isinstance(debt_equity, float) else 'N/A'
    current_ratio_text = f"{current_ratio:.2f}" if isinstance(current_ratio, float) else 'N/A'
    
    analysis_sections.append(f"""
    ## Risk Analysis
    - Beta: {beta_text}
    - Debt/Equity: {debt_equity_text}
    - Current Ratio: {current_ratio_text}
    """)
    
    return "\n\n".join(analysis_sections)

# Initialize services
ai_service = AIService()
sentiment_analyzer = SentimentAnalyzer(ai_service)
real_time_manager = RealTimeDataManager()

# Configure page
st.title("Financial AI Agent Dashboard")

# Initialize AI and API clients
def initialize_services():
    try:
        # Get API keys
        news_api_key = os.getenv("NEWS_API_KEY")
        
        # Initialize NewsAPI client
        news_client = NewsApiClient(api_key=news_api_key) if news_api_key else None
        
        return ai_service, news_client
    except Exception as e:
        st.error(f"Error initializing services: {str(e)}")
        return None, None

# Initialize services
ai_service, news_client = initialize_services()

# Sidebar for stock selection
st.sidebar.header("Stock Selection")
symbol = st.sidebar.text_input("Enter Stock Symbol:", "AAPL")
if symbol:
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        st.sidebar.write(f"**{info.get('longName', symbol)}**")
        st.sidebar.write(f"Current Price: ${info.get('currentPrice', 'N/A')}")
    except Exception as e:
        st.sidebar.error(f"Error loading stock info: {str(e)}")

# Main content area with tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Stock Analysis", 
    "News & Sentiment", 
    "Financial Metrics", 
    "AI Insights",
    "Predictive Analytics"
])

with tab1:
    st.header("Stock Analysis")
    if symbol and ai_service.genai_model:
        try:
            # Get stock data
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Display basic info in a more organized way
            col1, col2, col3 = st.columns(3)
            with col1:
                current_price = info.get('currentPrice', 0)
                change_percent = info.get('regularMarketChangePercent', 0)
                st.metric("Current Price", 
                         f"${current_price:,.2f}", 
                         f"{change_percent:.2f}%",
                         delta_color="normal")
            with col2:
                market_cap = info.get('marketCap', 0)
                st.metric("Market Cap", format_large_number(market_cap))
            with col3:
                pe_ratio = info.get('forwardPE', 'N/A')
                st.metric("P/E Ratio", f"{pe_ratio:.2f}" if isinstance(pe_ratio, float) else pe_ratio)

            # Technical Analysis Section
            st.subheader("Technical Analysis")
            
            # Get historical data
            hist_data = ticker.history(period="6mo")
            
            # Calculate technical indicators
            tech_data = calculate_technical_indicators(hist_data)
            
            # Plot interactive chart
            fig = plot_stock_data(tech_data, symbol)
            st.plotly_chart(fig, use_container_width=True)
            
            # Technical Signals
            st.subheader("Technical Signals")
            latest = tech_data.iloc[-1]
            
            signal_cols = st.columns(3)
            with signal_cols[0]:
                rsi = latest['RSI']
                rsi_signal = "Oversold" if rsi < 30 else "Overbought" if rsi > 70 else "Neutral"
                st.metric("RSI", f"{rsi:.2f}", rsi_signal)
            
            with signal_cols[1]:
                price = latest['Close']
                sma_50 = latest['SMA50']
                trend = "Bullish" if price > sma_50 else "Bearish"
                st.metric("Trend (SMA50)", f"${sma_50:.2f}", trend)
            
            with signal_cols[2]:
                macd = latest['MACD']
                macd_signal = latest['MACD_Signal']
                macd_trend = "Bullish" if macd > macd_signal else "Bearish"
                st.metric("MACD Signal", f"{macd:.2f}", macd_trend)

            # Competitor Analysis
            st.subheader("Competitor Analysis")
            competitors = get_competitor_analysis(symbol)
            
            if competitors:
                st.subheader("Competitor Analysis")
                comp_df = pd.DataFrame(competitors)
                
                # Format the columns
                comp_df['market_cap'] = comp_df['market_cap'].apply(format_large_number)
                comp_df['pe_ratio'] = comp_df['pe_ratio'].apply(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) and x > 0 else 'N/A')
                comp_df['price'] = comp_df['price'].apply(lambda x: f"${x:,.2f}" if isinstance(x, (int, float)) and x > 0 else 'N/A')
                comp_df['profit_margin'] = comp_df['profit_margin'].apply(lambda x: f"{x*100:.1f}%" if isinstance(x, (int, float)) and x > 0 else 'N/A')
                comp_df['revenue'] = comp_df['revenue'].apply(format_large_number)
                
                st.dataframe(
                    comp_df.set_index('symbol'),
                    column_config={
                        "name": "Company",
                        "market_cap": "Market Cap",
                        "pe_ratio": "P/E Ratio",
                        "price": "Price",
                        "revenue": "Revenue",
                        "profit_margin": "Profit Margin"
                    },
                    use_container_width=True
                )
            else:
                st.info("No competitor data available for this stock.")
            
            # AI Analysis button moved to bottom of analysis
            if st.button("Generate AI Analysis"):
                with st.spinner("Analyzing stock..."):
                    analysis_text = f"""
                    Company Name: {info.get('longName', '')}
                    Sector: {info.get('sector', '')}
                    Industry: {info.get('industry', '')}
                    Description: {info.get('longBusinessSummary', '')}
                    Market Cap: {format_large_number(info.get('marketCap', 0))}
                    P/E Ratio: {info.get('forwardPE', 'N/A')}
                    52-Week Range: ${info.get('fiftyTwoWeekLow', 'N/A')} - ${info.get('fiftyTwoWeekHigh', 'N/A')}
                    
                    Technical Analysis:
                    RSI: {rsi:.2f} ({rsi_signal})
                    Trend: {trend} (Price vs SMA50)
                    MACD: {macd_trend}
                    
                    Competitor Comparison:
                    {comp_df.to_string() if len(competitors) > 0 else 'No competitor data available'}
                    """
                    
                    analysis_query = """
                    Based on the company information and technical analysis, provide a detailed analysis including:
                    1. Company Overview and Business Model
                    2. Technical Analysis Summary
                    3. Competitive Position
                    4. Market Trends and Momentum
                    5. Key Risks and Opportunities
                    6. Investment Recommendation with Supporting Rationale
                    
                    Please provide specific insights based on the technical indicators and competitor data.
                    """
                    
                    result = ai_service.get_ai_response(analysis_query, analysis_text)
                    if result["success"]:
                        st.write(result["response"])
                        if "note" in result:
                            st.info(result["note"])
                    else:
                        st.error(result["response"])
                    
        except Exception as e:
            st.error(f"Error in stock analysis: {str(e)}")

with tab2:
    st.header("Advanced Sentiment Analysis")
    
    if symbol:
        try:
            # Get company info
            ticker = yf.Ticker(symbol)
            company_name = ticker.info.get('longName', symbol)
            
            # Sentiment Overview
            st.subheader("Sentiment Overview")
            
            # Get news and analyze sentiment
            news_data = []
            if news_client:
                try:
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=days_ago)
                    
                    news = news_client.get_everything(
                        q=f'"{company_name}" OR ${symbol}',
                        language='en',
                        sort_by='publishedAt',
                        from_param=start_date.strftime('%Y-%m-%d'),
                        to=end_date.strftime('%Y-%m-%d'),
                        page_size=20
                    )
                    
                    if news and 'articles' in news:
                        with st.spinner("Analyzing sentiment..."):
                            for article in news['articles']:
                                # Analyze sentiment
                                text = f"{article['title']} {article.get('description', '')}"
                                sentiment_data = sentiment_analyzer.analyze_text(text, "news article")
                                
                                if sentiment_data:
                                    news_data.append({
                                        'title': article['title'],
                                        'source': article['source']['name'],
                                        'published': article['publishedAt'],
                                        'sentiment': sentiment_data,
                                        'url': article['url']
                                    })
                                    
                                    # Track sentiment
                                    sentiment_analyzer.track_sentiment(
                                        symbol,
                                        sentiment_data,
                                        datetime.strptime(article['publishedAt'], "%Y-%m-%dT%H:%M:%SZ")
                                    )
                    
                    # Display sentiment trend
                    sentiment_trend = sentiment_analyzer.get_sentiment_trend(symbol)
                    if sentiment_trend:
                        # Create sentiment trend chart
                        fig = go.Figure()
                        
                        dates = [s['timestamp'] for s in sentiment_trend]
                        sentiment_scores = [
                            1 if s['data']['sentiment'] == 'positive'
                            else -1 if s['data']['sentiment'] == 'negative'
                            else 0
                            for s in sentiment_trend
                        ]
                        
                        fig.add_trace(go.Scatter(
                            x=dates,
                            y=sentiment_scores,
                            mode='lines+markers',
                            name='Sentiment Trend'
                        ))
                        
                        fig.update_layout(
                            title='Sentiment Trend Over Time',
                            yaxis_title='Sentiment Score',
                            yaxis=dict(
                                ticktext=['Negative', 'Neutral', 'Positive'],
                                tickvals=[-1, 0, 1]
                            )
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Display sentiment distribution
                    if news_data:
                        sentiments = [n['sentiment']['sentiment'] for n in news_data]
                        sentiment_counts = {
                            'Positive': sentiments.count('positive'),
                            'Negative': sentiments.count('negative'),
                            'Neutral': sentiments.count('neutral')
                        }
                        
                        # Create sentiment distribution pie chart
                        fig = go.Figure(data=[go.Pie(
                            labels=list(sentiment_counts.keys()),
                            values=list(sentiment_counts.values()),
                            hole=.3
                        )])
                        
                        fig.update_layout(title='Sentiment Distribution')
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display news with sentiment
                        st.subheader("Recent News with Sentiment Analysis")
                        for article in news_data:
                            with st.expander(article['title']):
                                col1, col2 = st.columns([3, 1])
                                with col1:
                                    st.write(f"**Source:** {article['source']}")
                                    st.write(f"**Published:** {article['published']}")
                                with col2:
                                    sentiment_color = {
                                        'positive': 'green',
                                        'negative': 'red',
                                        'neutral': 'gray'
                                    }[article['sentiment']['sentiment']]
                                    
                                    st.markdown(f"""
                                        <div style='background-color: {sentiment_color}; 
                                                  padding: 10px; 
                                                  border-radius: 5px; 
                                                  color: white; 
                                                  text-align: center;'>
                                            {article['sentiment']['sentiment'].title()}
                                            ({article['sentiment']['confidence']:.2f})
                                        </div>
                                    """, unsafe_allow_html=True)
                                
                                st.write("**Key Factors:**")
                                for factor in article['sentiment']['factors']:
                                    st.write(f"- {factor}")
                                
                                st.write(f"**Market Impact:** {article['sentiment']['market_impact']}")
                                st.markdown(f"[Read More]({article['url']})")
                except Exception as e:
                    st.error(f"Error fetching news: {str(e)}")
                    if "429" in str(e):
                        st.warning("""
                        News API rate limit reached. Please try again later.
                        
                        Alternative sources for news:
                        1. Yahoo Finance
                        2. Company's investor relations page
                        3. SEC EDGAR filings
                        """)
            else:
                st.warning("News API client not initialized. Please check your API key.")
        
        except Exception as e:
            st.error(f"Error in sentiment analysis: {str(e)}")
            st.info("Please check if the stock symbol is correct and try again.")

with tab3:
    st.header("Financial Metrics")
    if symbol:
        try:
            with st.spinner("Loading financial data..."):
                # Get financial analysis
                financial_analysis = get_financial_analysis(symbol)
                
                if financial_analysis:
                    # Display quarterly results
                    st.subheader("Quarterly Performance")
                    quarterly_metrics = st.columns(3)
                    
                    with quarterly_metrics[0]:
                        qoq_growth = financial_analysis['key_metrics'].get('revenue_growth_qoq')
                        if qoq_growth is not None:
                            st.metric("Revenue Growth (QoQ)", 
                                    f"{qoq_growth:.1f}%",
                                    delta_color="normal")
                    
                    with quarterly_metrics[1]:
                        profit_margin = financial_analysis['key_metrics'].get('profit_margin')
                        if profit_margin is not None:
                            st.metric("Profit Margin", 
                                    f"{profit_margin:.1f}%",
                                    delta_color="normal")
                    
                    # Display quarterly financials
                    st.subheader("Quarterly Financials")
                    if not financial_analysis['quarterly_financials'].empty:
                        st.dataframe(financial_analysis['quarterly_financials'].style.format("${:,.2f}"))
                        
                        # Add quarterly trend chart
                        quarterly_revenue = financial_analysis['quarterly_financials'].loc['Total Revenue']
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=quarterly_revenue.index,
                            y=quarterly_revenue.values,
                            name='Quarterly Revenue'
                        ))
                        fig.update_layout(
                            title='Quarterly Revenue Trend',
                            xaxis_title='Quarter',
                            yaxis_title='Revenue ($)',
                            showlegend=True
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Display annual financials
                    st.subheader("Annual Financials")
                    if not financial_analysis['annual_financials'].empty:
                        st.dataframe(financial_analysis['annual_financials'].style.format("${:,.2f}"))
                    
                    # Add source citation
                    st.markdown("""
                    **Data Sources:**
                    - Quarterly and Annual Financials: SEC Filings via Yahoo Finance
                    - Financial Metrics: Calculated based on company filings
                    - Market Data: Real-time market data via Yahoo Finance
                    
                    *Last updated: {}*
                    """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                
        except Exception as e:
            st.error(f"Error loading financial metrics: {str(e)}")
            st.info("Please try again later or check if the stock symbol is correct.")

with tab4:
    st.header("AI Insights")
    if symbol:
        try:
            with st.spinner("Loading comprehensive analysis..."):
                ticker = yf.Ticker(symbol)
                info = ticker.info
                financial_analysis = get_financial_analysis(symbol)
                
                # Display comprehensive analysis
                analysis = get_comprehensive_analysis(symbol, info, financial_analysis)
                st.markdown(analysis)
                
                # Custom query section with improved error handling
                st.subheader("Ask About the Stock")
                question = st.text_input("Enter your question about the stock:",
                                       placeholder="e.g., What are the main growth drivers for this company?")
                
                if question:
                    if st.button("Get AI Answer", key="ai_analysis_button"):
                        try:
                            with st.spinner("Analyzing..."):
                                # Create comprehensive context
                                context = f"""
                                Company Information:
                                {analysis}
                                
                                Recent Financial Performance:
                                {financial_analysis['quarterly_financials'].to_string() if financial_analysis and 'quarterly_financials' in financial_analysis else 'No quarterly data available'}
                                
                                Market Data:
                                - Current Price: ${info.get('currentPrice', 'N/A')}
                                - Market Cap: {format_large_number(info.get('marketCap', 0))}
                                - 52-Week Range: ${info.get('fiftyTwoWeekLow', 'N/A')} - ${info.get('fiftyTwoWeekHigh', 'N/A')}
                                """
                                
                                result = ai_service.get_ai_response(question, context)
                                if result["success"]:
                                    st.markdown(result["response"])
                                    
                                    # Add confidence level and sources
                                    st.info("""
                                    **Analysis Confidence Level:** High
                                    
                                    **Sources:**
                                    - Financial Data: SEC Filings, Yahoo Finance
                                    - Market Data: Real-time market feeds
                                    - Company Information: Official filings and reports
                                    
                                    *Last updated: {}*
                                    """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                                else:
                                    st.error(result["response"])
                        except Exception as e:
                            st.error(f"Error generating AI analysis: {str(e)}")
                            st.info("Please try rephrasing your question or try again later.")
        except Exception as e:
            st.error(f"Error loading AI insights: {str(e)}")
            st.info("Please check if the stock symbol is correct and try again.")

with tab5:
    st.header("Predictive Analytics")
    
    if symbol:
        try:
            predictor = PredictiveAnalytics(symbol)
            predictions = predictor.generate_predictions()
            
            if predictions:
                # Display current trend and volatility
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "Current Trend",
                        predictions['trend'].upper(),
                        delta_color="normal"
                    )
                with col2:
                    st.metric(
                        "Volatility",
                        f"{predictions['volatility']*100:.1f}%",
                        delta_color="normal"
                    )
                
                # Display price predictions
                st.subheader("Price Range Predictions")
                
                # Create prediction ranges table
                ranges_df = pd.DataFrame({
                    'Timeframe': ['Day', 'Week', 'Month'],
                    'Low': [
                        predictions['price_ranges']['day']['low'],
                        predictions['price_ranges']['week']['low'],
                        predictions['price_ranges']['month']['low']
                    ],
                    'High': [
                        predictions['price_ranges']['day']['high'],
                        predictions['price_ranges']['week']['high'],
                        predictions['price_ranges']['month']['high']
                    ]
                })
                
                # Format price columns
                ranges_df['Low'] = ranges_df['Low'].apply(lambda x: f"${x:.2f}")
                ranges_df['High'] = ranges_df['High'].apply(lambda x: f"${x:.2f}")
                
                st.dataframe(ranges_df.set_index('Timeframe'), use_container_width=True)
                
                # Add prediction visualization
                fig = go.Figure()
                
                # Add current price line
                fig.add_hline(
                    y=predictions['current_price'],
                    line_dash="dash",
                    annotation_text="Current Price",
                    line_color="green"
                )
                
                # Add prediction ranges
                timeframes = ['day', 'week', 'month']
                x_positions = [1, 7, 30]
                
                for i, (timeframe, x_pos) in enumerate(zip(timeframes, x_positions)):
                    fig.add_trace(go.Scatter(
                        x=[x_pos, x_pos],
                        y=[
                            predictions['price_ranges'][timeframe]['low'],
                            predictions['price_ranges'][timeframe]['high']
                        ],
                        mode='lines',
                        name=timeframe.capitalize(),
                        line=dict(width=10, color=f'rgb({50+i*70}, {100+i*50}, {150+i*30})')
                    ))
                
                fig.update_layout(
                    title='Price Range Predictions',
                    xaxis_title='Days Forward',
                    yaxis_title='Price ($)',
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add prediction notes
                st.info("""
                **Note:** These predictions are based on:
                - Historical price trends
                - Volatility analysis
                - Technical indicators
                
                Actual prices may vary significantly. Always do your own research and consider multiple factors before making investment decisions.
                """)
        
        except Exception as e:
            st.error(f"Error generating predictions: {str(e)}")
            st.info("Please check if the stock symbol is correct and try again.")

# Display environment info
st.sidebar.markdown("---")
st.sidebar.header("System Status")
st.sidebar.text(f"AI Service: {'✅ Ready' if ai_service.genai_model else '❌ Not Ready'}")
st.sidebar.text(f"NewsAPI: {'✅ Ready' if news_client else '❌ Not Ready'}")

# Add required packages info
if st.sidebar.button("Show Required Packages"):
    st.sidebar.code("""
    streamlit
    python-dotenv
    llama-index-core
    llama-index-llms-gemini
    yfinance
    newsapi-python
    pandas
    numpy
    datetime
    plotly
    ta
    """)

# Custom CSS for better UI
st.markdown("""
<style>
    .stAlert {
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 16px;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .css-1d391kg {
        padding: 1rem;
        border-radius: 5px;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Initialize real-time data manager
real_time_manager = RealTimeDataManager()

# Add to the sidebar
st.sidebar.header("Settings")
with st.sidebar.expander("Alert Settings"):
    alert_symbol = st.text_input("Symbol for Alert:", value=symbol if 'symbol' in locals() else "")
    alert_condition = st.selectbox("Condition:", ["above", "below"])
    alert_value = st.number_input("Price Target:", min_value=0.0, step=0.01)
    alert_message = st.text_input("Alert Message:", "Price target reached!")
    
    if st.button("Set Alert"):
        real_time_manager.alert_manager.add_alert(
            alert_symbol,
            alert_condition,
            alert_value,
            alert_message
        )
        st.success("Alert set successfully!")

# Show active alerts
with st.sidebar.expander("Active Alerts"):
    if 'symbol' in locals():
        active_alerts = real_time_manager.alert_manager.get_active_alerts(symbol)
        for alert in active_alerts:
            st.write(f"Target: ${alert['value']} ({alert['condition']})")
            st.write(f"Message: {alert['message']}")
            st.write("---")

# Add real-time data to the main dashboard
if 'symbol' in locals():
    real_time_manager.start_streaming([symbol])
    
    # Real-time price updates
    latest_data = real_time_manager.get_latest_data(symbol)
    if latest_data:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Real-Time Price",
                f"${latest_data['price']:.2f}",
                f"{latest_data['change_percent']:.2f}%"
            )
        with col2:
            st.metric(
                "Volume",
                f"{latest_data['volume']:,}"
            )
        with col3:
            st.metric(
                "Last Updated",
                latest_data['timestamp'].strftime("%H:%M:%S")
            ) 