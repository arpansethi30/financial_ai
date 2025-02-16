from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import yfinance as yf
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import google.generativeai as genai
import ta
import numpy as np
from typing import List, Optional, Dict, Literal
import time
from functools import wraps
import re
from app.trading import trading_service
from app.portfolio_optimizer import portfolio_optimizer
import logging
from app.portfolio_generator import portfolio_generator  # Import the new portfolio generator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Financial AI Agent API")

# Configure CORS with specific origins
origins = [
    "http://localhost:3000",    # Next.js development server
    "http://127.0.0.1:3000",
    "http://localhost:8000",    # Alternative port
    "http://127.0.0.1:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=86400,  # Cache preflight requests for 24 hours
)

class AIService:
    def __init__(self):
        self.model = None
        self.last_request_time = 0
        self.requests_this_minute = 0
        self.max_requests_per_minute = 60
        self.backoff_time = 1
        self.initialize_model()
    
    def initialize_model(self):
        """Initialize the Gemini model with error handling"""
        try:
            api_key = os.getenv("GOOGLE_AI_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_AI_API_KEY not found in environment variables")
            
            # Configure Gemini
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-pro')
            
            # Test the connection
            test_response = self.model.generate_content("Test connection")
            if not test_response:
                raise ValueError("Failed to initialize Gemini model")
            
        except Exception as e:
            raise ValueError(f"Error initializing AI service: {str(e)}")
    
    def generate_content(self, prompt):
        """Generate content with rate limiting and error handling"""
        current_time = time.time()
        
        # Rate limiting
        if current_time - self.last_request_time < 1:  # Minimum 1 second between requests
            time.sleep(1)
        
        if self.requests_this_minute >= self.max_requests_per_minute:
            time.sleep(self.backoff_time)
            self.requests_this_minute = 0
            self.backoff_time *= 2  # Exponential backoff
        else:
            self.backoff_time = 1  # Reset backoff time
        
        try:
            response = self.model.generate_content(prompt)
            self.last_request_time = time.time()
            self.requests_this_minute += 1
            return response
        except Exception as e:
            raise ValueError(f"Error generating content: {str(e)}")

# Initialize AI Service
ai_service = AIService()

# Initialize NewsAPI client
news_api_key = os.getenv("NEWS_API_KEY")
if news_api_key:
    from newsapi import NewsApiClient
    news_client = NewsApiClient(api_key=news_api_key)
else:
    news_client = None

# Pydantic models for request/response
class StockAnalysisRequest(BaseModel):
    symbol: str = Field(..., description="Stock symbol (e.g., AAPL)")
    period: str = Field(default="1y", description="Time period for analysis")
    interval: str = Field(default="1d", description="Data interval")

    model_config = {
        "json_schema_extra": {
            "example": {
                "symbol": "AAPL",
                "period": "1y",
                "interval": "1d"
            }
        }
    }

class SectorPreference(BaseModel):
    sector: str
    weight: float = Field(..., ge=0, le=100, description="Preferred weight for this sector (0-100)")

class PortfolioRequest(BaseModel):
    investment_amount: float = Field(..., ge=1000, description="Amount to invest")
    risk_appetite: Literal["conservative", "moderate", "aggressive"] = Field(..., description="Risk tolerance level")
    investment_period: int = Field(..., ge=1, le=30, description="Investment period in years")
    company_count: int = Field(..., ge=5, le=30, description="Number of companies to include in portfolio")

    model_config = {
        "json_schema_extra": {
            "example": {
                "investment_amount": 10000,
                "risk_appetite": "moderate",
                "investment_period": 5,
                "company_count": 10
            }
        }
    }

class SentimentRequest(BaseModel):
    symbol: str = Field(..., description="Stock symbol for sentiment analysis")
    days: int = Field(default=7, ge=1, le=30, description="Number of days for news analysis")

    model_config = {
        "json_schema_extra": {
            "example": {
                "symbol": "AAPL",
                "days": 7
            }
        }
    }

class TradingResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict] = None

class TestOrderRequest(BaseModel):
    symbol: str = Field(..., description="Stock symbol to buy (e.g., AAPL)")
    quantity: int = Field(..., gt=0, description="Number of shares to buy")

    model_config = {
        "json_schema_extra": {
            "example": {
                "symbol": "AAPL",
                "quantity": 1
            }
        }
    }

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Financial Product Backend API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/analyze/stock")
async def analyze_stock(request: StockAnalysisRequest):
    try:
        # Validate symbol
        if not request.symbol or not request.symbol.strip():
            raise HTTPException(
                status_code=400,
                detail="Stock symbol is required"
            )
        
        # Clean up the symbol
        symbol = request.symbol.strip().upper()
        if not re.match("^[A-Z]{1,5}$", symbol):
            raise HTTPException(
                status_code=400,
                detail="Invalid stock symbol format. Please use 1-5 letters."
            )

        # Get stock data with error handling
        try:
            stock = yf.Ticker(symbol)
            
            # First verify if the stock exists by trying to get info
            info = stock.info
            if not info or 'regularMarketPrice' not in info:
                raise HTTPException(
                    status_code=404,
                    detail=f"No data found for symbol {symbol}. Please verify the stock symbol is correct."
                )
            
            # Get historical data
            hist = stock.history(period=request.period)
            if hist.empty:
                raise HTTPException(
                    status_code=404,
                    detail=f"No historical data available for symbol {symbol} in the specified period."
                )
            
        except HTTPException as he:
            raise he
        except Exception as e:
            error_msg = str(e)
            if "not found" in error_msg.lower() or "not exist" in error_msg.lower():
                raise HTTPException(
                    status_code=404,
                    detail=f"Stock symbol {symbol} not found. Please verify the symbol is correct."
                )
            raise HTTPException(
                status_code=500,
                detail=f"Error fetching stock data: {error_msg}"
            )

        # Calculate technical indicators with error handling
        try:
            hist['SMA20'] = ta.trend.sma_indicator(hist['Close'], window=20)
            hist['SMA50'] = ta.trend.sma_indicator(hist['Close'], window=50)
            hist['RSI'] = ta.momentum.rsi(hist['Close'], window=14)
            hist['MACD'] = ta.trend.macd_diff(hist['Close'])
            bb_indicator = ta.volatility.BollingerBands(hist['Close'])
            hist['BB_upper'] = bb_indicator.bollinger_hband()
            hist['BB_middle'] = bb_indicator.bollinger_mavg()
            hist['BB_lower'] = bb_indicator.bollinger_lband()
            
            # Calculate volatility and returns
            hist['Returns'] = hist['Close'].pct_change()
            volatility = float(hist['Returns'].std() * np.sqrt(252))
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error calculating technical indicators: {str(e)}"
            )

        # Prepare data for analysis
        try:
            current_price = float(hist['Close'].iloc[-1])
            price_change = float(((current_price - hist['Close'].iloc[0]) / hist['Close'].iloc[0]) * 100)
            avg_volume = float(hist['Volume'].mean())
            rsi = float(hist['RSI'].iloc[-1])
            macd = float(hist['MACD'].iloc[-1])
            sma20 = float(hist['SMA20'].iloc[-1])
            sma50 = float(hist['SMA50'].iloc[-1])
            trend = 'Bullish' if sma20 > sma50 else 'Bearish'
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error preparing analysis data: {str(e)}"
            )

        # Generate analysis prompt
        analysis_prompt = f"""
        Provide a comprehensive stock analysis for {symbol} ({info.get('longName', 'Unknown Company')}) based on the following data:

        Technical Analysis:
        - Current Price: ${current_price:.2f}
        - Price Change: {price_change:.2f}%
        - Trend: {trend}
        - RSI (14): {rsi:.2f}
        - MACD: {macd:.2f}
        - 20-day SMA: ${sma20:.2f}
        - 50-day SMA: ${sma50:.2f}
        - Volatility (Annual): {volatility:.2f}%
        - Average Volume: {avg_volume:,.0f}

        Company Information:
        - Sector: {info.get('sector', 'N/A')}
        - Industry: {info.get('industry', 'N/A')}
        - Market Cap: ${info.get('marketCap', 0):,.2f}
        - P/E Ratio: {info.get('trailingPE', 'N/A')}
        - 52-Week High: ${info.get('fiftyTwoWeekHigh', 0):,.2f}
        - 52-Week Low: ${info.get('fiftyTwoWeekLow', 0):,.2f}

        Please provide:
        1. Technical Analysis Summary
        2. Key Support and Resistance Levels
        3. Volume Analysis
        4. Market Position and Trend Analysis
        5. Risk Assessment
        6. Short-term and Long-term Outlook
        7. Trading Recommendations

        Keep the analysis professional, concise, and actionable.
        """

        # Get analysis from AI service with error handling
        try:
            response = ai_service.generate_content(analysis_prompt)
            if not response or not response.text:
                raise ValueError("Failed to generate analysis")
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error generating AI analysis: {str(e)}"
            )

        return {
            "symbol": symbol,
            "company_name": info.get('longName', 'Unknown Company'),
            "sector": info.get('sector', 'N/A'),
            "industry": info.get('industry', 'N/A'),
            "current_price": current_price,
            "price_change": price_change,
            "market_cap": info.get('marketCap', 0),
            "pe_ratio": info.get('trailingPE', None),
            "fifty_two_week": {
                "high": info.get('fiftyTwoWeekHigh', 0),
                "low": info.get('fiftyTwoWeekLow', 0)
            },
            "technical_indicators": {
                "trend": trend,
                "rsi": rsi,
                "macd": macd,
                "sma20": sma20,
                "sma50": sma50,
                "volatility": volatility,
                "average_volume": avg_volume
            },
            "analysis": response.text
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error during stock analysis: {str(e)}"
        )

@app.post("/analyze/sentiment")
async def analyze_sentiment(request: SentimentRequest):
    if not news_api_key:
        raise HTTPException(status_code=500, detail="NewsAPI key not configured")
    
    try:
        # Get news articles
        news = news_client.get_everything(
            q=f"{request.symbol} stock",
            language='en',
            from_param=(datetime.now() - timedelta(days=request.days)).strftime('%Y-%m-%d'),
            to=datetime.now().strftime('%Y-%m-%d'),
            sort_by='relevancy'
        )
        
        if not news['articles']:
            return {
                "symbol": request.symbol,
                "sentiment_analysis": "No recent news found for analysis",
                "news_count": 0,
                "period_days": request.days,
                "overall_sentiment": "neutral",
                "confidence": 0
            }
        
        # Get company info for context
        stock = yf.Ticker(request.symbol)
        company_name = stock.info.get('longName', request.symbol)
        
        # Prepare articles for analysis
        articles = news['articles'][:10]  # Analyze top 10 most relevant articles
        articles_text = "\n\n".join([
            f"Title: {article['title']}\n"
            f"Source: {article['source']['name']}\n"
            f"Date: {article['publishedAt']}\n"
            f"Content: {article['description']}"
            for article in articles
        ])
        
        sentiment_prompt = f"""
        Analyze the sentiment of these news articles about {company_name} ({request.symbol}):

        {articles_text}

        Please provide:
        1. Overall market sentiment (positive/negative/neutral)
        2. Confidence level in the sentiment analysis (0-100%)
        3. Key factors influencing the sentiment
        4. Notable trends or patterns in the news coverage
        5. Potential impact on stock price
        6. Summary of major news events or developments

        Format your response in a clear, structured manner with distinct sections.
        """
        
        sentiment_analysis = ai_service.generate_content(sentiment_prompt)
        
        if not sentiment_analysis.text:
            raise HTTPException(
                status_code=500,
                detail="Failed to generate sentiment analysis"
            )
        
        return {
            "symbol": request.symbol,
            "company_name": company_name,
            "sentiment_analysis": sentiment_analysis.text,
            "news_count": len(news['articles']),
            "analyzed_articles": len(articles),
            "period_days": request.days,
            "sources": list(set(article['source']['name'] for article in articles))
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing sentiment: {str(e)}"
        )

@app.post("/portfolio/recommend")
async def recommend_portfolio(request: PortfolioRequest):
    """Generate optimized portfolio recommendations"""
    try:
        logger.info(f"Received portfolio request: {request.model_dump()}")
        
        # Generate portfolio using the new generator
        try:
            portfolio_data = portfolio_generator.generate_portfolio(request.model_dump())
            logger.info("Portfolio generation successful")
            
            # Generate AI analysis
            try:
                analysis_prompt = f"""
                Analyze the following stock portfolio recommendation:

                Investment Profile:
                - Investment Amount: ${request.investment_amount:,.2f}
                - Risk Appetite: {request.risk_appetite}
                - Investment Period: {request.investment_period} years
                
                Portfolio Summary:
                - Total Investment: ${portfolio_data['portfolio']['recommendations']['allocation_summary']['total_investment']:,.2f}
                - Total Stocks: {portfolio_data['portfolio']['recommendations']['allocation_summary']['total_stocks']}
                - Total Sectors: {portfolio_data['portfolio']['recommendations']['allocation_summary']['total_sectors']}
                
                Stock Recommendations:
                {chr(10).join([
                    f"- {sector}:" + chr(10) + chr(10).join([
                        f"  * {stock['symbol']}: {stock['weight']:.1f}% (${stock['amount']:,.2f}, {stock['suggested_shares']} shares)"
                        f"  Risk: {stock['risk_level']}"
                        for stock in stocks
                    ])
                    for sector, stocks in portfolio_data['portfolio']['recommendations']['stock_recommendations'].items()
                ])}

                Please provide:
                1. Overall Portfolio Strategy
                2. Risk Assessment
                3. Investment Timeline Strategy
                4. Rebalancing Recommendations
                5. Key Considerations and Warnings
                """
                
                ai_analysis = ai_service.generate_content(analysis_prompt)
                logger.info("AI analysis generated successfully")
                portfolio_data['analysis'] = ai_analysis.text if ai_analysis else "Analysis not available"
            except Exception as e:
                logger.error(f"AI analysis generation failed: {str(e)}")
                portfolio_data['analysis'] = "AI analysis generation failed"
            
            return portfolio_data
            
        except Exception as e:
            logger.error(f"Portfolio generation failed: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error during portfolio generation: {str(e)}"
            )
        
    except HTTPException as he:
        logger.error(f"HTTP Exception in portfolio recommendation: {str(he)}")
        raise he
    except Exception as e:
        logger.error(f"Unexpected error in portfolio recommendation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating portfolio recommendation: {str(e)}"
        )

@app.get("/trading/account")
async def get_trading_account():
    """Get paper trading account information"""
    try:
        account = trading_service.get_account()
        return TradingResponse(
            success=True,
            message="Account information retrieved successfully",
            data={
                "buying_power": float(account.buying_power),
                "cash": float(account.cash),
                "portfolio_value": float(account.portfolio_value),
                "currency": account.currency
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting account information: {str(e)}"
        )

@app.get("/trading/positions")
async def get_positions():
    """Get current positions in paper trading account"""
    try:
        positions = trading_service.get_positions()
        positions_data = [{
            "symbol": pos.symbol,
            "quantity": float(pos.qty),
            "market_value": float(pos.market_value),
            "avg_entry_price": float(pos.avg_entry_price),
            "current_price": float(pos.current_price),
            "unrealized_pl": float(pos.unrealized_pl)
        } for pos in positions]
        
        return TradingResponse(
            success=True,
            message="Positions retrieved successfully",
            data={"positions": positions_data}  # Wrap the list in a dictionary
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting positions: {str(e)}"
        )

@app.post("/trading/execute-portfolio")
async def execute_portfolio(portfolio_allocation: List[Dict]):
    """Execute trades based on portfolio allocation"""
    try:
        orders = trading_service.create_portfolio_orders(portfolio_allocation)
        return TradingResponse(
            success=True,
            message="Portfolio orders executed successfully",
            data=orders
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error executing portfolio orders: {str(e)}"
        )

@app.post("/trading/close-positions")
async def close_positions():
    """Close all positions in paper trading account"""
    try:
        result = trading_service.close_all_positions()
        return TradingResponse(
            success=True,
            message="All positions closed successfully",
            data=result
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error closing positions: {str(e)}"
        )

@app.post("/trading/test-buy")
async def test_buy_stock(request: TestOrderRequest):
    """Test endpoint to buy a single stock"""
    try:
        result = trading_service.test_buy_single_stock(
            symbol=request.symbol.upper(),
            quantity=request.quantity
        )
        return TradingResponse(
            success=True,
            message=f"Test order placed successfully for {request.quantity} shares of {request.symbol}",
            data=result
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error placing test order: {str(e)}"
        ) 