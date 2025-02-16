import yfinance as yf
import numpy as np
import pandas as pd
from typing import Dict, List
import logging
import ta
from newsapi import NewsApiClient
import google.generativeai as genai
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class PortfolioGenerator:
    def __init__(self):
        """Initialize the Portfolio Generator with necessary API keys"""
        # Initialize API clients
        self.newsapi = NewsApiClient(api_key=os.getenv('NEWS_API_KEY'))
        genai.configure(api_key=os.getenv('GOOGLE_AI_API_KEY'))
        self.ai_model = genai.GenerativeModel('gemini-pro')
        
        # Sector mappings with predefined weights
        self.sector_weights = {
            'conservative': {
                'Technology': 0.15,
                'Healthcare': 0.15,
                'Consumer Staples': 0.20,
                'Utilities': 0.20,
                'Industrial': 0.15,
                'Finance': 0.15
            },
            'moderate': {
                'Technology': 0.25,
                'Healthcare': 0.20,
                'Consumer Discretionary': 0.15,
                'Finance': 0.15,
                'Industrial': 0.15,
                'Energy': 0.10
            },
            'aggressive': {
                'Technology': 0.35,
                'Consumer Discretionary': 0.20,
                'Finance': 0.15,
                'Healthcare': 0.15,
                'Energy': 0.15
            }
        }
        
        # Sector tickers
        self.sector_tickers = {
            'Technology': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'AMD'],
            'Healthcare': ['JNJ', 'UNH', 'PFE', 'ABT', 'TMO'],
            'Finance': ['JPM', 'BAC', 'V', 'MA', 'GS'],
            'Consumer Staples': ['PG', 'KO', 'PEP', 'WMT', 'COST'],
            'Consumer Discretionary': ['AMZN', 'TSLA', 'HD', 'MCD', 'NKE'],
            'Industrial': ['HON', 'UPS', 'CAT', 'DE', 'BA'],
            'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG'],
            'Utilities': ['NEE', 'DUK', 'SO', 'D', 'AEP']
        }

    def analyze_fundamentals(self, ticker: str) -> Dict:
        """Analyze fundamental metrics for a given stock"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            fundamentals = {
                'current_price': info.get('regularMarketPrice', 0)
            }
            
            return fundamentals
        except Exception as e:
            logger.error(f"Error analyzing fundamentals for {ticker}: {str(e)}")
            return None

    def analyze_technicals(self, ticker: str) -> Dict:
        """Analyze technical indicators for a given stock"""
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period='1y')
            
            if len(hist) < 50:
                return None
            
            # Calculate technical indicators using ta library
            technicals = {
                'RSI': ta.momentum.rsi(hist['Close'], window=14).iloc[-1],
                'MACD': ta.trend.macd_diff(hist['Close']).iloc[-1],
                'BB_Upper': ta.volatility.bollinger_hband(hist['Close']).iloc[-1],
                'BB_Middle': ta.volatility.bollinger_mavg(hist['Close']).iloc[-1],
                'BB_Lower': ta.volatility.bollinger_lband(hist['Close']).iloc[-1],
                'ADX': ta.trend.adx(hist['High'], hist['Low'], hist['Close']).iloc[-1],
                'Volume_MA': hist['Volume'].rolling(window=20).mean().iloc[-1]
            }
            
            score = self._score_technicals(technicals, hist['Close'].iloc[-1])
            technicals['Overall_Score'] = score
            
            return technicals
        except Exception as e:
            logger.error(f"Error analyzing technicals for {ticker}: {str(e)}")
            return None

    def _score_technicals(self, metrics: Dict, current_price: float) -> float:
        """Score technical metrics"""
        scores = []
        
        # RSI Score (0-100)
        rsi_score = 100 - abs(50 - metrics['RSI'])
        scores.append(rsi_score)
        
        # MACD Score (normalized to 0-100)
        macd_score = 50 + (metrics['MACD'] * 10)
        scores.append(max(min(macd_score, 100), 0))
        
        # Bollinger Bands Position Score (0-100)
        bb_position = (current_price - metrics['BB_Lower']) / (metrics['BB_Upper'] - metrics['BB_Lower'])
        bb_score = 100 - abs(0.5 - bb_position) * 100
        scores.append(bb_score)
        
        # ADX Score (0-100)
        adx_score = min(metrics['ADX'], 100)
        scores.append(adx_score)
        
        return np.mean(scores)

    def get_news_sentiment(self, ticker: str) -> Dict:
        """Analyze news sentiment for a given stock"""
        try:
            news = self.newsapi.get_everything(
                q=ticker,
                language='en',
                sort_by='relevancy',
                page_size=10
            )
            
            articles_text = " ".join([article['title'] + " " + article['description'] for article in news['articles']])
            sentiment_prompt = f"Analyze the sentiment of these news articles about {ticker}: {articles_text}"
            
            sentiment_response = self.ai_model.generate_content(sentiment_prompt)
            sentiment_score = self._process_sentiment(sentiment_response.text)
            
            return {
                'sentiment_score': sentiment_score,
                'articles': news['articles'][:3]
            }
        except Exception as e:
            logger.error(f"Error getting news sentiment for {ticker}: {str(e)}")
            return None

    def _process_sentiment(self, sentiment_text: str) -> float:
        """Process sentiment analysis text to get a numerical score"""
        positive_words = ['positive', 'bullish', 'growth', 'increase', 'gain']
        negative_words = ['negative', 'bearish', 'decline', 'decrease', 'loss']
        
        sentiment_score = 0
        for word in positive_words:
            if word in sentiment_text.lower():
                sentiment_score += 0.2
        for word in negative_words:
            if word in sentiment_text.lower():
                sentiment_score -= 0.2
                
        return max(min(sentiment_score, 1), -1)

    def generate_portfolio(self, request_data: Dict) -> Dict:
        """Generate a portfolio based on risk appetite and investment amount"""
        try:
            risk_appetite = request_data['risk_appetite']
            investment_amount = request_data['investment_amount']
            investment_period = request_data.get('investment_period', 5)
            
            sector_weights = self.sector_weights[risk_appetite]
            stock_recommendations = {}
            total_investment = 0
            total_stocks = 0
            
            for sector, weight in sector_weights.items():
                sector_amount = investment_amount * weight
                sector_stocks = []
                
                for ticker in self.sector_tickers[sector]:
                    fundamentals = self.analyze_fundamentals(ticker)
                    if not fundamentals:
                        continue
                    
                    current_price = fundamentals['current_price']
                    if current_price <= 0:
                        continue
                    
                    # Calculate shares and amount
                    suggested_shares = max(1, int((sector_amount * 0.2) / current_price))  # 20% of sector amount per stock
                    actual_amount = suggested_shares * current_price
                    
                    # Determine risk level based on sector and position
                    risk_level = 'High' if sector in ['Technology', 'Energy'] else \
                                'Medium' if sector in ['Finance', 'Consumer Discretionary'] else 'Low'
                    
                    sector_stocks.append({
                        'symbol': ticker,
                        'weight': 20,  # Equal weight within sector
                        'amount': actual_amount,
                        'suggested_shares': suggested_shares,
                        'risk_level': risk_level,
                        'fundamentals': fundamentals
                    })
                    
                    total_investment += actual_amount
                    total_stocks += 1
                
                if sector_stocks:
                    stock_recommendations[sector] = sector_stocks
            
            # Create the response structure that matches frontend expectations
            return {
                'portfolio': {
                    'recommendations': {
                        'stock_recommendations': stock_recommendations,
                        'allocation_summary': {
                            'total_investment': total_investment,
                            'total_stocks': total_stocks,
                            'total_sectors': len(stock_recommendations)
                        }
                    }
                },
                'analysis': f"""Investment Strategy Analysis:
                
Risk Profile: {risk_appetite.title()}
Investment Period: {investment_period} years
Total Investment: ${total_investment:,.2f}
Total Stocks: {total_stocks}
Total Sectors: {len(stock_recommendations)}

Portfolio Allocation:
{chr(10).join([f'- {sector}: {weight*100:.1f}%' for sector, weight in sector_weights.items()])}

This {risk_appetite} portfolio is designed for a {investment_period}-year investment horizon, with diversification across {len(stock_recommendations)} major sectors. The allocation reflects a {risk_appetite} risk tolerance, with a focus on {'growth and technology sectors' if risk_appetite == 'aggressive' else 'balanced sector exposure' if risk_appetite == 'moderate' else 'stable, dividend-paying sectors'}.

Rebalancing Recommendation:
- Review and rebalance the portfolio {'quarterly' if risk_appetite == 'aggressive' else 'semi-annually' if risk_appetite == 'moderate' else 'annually'}.
- Maintain sector weights within 5% of target allocation.
- Monitor individual positions for any significant changes in fundamentals.
"""
            }
            
        except Exception as e:
            logger.error(f"Error generating portfolio: {str(e)}")
            raise e

# Initialize generator
portfolio_generator = PortfolioGenerator() 