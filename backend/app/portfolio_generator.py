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
        
        # Extended sector tickers with more options
        self.sector_tickers = {
            'Technology': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'AMD', 'ADBE', 'CRM', 'INTC', 'CSCO', 'ORCL'],
            'Healthcare': ['JNJ', 'UNH', 'PFE', 'ABT', 'TMO', 'MRK', 'ABBV', 'DHR', 'BMY', 'AMGN'],
            'Finance': ['JPM', 'BAC', 'V', 'MA', 'GS', 'MS', 'BLK', 'C', 'AXP', 'SPGI'],
            'Consumer Staples': ['PG', 'KO', 'PEP', 'WMT', 'COST', 'PM', 'TGT', 'EL', 'CL', 'KMB'],
            'Consumer Discretionary': ['AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'TJX', 'LOW', 'BKNG', 'MAR'],
            'Industrial': ['HON', 'UPS', 'CAT', 'DE', 'BA', 'MMM', 'GE', 'LMT', 'RTX', 'UNP'],
            'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'PSX', 'PXD', 'VLO', 'MPC', 'OXY'],
            'Utilities': ['NEE', 'DUK', 'SO', 'D', 'AEP', 'SRE', 'EXC', 'XEL', 'WEC', 'ES']
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
        """Generate a portfolio based on risk appetite, investment amount, and company count"""
        try:
            risk_appetite = request_data['risk_appetite']
            investment_amount = request_data['investment_amount']
            investment_period = request_data.get('investment_period', 5)
            company_count = request_data.get('company_count', 10)  # Get desired company count
            
            sector_weights = self.sector_weights[risk_appetite]
            stock_recommendations = {}
            total_investment = 0
            total_stocks = 0
            
            # Calculate how many companies to pick from each sector
            total_sectors = len(sector_weights)
            base_companies_per_sector = company_count // total_sectors
            extra_companies = company_count % total_sectors
            
            # Sort sectors by weight to allocate extra companies to highest weight sectors
            sorted_sectors = sorted(sector_weights.items(), key=lambda x: x[1], reverse=True)
            
            for sector, weight in sorted_sectors:
                sector_amount = investment_amount * weight
                sector_stocks = []
                
                # Determine number of companies for this sector
                sector_company_count = base_companies_per_sector
                if extra_companies > 0:
                    sector_company_count += 1
                    extra_companies -= 1
                
                # Get all available stocks for this sector
                available_stocks = []
                for ticker in self.sector_tickers[sector]:
                    fundamentals = self.analyze_fundamentals(ticker)
                    if not fundamentals or fundamentals['current_price'] <= 0:
                        continue
                        
                    technicals = self.analyze_technicals(ticker)
                    if not technicals:
                        continue
                        
                    available_stocks.append({
                        'ticker': ticker,
                        'fundamentals': fundamentals,
                        'technical_score': technicals['Overall_Score']
                    })
                
                # Sort stocks by technical score and take the top N for this sector
                available_stocks.sort(key=lambda x: x['technical_score'], reverse=True)
                selected_stocks = available_stocks[:sector_company_count]
                
                # Calculate per-stock allocation for this sector
                per_stock_amount = sector_amount / len(selected_stocks)
                
                for stock in selected_stocks:
                    current_price = stock['fundamentals']['current_price']
                    suggested_shares = max(1, int(per_stock_amount / current_price))
                    actual_amount = suggested_shares * current_price
                    
                    # Determine risk level based on technical score and sector
                    risk_level = self._determine_risk_level(stock['technical_score'], sector, risk_appetite)
                    
                    sector_stocks.append({
                        'symbol': stock['ticker'],
                        'weight': (actual_amount / investment_amount) * 100,  # Calculate actual weight
                        'amount': actual_amount,
                        'suggested_shares': suggested_shares,
                        'risk_level': risk_level,
                        'fundamentals': stock['fundamentals']
                    })
                    
                    total_investment += actual_amount
                    total_stocks += 1
                
                if sector_stocks:
                    stock_recommendations[sector] = sector_stocks
            
            # Generate detailed analysis
            analysis = self._generate_detailed_analysis(
                risk_appetite,
                investment_period,
                total_investment,
                total_stocks,
                stock_recommendations,
                sector_weights
            )
            
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
                'analysis': analysis
            }
        except Exception as e:
            logger.error(f"Error generating portfolio: {str(e)}")
            raise

    def _determine_risk_level(self, technical_score: float, sector: str, risk_appetite: str) -> str:
        """Determine risk level based on technical score, sector, and risk appetite"""
        high_risk_sectors = ['Technology', 'Energy']
        medium_risk_sectors = ['Finance', 'Consumer Discretionary', 'Industrial']
        
        # Base risk on technical score
        if technical_score >= 75:
            base_risk = 'low'
        elif technical_score >= 50:
            base_risk = 'medium'
        else:
            base_risk = 'high'
            
        # Adjust for sector
        if sector in high_risk_sectors:
            base_risk = 'high' if base_risk != 'low' else 'medium'
        elif sector in medium_risk_sectors:
            base_risk = 'medium' if base_risk == 'low' else base_risk
            
        # Adjust for risk appetite
        if risk_appetite == 'aggressive':
            return base_risk
        elif risk_appetite == 'moderate':
            return 'medium' if base_risk == 'high' else base_risk
        else:  # conservative
            return 'medium' if base_risk == 'high' else 'low'

    def _generate_detailed_analysis(self, risk_appetite: str, investment_period: int,
                                  total_investment: float, total_stocks: int,
                                  stock_recommendations: Dict, sector_weights: Dict) -> str:
        """Generate detailed portfolio analysis"""
        sector_allocations = []
        for sector, stocks in stock_recommendations.items():
            sector_total = sum(stock['amount'] for stock in stocks)
            sector_weight = (sector_total / total_investment) * 100
            sector_allocations.append(f"- {sector}: {sector_weight:.1f}% ({len(stocks)} stocks)")
        
        risk_descriptions = {
            'conservative': "focuses on stable, established companies with strong fundamentals and lower volatility",
            'moderate': "balances growth potential with stability through a mix of established and growing companies",
            'aggressive': "emphasizes growth potential through technology and emerging market leaders"
        }
        
        rebalancing_frequency = {
            'conservative': 'annually',
            'moderate': 'semi-annually',
            'aggressive': 'quarterly'
        }
        
        analysis = f"""Investment Strategy Analysis:

Risk Profile: {risk_appetite.title()}
Investment Period: {investment_period} years
Total Investment: ${total_investment:,.2f}
Total Stocks: {total_stocks}
Total Sectors: {len(stock_recommendations)}

Portfolio Allocation:
{chr(10).join(sector_allocations)}

Strategy Overview:
This {risk_appetite} portfolio {risk_descriptions[risk_appetite]}, designed for a {investment_period}-year investment horizon. The portfolio is diversified across {len(stock_recommendations)} major sectors, with each stock selected based on technical analysis, fundamental strength, and market position.

Key Features:
- Sector diversification to manage risk and capture growth opportunities
- Stock selection based on technical and fundamental analysis
- Risk-adjusted position sizing within sectors
- Focus on liquid, established companies

Rebalancing Recommendation:
- Review and rebalance the portfolio {rebalancing_frequency[risk_appetite]}
- Maintain sector weights within 5% of target allocation
- Monitor individual positions for changes in fundamentals or technical indicators
- Consider tax implications when rebalancing

Risk Management:
- Position sizes are adjusted based on individual stock risk levels
- Sector weights aligned with {risk_appetite} risk profile
- Regular monitoring of technical indicators and fundamentals
- Stop-loss recommendations: {'15-20%' if risk_appetite == 'aggressive' else '10-15%' if risk_appetite == 'moderate' else '5-10%'} below purchase price
"""
        return analysis

# Initialize generator
portfolio_generator = PortfolioGenerator() 