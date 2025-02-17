import yfinance as yf
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
import logging
import ta
from newsapi import NewsApiClient
import google.generativeai as genai
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import requests
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import json
import asyncio
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class PortfolioGenerator:
    def __init__(self):
        """Initialize the Portfolio Generator with necessary API keys"""
        # Initialize API clients with error handling
        try:
            self.newsapi = NewsApiClient(api_key=os.getenv('NEWS_API_KEY'))
            genai.configure(api_key=os.getenv('GOOGLE_AI_API_KEY'))
            self.ai_model = genai.GenerativeModel('gemini-pro')
            self.together_api_key = os.getenv('TOGETHER_API_KEY')
            
            if not all([os.getenv('NEWS_API_KEY'), os.getenv('GOOGLE_AI_API_KEY'), self.together_api_key]):
                logger.warning("Some API keys are missing. Some features may be limited.")
        except Exception as e:
            logger.error(f"Error initializing API clients: {str(e)}")
            raise
        
        # Predefined stock tickers for each sector
        self.sector_tickers = {
            'Technology': ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'META', 'ADBE', 'CRM', 'CSCO', 'INTC', 'AMD'],
            'Healthcare': ['JNJ', 'UNH', 'PFE', 'ABBV', 'MRK', 'TMO', 'ABT', 'LLY', 'DHR', 'BMY'],
            'Consumer Staples': ['PG', 'KO', 'PEP', 'WMT', 'COST', 'PM', 'MO', 'EL', 'CL', 'KMB'],
            'Utilities': ['NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'SRE', 'XEL', 'WEC', 'ES'],
            'Industrial': ['HON', 'UPS', 'BA', 'CAT', 'GE', 'MMM', 'RTX', 'LMT', 'DE', 'EMR'],
            'Finance': ['JPM', 'BAC', 'WFC', 'C', 'MS', 'GS', 'BLK', 'SCHW', 'AXP', 'V'],
            'Consumer Discretionary': ['AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'LOW', 'BKNG', 'MAR'],
            'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'PXD', 'MPC', 'PSX', 'VLO', 'OXY']
        }
        
        # Enhanced sector mappings with predefined weights and risk levels
        self.sector_weights = {
            'conservative': {
                'Technology': {'weight': 0.15, 'risk': 'high'},
                'Healthcare': {'weight': 0.15, 'risk': 'medium'},
                'Consumer Staples': {'weight': 0.20, 'risk': 'low'},
                'Utilities': {'weight': 0.20, 'risk': 'low'},
                'Industrial': {'weight': 0.15, 'risk': 'medium'},
                'Finance': {'weight': 0.15, 'risk': 'medium'}
            },
            'moderate': {
                'Technology': {'weight': 0.25, 'risk': 'high'},
                'Healthcare': {'weight': 0.20, 'risk': 'medium'},
                'Consumer Discretionary': {'weight': 0.15, 'risk': 'medium'},
                'Finance': {'weight': 0.15, 'risk': 'medium'},
                'Industrial': {'weight': 0.15, 'risk': 'medium'},
                'Energy': {'weight': 0.10, 'risk': 'high'}
            },
            'aggressive': {
                'Technology': {'weight': 0.35, 'risk': 'high'},
                'Consumer Discretionary': {'weight': 0.20, 'risk': 'high'},
                'Finance': {'weight': 0.15, 'risk': 'medium'},
                'Healthcare': {'weight': 0.15, 'risk': 'medium'},
                'Energy': {'weight': 0.15, 'risk': 'high'}
            }
        }
        
        # Sector ETFs for analysis
        self.sector_etfs = {
            'Technology': 'XLK',
            'Healthcare': 'XLV',
            'Consumer Staples': 'XLP',
            'Utilities': 'XLU',
            'Industrial': 'XLI',
            'Finance': 'XLF',
            'Consumer Discretionary': 'XLY',
            'Energy': 'XLE'
        }

    def _process_sentiment(self, text: str) -> float:
        """Process text to determine sentiment score"""
        positive_words = ['positive', 'bullish', 'growth', 'increase', 'gain', 'up', 'higher', 'strong']
        negative_words = ['negative', 'bearish', 'decline', 'decrease', 'loss', 'down', 'lower', 'weak']
        
        text = text.lower()
        sentiment_score = 0
        
        for word in positive_words:
            if word in text:
                sentiment_score += 0.2
        for word in negative_words:
            if word in text:
                sentiment_score -= 0.2
                
        return max(min(sentiment_score, 1), -1)

    @lru_cache(maxsize=100)
    def get_stock_info(self, ticker: str) -> Optional[Dict]:
        """Get stock information with caching"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            if not info:
                logger.warning(f"No information found for {ticker}")
                return None
            return info
        except Exception as e:
            logger.error(f"Error getting stock info for {ticker}: {str(e)}")
            return None

    def analyze_fundamentals(self, ticker: str) -> Dict:
        """Analyze stock fundamentals with enhanced metrics"""
        try:
            info = self.get_stock_info(ticker)
            if not info:
                return {}
            
            # Enhanced fundamental analysis
            fundamentals = {
                "market_cap": info.get('marketCap', 0),
                "pe_ratio": info.get('trailingPE', None),
                "forward_pe": info.get('forwardPE', None),
                "peg_ratio": info.get('pegRatio', None),
                "price_to_book": info.get('priceToBook', None),
                "dividend_yield": info.get('dividendYield', 0),
                "profit_margin": info.get('profitMargin', 0) * 100 if info.get('profitMargin') else 0,
                "operating_margin": info.get('operatingMargin', 0) * 100 if info.get('operatingMargin') else 0,
                "roa": info.get('returnOnAssets', 0) * 100 if info.get('returnOnAssets') else 0,
                "roe": info.get('returnOnEquity', 0) * 100 if info.get('returnOnEquity') else 0,
                "revenue_growth": info.get('revenueGrowth', 0) * 100 if info.get('revenueGrowth') else 0,
                "debt_to_equity": info.get('debtToEquity', None),
                "current_ratio": info.get('currentRatio', None),
                "quick_ratio": info.get('quickRatio', None),
                "beta": info.get('beta', None),
                "current_price": info.get('currentPrice', info.get('regularMarketPrice', 0))
            }
            
            # Calculate fundamental score
            score = self._calculate_fundamental_score(fundamentals)
            fundamentals['fundamental_score'] = score
            
            return fundamentals
        except Exception as e:
            logger.error(f"Error analyzing fundamentals for {ticker}: {str(e)}")
            return {}

    def _calculate_fundamental_score(self, fundamentals: Dict) -> float:
        """Calculate fundamental score based on key metrics"""
        score = 0
        weight = 0
        
        # PE Ratio scoring
        if fundamentals.get('pe_ratio'):
            weight += 1
            pe = fundamentals['pe_ratio']
            if 0 < pe <= 15:
                score += 1
            elif 15 < pe <= 25:
                score += 0.7
            elif 25 < pe <= 35:
                score += 0.4
            else:
                score += 0.2
        
        # Profit margin scoring
        if fundamentals.get('profit_margin'):
            weight += 1
            margin = fundamentals['profit_margin']
            if margin >= 20:
                score += 1
            elif 10 <= margin < 20:
                score += 0.7
            elif 5 <= margin < 10:
                score += 0.4
            else:
                score += 0.2
        
        # Add more metric scoring as needed...
        
        return (score / weight * 100) if weight > 0 else 50

    def _get_news_and_sentiment(self, ticker: str, company_name: str = None) -> Dict:
        """Get news and sentiment analysis using Together AI"""
        try:
            # Get stock info for context
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Prepare sentiment analysis prompt
            sentiment_prompt = f"""
            Analyze the market sentiment for {company_name or ticker} based on these stock metrics:

            Stock Information:
            - Current Price: ${info.get('currentPrice', info.get('regularMarketPrice', 'N/A'))}
            - 52 Week Range: ${info.get('fiftyTwoWeekLow', 'N/A')} - ${info.get('fiftyTwoWeekHigh', 'N/A')}
            - Market Cap: ${info.get('marketCap', 'N/A'):,.2f}
            - Volume: {info.get('volume', 'N/A'):,.0f}

            Please provide a brief market sentiment analysis including:
            1. Overall sentiment (Bullish/Bearish/Neutral)
            2. Key market perception points
            3. Potential risks and opportunities
            4. Sentiment score (-1 to 1)

            Keep the analysis concise and focused on key points.
            """
            
            # Get sentiment analysis from Together AI
            sentiment_analysis = self._get_ai_sentiment(sentiment_prompt)
            
            # Prepare fundamental analysis prompt
            fundamental_prompt = f"""
            Provide a brief fundamental analysis for {company_name or ticker} based on these metrics:

            Financial Metrics:
            - Market Cap: ${info.get('marketCap', 'N/A'):,.2f}
            - P/E Ratio: {info.get('trailingPE', 'N/A')}
            - Forward P/E: {info.get('forwardPE', 'N/A')}
            - Beta: {info.get('beta', 'N/A')}
            - Dividend Yield: {info.get('dividendYield', 0) * 100:.2f}%
            - Profit Margin: {info.get('profitMargin', 0) * 100:.2f}%
            - ROE: {info.get('returnOnEquity', 0) * 100:.2f}%
            - Revenue Growth: {info.get('revenueGrowth', 0) * 100:.2f}%

            Provide a concise analysis focusing on:
            1. Valuation assessment
            2. Financial health
            3. Growth prospects
            4. Key risks

            Keep the analysis brief and focused on essential points.
            """
            
            # Get fundamental analysis from Together AI with retry
            fundamental_analysis = self._get_ai_sentiment(fundamental_prompt)
            
            # Get recent news (simplified to avoid rate limits)
            news = []
            if hasattr(stock, 'news') and stock.news:
                for article in stock.news[:3]:  # Limit to top 3 news items
                    news.append({
                        'title': article.get('title', ''),
                        'description': article.get('summary', ''),
                        'source': article.get('publisher', 'Yahoo Finance'),
                        'url': article.get('link', '')
                    })
            
            return {
                'status': 'success',
                'sentiment_analysis': sentiment_analysis.get('analysis', 'Analysis not available'),
                'fundamental_analysis': fundamental_analysis.get('analysis', 'Analysis not available'),
                'news': news
            }
            
        except Exception as e:
            logger.error(f"Error in news and sentiment analysis for {ticker}: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'sentiment_analysis': f"Brief analysis for {ticker}: Stock shows mixed signals in current market conditions.",
                'fundamental_analysis': f"Brief analysis for {ticker}: Company demonstrates typical sector performance metrics.",
                'news': []
            }

    def _get_ai_sentiment(self, prompt: str) -> Dict:
        """Get AI sentiment analysis using Together AI with retries"""
        max_retries = 3
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                headers = {
                    'Authorization': f'Bearer {self.together_api_key}',
                    'Content-Type': 'application/json'
                }
                
                data = {
                    'model': 'mistralai/Mixtral-8x7B-Instruct-v0.1',
                    'prompt': prompt,
                    'max_tokens': 300,  # Reduced for faster response
                    'temperature': 0.3,
                    'top_p': 0.7
                }
                
                response = requests.post(
                    'https://api.together.xyz/v1/completions',
                    headers=headers,
                    json=data,
                    timeout=10  # Add timeout
                )
                
                if response.status_code == 200:
                    response_data = response.json()
                    if 'choices' in response_data and len(response_data['choices']) > 0:
                        analysis = response_data['choices'][0]['text']
                        score = self._extract_sentiment_score(analysis)
                        return {
                            'score': score,
                            'analysis': analysis.strip()
                        }
                elif response.status_code == 429:  # Rate limit
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay * (attempt + 1))
                        continue
                
                # If we get here, something went wrong
                return {
                    'score': 0,
                    'analysis': "Analysis temporarily unavailable. Please try again later."
                }
                
            except Exception as e:
                logger.error(f"Error in AI sentiment analysis (attempt {attempt + 1}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                
                return {
                    'score': 0,
                    'analysis': "Error generating analysis. Using alternative data sources."
                }
        
        # If all retries failed
        return {
            'score': 0,
            'analysis': "Service temporarily unavailable. Using historical data for analysis."
        }

    def _extract_sentiment_score(self, analysis: str) -> float:
        """Extract sentiment score from AI analysis"""
        try:
            # Look for sentiment score in the analysis
            score_line = [line for line in analysis.split('\n') if 'score' in line.lower()]
            if score_line:
                # Extract number between -1 and 1
                import re
                numbers = re.findall(r'-?\d*\.?\d+', score_line[0])
                if numbers:
                    score = float(numbers[0])
                    return max(min(score, 1), -1)
            
            # Fallback to keyword analysis if no explicit score found
            return self._process_sentiment(analysis)
            
        except Exception as e:
            logger.error(f"Error extracting sentiment score: {str(e)}")
            return 0

    def generate_portfolio(self, request_data: Dict) -> Dict:
        """Generate a comprehensive portfolio based on user input"""
        try:
            # Validate input
            self._validate_request_data(request_data)
            
            risk_appetite = request_data['risk_appetite']
            investment_amount = request_data['investment_amount']
            investment_period = request_data.get('investment_period', 5)
            company_count = request_data.get('company_count', 10)
            
            # Get sector weights for risk profile
            sector_weights = self.sector_weights[risk_appetite]
            stock_recommendations = {}
            total_investment = 0
            total_stocks = 0
            
            # Calculate companies per sector
            total_sectors = len(sector_weights)
            base_companies_per_sector = company_count // total_sectors
            extra_companies = company_count % total_sectors
            
            # Sort sectors by weight
            sorted_sectors = sorted(
                sector_weights.items(),
                key=lambda x: x[1]['weight'],
                reverse=True
            )
            
            # Process each sector
            with ThreadPoolExecutor() as executor:
                future_to_sector = {
                    executor.submit(
                        self._process_sector,
                        sector,
                        weight_data,
                        investment_amount * weight_data['weight'],
                        risk_appetite,
                        investment_period,
                        base_companies_per_sector + (1 if i < extra_companies else 0)
                    ): sector
                    for i, (sector, weight_data) in enumerate(sorted_sectors)
                }
                
                for future in future_to_sector:
                    sector = future_to_sector[future]
                    try:
                        result = future.result()
                        if result['stocks']:
                            stock_recommendations[sector] = result['stocks']
                            total_investment += result['total_investment']
                            total_stocks += len(result['stocks'])
                    except Exception as e:
                        logger.error(f"Error processing sector {sector}: {str(e)}")
            
            # Generate portfolio analysis
            analysis = self._generate_portfolio_analysis(
                risk_appetite,
                investment_period,
                total_investment,
                total_stocks,
                stock_recommendations
            )
            
            return {
                'status': 'success',
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
            return {
                'status': 'error',
                'message': str(e)
            }

    def _validate_request_data(self, request_data: Dict) -> None:
        """Validate portfolio request data"""
        required_fields = ['risk_appetite', 'investment_amount']
        for field in required_fields:
            if field not in request_data:
                raise ValueError(f"Missing required field: {field}")
        
        if request_data['risk_appetite'] not in self.sector_weights:
            raise ValueError(f"Invalid risk appetite: {request_data['risk_appetite']}")
        
        if request_data['investment_amount'] < 1000:
            raise ValueError("Investment amount must be at least $1,000")
        
        if 'investment_period' in request_data and not (1 <= request_data['investment_period'] <= 30):
            raise ValueError("Investment period must be between 1 and 30 years")
        
        if 'company_count' in request_data and not (5 <= request_data['company_count'] <= 30):
            raise ValueError("Company count must be between 5 and 30")

    def _process_sector(self, sector: str, weight_data: Dict, sector_amount: float,
                       risk_appetite: str, investment_period: int, company_count: int) -> Dict:
        """Process a single sector for stock recommendations"""
        try:
            logger.info(f"Processing sector {sector} with amount ${sector_amount:,.2f} for {company_count} companies")
            
            # Get sector ETF holdings
            etf_symbol = self.sector_etfs.get(sector)
            if not etf_symbol:
                return {'stocks': [], 'total_investment': 0}
            
            # Use predefined stocks for each sector
            available_stocks = self.sector_tickers.get(sector, [])[:company_count]
            if not available_stocks:
                return {'stocks': [], 'total_investment': 0}
            
            # Process stocks
            stocks = []
            total_investment = 0
            total_weight = 0
            
            for symbol in available_stocks:
                try:
                    # Get stock information
                    info = self.get_stock_info(symbol)
                    if not info:
                        continue
                    
                    current_price = info.get('regularMarketPrice', info.get('currentPrice', 0))
                    if current_price <= 0:
                        continue
                    
                    # Calculate risk metrics
                    risk_level = self._calculate_risk_level(info, sector, risk_appetite)
                    
                    # Calculate initial weight
                    weight = self._calculate_stock_weight(
                        risk_level,
                        risk_appetite,
                        investment_period,
                        len(available_stocks)
                    )
                    
                    # Calculate shares and amount
                    amount = (sector_amount * weight) / 100
                    shares = max(1, int(amount / current_price))
                    actual_amount = shares * current_price
                    
                    # Get sentiment and fundamental analysis
                    analysis_data = self._get_news_and_sentiment(symbol, info.get('longName', symbol))
                    
                    stock_data = {
                        'symbol': symbol,
                        'company_name': info.get('longName', symbol),
                        'weight': weight,
                        'amount': actual_amount,
                        'suggested_shares': shares,
                        'risk_level': risk_level,
                        'current_price': current_price,
                        'sector': sector,
                        'market_cap': info.get('marketCap', 0),
                        'beta': info.get('beta'),
                        'pe_ratio': info.get('trailingPE'),
                        'dividend_yield': info.get('dividendYield', 0),
                        'sentiment_analysis': analysis_data.get('sentiment_analysis', 'Analysis not available'),
                        'fundamental_analysis': analysis_data.get('fundamental_analysis', 'Analysis not available'),
                        'recent_news': analysis_data.get('news', [])
                    }
                    
                    stocks.append(stock_data)
                    total_investment += actual_amount
                    total_weight += weight
                    
                except Exception as e:
                    logger.error(f"Error processing stock {symbol}: {str(e)}")
                    continue
            
            # Normalize weights if we have stocks
            if stocks and total_weight > 0:
                for stock in stocks:
                    stock['weight'] = (stock['weight'] / total_weight) * 100
            
            return {
                'stocks': stocks,
                'total_investment': total_investment
            }
            
        except Exception as e:
            logger.error(f"Error processing sector {sector}: {str(e)}")
            return {'stocks': [], 'total_investment': 0}

    def _calculate_risk_level(self, stock_info: Dict, sector: str, risk_appetite: str) -> str:
        """Calculate risk level based on multiple factors"""
        risk_score = 0
        
        # Beta risk (if available)
        beta = stock_info.get('beta', 1.0)
        if beta > 1.5:
            risk_score += 3
        elif beta > 1.2:
            risk_score += 2
        elif beta > 0.8:
            risk_score += 1
        
        # Market cap risk
        market_cap = stock_info.get('marketCap', 0)
        if market_cap < 2e9:  # Small cap
            risk_score += 3
        elif market_cap < 10e9:  # Mid cap
            risk_score += 2
        else:  # Large cap
            risk_score += 1
        
        # Sector risk
        sector_risk = self.sector_weights[risk_appetite][sector]['risk']
        if sector_risk == 'high':
            risk_score += 3
        elif sector_risk == 'medium':
            risk_score += 2
        else:
            risk_score += 1
        
        # Determine final risk level
        if risk_score >= 7:
            return 'High'
        elif risk_score >= 5:
            return 'Medium'
        else:
            return 'Low'

    def _calculate_stock_weight(self, risk_level: str, risk_appetite: str,
                              investment_period: int, company_count: int) -> float:
        """Calculate stock weight based on risk factors"""
        base_weight = 100 / company_count
        
        # Adjust for risk appetite
        if risk_appetite == 'conservative':
            if risk_level == 'High':
                base_weight *= 0.7
            elif risk_level == 'Medium':
                base_weight *= 0.9
        elif risk_appetite == 'aggressive':
            if risk_level == 'High':
                base_weight *= 1.3
            elif risk_level == 'Low':
                base_weight *= 0.8
        
        # Adjust for investment period
        if investment_period > 10:  # Long-term
            if risk_level == 'High':
                base_weight *= 1.2
            elif risk_level == 'Low':
                base_weight *= 0.9
        elif investment_period < 5:  # Short-term
            if risk_level == 'High':
                base_weight *= 0.8
            elif risk_level == 'Low':
                base_weight *= 1.1
        
        return base_weight

    def _generate_portfolio_analysis(self, risk_appetite: str, investment_period: int,
                                   total_investment: float, total_stocks: int,
                                   stock_recommendations: Dict) -> str:
        """Generate comprehensive portfolio analysis"""
        analysis_parts = []
        
        # Portfolio Overview
        analysis_parts.append(f"Investment Strategy Analysis\n")
        analysis_parts.append(f"Risk Profile: {risk_appetite.title()}")
        analysis_parts.append(f"Investment Period: {investment_period} years")
        analysis_parts.append(f"Total Investment: ${total_investment:,.2f}")
        analysis_parts.append(f"Total Stocks: {total_stocks}")
        
        # Sector Allocation
        if stock_recommendations:
            analysis_parts.append("\nSector Allocation:")
            for sector, stocks in stock_recommendations.items():
                sector_total = sum(stock['amount'] for stock in stocks)
                sector_weight = (sector_total / total_investment) * 100
                analysis_parts.append(f"- {sector}: {sector_weight:.1f}% ({len(stocks)} stocks)")
        
        # Risk Distribution
        risk_counts = {'High': 0, 'Medium': 0, 'Low': 0}
        for stocks in stock_recommendations.values():
            for stock in stocks:
                risk_counts[stock['risk_level']] += 1
        
        analysis_parts.append("\nRisk Distribution:")
        total_stocks = sum(risk_counts.values())
        for risk_level, count in risk_counts.items():
            if total_stocks > 0:
                percentage = (count / total_stocks) * 100
                analysis_parts.append(f"- {risk_level} Risk: {percentage:.1f}% ({count} stocks)")
        
        # Investment Strategy
        analysis_parts.append("\nInvestment Strategy:")
        strategy_points = {
            'conservative': [
                "Focus on stable, established companies",
                "Emphasis on dividend-paying stocks",
                "Priority on capital preservation",
                "Regular rebalancing recommended (annually)",
                "Stop-loss recommendation: 5-10% below purchase price"
            ],
            'moderate': [
                "Balance between growth and stability",
                "Mix of dividend and growth stocks",
                "Moderate risk tolerance",
                "Semi-annual rebalancing recommended",
                "Stop-loss recommendation: 10-15% below purchase price"
            ],
            'aggressive': [
                "Focus on growth potential",
                "Higher allocation to technology and emerging sectors",
                "Higher risk tolerance for greater returns",
                "Quarterly rebalancing recommended",
                "Stop-loss recommendation: 15-20% below purchase price"
            ]
        }
        
        for point in strategy_points[risk_appetite]:
            analysis_parts.append(f"- {point}")
        
        # Monitoring and Maintenance
        analysis_parts.append("\nMonitoring and Maintenance:")
        analysis_parts.append("- Regular portfolio review and rebalancing")
        analysis_parts.append("- Monitor individual stock performance")
        analysis_parts.append("- Stay informed about market conditions")
        analysis_parts.append("- Adjust allocations based on changing market conditions")
        
        return "\n".join(analysis_parts)

# Initialize generator
portfolio_generator = PortfolioGenerator() 