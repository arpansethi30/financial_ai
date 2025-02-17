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
                return {
                    'status': 'error',
                    'analysis': '\n'.join([
                        "Fundamental Metrics: Data temporarily unavailable",
                        "Growth Analysis: Historical sector averages apply",
                        "Risk Metrics: Standard sector risk profile",
                        "Financial Health: Review latest filings for details"
                    ])
                }
            
            # Calculate key metrics
            metrics = {
                "pe_ratio": info.get('trailingPE', 'N/A'),
                "forward_pe": info.get('forwardPE', 'N/A'),
                "peg_ratio": info.get('pegRatio', 'N/A'),
                "profit_margin": info.get('profitMargin', 0) * 100 if info.get('profitMargin') else 0,
                "revenue_growth": info.get('revenueGrowth', 0) * 100 if info.get('revenueGrowth') else 0,
                "debt_to_equity": info.get('debtToEquity', 'N/A'),
                "current_ratio": info.get('currentRatio', 'N/A'),
                "roa": info.get('returnOnAssets', 0) * 100 if info.get('returnOnAssets') else 0,
                "roe": info.get('returnOnEquity', 0) * 100 if info.get('returnOnEquity') else 0
            }
            
            # Generate 4-line analysis
            analysis_lines = [
                f"Fundamental Metrics: P/E {metrics['pe_ratio']}, PEG {metrics['peg_ratio']}, D/E {metrics['debt_to_equity']}",
                f"Growth Analysis: Revenue {metrics['revenue_growth']:.1f}% YoY, Margin {metrics['profit_margin']:.1f}%",
                f"Risk Metrics: ROE {metrics['roe']:.1f}%, ROA {metrics['roa']:.1f}%, Beta {info.get('beta', 'N/A')}",
                f"Financial Health: {'Strong' if metrics['profit_margin'] > 15 and metrics['revenue_growth'] > 10 else 'Moderate' if metrics['profit_margin'] > 8 and metrics['revenue_growth'] > 5 else 'Needs Improvement'}"
            ]
            
            return {
                'status': 'success',
                'metrics': metrics,
                'analysis': '\n'.join(analysis_lines)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing fundamentals for {ticker}: {str(e)}")
            return {
                'status': 'error',
                'analysis': '\n'.join([
                    "Fundamental Metrics: Analysis temporarily unavailable",
                    "Growth Analysis: Refer to latest quarterly reports",
                    "Risk Metrics: Standard industry risk profile",
                    "Financial Health: Review latest financial statements"
                ])
            }

    def _get_news_and_sentiment(self, ticker: str, company_name: str = None) -> Dict:
        """Get enhanced news and sentiment analysis using Together AI"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Enhanced market analysis prompt
            market_prompt = f"""
            Analyze {company_name or ticker} (${info.get('currentPrice', info.get('regularMarketPrice', 'N/A'))})
            Return exactly these 4 lines:
            1. Market Overview: [Bullish/Neutral/Bearish] - Key market position and trend
            2. Technical Signals: Major support/resistance levels and momentum
            3. Volume Analysis: Trading patterns and institutional activity
            4. Risk Indicators: Volatility metrics and market sentiment
            """
            
            # Get market analysis
            market_analysis = self._get_ai_sentiment(market_prompt)
            
            # Enhanced fundamental analysis prompt
            fundamental_prompt = f"""
            Analyze fundamentals for {company_name or ticker}
            Return exactly these 4 lines:
            1. Financial Health: Revenue ${info.get('totalRevenue', 0)/1e9:.1f}B, Margins {info.get('profitMargin', 0)*100:.1f}%
            2. Growth Profile: Revenue growth {info.get('revenueGrowth', 0)*100:.1f}%, Market position
            3. Valuation: P/E {info.get('trailingPE', 'N/A')}, Industry comparison
            4. Risk Metrics: Beta {info.get('beta', 'N/A')}, Key risk factors
            """
            
            # Get fundamental analysis
            fundamental_analysis = self._get_ai_sentiment(fundamental_prompt)
            
            # Enhanced AI overview prompt
            overview_prompt = f"""
            Provide investment overview for {company_name or ticker}
            Return exactly these 4 lines:
            1. Investment Rating: [Strong Buy/Buy/Hold/Sell] based on analysis
            2. Business Model: Core strengths and competitive position
            3. Growth Catalysts: Key drivers and market opportunities
            4. Risk Assessment: Primary challenges and mitigation factors
            """
            
            # Get AI overview
            ai_overview = self._get_ai_sentiment(overview_prompt)
            
            # Process news with enhanced sentiment
            news = []
            if hasattr(stock, 'news') and stock.news:
                for article in stock.news[:3]:
                    news.append({
                        'title': article.get('title', ''),
                        'description': article.get('summary', ''),
                        'source': article.get('publisher', 'Yahoo Finance'),
                        'url': article.get('link', ''),
                        'date': datetime.fromtimestamp(article.get('providerPublishTime', 0)).strftime('%Y-%m-%d'),
                        'sentiment': self._process_sentiment(article.get('title', '') + ' ' + article.get('summary', ''))
                    })
            
            return {
                'status': 'success',
                'market_analysis': market_analysis.get('analysis', 'Analysis not available'),
                'fundamental_analysis': fundamental_analysis.get('analysis', 'Analysis not available'),
                'ai_overview': ai_overview.get('analysis', 'Analysis not available'),
                'news': news
            }
            
        except Exception as e:
            logger.error(f"Error in news and sentiment analysis for {ticker}: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'market_analysis': '\n'.join([
                    "1. Market Overview: Standard market patterns observed",
                    "2. Technical Signals: Key levels pending confirmation",
                    "3. Volume Analysis: Normal trading activity",
                    "4. Risk Indicators: Standard volatility levels"
                ]),
                'fundamental_analysis': '\n'.join([
                    "1. Financial Health: Standard industry metrics",
                    "2. Growth Profile: Sector-aligned growth",
                    "3. Valuation: Near sector median",
                    "4. Risk Metrics: Typical sector risk"
                ]),
                'ai_overview': '\n'.join([
                    "1. Investment Rating: Hold - Limited data",
                    "2. Business Model: Standard sector operation",
                    "3. Growth Catalysts: Industry-aligned opportunities",
                    "4. Risk Assessment: Standard sector risks"
                ]),
                'news': []
            }

    def _get_ai_sentiment(self, prompt: str) -> Dict:
        """Get AI sentiment analysis using Together AI with retries"""
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                headers = {
                    'Authorization': f'Bearer {self.together_api_key}',
                    'Content-Type': 'application/json'
                }
                
                data = {
                    'model': 'mistralai/Mixtral-8x7B-Instruct-v0.1',
                    'prompt': f"{prompt}\nProvide specific, data-driven analysis with market metrics.",
                    'max_tokens': 300,
                    'temperature': 0.3,
                    'top_p': 0.7
                }
                
                response = requests.post(
                    'https://api.together.xyz/v1/completions',
                    headers=headers,
                    json=data,
                    timeout=10
                )
                
                if response.status_code == 200:
                    response_data = response.json()
                    if 'choices' in response_data and len(response_data['choices']) > 0:
                        analysis = response_data['choices'][0]['text'].strip()
                        lines = analysis.split('\n')
                        # Ensure exactly 4 lines
                        while len(lines) < 4:
                            lines.append("Additional analysis pending confirmation")
                        return {
                            'score': self._process_sentiment(analysis),
                            'analysis': '\n'.join(lines[:4])
                        }
                
                # Return structured fallback if API fails
                return {
                    'score': 0,
                    'analysis': '\n'.join([
                        "1. Market Overview: Standard market patterns observed",
                        "2. Technical Signals: Support and resistance levels being established",
                        "3. Volume Analysis: Average trading volume with normal activity",
                        "4. Risk Assessment: Typical market volatility levels"
                    ])
                }
                
            except Exception as e:
                logger.error(f"Error in AI sentiment analysis (attempt {attempt + 1}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                    continue
        
        # Final fallback with more informative analysis
        return {
            'score': 0,
            'analysis': '\n'.join([
                "1. Market Overview: Sector performance aligned with broader market trends",
                "2. Technical Signals: Historical support/resistance levels remain relevant",
                "3. Volume Analysis: Trading activity consistent with sector averages",
                "4. Risk Assessment: Standard market risk metrics apply"
            ])
        }

    def _generate_fallback_analysis(self, prompt: str) -> Dict:
        """Generate fallback analysis when AI service fails"""
        try:
            # Extract ticker/company name from prompt
            import re
            company_match = re.search(r'analyzing ([A-Z]+|[^:]+) at \$([0-9.]+)', prompt)
            if company_match:
                company = company_match.group(1)
                price = float(company_match.group(2))
                
                # Get stock info
                info = self.get_stock_info(company)
                if info:
                    # For market analysis
                    if 'Bullish/Neutral/Bearish' in prompt:
                        rec_key = info.get('recommendationKey', '').lower()
                        sentiment = 'Bullish' if rec_key in ['strong_buy', 'buy'] else 'Bearish' if rec_key in ['sell', 'strong_sell'] else 'Neutral'
                        target = info.get('targetMeanPrice', price * 1.1)
                        analysis = f"{sentiment} - {company} at ${price:.2f}: Key driver - Market consensus. Target: ${target:.2f}"
                        return {'score': 1 if sentiment == 'Bullish' else -1 if sentiment == 'Bearish' else 0, 'analysis': analysis}
                    
                    # For fundamental analysis
                    if 'P/E' in prompt and 'Beta' in prompt:
                        pe = info.get('trailingPE', 'N/A')
                        beta = info.get('beta', 'N/A')
                        margin = info.get('profitMargin', 0) * 100
                        growth = info.get('revenueGrowth', 0) * 100
                        strength = 'Strong' if margin > 15 and growth > 10 else 'Moderate' if margin > 8 and growth > 5 else 'Weak'
                        analysis = f"{company} ({info.get('sector', '')}): P/E {pe}, Beta {beta} | {margin:.1f}% margin, {growth:.1f}% growth | {strength} fundamentals"
                        return {'score': 0, 'analysis': analysis}
                    
                    # For AI overview
                    if 'investment case' in prompt:
                        rec = 'Strong Buy' if info.get('recommendationKey', '').lower() == 'strong_buy' else 'Buy' if info.get('recommendationKey', '').lower() == 'buy' else 'Hold' if info.get('recommendationKey', '').lower() == 'hold' else 'Sell' if info.get('recommendationKey', '').lower() == 'sell' else 'Neutral'
                        summary = info.get('longBusinessSummary', 'Leading company in its sector')[:100]
                        analysis = f"{company}: {rec} - {summary}..."
                        return {'score': 0, 'analysis': analysis}
            
            # Default fallback if we can't parse the prompt
            return {
                'score': 0,
                'analysis': "Analysis based on current market data and fundamentals"
            }
            
        except Exception as e:
            logger.error(f"Error generating fallback analysis: {str(e)}")
            return {
                'score': 0,
                'analysis': "Analysis temporarily unavailable"
            }

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
                        'analysis': {
                            'market': analysis_data.get('market_analysis', 'Market analysis not available'),
                            'fundamental': analysis_data.get('fundamental_analysis', 'Fundamental analysis not available'),
                            'overview': analysis_data.get('ai_overview', 'AI overview not available')
                        },
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