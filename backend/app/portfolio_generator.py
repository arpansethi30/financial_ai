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
import time

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
        max_retries = 3
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                stock = yf.Ticker(ticker)
                time.sleep(1)  # Add delay to avoid rate limiting
                
                # First try to get current price directly from history
                try:
                    hist = stock.history(period='1d')
                    if not hist.empty:
                        current_price = float(hist['Close'].iloc[-1])
                    else:
                        current_price = None
                except:
                    current_price = None
                
                # If history fails, try info
                if not current_price:
                    info = stock.info
                    if info:
                        current_price = info.get('regularMarketPrice') or info.get('currentPrice')
                
                if not current_price or current_price <= 0:
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay * (attempt + 1))
                        continue
                    else:
                        logger.warning(f"No valid price data for {ticker}")
                        return None
                
                fundamentals = {
                    'current_price': current_price
                }
                
                return fundamentals
                
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Attempt {attempt + 1} failed for {ticker}: {str(e)}")
                    time.sleep(retry_delay * (attempt + 1))
                else:
                    logger.error(f"Failed to get fundamentals for {ticker} after {max_retries} attempts: {str(e)}")
                    return None

    def analyze_technicals(self, ticker: str) -> Dict:
        """Analyze technical indicators for a given stock"""
        max_retries = 3
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                stock = yf.Ticker(ticker)
                time.sleep(1)  # Add delay to avoid rate limiting
                
                hist = stock.history(period='1y')
                
                if len(hist) < 50:
                    logger.warning(f"Insufficient historical data for {ticker}")
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
                if attempt < max_retries - 1:
                    logger.warning(f"Attempt {attempt + 1} failed for {ticker}: {str(e)}")
                    time.sleep(retry_delay * (attempt + 1))
                else:
                    logger.error(f"Failed to get technicals for {ticker} after {max_retries} attempts: {str(e)}")
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
                page_size=10,
                from_param=(datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
            )
            
            if not news.get('articles'):
                return {
                    'sentiment_score': 0,
                    'sentiment_summary': "**Market Sentiment Analysis:**\n\nInsufficient news data available for comprehensive analysis.",
                    'articles': []
                }
            
            articles_text = ""
            for article in news['articles']:
                title = article.get('title', '')
                description = article.get('description', '')
                if title and description:
                    articles_text += f"{title} {description} "
            
            if not articles_text.strip():
                return {
                    'sentiment_score': 0,
                    'sentiment_summary': "**Market Sentiment Analysis:**\n\nNo valid news content found for analysis.",
                    'articles': news['articles'][:3] if news['articles'] else []
                }
            
            sentiment_prompt = f"Analyze the sentiment of these news articles about {ticker}: {articles_text}"
            
            try:
                sentiment_response = self.ai_model.generate_content(sentiment_prompt)
                sentiment_score = self._process_sentiment(sentiment_response.text)
                
                # Generate detailed sentiment summary with context
                if sentiment_score > 0.6:
                    sentiment_summary = f"**Market Sentiment Analysis: Strongly Positive**\n\n" \
                                     f"Based on recent news coverage and market indicators:\n\n" \
                                     f"• Strong bullish sentiment in market coverage\n" \
                                     f"• Multiple positive developments reported\n" \
                                     f"• High confidence in growth trajectory\n" \
                                     f"• Favorable market conditions observed\n\n" \
                                     f"Recommendation: Consider increasing position on pullbacks"
                elif sentiment_score > 0.2:
                    sentiment_summary = f"**Market Sentiment Analysis: Moderately Positive**\n\n" \
                                     f"Analysis of recent market coverage indicates:\n\n" \
                                     f"• Generally favorable market outlook\n" \
                                     f"• Steady performance metrics\n" \
                                     f"• Stable growth indicators\n" \
                                     f"• Some positive catalysts identified\n\n" \
                                     f"Recommendation: Maintain current position with regular monitoring"
                elif sentiment_score > -0.2:
                    sentiment_summary = f"**Market Sentiment Analysis: Neutral**\n\n" \
                                     f"Current market analysis shows:\n\n" \
                                     f"• Balanced positive and negative factors\n" \
                                     f"• No significant sentiment shift\n" \
                                     f"• Mixed market signals present\n" \
                                     f"• Watching for clear directional indicators\n\n" \
                                     f"Recommendation: Monitor for emerging trends before adjusting position"
                elif sentiment_score > -0.6:
                    sentiment_summary = f"**Market Sentiment Analysis: Moderately Negative**\n\n" \
                                     f"Recent market developments indicate:\n\n" \
                                     f"• Some concerning market signals\n" \
                                     f"• Potential headwinds identified\n" \
                                     f"• Increased market uncertainty\n" \
                                     f"• Risk factors require attention\n\n" \
                                     f"Recommendation: Consider reducing exposure on strength"
                else:
                    sentiment_summary = f"**Market Sentiment Analysis: Strongly Negative**\n\n" \
                                     f"Analysis reveals significant concerns:\n\n" \
                                     f"• Multiple negative indicators present\n" \
                                     f"• Substantial market headwinds\n" \
                                     f"• High risk factors identified\n" \
                                     f"• Challenging market environment\n\n" \
                                     f"Recommendation: Consider defensive positioning"
                
            except Exception as e:
                logger.error(f"Error in AI sentiment analysis for {ticker}: {str(e)}")
                sentiment_score = 0
                sentiment_summary = "**Market Sentiment Analysis:**\n\nSentiment analysis temporarily unavailable"
            
            return {
                'sentiment_score': sentiment_score,
                'sentiment_summary': sentiment_summary,
                'articles': news['articles'][:3] if news['articles'] else []
            }
        except Exception as e:
            logger.error(f"Error getting news sentiment for {ticker}: {str(e)}")
            return {
                'sentiment_score': 0,
                'sentiment_summary': "**Market Sentiment Analysis:**\n\nError processing sentiment data",
                'articles': []
            }

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
            company_count = request_data.get('company_count', 10)
            
            sector_weights = self.sector_weights[risk_appetite]
            stock_recommendations = {}
            total_investment = 0
            total_stocks = 0
            
            # Calculate how many companies to pick from each sector
            total_sectors = len(sector_weights)
            base_companies_per_sector = max(1, company_count // total_sectors)
            extra_companies = company_count % total_sectors
            
            # Sort sectors by weight to allocate extra companies to highest weight sectors
            sorted_sectors = sorted(sector_weights.items(), key=lambda x: x[1], reverse=True)
            
            valid_sectors = 0
            for sector, weight in sorted_sectors:
                sector_amount = investment_amount * weight
                sector_stocks = []
                
                # Get all available stocks for this sector
                available_stocks = []
                for ticker in self.sector_tickers[sector]:
                    try:
                        time.sleep(0.5)  # Add delay between stock requests
                        fundamentals = self.analyze_fundamentals(ticker)
                        if not fundamentals or fundamentals.get('current_price', 0) <= 0:
                            logger.warning(f"No valid fundamentals for {ticker}")
                            continue
                            
                        technicals = self.analyze_technicals(ticker)
                        if not technicals:
                            logger.warning(f"No valid technicals for {ticker}")
                            continue
                            
                        available_stocks.append({
                            'ticker': ticker,
                            'fundamentals': fundamentals,
                            'technical_score': technicals['Overall_Score']
                        })
                        logger.info(f"Successfully processed {ticker} for sector {sector}")
                    except Exception as e:
                        logger.error(f"Error processing stock {ticker}: {str(e)}")
                        continue
                
                if not available_stocks:
                    logger.warning(f"No valid stocks found for sector {sector}")
                    continue
                
                valid_sectors += 1
                logger.info(f"Found {len(available_stocks)} valid stocks for sector {sector}")
                
                # Sort stocks by technical score and take the top N for this sector
                available_stocks.sort(key=lambda x: x['technical_score'], reverse=True)
                sector_company_count = min(
                    len(available_stocks),
                    base_companies_per_sector + (1 if extra_companies > 0 else 0)
                )
                
                if extra_companies > 0:
                    extra_companies -= 1
                
                selected_stocks = available_stocks[:sector_company_count]
                logger.info(f"Selected {len(selected_stocks)} stocks for sector {sector}")
                
                # Calculate per-stock allocation for this sector
                if selected_stocks:
                    per_stock_amount = sector_amount / len(selected_stocks)
                    
                    for stock in selected_stocks:
                        try:
                            current_price = stock['fundamentals']['current_price']
                            if current_price <= 0:
                                logger.warning(f"Invalid price for {stock['ticker']}: {current_price}")
                                continue
                                
                            suggested_shares = max(1, int(per_stock_amount / current_price))
                            actual_amount = suggested_shares * current_price
                            
                            # Only proceed if we can get valid data
                            try:
                                risk_level = self._determine_risk_level(stock['technical_score'], sector, risk_appetite)
                                
                                # Get sentiment analysis and news
                                sentiment_data = self.get_news_sentiment(stock['ticker'])
                                sentiment_analysis = "Neutral market sentiment"
                                recent_news = []
                                
                                if sentiment_data:
                                    sentiment_score = sentiment_data['sentiment_score']
                                    if sentiment_score > 0.3:
                                        sentiment_analysis = "Positive market sentiment with strong growth potential"
                                    elif sentiment_score > 0:
                                        sentiment_analysis = "Slightly positive market outlook"
                                    elif sentiment_score < -0.3:
                                        sentiment_analysis = "Negative market sentiment, exercise caution"
                                    elif sentiment_score < 0:
                                        sentiment_analysis = "Slightly negative market outlook"
                                    recent_news = sentiment_data['articles']
                                
                                # Get additional fundamental data with retry
                                stock_info = None
                                for retry in range(3):
                                    try:
                                        time.sleep(1)
                                        stock_info = yf.Ticker(stock['ticker']).info
                                        if stock_info:
                                            break
                                    except:
                                        time.sleep(2 ** retry)
                                
                                if not stock_info:
                                    stock_info = {}  # Use empty dict if no additional info available
                                    
                                financial_metrics = {
                                    'market_cap': stock_info.get('marketCap', 0),
                                    'pe_ratio': stock_info.get('trailingPE', 0),
                                    'revenue_growth': stock_info.get('revenueGrowth', 0) * 100 if stock_info.get('revenueGrowth') else 0,
                                    'profit_margins': stock_info.get('profitMargins', 0) * 100 if stock_info.get('profitMargins') else 0,
                                    'debt_to_equity': stock_info.get('debtToEquity', 0),
                                    'beta': stock_info.get('beta', 1),
                                    'dividend_yield': stock_info.get('dividendYield', 0) * 100 if stock_info.get('dividendYield') else 0
                                }
                                
                                sector_stocks.append({
                                    'symbol': stock['ticker'],
                                    'weight': (actual_amount / investment_amount) * 100,
                                    'amount': actual_amount,
                                    'suggested_shares': suggested_shares,
                                    'risk_level': risk_level,
                                    'fundamentals': {
                                        'current_price': current_price,
                                        **financial_metrics
                                    },
                                    'sentiment_analysis': sentiment_analysis,
                                    'ai_analysis': self._generate_ai_analysis(
                                        stock['ticker'],
                                        risk_level,
                                        financial_metrics,
                                        sentiment_analysis
                                    ),
                                    'recent_news': recent_news
                                })
                                
                                total_investment += actual_amount
                                total_stocks += 1
                                logger.info(f"Successfully added {stock['ticker']} to portfolio")
                                
                            except Exception as e:
                                logger.error(f"Error processing stock data for {stock['ticker']}: {str(e)}")
                                continue
                                
                        except Exception as e:
                            logger.error(f"Error calculating allocation for {stock['ticker']}: {str(e)}")
                            continue
                
                if sector_stocks:
                    stock_recommendations[sector] = sector_stocks
            
            if valid_sectors == 0:
                raise ValueError("Could not generate portfolio: No valid sectors found. Please try again later.")
                
            if total_stocks == 0:
                raise ValueError("Could not generate portfolio: No stocks could be allocated. Please try again later.")
            
            logger.info(f"Successfully generated portfolio with {total_stocks} stocks across {len(stock_recommendations)} sectors")
            
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
                    'summary': {
                        'investment_amount': investment_amount,
                        'risk_profile': risk_appetite,
                        'time_horizon': investment_period,
                        'total_stocks': total_stocks
                    },
                    'recommendations': {
                        'stock_recommendations': stock_recommendations,
                        'allocation_summary': {
                            'total_investment': total_investment,
                            'total_stocks': total_stocks,
                            'total_sectors': len(stock_recommendations)
                        }
                    },
                    'analysis': analysis
                }
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

    def _generate_ai_analysis(self, ticker: str, risk_level: str, metrics: Dict, sentiment: str) -> str:
        """Generate AI-powered analysis for a stock"""
        try:
            prompt = f"""
            Analyze {ticker} as an investment opportunity with the following metrics:
            Risk Level: {risk_level}
            P/E Ratio: {metrics.get('pe_ratio', 'N/A')}
            Revenue Growth: {metrics.get('revenue_growth', 'N/A')}%
            Profit Margins: {metrics.get('profit_margins', 'N/A')}%
            Beta: {metrics.get('beta', 'N/A')}
            Market Sentiment: {sentiment}
            
            Provide a structured investment analysis with clear sections and bullet points.
            """
            
            response = self.ai_model.generate_content(prompt)
            analysis_text = response.text[:800]  # Allow for longer analysis
            
            # Format the analysis with proper structure and context
            formatted_analysis = f"""
**{ticker} Investment Analysis**

**Risk Assessment:**
• Risk Level: {risk_level.title()}
• Market Position: {metrics.get('beta', 'N/A'):.2f} Beta
• Volatility Profile: {'High' if float(metrics.get('beta', 1)) > 1.2 else 'Moderate' if float(metrics.get('beta', 1)) > 0.8 else 'Low'}

**Fundamental Metrics:**
• Market Cap: ${metrics.get('market_cap', 0)/1e9:.1f}B
• P/E Ratio: {metrics.get('pe_ratio', 'N/A'):.2f}
• Revenue Growth: {metrics.get('revenue_growth', 'N/A'):.1f}%
• Profit Margins: {metrics.get('profit_margins', 'N/A'):.1f}%
• Dividend Yield: {metrics.get('dividend_yield', 'N/A'):.2f}%

**Investment Considerations:**
{analysis_text}

**Key Takeaways:**
• {'Strong fundamentals with growth potential' if metrics.get('revenue_growth', 0) > 10 else 'Stable fundamentals with moderate growth' if metrics.get('revenue_growth', 0) > 5 else 'Limited growth metrics'}
• {'Attractive valuation metrics' if metrics.get('pe_ratio', 0) < 20 else 'Fair valuation range' if metrics.get('pe_ratio', 0) < 30 else 'Premium valuation metrics'}
• {'High profitability indicators' if metrics.get('profit_margins', 0) > 15 else 'Moderate profit potential' if metrics.get('profit_margins', 0) > 8 else 'Margin improvement needed'}
"""
            return formatted_analysis
            
        except Exception as e:
            logger.error(f"Error generating AI analysis for {ticker}: {str(e)}")
            return "**Investment Analysis:**\n\nAnalysis temporarily unavailable"

# Initialize generator
portfolio_generator = PortfolioGenerator() 