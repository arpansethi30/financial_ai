import os
import requests
from datetime import datetime, timedelta
from typing import List, Dict
import pandas as pd
import yfinance as yf
from newsapi import NewsApiClient
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

# API Keys
TOGETHER_API_KEY = os.getenv('TOGETHER_API_KEY')
NEWS_API_KEY = os.getenv('NEWS_API_KEY')

# Initialize News API client
newsapi = NewsApiClient(api_key=NEWS_API_KEY)

class EnhancedStockAnalyzer:
    def __init__(self, ticker: str):
        """Initialize EnhancedStockAnalyzer with a stock ticker"""
        self.ticker = ticker
        self.stock = yf.Ticker(ticker)
        
    def fetch_comprehensive_news(self, days_back: int = 7) -> List[Dict]:
        """Fetch news from multiple sources"""
        try:
            # Get news from News API
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            newsapi_articles = newsapi.get_everything(
                q=f"{self.ticker} OR {self.stock.info.get('longName', '')}",
                from_param=start_date.strftime('%Y-%m-%d'),
                to=end_date.strftime('%Y-%m-%d'),
                language='en',
                sort_by='relevancy'
            )
            
            # Get news from Yahoo Finance
            yf_news = self.stock.news
            
            # Combine and format news
            all_news = []
            
            # Add NewsAPI articles
            for article in newsapi_articles.get('articles', []):
                all_news.append({
                    'source': 'NewsAPI',
                    'title': article.get('title', ''),
                    'summary': article.get('description', ''),
                    'link': article.get('url', ''),
                    'published': article.get('publishedAt', '')
                })
            
            # Add Yahoo Finance articles
            for article in yf_news:
                all_news.append({
                    'source': 'Yahoo Finance',
                    'title': article.get('title', ''),
                    'summary': article.get('summary', ''),
                    'link': article.get('link', ''),
                    'published': article.get('published', '')
                })
            
            return all_news[:10]  # Return top 10 most relevant articles
            
        except Exception as e:
            print(f"Error fetching news: {e}")
            return []

    def get_comprehensive_financials(self) -> Dict:
        """Get financial data from Yahoo Finance"""
        try:
            # Get Yahoo Finance data
            info = self.stock.info
            
            # Get historical data for technical analysis
            hist = self.stock.history(period="1y")
            
            # Calculate additional metrics
            metrics = {
                'Company Overview': {
                    'Market Cap': info.get('marketCap'),
                    'PE Ratio': info.get('forwardPE'),
                    'PEG Ratio': info.get('pegRatio'),
                    'Dividend Yield': info.get('dividendYield'),
                    'Profit Margins': info.get('profitMargins'),
                    'Operating Margins': info.get('operatingMargins'),
                    'Return on Equity': info.get('returnOnEquity'),
                    'Beta': info.get('beta')
                },
                'Growth & Performance': {
                    'Revenue Growth': info.get('revenueGrowth'),
                    'Earnings Growth': info.get('earningsGrowth'),
                    'Quarterly Growth': info.get('earningsQuarterlyGrowth'),
                    'Gross Profits': info.get('grossProfits'),
                    'Free Cash Flow': info.get('freeCashflow')
                },
                'Technical Indicators': {
                    '50 Day MA': info.get('fiftyDayAverage'),
                    '200 Day MA': info.get('twoHundredDayAverage'),
                    '52 Week High': info.get('fiftyTwoWeekHigh'),
                    '52 Week Low': info.get('fiftyTwoWeekLow'),
                    'RSI': self._calculate_rsi(hist['Close']),
                    'Current Price': info.get('currentPrice'),
                    'Volume': info.get('volume'),
                    'Avg Volume': info.get('averageVolume')
                },
                'Financial Health': {
                    'Total Revenue': info.get('totalRevenue'),
                    'Total Cash': info.get('totalCash'),
                    'Total Debt': info.get('totalDebt'),
                    'Quick Ratio': info.get('quickRatio'),
                    'Current Ratio': info.get('currentRatio'),
                    'Debt to Equity': info.get('debtToEquity'),
                    'Book Value': info.get('bookValue')
                }
            }
            
            return metrics
        except Exception as e:
            print(f"Error fetching financial data: {e}")
            return {}

    def _calculate_rsi(self, prices, periods=14):
        """Calculate RSI technical indicator"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
            rs = gain / loss
            return round(float(100 - (100 / (1 + rs)).iloc[-1]), 2)
        except:
            return 'N/A'

    def analyze_with_deepseek(self, news_articles: List[Dict], financials: Dict) -> Dict:
        """Perform comprehensive analysis using DeepSeek model"""
        try:
            # Prepare the prompt with all available data
            prompt = f"""
            Perform a comprehensive analysis of {self.ticker} stock based on the following data:
            
            1. Financial Metrics:
            {pd.json_normalize(financials).to_string()}
            
            2. Recent News Headlines:
            {', '.join([article['title'] for article in news_articles[:5]])}
            
            Please provide a detailed analysis in the following format:
            1. Financial Health Assessment
               - Current Financial Position
               - Growth Prospects
               - Risk Analysis
            
            2. Market Sentiment Analysis
               - News Sentiment
               - Market Perception
               - Key Developments
            
            3. Technical Analysis
               - Price Trends
               - Key Technical Indicators
               - Support/Resistance Levels
            
            4. Investment Recommendation
               - Short-term Outlook (1-3 months)
               - Long-term Outlook (1-2 years)
               - Risk Factors
               - Target Price Range
            """
            
            # Make API request to Together AI
            headers = {
                'Authorization': f'Bearer {TOGETHER_API_KEY}',
                'Content-Type': 'application/json'
            }
            
            data = {
                'model': 'mistralai/Mixtral-8x7B-Instruct-v0.1',
                'prompt': prompt,
                'max_tokens': 1500,
                'temperature': 0.3,
                'top_p': 0.7,
                'top_k': 50,
                'repetition_penalty': 1.1
            }
            
            response = requests.post(
                'https://api.together.xyz/v1/completions',
                headers=headers,
                json=data
            )
            
            if response.status_code == 200:
                response_data = response.json()
                if 'choices' in response_data and len(response_data['choices']) > 0:
                    analysis = response_data['choices'][0]['text']
                    return {
                        'status': 'success',
                        'analysis': analysis
                    }
                else:
                    return {
                        'status': 'error',
                        'message': 'No analysis generated'
                    }
            else:
                print(f"API Response: {response.text}")  # Debug line
                return {
                    'status': 'error',
                    'message': f"API Error: {response.status_code}"
                }
                
        except Exception as e:
            print(f"Exception details: {str(e)}")  # Debug line
            return {
                'status': 'error',
                'message': str(e)
            }

def main():
    # Analyze NVIDIA stock
    ticker = "NVDA"
    print(f"\n=== Comprehensive Analysis for {ticker} ===")
    
    analyzer = EnhancedStockAnalyzer(ticker)
    
    # Fetch all data
    print("\nFetching news and financial data...")
    news = analyzer.fetch_comprehensive_news(days_back=7)
    financials = analyzer.get_comprehensive_financials()
    
    # Perform analysis
    print("\nPerforming comprehensive analysis...")
    analysis = analyzer.analyze_with_deepseek(news, financials)
    
    # Display results
    if analysis['status'] == 'success':
        print("\nAnalysis Results:")
        print(analysis['analysis'])
        
        print("\nKey Financial Metrics:")
        for category, metrics in financials.items():
            print(f"\n{category}:")
            for metric, value in metrics.items():
                print(f"{metric}: {value}")
            
        print("\nRecent News Headlines:")
        for idx, article in enumerate(news[:5], 1):
            print(f"\n{idx}. {article['title']}")
            print(f"Source: {article['source']}")
            print(f"Link: {article['link']}")
    else:
        print(f"\nError in analysis: {analysis['message']}")

if __name__ == "__main__":
    main() 