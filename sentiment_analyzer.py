import os
import requests
from datetime import datetime, timedelta
from typing import List, Dict
import yfinance as yf
from newsapi import NewsApiClient
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()
TOGETHER_API_KEY = os.getenv('TOGETHER_API_KEY')
NEWS_API_KEY = os.getenv('NEWS_API_KEY')

class StockSentimentAnalyzer:
    def __init__(self, ticker: str):
        """Initialize with a stock ticker"""
        self.ticker = ticker
        self.stock = yf.Ticker(ticker)
        self.company_name = self.stock.info.get('longName', ticker)
        self.newsapi = NewsApiClient(api_key=NEWS_API_KEY)

    def get_news_and_sentiment(self) -> Dict:
        """Get news and analyze sentiment"""
        try:
            # Get recent news
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
            
            news_articles = self.newsapi.get_everything(
                q=f'({self.ticker} OR "{self.company_name}") AND (stock OR market)',
                from_param=start_date.strftime('%Y-%m-%d'),
                to=end_date.strftime('%Y-%m-%d'),
                language='en',
                sort_by='relevancy'
            )
            
            # Get stock price info
            info = self.stock.info
            current_price = info.get('currentPrice', 0)
            price_change = info.get('regularMarketChangePercent', 0)
            
            # Prepare news summary
            news_summary = []
            for article in news_articles.get('articles', [])[:5]:
                news_summary.append(f"Title: {article['title']}\nSummary: {article['description']}")
            
            # Create analysis prompt
            prompt = f"""
            Analyze the market sentiment for {self.ticker} ({self.company_name}) based on recent news and market data:

            Stock Information:
            - Current Price: ${current_price}
            - Price Change: {price_change:.2f}%

            Recent News:
            {' '.join(news_summary)}

            Please provide a concise sentiment analysis covering:
            1. Overall Market Sentiment (Bullish/Bearish/Neutral)
            2. Key Themes from News
            3. Short-term Sentiment Outlook
            4. Key Risks or Opportunities
            """
            
            # Get sentiment analysis from AI
            headers = {
                'Authorization': f'Bearer {TOGETHER_API_KEY}',
                'Content-Type': 'application/json'
            }
            
            data = {
                'model': 'mistralai/Mixtral-8x7B-Instruct-v0.1',
                'prompt': prompt,
                'max_tokens': 500,
                'temperature': 0.3,
                'top_p': 0.7
            }
            
            response = requests.post(
                'https://api.together.xyz/v1/completions',
                headers=headers,
                json=data
            )
            
            if response.status_code == 200:
                response_data = response.json()
                if 'choices' in response_data and len(response_data['choices']) > 0:
                    return {
                        'status': 'success',
                        'analysis': response_data['choices'][0]['text'].strip(),
                        'news': news_articles.get('articles', [])[:5]
                    }
            
            return {
                'status': 'error',
                'message': 'Failed to get sentiment analysis'
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }

def main():
    try:
        # Get stock ticker
        ticker = input("Enter stock ticker (default: NVDA): ").strip().upper() or "NVDA"
        print(f"\nAnalyzing sentiment for {ticker}...")
        
        # Get sentiment analysis
        analyzer = StockSentimentAnalyzer(ticker)
        result = analyzer.get_news_and_sentiment()
        
        if result['status'] == 'success':
            # Print sentiment analysis
            print("\nSentiment Analysis:")
            print("=" * 80)
            print(result['analysis'])
            print("=" * 80)
            
            # Print headlines
            print("\nRecent Headlines:")
            for idx, article in enumerate(result['news'], 1):
                print(f"\n{idx}. {article['title']}")
                print(f"Source: {article['source']['name']}")
        else:
            print(f"\nError: {result['message']}")
            
    except Exception as e:
        print(f"\nError: {str(e)}")

if __name__ == "__main__":
    main() 