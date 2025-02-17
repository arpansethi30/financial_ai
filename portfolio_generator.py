from typing import List, Dict

class PortfolioGenerator:
    def _prepare_sentiment_prompt(self, ticker: str, articles: List[Dict]) -> str:
        """Prepare prompt for sentiment analysis"""
        article_texts = []
        for i, article in enumerate(articles[:5]):
            article_text = (
                f"Article {i+1}:\n"
                f"Title: {article['title']}\n"
                f"Summary: {article['description']}\n"
                f"Source: {article['source']}\n"
            )
            article_texts.append(article_text)
            
        return f"""
        Analyze the market sentiment for {ticker} based on these recent news articles:
        
        {chr(10).join(article_texts)}
        
        Please provide:
        1. Overall sentiment (Bullish/Bearish/Neutral)
        2. Key themes or trends
        3. Potential impact on stock price
        4. Risk factors mentioned
        5. Sentiment score (-1 to 1, where -1 is very bearish and 1 is very bullish)
        """ 