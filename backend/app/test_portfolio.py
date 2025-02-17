import os
from dotenv import load_dotenv
from portfolio_generator import PortfolioGenerator
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_portfolio_generation():
    """Test portfolio generation with sample data"""
    try:
        # Initialize portfolio generator
        generator = PortfolioGenerator()
        
        # Test data
        request_data = {
            'investment_amount': 10000,
            'risk_appetite': 'moderate',
            'investment_period': 5,
            'company_count': 10
        }
        
        # Generate portfolio
        logger.info("Generating portfolio...")
        result = generator.generate_portfolio(request_data)
        
        if result['status'] == 'success':
            logger.info("Portfolio generated successfully!")
            
            # Print portfolio summary
            portfolio = result['portfolio']
            summary = portfolio['recommendations']['allocation_summary']
            
            print("\nPortfolio Summary:")
            print(f"Total Investment: ${summary['total_investment']:,.2f}")
            print(f"Total Stocks: {summary['total_stocks']}")
            print(f"Total Sectors: {summary['total_sectors']}")
            
            # Print sector allocations
            print("\nSector Allocations:")
            for sector, stocks in portfolio['recommendations']['stock_recommendations'].items():
                sector_total = sum(stock['amount'] for stock in stocks)
                print(f"\n{sector}:")
                for stock in stocks:
                    print(f"- {stock['symbol']}: ${stock['amount']:,.2f} ({stock['suggested_shares']} shares)")
            
            # Print analysis
            print("\nPortfolio Analysis:")
            print(result['analysis'])
            
        else:
            logger.error(f"Portfolio generation failed: {result['message']}")
            
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    # Verify required environment variables
    required_vars = ['NEWS_API_KEY', 'GOOGLE_AI_API_KEY', 'TOGETHER_API_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        logger.error("Please set these variables in your .env file")
    else:
        test_portfolio_generation() 