import os
import time
import requests
import pandas as pd
import yfinance as yf
from typing import Dict, List
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
TOGETHER_API_KEY = os.getenv('TOGETHER_API_KEY')

class FinancialAnalyzer:
    def __init__(self, ticker: str):
        """Initialize FinancialAnalyzer with retry logic"""
        self.ticker = ticker
        max_retries = 3
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                print(f"Attempting to fetch stock data (attempt {attempt + 1}/{max_retries})...")
                self.stock = yf.Ticker(ticker)
                
                # Force info fetch to verify stock exists
                time.sleep(1)  # Add delay before fetching
                info = self.stock.info
                
                if not info or len(info) < 5:  # Basic validation of info
                    raise ValueError("Insufficient stock data received")
                
                self.company_name = info.get('longName', ticker)
                print(f"Successfully fetched data for {self.company_name}")
                return
                
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Attempt {attempt + 1} failed: {str(e)}")
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    raise ValueError(f"Failed to initialize stock {ticker} after {max_retries} attempts: {str(e)}")

    def get_balance_sheet(self) -> Dict:
        """Get quarterly and annual balance sheets"""
        try:
            quarterly_bs = self.stock.quarterly_balance_sheet
            annual_bs = self.stock.balance_sheet
            
            return {
                'quarterly': quarterly_bs.to_dict(),
                'annual': annual_bs.to_dict()
            }
        except Exception as e:
            print(f"Error fetching balance sheets: {e}")
            return {}
            
    def get_financial_statements(self) -> Dict:
        """Get comprehensive financial statements with error handling"""
        try:
            print("Fetching balance sheet data...")
            time.sleep(1)
            quarterly_balance_sheet = self.stock.quarterly_balance_sheet
            annual_balance_sheet = self.stock.balance_sheet
            
            print("Fetching income statement data...")
            time.sleep(1)
            quarterly_income_stmt = self.stock.quarterly_income_stmt
            annual_income_stmt = self.stock.income_stmt
            
            print("Fetching cash flow data...")
            time.sleep(1)
            quarterly_cashflow = self.stock.quarterly_cashflow
            annual_cashflow = self.stock.cashflow
            
            # Validate data
            if quarterly_balance_sheet.empty and annual_balance_sheet.empty:
                print("Warning: No balance sheet data available")
            if quarterly_income_stmt.empty and annual_income_stmt.empty:
                print("Warning: No income statement data available")
            if quarterly_cashflow.empty and annual_cashflow.empty:
                print("Warning: No cash flow data available")
            
            return {
                'Income Statement': {
                    'Quarterly': quarterly_income_stmt.to_dict() if not quarterly_income_stmt.empty else {},
                    'Annual': annual_income_stmt.to_dict() if not annual_income_stmt.empty else {}
                },
                'Balance Sheet': {
                    'Quarterly': quarterly_balance_sheet.to_dict() if not quarterly_balance_sheet.empty else {},
                    'Annual': annual_balance_sheet.to_dict() if not annual_balance_sheet.empty else {}
                },
                'Cash Flow': {
                    'Quarterly': quarterly_cashflow.to_dict() if not quarterly_cashflow.empty else {},
                    'Annual': annual_cashflow.to_dict() if not annual_cashflow.empty else {}
                }
            }
        except Exception as e:
            print(f"Error fetching financial statements: {e}")
            return {}

    def get_key_metrics(self) -> Dict:
        """Get key financial metrics with validation"""
        try:
            # Add delay to avoid rate limiting
            time.sleep(1)
            
            info = self.stock.info
            hist = self.stock.history(period="1y")
            
            if hist.empty:
                raise ValueError("No historical data available")
            
            # Helper function to format values
            def format_value(value, is_percentage=False):
                if value is None:
                    return None
                try:
                    value = float(value)
                    if is_percentage:
                        return value * 100 if abs(value) < 100 else value
                    return value
                except:
                    return None
            
            metrics = {
                'Valuation Metrics': {
                    'Market Cap': format_value(info.get('marketCap')),
                    'PE Ratio': format_value(info.get('forwardPE')),
                    'PEG Ratio': format_value(info.get('pegRatio')),
                    'Price to Book': format_value(info.get('priceToBook')),
                    'Price to Sales': format_value(info.get('priceToSalesTrailing12Months')),
                    'Current Price': format_value(info.get('currentPrice'))
                },
                'Profitability Metrics': {
                    'Profit Margins': format_value(info.get('profitMargins'), True),
                    'Operating Margins': format_value(info.get('operatingMargins'), True),
                    'Gross Margins': format_value(info.get('grossMargins'), True),
                    'ROE': format_value(info.get('returnOnEquity'), True),
                    'ROA': format_value(info.get('returnOnAssets'), True)
                },
                'Growth Metrics': {
                    'Revenue Growth': format_value(info.get('revenueGrowth'), True),
                    'Earnings Growth': format_value(info.get('earningsGrowth'), True),
                    'Quarterly Growth': format_value(info.get('earningsQuarterlyGrowth'), True)
                },
                'Financial Health': {
                    'Current Ratio': format_value(info.get('currentRatio')),
                    'Debt to Equity': format_value(info.get('debtToEquity')),
                    'Total Cash': format_value(info.get('totalCash')),
                    'Total Debt': format_value(info.get('totalDebt')),
                    'Operating Cash Flow': format_value(info.get('operatingCashflow'))
                }
            }
            
            # Remove None values
            for category in metrics:
                metrics[category] = {k: v for k, v in metrics[category].items() if v is not None}
            
            return metrics
        except Exception as e:
            print(f"Error fetching key metrics: {e}")
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

    def get_earnings_analysis(self) -> Dict:
        """Get detailed earnings analysis"""
        try:
            earnings = self.stock.earnings
            earnings_dates = self.stock.earnings_dates
            
            return {
                'Historical Earnings': earnings.to_dict(),
                'Recent Earnings Dates': earnings_dates.to_dict(),
                'Earnings Trend': {
                    'Quarterly Growth': self.stock.info.get('earningsQuarterlyGrowth'),
                    'Forward EPS': self.stock.info.get('forwardEps'),
                    'Trailing EPS': self.stock.info.get('trailingEps')
                }
            }
        except Exception as e:
            print(f"Error fetching earnings analysis: {e}")
            return {}

    def analyze_financials(self) -> Dict:
        """Perform comprehensive financial analysis"""
        try:
            # Gather all financial data
            print("Fetching financial statements...")
            financials = self.get_financial_statements()
            
            print("Fetching key metrics...")
            metrics = self.get_key_metrics()
            
            if not metrics or not financials:
                raise ValueError("Failed to fetch required financial data")
            
            # Format financial data for analysis
            try:
                recent_bs = pd.DataFrame(financials['Balance Sheet']['Quarterly']).iloc[:, 0]
                recent_is = pd.DataFrame(financials['Income Statement']['Quarterly']).iloc[:, 0]
                
                # Format numbers to millions/billions for readability
                def format_number(num):
                    if num is None or num == 'N/A':
                        return 'N/A'
                    try:
                        num = float(num)
                        if abs(num) >= 1e9:
                            return f"${num/1e9:.2f}B"
                        elif abs(num) >= 1e6:
                            return f"${num/1e6:.2f}M"
                        else:
                            return f"${num:,.2f}"
                    except:
                        return 'N/A'
                
                # Prepare financial summaries
                balance_sheet_summary = {
                    'Total Assets': format_number(recent_bs.get('Total Assets')),
                    'Total Liabilities': format_number(recent_bs.get('Total Liabilities Net Minority Interest')),
                    'Total Equity': format_number(recent_bs.get('Total Equity Gross Minority Interest')),
                    'Cash and Equivalents': format_number(recent_bs.get('Cash and Cash Equivalents')),
                    'Total Debt': format_number(recent_bs.get('Total Debt'))
                }
                
                income_stmt_summary = {
                    'Revenue': format_number(recent_is.get('Total Revenue')),
                    'Gross Profit': format_number(recent_is.get('Gross Profit')),
                    'Operating Income': format_number(recent_is.get('Operating Income')),
                    'Net Income': format_number(recent_is.get('Net Income')),
                    'EPS': recent_is.get('Diluted EPS', 'N/A')
                }
            except Exception as e:
                print(f"Error formatting financial statements: {e}")
                balance_sheet_summary = {}
                income_stmt_summary = {}
            
            # Create analysis prompt
            prompt = f"""
            Analyze {self.ticker} ({self.company_name}) based on the following financial data:

            1. Key Metrics Summary:
            Valuation:
            - Market Cap: {format_number(metrics['Valuation Metrics'].get('Market Cap'))}
            - P/E Ratio: {metrics['Valuation Metrics'].get('PE Ratio', 'N/A'):.2f}
            - Price/Book: {metrics['Valuation Metrics'].get('Price to Book', 'N/A'):.2f}
            - Current Price: ${metrics['Valuation Metrics'].get('Current Price', 'N/A'):.2f}

            Profitability:
            - Profit Margin: {metrics['Profitability Metrics'].get('Profit Margins', 'N/A'):.1f}%
            - Operating Margin: {metrics['Profitability Metrics'].get('Operating Margins', 'N/A'):.1f}%
            - ROE: {metrics['Profitability Metrics'].get('ROE', 'N/A'):.1f}%

            Growth:
            - Revenue Growth: {metrics['Growth Metrics'].get('Revenue Growth', 'N/A'):.1f}%
            - Earnings Growth: {metrics['Growth Metrics'].get('Earnings Growth', 'N/A'):.1f}%

            Recent Quarter Performance:
            Income Statement:
            {pd.Series(income_stmt_summary).to_string()}

            Balance Sheet:
            {pd.Series(balance_sheet_summary).to_string()}

            Please provide a concise analysis covering:

            1. Financial Health Assessment
            - Current financial position and stability
            - Profitability and operational efficiency
            - Key risks and concerns

            2. Growth & Valuation Analysis
            - Growth trajectory and sustainability
            - Current valuation assessment
            - Fair value estimate

            3. Investment Recommendation
            - Key strengths and competitive advantages
            - Major risks and challenges
            - Target price range
            - Suggested investment horizon
            """
            
            print("Performing analysis...")
            # Make API request to Together AI
            headers = {
                'Authorization': f'Bearer {TOGETHER_API_KEY}',
                'Content-Type': 'application/json'
            }
            
            data = {
                'model': 'mistralai/Mixtral-8x7B-Instruct-v0.1',
                'prompt': prompt,
                'max_tokens': 1000,
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
                        'analysis': analysis,
                        'raw_data': {
                            'metrics': metrics,
                            'recent_quarter': {
                                'balance_sheet': balance_sheet_summary,
                                'income_statement': income_stmt_summary
                            }
                        }
                    }
            
            print(f"API Response: {response.text}")
            return {
                'status': 'error',
                'message': f"API Error: {response.status_code}"
            }
            
        except Exception as e:
            print(f"Exception details: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }

def main():
    try:
        # Analyze stock (default: NVIDIA)
        ticker = input("Enter stock ticker (default: NVDA): ").strip().upper() or "NVDA"
        print(f"\n=== Comprehensive Financial Analysis for {ticker} ===")
        
        analyzer = FinancialAnalyzer(ticker)
        
        print("\nGathering financial data...")
        analysis = analyzer.analyze_financials()
        
        if analysis['status'] == 'success':
            print("\nAnalysis Results:")
            print("=" * 80)
            print(analysis['analysis'])
            print("=" * 80)
            
            print("\nKey Financial Metrics:")
            for category, data in analysis['raw_data']['metrics'].items():
                if data:  # Only print if category has data
                    print(f"\n{category}:")
                    for metric, value in data.items():
                        if isinstance(value, float):
                            if 'Margin' in metric or 'Growth' in metric or 'ROE' in metric or 'ROA' in metric:
                                print(f"{metric}: {value:.2f}%")
                            elif value > 1e9:
                                print(f"{metric}: ${value/1e9:.2f}B")
                            elif value > 1e6:
                                print(f"{metric}: ${value/1e6:.2f}M")
                            else:
                                print(f"{metric}: ${value:,.2f}")
                        else:
                            print(f"{metric}: {value}")
            
            if analysis['raw_data']['recent_quarter']['income_statement']:
                print("\nMost Recent Quarter Results:")
                print("\nIncome Statement Summary:")
                for metric, value in analysis['raw_data']['recent_quarter']['income_statement'].items():
                    print(f"{metric}: {value}")
            
            if analysis['raw_data']['recent_quarter']['balance_sheet']:
                print("\nBalance Sheet Summary:")
                for metric, value in analysis['raw_data']['recent_quarter']['balance_sheet'].items():
                    print(f"{metric}: {value}")
        else:
            print(f"\nError in analysis: {analysis['message']}")
    
    except Exception as e:
        print(f"\nError analyzing stock: {str(e)}")
        print("Please verify the stock ticker and try again.")

if __name__ == "__main__":
    main() 