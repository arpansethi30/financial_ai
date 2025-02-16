from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from typing import Dict, List
import os
from dotenv import load_dotenv
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class AlpacaTradingService:
    def __init__(self):
        self.api_key = os.getenv("ALPACA_API_KEY")
        self.api_secret = os.getenv("ALPACA_API_SECRET")
        
        logger.info(f"Initializing Alpaca Trading Service")
        logger.info(f"API Key present: {bool(self.api_key)}")
        logger.info(f"API Secret present: {bool(self.api_secret)}")
        
        if not self.api_key or not self.api_secret:
            raise ValueError("Alpaca API credentials not found in environment variables")
        
        try:
            self.trading_client = TradingClient(
                api_key=self.api_key,
                secret_key=self.api_secret,
                paper=True  # Ensure we're using paper trading
            )
            logger.info("Successfully created Alpaca Trading Client")
        except Exception as e:
            logger.error(f"Error creating Alpaca Trading Client: {str(e)}")
            raise

    def get_account(self):
        """Get account information"""
        try:
            logger.info("Attempting to get account information")
            account = self.trading_client.get_account()
            logger.info(f"Successfully retrieved account information: {account}")
            return account
        except Exception as e:
            logger.error(f"Error getting account information: {str(e)}")
            raise

    def create_portfolio_orders(self, portfolio_allocation: List[Dict]):
        """
        Create orders based on portfolio allocation
        portfolio_allocation: List of dicts with format:
        [
            {
                "symbol": "AAPL",
                "percentage": 20,  # Percentage of portfolio to allocate
                "quantity": 10     # Number of shares to buy
            },
            ...
        ]
        """
        orders = []
        for allocation in portfolio_allocation:
            try:
                # Create market order
                market_order = MarketOrderRequest(
                    symbol=allocation["symbol"],
                    qty=allocation["quantity"],
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY
                )
                
                # Submit order
                order = self.trading_client.submit_order(market_order)
                orders.append({
                    "symbol": allocation["symbol"],
                    "quantity": allocation["quantity"],
                    "order_id": order.id,
                    "status": order.status
                })
            except Exception as e:
                orders.append({
                    "symbol": allocation["symbol"],
                    "error": str(e)
                })
        
        return orders

    def get_positions(self):
        """Get current positions"""
        return self.trading_client.get_all_positions()

    def close_all_positions(self):
        """Close all open positions"""
        return self.trading_client.close_all_positions(cancel_orders=True)

    def test_buy_single_stock(self, symbol: str, quantity: int):
        """
        Test method to buy a single stock
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            quantity: Number of shares to buy
        """
        try:
            # Create market order
            market_order = MarketOrderRequest(
                symbol=symbol,
                qty=quantity,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY
            )
            
            # Submit order
            order = self.trading_client.submit_order(market_order)
            return {
                "symbol": symbol,
                "quantity": quantity,
                "order_id": order.id,
                "status": order.status,
                "created_at": order.created_at,
                "filled_at": order.filled_at,
                "filled_qty": order.filled_qty,
                "filled_avg_price": order.filled_avg_price
            }
        except Exception as e:
            raise Exception(f"Error placing order: {str(e)}")

# Initialize trading service
trading_service = AlpacaTradingService() 