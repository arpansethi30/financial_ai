from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from typing import Dict, List
import os
from dotenv import load_dotenv
import logging
import time
import alpaca_trade_api as tradeapi

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class AlpacaTradingService:
    def __init__(self):
        self.api_key = os.getenv("ALPACA_API_KEY")
        self.api_secret = os.getenv("ALPACA_API_SECRET")
        self.base_url = "https://paper-api.alpaca.markets"  # Paper trading URL
        
        logger.info("Initializing Alpaca Trading Service")
        logger.info(f"API Key present: {bool(self.api_key)}")
        logger.info(f"API Secret present: {bool(self.api_secret)}")
        
        if not self.api_key or not self.api_secret:
            raise ValueError("Alpaca API credentials not found in environment variables")
        
        try:
            # Initialize both API clients
            self.trading_client = TradingClient(
                api_key=self.api_key,
                secret_key=self.api_secret,
                paper=True
            )
            self.api = tradeapi.REST(
                self.api_key,
                self.api_secret,
                self.base_url,
                api_version='v2'
            )
            logger.info("Successfully created Alpaca Trading Clients")
        except Exception as e:
            logger.error(f"Error creating Alpaca Trading Client: {str(e)}")
            raise

    def create_portfolio_orders(self, portfolio_allocation: List[Dict]):
        """
        Create orders based on portfolio allocation using a simpler approach
        """
        logger.info(f"Creating portfolio orders for {len(portfolio_allocation)} positions")
        
        # First, close all existing positions
        try:
            logger.info("Closing existing positions")
            self.close_all_positions()
            logger.info("Successfully closed existing positions")
            # Wait for positions to be closed
            time.sleep(2)
        except Exception as e:
            logger.error(f"Error closing existing positions: {str(e)}")
        
        orders = []
        successful_orders = 0
        
        for allocation in portfolio_allocation:
            try:
                symbol = allocation["symbol"].upper()
                quantity = int(allocation["quantity"])
                
                if quantity <= 0:
                    logger.warning(f"Skipping {symbol} - quantity is 0")
                    continue
                
                logger.info(f"Attempting to buy {quantity} shares of {symbol}")
                
                # Submit order using the simpler API
                try:
                    order = self.api.submit_order(
                        symbol=symbol,
                        qty=quantity,
                        side='buy',
                        type='market',
                        time_in_force='day'
                    )
                    
                    # Wait briefly for order to be processed
                    time.sleep(1)
                    
                    # Get updated order status
                    updated_order = self.api.get_order(order.id)
                    
                    order_response = {
                        "symbol": symbol,
                        "quantity": quantity,
                        "order_id": updated_order.id,
                        "status": updated_order.status,
                        "created_at": str(updated_order.created_at),
                        "filled_qty": float(updated_order.filled_qty) if hasattr(updated_order, 'filled_qty') and updated_order.filled_qty else None,
                        "filled_avg_price": float(updated_order.filled_avg_price) if hasattr(updated_order, 'filled_avg_price') and updated_order.filled_avg_price else None
                    }
                    
                    successful_orders += 1
                    orders.append(order_response)
                    logger.info(f"Successfully bought {quantity} shares of {symbol}")
                    
                except Exception as e:
                    error_msg = str(e)
                    logger.error(f"Error buying {symbol}: {error_msg}")
                    orders.append({
                        "symbol": symbol,
                        "quantity": quantity,
                        "status": "failed",
                        "error": error_msg
                    })
                    
            except Exception as e:
                logger.error(f"Error processing order for {allocation['symbol']}: {str(e)}")
                orders.append({
                    "symbol": allocation["symbol"],
                    "quantity": allocation["quantity"],
                    "status": "failed",
                    "error": str(e)
                })
        
        logger.info(f"Completed portfolio orders. Success: {successful_orders}/{len(portfolio_allocation)}")
        
        return {
            "orders": orders,
            "total_orders": len(portfolio_allocation),
            "successful_orders": successful_orders
        }

    def get_account(self):
        """Get account information"""
        try:
            logger.info("Attempting to get account information")
            account = self.api.get_account()
            logger.info(f"Successfully retrieved account information")
            return account
        except Exception as e:
            logger.error(f"Error getting account information: {str(e)}")
            raise

    def get_positions(self):
        """Get current positions"""
        try:
            logger.info("Getting current positions")
            positions = self.api.list_positions()
            logger.info(f"Successfully retrieved {len(positions)} positions")
            return positions
        except Exception as e:
            logger.error(f"Error getting positions: {str(e)}")
            raise

    def close_all_positions(self):
        """Close all open positions"""
        try:
            logger.info("Attempting to close all positions")
            self.api.close_all_positions()
            logger.info("Successfully closed all positions")
        except Exception as e:
            logger.error(f"Error closing positions: {str(e)}")
            raise

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