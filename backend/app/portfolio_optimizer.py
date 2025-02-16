import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging
from datetime import datetime, timedelta
import ta
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class PortfolioOptimizer:
    def __init__(self):
        # Define sector ETFs for sector analysis
        self.sector_etfs = {
            "Technology": "XLK",
            "Healthcare": "XLV",
            "Financial": "XLF",
            "Consumer Discretionary": "XLY",
            "Consumer Staples": "XLP",
            "Energy": "XLE",
            "Materials": "XLB",
            "Industrial": "XLI",
            "Utilities": "XLU",
            "Real Estate": "XLRE",
            "Communication Services": "XLC"
        }
        
        # Define risk profiles with fundamental criteria
        self.risk_profiles = {
            "conservative": {
                "max_stock_allocation": 50,
                "min_large_cap": 70,
                "max_per_sector": 20,
                "min_dividend_yield": 2.0,
                "min_pe_ratio": 5,
                "max_pe_ratio": 20,
                "min_profit_margin": 10,
                "min_current_ratio": 1.5,
                "max_debt_to_equity": 0.5
            },
            "moderate": {
                "max_stock_allocation": 70,
                "min_large_cap": 50,
                "max_per_sector": 30,
                "min_dividend_yield": 1.0,
                "min_pe_ratio": 0,
                "max_pe_ratio": 30,
                "min_profit_margin": 5,
                "min_current_ratio": 1.2,
                "max_debt_to_equity": 1.0
            },
            "aggressive": {
                "max_stock_allocation": 90,
                "min_large_cap": 30,
                "max_per_sector": 40,
                "min_dividend_yield": 0,
                "min_pe_ratio": 0,
                "max_pe_ratio": 50,
                "min_profit_margin": 0,
                "min_current_ratio": 1.0,
                "max_debt_to_equity": 2.0
            }
        }

    def analyze_fundamentals(self, symbol: str) -> Dict:
        """Analyze stock fundamentals"""
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            
            return {
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
                "analyst_rating": info.get('recommendationMean', None),
                "target_price": info.get('targetMeanPrice', None),
                "current_price": info.get('currentPrice', None)
            }
        except Exception as e:
            logger.error(f"Error analyzing fundamentals for {symbol}: {str(e)}")
            return {}

    def analyze_technicals(self, symbol: str, period: str = "1y") -> Dict:
        """Analyze technical indicators"""
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(period=period)
            
            if not hist.empty:
                # Calculate technical indicators
                hist['SMA20'] = ta.trend.sma_indicator(hist['Close'], window=20)
                hist['SMA50'] = ta.trend.sma_indicator(hist['Close'], window=50)
                hist['SMA200'] = ta.trend.sma_indicator(hist['Close'], window=200)
                hist['RSI'] = ta.momentum.rsi(hist['Close'], window=14)
                hist['MACD_diff'] = ta.trend.macd_diff(hist['Close'])
                bb_indicator = ta.volatility.BollingerBands(hist['Close'])
                hist['BB_upper'] = bb_indicator.bollinger_hband()
                hist['BB_lower'] = bb_indicator.bollinger_lband()
                
                current_price = hist['Close'].iloc[-1]
                sma20 = hist['SMA20'].iloc[-1]
                sma50 = hist['SMA50'].iloc[-1]
                sma200 = hist['SMA200'].iloc[-1]
                
                # Calculate momentum and trend scores
                trend_score = 0
                trend_score += 1 if current_price > sma20 else -1
                trend_score += 1 if current_price > sma50 else -1
                trend_score += 1 if current_price > sma200 else -1
                trend_score += 1 if sma20 > sma50 else -1
                
                return {
                    "current_price": current_price,
                    "sma20": sma20,
                    "sma50": sma50,
                    "sma200": sma200,
                    "rsi": float(hist['RSI'].iloc[-1]),
                    "macd": float(hist['MACD_diff'].iloc[-1]),
                    "volatility": float(hist['Close'].pct_change().std() * np.sqrt(252) * 100),
                    "trend_score": trend_score,
                    "above_200_sma": current_price > sma200,
                    "above_50_sma": current_price > sma50,
                    "above_20_sma": current_price > sma20,
                    "volume_trend": float(hist['Volume'].tail(20).mean() / hist['Volume'].tail(50).mean())
                }
            return {}
        except Exception as e:
            logger.error(f"Error analyzing technicals for {symbol}: {str(e)}")
            return {}

    def get_stock_score(self, fundamentals: Dict, technicals: Dict, risk_profile: Dict) -> float:
        """Calculate overall stock score based on fundamentals and technicals"""
        score = 0
        max_score = 0
        
        # Fundamental Scoring (60% weight)
        if fundamentals:
            # Valuation (20%)
            if fundamentals.get('pe_ratio'):
                max_score += 20
                if risk_profile['min_pe_ratio'] <= fundamentals['pe_ratio'] <= risk_profile['max_pe_ratio']:
                    score += 20
            
            # Profitability (15%)
            if fundamentals.get('profit_margin'):
                max_score += 15
                if fundamentals['profit_margin'] >= risk_profile['min_profit_margin']:
                    score += 15
            
            # Financial Health (15%)
            if fundamentals.get('debt_to_equity') and fundamentals.get('current_ratio'):
                max_score += 15
                if fundamentals['debt_to_equity'] <= risk_profile['max_debt_to_equity']:
                    score += 7.5
                if fundamentals['current_ratio'] >= risk_profile['min_current_ratio']:
                    score += 7.5
            
            # Growth & Returns (10%)
            if fundamentals.get('revenue_growth') and fundamentals.get('roe'):
                max_score += 10
                if fundamentals['revenue_growth'] > 0:
                    score += 5
                if fundamentals['roe'] > 10:
                    score += 5
        
        # Technical Scoring (40% weight)
        if technicals:
            # Trend (20%)
            max_score += 20
            score += (technicals['trend_score'] + 4) * 2.5  # Convert -4 to +4 scale to 0-20
            
            # Momentum (10%)
            if 30 <= technicals.get('rsi', 0) <= 70:
                max_score += 10
                score += 10
            
            # Volume (10%)
            if technicals.get('volume_trend', 0) > 1:
                max_score += 10
                score += 10
        
        return (score / max_score * 100) if max_score > 0 else 0

    def get_stock_recommendations(self, sector: str, risk_appetite: str, sector_amount: float, investment_period: int) -> List[Dict]:
        """Get stock recommendations for a specific sector."""
        try:
            etf = yf.Ticker(self.sector_etfs[sector])
            
            # Get holdings with error handling
            try:
                holdings = etf.get_holdings()
            except:
                # Fallback to getting top holdings from info
                info = etf.info
                holdings = [h['symbol'] for h in info.get('holdings', [])][:5] if info.get('holdings') else []
            
            if not holdings:
                return []

            recommendations = []
            
            # Get top 5 holdings
            for symbol in list(holdings.keys())[:5] if isinstance(holdings, dict) else holdings[:5]:
                try:
                    stock = yf.Ticker(symbol)
                    info = stock.info
                    
                    if not info:
                        continue
                    
                    current_price = info.get('regularMarketPrice', 0)
                    if current_price <= 0:
                        continue

                    # Determine risk level based on beta
                    beta = info.get('beta', 1.0)
                    risk_level = 'High' if beta > 1.2 else 'Medium' if beta > 0.8 else 'Low'

                    # Calculate weight based on risk appetite and investment period
                    weight = 20  # Equal weight by default
                    if risk_appetite == 'conservative':
                        weight = 15 if risk_level == 'Low' else 10
                    elif risk_appetite == 'aggressive':
                        weight = 25 if risk_level == 'High' else 20

                    # Adjust for investment period
                    if investment_period > 10:  # Long-term
                        weight *= 1.2
                    elif investment_period < 5:  # Short-term
                        weight *= 0.8

                    # Calculate amount and shares
                    amount = (sector_amount * weight) / 100
                    suggested_shares = max(1, int(amount / current_price))
                    actual_amount = suggested_shares * current_price

                    recommendations.append({
                        'symbol': symbol,
                        'weight': weight,
                        'amount': actual_amount,
                        'suggested_shares': suggested_shares,
                        'risk_level': risk_level,
                        'fundamentals': {
                            'current_price': current_price
                        }
                    })

                except Exception as e:
                    logger.error(f"Error processing stock {symbol}: {str(e)}")
                    continue

            return recommendations

        except Exception as e:
            logger.error(f"Error getting recommendations for {sector}: {str(e)}")
            return []

    def _generate_recommendation_reason(self, fundamentals: Dict, technicals: Dict, score: float) -> str:
        """Generate a human-readable reason for the stock recommendation"""
        reasons = []
        
        # Fundamental analysis reasons
        if fundamentals.get('pe_ratio') and fundamentals['pe_ratio'] < 20:
            reasons.append("Attractive valuation")
        if fundamentals.get('dividend_yield', 0) > 2:
            reasons.append("Strong dividend yield")
        if fundamentals.get('profit_margin', 0) > 15:
            reasons.append("High profit margins")
        
        # Technical analysis reasons
        if technicals.get('rsi', 0) < 70 and technicals.get('rsi', 0) > 30:
            reasons.append("Healthy momentum")
        if technicals.get('trend_score', 0) > 2:
            reasons.append("Strong upward trend")
        
        # Score-based summary
        if score >= 80:
            reasons.append("Excellent overall metrics")
        elif score >= 60:
            reasons.append("Strong performance potential")
        
        return "; ".join(reasons) if reasons else "Balanced risk-reward profile"

    def optimize_portfolio(self, request_data: Dict) -> Dict:
        """
        Generate a stock portfolio based on investment amount and risk appetite.
        """
        try:
            investment_amount = request_data['investment_amount']
            risk_appetite = request_data['risk_appetite']
            investment_period = request_data['investment_period']

            # Define sector weights based on risk appetite
            sector_weights = {
                'conservative': {
                    'Technology': 15,
                    'Healthcare': 20,
                    'Consumer Staples': 25,
                    'Utilities': 25,
                    'Communication Services': 15
                },
                'moderate': {
                    'Technology': 25,
                    'Healthcare': 20,
                    'Financial': 20,
                    'Consumer Discretionary': 20,
                    'Industrial': 15
                },
                'aggressive': {
                    'Technology': 35,
                    'Financial': 25,
                    'Consumer Discretionary': 20,
                    'Energy': 10,
                    'Communication Services': 10
                }
            }

            selected_sectors = sector_weights[risk_appetite]
            stock_recommendations = {}
            total_stocks = 0

            # Get stock recommendations for each sector
            for sector, weight in selected_sectors.items():
                sector_amount = (investment_amount * weight) / 100
                sector_stocks = self.get_stock_recommendations(
                    sector=sector,
                    risk_appetite=risk_appetite,
                    sector_amount=sector_amount,
                    investment_period=investment_period
                )
                
                if sector_stocks:
                    stock_recommendations[sector] = sector_stocks
                    total_stocks += len(sector_stocks)

            # Calculate total actual investment
            total_investment = sum(
                sum(stock['amount'] for stock in stocks)
                for stocks in stock_recommendations.values()
            )

            return {
                'portfolio': {
                    'recommendations': {
                        'stock_recommendations': stock_recommendations,
                        'allocation_summary': {
                            'total_investment': total_investment,
                            'total_stocks': total_stocks,
                            'total_sectors': len(stock_recommendations)
                        }
                    }
                }
            }

        except Exception as e:
            logger.error(f"Error in portfolio optimization: {str(e)}")
            raise Exception(f"Failed to optimize portfolio: {str(e)}")

    def _generate_stock_strategy(
        self,
        request_data: Dict,
        stock_recommendations: Dict
    ) -> str:
        """Generate personalized stock investment strategy"""
        
        strategy_parts = []
        
        # Introduction
        strategy_parts.append(f"Stock Investment Strategy for {request_data['investment_period']} Year Horizon")
        strategy_parts.append(f"\nRisk Profile: {request_data['risk_appetite'].title()}")
        
        # Stock Portfolio
        if stock_recommendations and any(stocks for stocks in stock_recommendations.values()):
            strategy_parts.append("\nRecommended Stock Portfolio:")
            for sector, stocks in stock_recommendations.items():
                if stocks:  # Only process non-empty sector stocks
                    strategy_parts.append(f"\n{sector} Sector:")
                    for stock in stocks:
                        strategy_parts.append(
                            f"- {stock['symbol']}: {stock['weight']:.1f}% allocation "
                            f"(${stock['amount']:,.2f}, {stock['suggested_shares']} shares)"
                        )
                        strategy_parts.append(f"  Reason: {stock['recommendation_reason']}")
            
            # Calculate max exposure
            try:
                max_exposure = max(
                    stock['weight'] 
                    for stocks in stock_recommendations.values() 
                    for stock in stocks
                )
            except ValueError:
                max_exposure = 0
        else:
            strategy_parts.append("\nNo specific stock recommendations at this time.")
            max_exposure = 0
        
        # Investment Approach
        strategy_parts.append("\nInvestment Approach:")
        if request_data['investment_period'] > 10:
            strategy_parts.append("- Long-term growth focus with emphasis on quality companies")
            strategy_parts.append("- Regular dividend reinvestment recommended")
            strategy_parts.append("- Quarterly portfolio rebalancing suggested")
        elif request_data['investment_period'] > 5:
            strategy_parts.append("- Medium-term balanced approach")
            strategy_parts.append("- Mix of growth and value stocks")
            strategy_parts.append("- Semi-annual portfolio rebalancing recommended")
        else:
            strategy_parts.append("- Short-term focus on stability")
            strategy_parts.append("- Focus on blue-chip stocks")
            strategy_parts.append("- More frequent portfolio monitoring advised")
        
        # Regular Investment Plan
        if request_data.get('regular_investment'):
            strategy_parts.append(f"\nMonthly Investment Plan:")
            strategy_parts.append(f"- Recommended monthly investment: ${request_data['regular_investment']:,.2f}")
            strategy_parts.append("- Implement dollar-cost averaging strategy")
        
        # Risk Management
        strategy_parts.append("\nRisk Management:")
        strategy_parts.append("- Diversification across sectors and companies")
        strategy_parts.append(f"- Maximum single stock exposure: {max_exposure:.1f}%")
        strategy_parts.append("- Regular monitoring and rebalancing recommended")
        
        return "\n".join(strategy_parts)

    def analyze_sector_performance(self, sectors: List[str], period: str = "1y") -> Dict:
        """Analyze performance of specified sectors"""
        sector_performance = {}
        
        for sector in sectors:
            if sector in self.sector_etfs:
                try:
                    etf = yf.Ticker(self.sector_etfs[sector])
                    hist = etf.history(period=period)
                    
                    returns = hist['Close'].pct_change()
                    sector_performance[sector] = {
                        "annual_return": float(((hist['Close'].iloc[-1] / hist['Close'].iloc[0]) - 1) * 100),
                        "volatility": float(returns.std() * np.sqrt(252) * 100),
                        "sharpe_ratio": float(returns.mean() / returns.std() * np.sqrt(252))
                    }
                except Exception as e:
                    logger.error(f"Error analyzing sector {sector}: {str(e)}")
                    
        return sector_performance

    def _generate_asset_allocation(self, risk_profile: Dict, request_data: Dict) -> Dict:
        """Generate high-level asset allocation"""
        age_factor = min((100 - request_data['age']) / 100, 0.9)
        stock_allocation = min(
            risk_profile['max_stock_allocation'],
            risk_profile['max_stock_allocation'] * age_factor
        )
        
        return {
            "stocks": stock_allocation,
            "bonds": 90 - stock_allocation,
            "cash": 10
        }

    def _generate_sector_allocation(self, sector_analysis: Dict, risk_profile: Dict, preferred_sectors: List[str]) -> Dict:
        """Generate sector allocation based on analysis and preferences"""
        allocation = {}
        total_weight = 0
        
        # Sort sectors by Sharpe ratio
        sorted_sectors = sorted(
            sector_analysis.items(),
            key=lambda x: x[1]['sharpe_ratio'],
            reverse=True
        )
        
        # Allocate based on performance and preferences
        for sector, metrics in sorted_sectors:
            if sector in preferred_sectors:
                weight = min(risk_profile['max_per_sector'], 30)
            else:
                weight = min(risk_profile['max_per_sector'], 15)
                
            if total_weight + weight <= 100:
                allocation[sector] = weight
                total_weight += weight
        
        # Normalize weights to 100%
        if total_weight > 0:
            for sector in allocation:
                allocation[sector] = (allocation[sector] / total_weight) * 100
                
        return allocation

# Initialize optimizer
portfolio_optimizer = PortfolioOptimizer() 