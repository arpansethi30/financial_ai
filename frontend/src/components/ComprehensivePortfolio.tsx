'use client';

import React, { useState } from 'react';
import {
  TrendingUp,
  TrendingDown,
  AttachMoney,
  Timeline,
  Assessment,
  ShowChart,
} from '@mui/icons-material';

interface PortfolioRequest {
  investment_amount: number;
  risk_appetite: 'conservative' | 'moderate' | 'aggressive';
  investment_period: number;
  company_count: number;
}

interface StockRecommendation {
  symbol: string;
  company_name: string;
  weight: number;
  amount: number;
  suggested_shares: number;
  risk_level: string;
  current_price: number;
  sector: string;
  market_cap: number;
  beta: number | null;
  pe_ratio: number | null;
  dividend_yield: number;
  analysis: {
    market: string;
    fundamental: string;
    overview: string;
  };
  recent_news?: Array<{
    title: string;
    url: string;
    source: string;
    description?: string;
    date: string;
    sentiment: number;
  }>;
}

interface PortfolioResponse {
  status: string;
  message?: string;
  portfolio: {
    recommendations: {
      stock_recommendations: Record<string, StockRecommendation[]>;
      allocation_summary: {
        total_investment: number;
        total_stocks: number;
        total_sectors: number;
      };
    };
    analysis: string;
  };
}

interface TradingResponse {
  success: boolean;
  message: string;
  data?: {
    orders: Array<{
      symbol: string;
      quantity: number;
      order_id?: string;
      status?: string;
      error?: string;
    }>;
    total_orders: number;
    successful_orders: number;
  };
}

const ComprehensivePortfolio: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const [tradingLoading, setTradingLoading] = useState(false);
  const [error, setError] = useState('');
  const [tradingError, setTradingError] = useState('');
  const [tradingSuccess, setTradingSuccess] = useState('');
  const [portfolioData, setPortfolioData] = useState<PortfolioResponse | null>(null);
  const [formData, setFormData] = useState<PortfolioRequest>({
    investment_amount: 10000,
    risk_appetite: 'moderate',
    investment_period: 5,
    company_count: 10,
  });

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value, type } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: type === 'number' ? Number(value) : value,
    }));
  };

  const generatePortfolio = async () => {
    setLoading(true);
    setError('');
    setPortfolioData(null);

    try {
      const response = await fetch('http://localhost:8000/api/portfolio/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to generate portfolio');
      }

      const data = await response.json();
      setPortfolioData(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to generate portfolio');
      console.error('Portfolio generation error:', err);
    } finally {
      setLoading(false);
    }
  };

  const executeTrades = async () => {
    if (!portfolioData) return;
    
    setTradingLoading(true);
    setTradingError('');
    setTradingSuccess('');

    try {
      const response = await fetch('http://localhost:8000/portfolio/execute', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(portfolioData),
      });

      const data: TradingResponse = await response.json();
      
      if (data.success) {
        setTradingSuccess(data.message);
      } else {
        setTradingError(data.message);
      }
    } catch (err) {
      setTradingError(err instanceof Error ? err.message : 'Failed to execute trades');
      console.error('Trading error:', err);
    } finally {
      setTradingLoading(false);
    }
  };

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
    }).format(amount);
  };

  const formatPercentage = (value: number) => {
    return `${(value * 100).toFixed(2)}%`;
  };

  return (
    <div className="max-w-3xl mx-auto">
      <form className="space-y-8 bg-white p-8 rounded-2xl border border-gray-200 shadow-lg">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <label htmlFor="investment_amount" className="block text-sm font-medium text-gray-700 mb-2">
              Investment Amount ($)
            </label>
            <input
              type="number"
              id="investment_amount"
              name="investment_amount"
              value={formData.investment_amount}
              onChange={handleInputChange}
              className="w-full p-3 border border-gray-200 rounded-xl focus:ring-2 focus:ring-gray-200 focus:border-gray-400 transition-all
                        text-gray-900 placeholder-gray-400"
              placeholder="Enter investment amount"
              required
              min="1000"
              step="100"
            />
          </div>

          <div>
            <label htmlFor="risk_appetite" className="block text-sm font-medium text-gray-700 mb-2">
              Risk Appetite
            </label>
            <select
              id="risk_appetite"
              name="risk_appetite"
              value={formData.risk_appetite}
              onChange={handleInputChange}
              className="w-full p-3 border border-gray-200 rounded-xl focus:ring-2 focus:ring-gray-200 focus:border-gray-400 transition-all
                        text-gray-900"
              required
            >
              <option value="conservative">Conservative</option>
              <option value="moderate">Moderate</option>
              <option value="aggressive">Aggressive</option>
            </select>
          </div>

          <div>
            <label htmlFor="investment_period" className="block text-sm font-medium text-gray-700 mb-2">
              Investment Period (Years)
            </label>
            <input
              type="number"
              id="investment_period"
              name="investment_period"
              value={formData.investment_period}
              onChange={handleInputChange}
              className="w-full p-3 border border-gray-200 rounded-xl focus:ring-2 focus:ring-gray-200 focus:border-gray-400 transition-all
                        text-gray-900 placeholder-gray-400"
              placeholder="Enter investment period"
              required
              min="1"
              max="30"
            />
          </div>

          <div>
            <label htmlFor="company_count" className="block text-sm font-medium text-gray-700 mb-2">
              Number of Companies
            </label>
            <input
              type="number"
              id="company_count"
              name="company_count"
              value={formData.company_count}
              onChange={handleInputChange}
              className="w-full p-3 border border-gray-200 rounded-xl focus:ring-2 focus:ring-gray-200 focus:border-gray-400 transition-all
                        text-gray-900 placeholder-gray-400"
              placeholder="Enter number of companies"
              required
              min="5"
              max="30"
            />
          </div>
        </div>

        <button
          onClick={generatePortfolio}
          disabled={loading}
          className="w-full px-8 py-4 text-base font-medium rounded-xl 
                    bg-gray-900 text-white
                    hover:bg-gray-800 transition-all duration-200
                    shadow-lg disabled:bg-gray-300 disabled:cursor-not-allowed
                    flex items-center justify-center"
        >
          {loading ? (
            <div className="flex items-center justify-center">
              <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
            </div>
          ) : (
            'Generate Portfolio'
          )}
        </button>
      </form>

      {error && (
        <div className="mt-6 p-4 rounded-xl bg-red-50 border border-red-200">
          <p className="text-red-600">{error}</p>
        </div>
      )}

      {portfolioData && (
        <div className="mt-12 space-y-8">
          {/* Portfolio Summary */}
          <div className="bg-white p-8 rounded-2xl border border-gray-200 shadow-lg">
            <h2 className="text-2xl font-bold text-gray-900 mb-6">Portfolio Summary</h2>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div>
                <p className="text-sm text-gray-600">Total Investment</p>
                <p className="text-xl font-bold text-gray-900">
                  {formatCurrency(portfolioData.portfolio.recommendations.allocation_summary.total_investment)}
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-600">Risk Profile</p>
                <p className="text-xl font-bold text-gray-900 capitalize">
                  {formData.risk_appetite}
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-600">Time Horizon</p>
                <p className="text-xl font-bold text-gray-900">
                  {formData.investment_period} years
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-600">Total Stocks</p>
                <p className="text-xl font-bold text-gray-900">
                  {portfolioData.portfolio.recommendations.allocation_summary.total_stocks}
                </p>
              </div>
            </div>
          </div>

          {/* Stock Recommendations */}
          {Object.entries(portfolioData.portfolio.recommendations.stock_recommendations).map(([sector, stocks]) => (
            <div key={sector} className="bg-white p-8 rounded-2xl border border-gray-200 shadow-lg">
              <h3 className="text-xl font-bold text-gray-900 mb-6">{sector} Sector</h3>
              <div className="space-y-6">
                {stocks.map((stock) => (
                  <div key={stock.symbol} className="border border-gray-200 rounded-xl p-6">
                    <div className="flex justify-between items-start mb-4">
                      <div>
                        <h4 className="text-lg font-bold text-gray-900">{stock.symbol}</h4>
                        <p className="text-sm text-gray-600">
                          Risk Level: <span className={`font-medium ${
                            stock.risk_level === 'High' ? 'text-red-600' :
                            stock.risk_level === 'Medium' ? 'text-yellow-600' :
                            'text-green-600'
                          }`}>{stock.risk_level}</span>
                        </p>
                      </div>
                      <div className="text-right">
                        <p className="text-sm text-gray-600">Allocation</p>
                        <p className="text-lg font-bold text-gray-900">{stock.weight.toFixed(2)}%</p>
                      </div>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                      <div>
                        <p className="text-sm text-gray-600">Shares to Buy</p>
                        <p className="text-base font-medium text-gray-900">{stock.suggested_shares}</p>
                      </div>
                      <div>
                        <p className="text-sm text-gray-600">Current Price</p>
                        <p className="text-base font-medium text-gray-900">
                          {formatCurrency(stock.current_price)}
                        </p>
                      </div>
                      <div>
                        <p className="text-sm text-gray-600">Total Investment</p>
                        <p className="text-base font-medium text-gray-900">{formatCurrency(stock.amount)}</p>
                      </div>
                    </div>

                    {/* Stock Analysis Sections */}
                    <div className="mt-6 space-y-4">
                      {/* Market Sentiment and Fundamental Analysis */}
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div className="bg-gray-50 p-4 rounded-lg">
                          <h5 className="text-sm font-semibold text-gray-900 mb-2">Market Analysis</h5>
                          <p className="text-sm text-gray-700 whitespace-pre-line">
                            {stock.analysis.market}
                          </p>
                        </div>
                        
                        <div className="bg-gray-50 p-4 rounded-lg">
                          <h5 className="text-sm font-semibold text-gray-900 mb-2">Fundamental Analysis</h5>
                          <p className="text-sm text-gray-700 whitespace-pre-line">
                            {stock.analysis.fundamental}
                          </p>
                        </div>
                      </div>

                      {/* AI Overview */}
                      <div className="bg-gray-50 p-4 rounded-lg">
                        <h5 className="text-sm font-semibold text-gray-900 mb-2">Investment Overview</h5>
                        <p className="text-sm text-gray-700 whitespace-pre-line">
                          {stock.analysis.overview}
                        </p>
                      </div>

                      {/* Financial Metrics */}
                      <div className="bg-white p-6 rounded-xl border border-gray-200">
                        <h5 className="text-base font-semibold text-gray-900 mb-3">Key Metrics</h5>
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                          <div className="bg-gray-50 p-4 rounded-lg">
                            <p className="text-xs text-gray-600 mb-1">Market Cap</p>
                            <p className="text-sm font-medium text-gray-900">
                              {formatCurrency(stock.market_cap)}
                            </p>
                          </div>
                          <div className="bg-gray-50 p-4 rounded-lg">
                            <p className="text-xs text-gray-600 mb-1">P/E Ratio</p>
                            <p className="text-sm font-medium text-gray-900">
                              {stock.pe_ratio ? stock.pe_ratio.toFixed(2) : 'N/A'}
                            </p>
                          </div>
                          <div className="bg-gray-50 p-4 rounded-lg">
                            <p className="text-xs text-gray-600 mb-1">Beta</p>
                            <p className="text-sm font-medium text-gray-900">
                              {stock.beta ? stock.beta.toFixed(2) : 'N/A'}
                            </p>
                          </div>
                          <div className="bg-gray-50 p-4 rounded-lg">
                            <p className="text-xs text-gray-600 mb-1">Dividend Yield</p>
                            <p className="text-sm font-medium text-gray-900">
                              {stock.dividend_yield ? `${(stock.dividend_yield * 100).toFixed(2)}%` : 'N/A'}
                            </p>
                          </div>
                        </div>
                      </div>

                      {/* Recent News */}
                      {stock.recent_news && stock.recent_news.length > 0 && (
                        <div className="bg-white p-6 rounded-xl border border-gray-200">
                          <h5 className="text-base font-semibold text-gray-900 mb-3">Recent News</h5>
                          <div className="space-y-3">
                            {stock.recent_news.map((news, index) => (
                              <a
                                key={index}
                                href={news.url}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="block p-4 bg-gray-50 hover:bg-gray-100 rounded-lg transition-colors"
                              >
                                <p className="text-sm font-medium text-gray-900">{news.title}</p>
                                <p className="text-xs text-gray-600 mt-1">Source: {news.source}</p>
                                <p className="text-xs text-gray-500 mt-1">Date: {news.date}</p>
                                {news.sentiment && (
                                  <p className={`text-xs mt-1 ${
                                    news.sentiment > 0 ? 'text-green-600' :
                                    news.sentiment < 0 ? 'text-red-600' :
                                    'text-gray-600'
                                  }`}>
                                    Sentiment: {news.sentiment > 0 ? 'Positive' : news.sentiment < 0 ? 'Negative' : 'Neutral'}
                                  </p>
                                )}
                              </a>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ))}

          {/* Portfolio Analysis */}
          <div className="bg-white p-8 rounded-2xl border border-gray-200 shadow-lg">
            <h3 className="text-xl font-bold text-gray-900 mb-4">Portfolio Analysis</h3>
            <div className="prose prose-sm max-w-none">
              <p className="whitespace-pre-line">{portfolioData.portfolio.analysis}</p>
            </div>
          </div>

          {/* Trading Button */}
          <div className="bg-white p-8 rounded-2xl border border-gray-200 shadow-lg">
            <button
              onClick={executeTrades}
              disabled={tradingLoading}
              className="w-full px-8 py-4 text-base font-medium rounded-xl 
                        bg-blue-600 text-white
                        hover:bg-blue-700 transition-all duration-200
                        shadow-lg disabled:bg-gray-300 disabled:cursor-not-allowed
                        flex items-center justify-center"
            >
              {tradingLoading ? (
                <div className="flex items-center justify-center">
                  <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                </div>
              ) : (
                'Execute Trades in Alpaca'
              )}
            </button>
            
            {tradingSuccess && (
              <div className="mt-4 p-4 rounded-xl bg-green-50 border border-green-200">
                <p className="text-green-600">{tradingSuccess}</p>
              </div>
            )}
            
            {tradingError && (
              <div className="mt-4 p-4 rounded-xl bg-red-50 border border-red-200">
                <p className="text-red-600">{tradingError}</p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default ComprehensivePortfolio; 