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
  sectors?: string[];
}

interface StockRecommendation {
  symbol: string;
  weight: number;
  amount: number;
  risk_level: string;
  sentiment_analysis: string;
  ai_analysis: string;
  suggested_shares: number;
  fundamentals: {
    current_price: number;
    market_cap: number;
    pe_ratio: number;
    revenue_growth: number;
    profit_margins: number;
  };
  recent_news: Array<{
    title: string;
    url: string;
    source: {
      name: string;
    };
  }>;
}

interface PortfolioResponse {
  status: string;
  portfolio: {
    summary: {
      investment_amount: number;
      risk_profile: string;
      time_horizon: number;
      total_stocks: number;
    };
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

const ComprehensivePortfolio: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
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
      const response = await fetch('http://localhost:8001/api/portfolio/comprehensive', {
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
                  {formatCurrency(portfolioData.portfolio.summary.investment_amount)}
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-600">Risk Profile</p>
                <p className="text-xl font-bold text-gray-900 capitalize">
                  {portfolioData.portfolio.summary.risk_profile}
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-600">Time Horizon</p>
                <p className="text-xl font-bold text-gray-900">
                  {portfolioData.portfolio.summary.time_horizon} years
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-600">Total Stocks</p>
                <p className="text-xl font-bold text-gray-900">
                  {portfolioData.portfolio.summary.total_stocks}
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
                            stock.risk_level === 'high' ? 'text-red-600' :
                            stock.risk_level === 'medium' ? 'text-yellow-600' :
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
                          {formatCurrency(stock.fundamentals.current_price)}
                        </p>
                      </div>
                      <div>
                        <p className="text-sm text-gray-600">Total Investment</p>
                        <p className="text-base font-medium text-gray-900">{formatCurrency(stock.amount)}</p>
                      </div>
                    </div>

                    {/* Sentiment and Analysis */}
                    <div className="mt-4 space-y-4">
                      <div className="bg-white p-4 rounded-lg border border-gray-200">
                        <div className="prose prose-sm max-w-none text-gray-900">
                          <div dangerouslySetInnerHTML={{ 
                            __html: stock.sentiment_analysis
                              .replace(/\n/g, '<br />')
                              .replace(/•/g, '&#8226;')
                              .replace(/\*\*(.*?)\*\*/g, '<strong class="text-gray-900">$1</strong>')
                          }} />
                        </div>
                      </div>
                      <div className="bg-white p-4 rounded-lg border border-gray-200">
                        <div className="prose prose-sm max-w-none text-gray-900">
                          <div dangerouslySetInnerHTML={{ 
                            __html: stock.ai_analysis
                              .replace(/\n/g, '<br />')
                              .replace(/•/g, '&#8226;')
                              .replace(/\*\*(.*?)\*\*/g, '<strong class="text-gray-900">$1</strong>')
                          }} />
                        </div>
                      </div>
                    </div>

                    {/* Financial Metrics */}
                    <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-4">
                      <div>
                        <p className="text-xs text-gray-600">Market Cap</p>
                        <p className="text-sm font-medium text-gray-900">
                          {formatCurrency(stock.fundamentals.market_cap)}
                        </p>
                      </div>
                      <div>
                        <p className="text-xs text-gray-600">P/E Ratio</p>
                        <p className="text-sm font-medium text-gray-900">
                          {stock.fundamentals.pe_ratio?.toFixed(2) || 'N/A'}
                        </p>
                      </div>
                      <div>
                        <p className="text-xs text-gray-600">Revenue Growth</p>
                        <p className="text-sm font-medium text-gray-900">
                          {stock.fundamentals.revenue_growth?.toFixed(2)}%
                        </p>
                      </div>
                      <div>
                        <p className="text-xs text-gray-600">Profit Margins</p>
                        <p className="text-sm font-medium text-gray-900">
                          {stock.fundamentals.profit_margins?.toFixed(2)}%
                        </p>
                      </div>
                    </div>

                    {/* Recent News */}
                    {stock.recent_news && stock.recent_news.length > 0 && (
                      <div className="mt-4">
                        <p className="text-sm font-medium text-gray-900 mb-2">Recent News</p>
                        <div className="space-y-2">
                          {stock.recent_news.map((news, index) => (
                            <a
                              key={index}
                              href={news.url}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="block p-2 hover:bg-gray-50 rounded-lg transition-colors"
                            >
                              <p className="text-sm font-medium text-gray-900">{news.title}</p>
                              <p className="text-xs text-gray-600">Source: {news.source.name}</p>
                            </a>
                          ))}
                        </div>
                      </div>
                    )}
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
        </div>
      )}
    </div>
  );
};

export default ComprehensivePortfolio; 