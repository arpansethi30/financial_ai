'use client';

import React, { useState } from 'react';
import {
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from 'recharts';

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8', '#82CA9D'];

interface PortfolioRequest {
  investment_amount: number;
  risk_appetite: 'conservative' | 'moderate' | 'aggressive';
  investment_period: number;
  company_count: number;
}

interface PortfolioData {
  portfolio: {
    recommendations: {
      stock_recommendations: Record<string, Array<{
        symbol: string;
        weight: number;
        amount: number;
        suggested_shares: number;
        risk_level: string;
        fundamentals: {
          current_price: number;
        };
      }>>;
      allocation_summary: {
        total_investment: number;
        total_stocks: number;
        total_sectors: number;
      };
    };
  };
  analysis: string;
}

export default function PortfolioAnalysis() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [portfolioData, setPortfolioData] = useState<PortfolioData | null>(null);
  const [formData, setFormData] = useState<PortfolioRequest>({
    investment_amount: 10000,
    risk_appetite: 'moderate',
    investment_period: 5,
    company_count: 10,
  });

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch('http://localhost:8000/portfolio/recommend', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      });
      
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Server error: ${response.status} - ${errorText}`);
      }
      
      const data = await response.json();
      setPortfolioData(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: name === 'risk_appetite' ? value : Number(value),
    }));
  };

  // Prepare data for pie chart
  const getPieChartData = () => {
    if (!portfolioData) return [];
    
    return Object.entries(portfolioData.portfolio.recommendations.stock_recommendations)
      .map(([sector, stocks]) => ({
        name: sector,
        value: stocks.reduce((sum, stock) => sum + stock.amount, 0)
      }));
  };

  return (
    <div className="min-h-screen bg-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-28 pb-16">
        <div className="max-w-5xl mx-auto">
          <h2 className="text-3xl font-bold text-gray-900 mb-8">AI Portfolio Generator</h2>
          
          <form onSubmit={handleSubmit} className="bg-white rounded-lg shadow-md px-8 pt-6 pb-8 mb-8">
            <h2 className="text-2xl font-bold text-gray-900 mb-6">Generate Portfolio</h2>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              <div className="space-y-6">
                {/* Investment Amount */}
                <div className="form-group">
                  <label className="block text-gray-700 text-sm font-bold mb-2">
                    Investment Amount
                    <div className="mt-1 relative rounded-md shadow-sm">
                      <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                        <span className="text-gray-500 sm:text-sm">$</span>
                      </div>
                      <input
                        type="number"
                        name="investment_amount"
                        value={formData.investment_amount}
                        onChange={handleInputChange}
                        className="block w-full pl-8 pr-3 py-3 text-gray-900 border border-gray-200 rounded-lg focus:ring-2 focus:ring-gray-900 focus:border-gray-900"
                        min="1000"
                        placeholder="Enter amount"
                      />
                    </div>
                  </label>
                </div>

                {/* Risk Appetite */}
                <div className="form-group">
                  <label className="block text-gray-700 text-sm font-bold mb-2">
                    Risk Appetite
                    <div className="mt-1">
                      <select
                        name="risk_appetite"
                        value={formData.risk_appetite}
                        onChange={handleInputChange}
                        className="block w-full py-3 px-3 text-gray-900 border border-gray-200 rounded-lg focus:ring-2 focus:ring-gray-900 focus:border-gray-900"
                      >
                        <option value="conservative">Conservative</option>
                        <option value="moderate">Moderate</option>
                        <option value="aggressive">Aggressive</option>
                      </select>
                    </div>
                  </label>
                </div>
              </div>

              <div className="space-y-6">
                {/* Number of Companies */}
                <div className="form-group">
                  <label className="block text-gray-700 text-sm font-bold mb-2">
                    Number of Stocks
                    <div className="mt-1">
                      <input
                        type="number"
                        name="company_count"
                        value={formData.company_count}
                        onChange={handleInputChange}
                        className="block w-full py-3 px-3 text-gray-900 border border-gray-200 rounded-lg focus:ring-2 focus:ring-gray-900 focus:border-gray-900"
                        min="5"
                        max="30"
                        placeholder="Enter number of stocks"
                      />
                      <p className="mt-1 text-sm text-gray-500">Select between 5 and 30 stocks</p>
                    </div>
                  </label>
                </div>

                {/* Investment Period */}
                <div className="form-group">
                  <label className="block text-gray-700 text-sm font-bold mb-2">
                    Investment Period
                    <div className="mt-1">
                      <input
                        type="number"
                        name="investment_period"
                        value={formData.investment_period}
                        onChange={handleInputChange}
                        className="block w-full py-3 px-3 text-gray-900 border border-gray-200 rounded-lg focus:ring-2 focus:ring-gray-900 focus:border-gray-900"
                        min="1"
                        max="30"
                        placeholder="Enter years"
                      />
                      <p className="mt-1 text-sm text-gray-500">Enter period in years (1-30)</p>
                    </div>
                  </label>
                </div>
              </div>
            </div>

            <div className="mt-8">
              <button
                type="submit"
                disabled={loading}
                className="w-full bg-gray-900 hover:bg-gray-800 text-white font-medium py-4 px-6 rounded-lg focus:outline-none focus:ring-2 focus:ring-gray-900 focus:ring-offset-2 disabled:opacity-50 transition-all duration-200 text-lg"
              >
                {loading ? (
                  <div className="flex items-center justify-center">
                    <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    Generating Your Portfolio...
                  </div>
                ) : 'Generate Portfolio'}
              </button>
            </div>
          </form>

          {error && (
            <div className="bg-red-50 border-l-4 border-red-500 text-red-700 p-4 rounded-lg mb-8" role="alert">
              <div className="flex">
                <div className="flex-shrink-0">
                  <svg className="h-5 w-5 text-red-500" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                  </svg>
                </div>
                <div className="ml-3">
                  <p className="text-sm">{error}</p>
                </div>
              </div>
            </div>
          )}

          {/* Results Display */}
          {portfolioData && (
            <div className="bg-white rounded-lg shadow-md overflow-hidden">
              {/* Portfolio Summary */}
              <div className="p-8 border-b border-gray-200">
                <h2 className="text-2xl font-bold text-gray-900 mb-6">Portfolio Summary</h2>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  <div className="bg-gray-50 p-6 rounded-lg">
                    <h3 className="text-lg font-semibold text-gray-900">Total Investment</h3>
                    <p className="text-3xl font-bold text-gray-900 mt-2">
                      ${portfolioData.portfolio.recommendations.allocation_summary.total_investment.toLocaleString()}
                    </p>
                  </div>
                  <div className="bg-gray-50 p-6 rounded-lg">
                    <h3 className="text-lg font-semibold text-gray-900">Total Stocks</h3>
                    <p className="text-3xl font-bold text-gray-900 mt-2">
                      {portfolioData.portfolio.recommendations.allocation_summary.total_stocks}
                    </p>
                  </div>
                  <div className="bg-gray-50 p-6 rounded-lg">
                    <h3 className="text-lg font-semibold text-gray-900">Total Sectors</h3>
                    <p className="text-3xl font-bold text-gray-900 mt-2">
                      {portfolioData.portfolio.recommendations.allocation_summary.total_sectors}
                    </p>
                  </div>
                </div>
              </div>

              {/* Sector Allocation Chart */}
              <div className="p-8 border-b border-gray-200">
                <h3 className="text-xl font-bold mb-4 text-gray-800">Sector Allocation</h3>
                <div className="h-[400px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={getPieChartData()}
                        dataKey="value"
                        nameKey="name"
                        cx="50%"
                        cy="50%"
                        outerRadius={150}
                        fill="#8884d8"
                        label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                      >
                        {getPieChartData().map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                        ))}
                      </Pie>
                      <Tooltip />
                      <Legend />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Stock Recommendations */}
              <div className="p-8">
                <h3 className="text-xl font-bold mb-4 text-gray-800">Stock Recommendations</h3>
                <div className="overflow-x-auto">
                  <table className="min-w-full divide-y divide-gray-200">
                    <thead className="bg-gray-50">
                      <tr>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Stock</th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Shares</th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Price</th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Amount</th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Weight</th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Risk</th>
                      </tr>
                    </thead>
                    <tbody className="bg-white divide-y divide-gray-200">
                      {Object.entries(portfolioData.portfolio.recommendations.stock_recommendations).map(([sector, stocks]) =>
                        stocks.map((stock) => (
                          <tr key={stock.symbol} className="hover:bg-gray-50">
                            <td className="px-6 py-4">
                              <div className="font-medium text-gray-900">{stock.symbol}</div>
                              <div className="text-xs text-gray-500">{sector}</div>
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap">{stock.suggested_shares}</td>
                            <td className="px-6 py-4 whitespace-nowrap">${stock.fundamentals.current_price.toFixed(2)}</td>
                            <td className="px-6 py-4 whitespace-nowrap">${stock.amount.toFixed(2)}</td>
                            <td className="px-6 py-4 whitespace-nowrap">{stock.weight.toFixed(1)}%</td>
                            <td className="px-6 py-4 whitespace-nowrap">
                              <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full 
                                ${stock.risk_level === 'High' ? 'bg-red-100 text-red-800' : 
                                  stock.risk_level === 'Medium' ? 'bg-yellow-100 text-yellow-800' : 
                                  'bg-green-100 text-green-800'}`}>
                                {stock.risk_level}
                              </span>
                            </td>
                          </tr>
                        ))
                      )}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* AI Analysis */}
              {portfolioData.analysis && (
                <div className="p-8 bg-gray-50">
                  <h3 className="text-xl font-bold mb-4 text-gray-800">Portfolio Analysis</h3>
                  <div className="prose max-w-none">
                    <p className="whitespace-pre-line text-gray-700">{portfolioData.analysis}</p>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
} 