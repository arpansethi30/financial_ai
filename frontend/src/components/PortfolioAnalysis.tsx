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
    <div className="space-y-8 p-6">
      {/* Input Form */}
      <form onSubmit={handleSubmit} className="bg-white shadow-lg rounded-lg px-8 pt-6 pb-8 mb-4">
        <h2 className="text-2xl font-bold mb-6 text-gray-800">Generate Portfolio</h2>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div>
            <label className="block text-gray-700 text-sm font-bold mb-2">
              Investment Amount ($)
              <input
                type="number"
                name="investment_amount"
                value={formData.investment_amount}
                onChange={handleInputChange}
                className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline mt-1"
                min="1000"
              />
            </label>
          </div>
          
          <div>
            <label className="block text-gray-700 text-sm font-bold mb-2">
              Risk Appetite
              <select
                name="risk_appetite"
                value={formData.risk_appetite}
                onChange={handleInputChange}
                className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline mt-1"
              >
                <option value="conservative">Conservative</option>
                <option value="moderate">Moderate</option>
                <option value="aggressive">Aggressive</option>
              </select>
            </label>
          </div>

          <div>
            <label className="block text-gray-700 text-sm font-bold mb-2">
              Investment Period (years)
              <input
                type="number"
                name="investment_period"
                value={formData.investment_period}
                onChange={handleInputChange}
                className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline mt-1"
                min="1"
                max="30"
              />
            </label>
          </div>
        </div>

        <div className="mt-6">
          <button
            type="submit"
            disabled={loading}
            className="w-full bg-blue-500 hover:bg-blue-700 text-white font-bold py-3 px-4 rounded focus:outline-none focus:shadow-outline disabled:opacity-50"
          >
            {loading ? 'Generating Portfolio...' : 'Generate Portfolio'}
          </button>
        </div>
      </form>

      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative" role="alert">
          <strong className="font-bold">Error: </strong>
          <span className="block sm:inline">{error}</span>
        </div>
      )}

      {/* Results Display */}
      {portfolioData && (
        <div className="bg-white shadow-lg rounded-lg overflow-hidden">
          {/* Portfolio Summary */}
          <div className="p-6 border-b">
            <h2 className="text-2xl font-bold mb-4 text-gray-800">Portfolio Summary</h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-blue-50 p-4 rounded-lg">
                <h3 className="text-lg font-semibold text-blue-800">Total Investment</h3>
                <p className="text-2xl font-bold text-blue-600">
                  ${portfolioData.portfolio.recommendations.allocation_summary.total_investment.toLocaleString()}
                </p>
              </div>
              <div className="bg-green-50 p-4 rounded-lg">
                <h3 className="text-lg font-semibold text-green-800">Total Stocks</h3>
                <p className="text-2xl font-bold text-green-600">
                  {portfolioData.portfolio.recommendations.allocation_summary.total_stocks}
                </p>
              </div>
              <div className="bg-purple-50 p-4 rounded-lg">
                <h3 className="text-lg font-semibold text-purple-800">Total Sectors</h3>
                <p className="text-2xl font-bold text-purple-600">
                  {portfolioData.portfolio.recommendations.allocation_summary.total_sectors}
                </p>
              </div>
            </div>
          </div>

          {/* Sector Allocation Chart */}
          <div className="p-6 border-b">
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
          <div className="p-6">
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
            <div className="p-6 bg-gray-50">
              <h3 className="text-xl font-bold mb-4 text-gray-800">Portfolio Analysis</h3>
              <div className="prose max-w-none">
                <p className="whitespace-pre-line text-gray-700">{portfolioData.analysis}</p>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
} 