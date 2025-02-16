'use client';

import { useState } from 'react';

interface PortfolioRecommendation {
  success: boolean;
  message: string;
  data: {
    allocation: Array<{
      symbol: string;
      weight: number;
      amount: number;
    }>;
    total_amount: number;
  };
}

export default function PortfolioPage() {
  const [formData, setFormData] = useState({
    investment_amount: '',
    risk_appetite: 'moderate',
    investment_period: '',
    company_count: '10',
  });
  const [loading, setLoading] = useState(false);
  const [recommendation, setRecommendation] = useState<PortfolioRecommendation | null>(null);
  const [error, setError] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setRecommendation(null);

    try {
      const response = await fetch('http://localhost:8000/portfolio/recommend', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          investment_amount: parseFloat(formData.investment_amount),
          risk_appetite: formData.risk_appetite,
          investment_period: parseInt(formData.investment_period),
          company_count: parseInt(formData.company_count),
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to get portfolio recommendation');
      }

      const data = await response.json();
      setRecommendation(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to get portfolio recommendation. Please try again.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  return (
    <div className="max-w-4xl mx-auto py-8">
      <h1 className="text-3xl font-bold mb-8">Portfolio Recommendation</h1>

      <form onSubmit={handleSubmit} className="mb-8">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
          <div>
            <label htmlFor="investment_amount" className="block text-sm font-medium text-gray-700 mb-1">
              Investment Amount ($)
            </label>
            <input
              type="number"
              id="investment_amount"
              name="investment_amount"
              value={formData.investment_amount}
              onChange={handleInputChange}
              className="w-full p-2 border rounded"
              placeholder="Enter investment amount"
              required
              min="1000"
              step="100"
            />
          </div>

          <div>
            <label htmlFor="risk_appetite" className="block text-sm font-medium text-gray-700 mb-1">
              Risk Appetite
            </label>
            <select
              id="risk_appetite"
              name="risk_appetite"
              value={formData.risk_appetite}
              onChange={handleInputChange}
              className="w-full p-2 border rounded"
              required
            >
              <option value="conservative">Conservative</option>
              <option value="moderate">Moderate</option>
              <option value="aggressive">Aggressive</option>
            </select>
          </div>

          <div>
            <label htmlFor="investment_period" className="block text-sm font-medium text-gray-700 mb-1">
              Investment Period (Years)
            </label>
            <input
              type="number"
              id="investment_period"
              name="investment_period"
              value={formData.investment_period}
              onChange={handleInputChange}
              className="w-full p-2 border rounded"
              placeholder="Enter investment period"
              required
              min="1"
              max="30"
            />
          </div>

          <div>
            <label htmlFor="company_count" className="block text-sm font-medium text-gray-700 mb-1">
              Number of Companies
            </label>
            <input
              type="number"
              id="company_count"
              name="company_count"
              value={formData.company_count}
              onChange={handleInputChange}
              className="w-full p-2 border rounded"
              placeholder="Enter number of companies"
              required
              min="5"
              max="30"
            />
          </div>
        </div>

        <button
          type="submit"
          disabled={loading}
          className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 disabled:bg-blue-300"
        >
          {loading ? 'Generating Portfolio...' : 'Get Portfolio Recommendation'}
        </button>
      </form>

      {error && (
        <div className="bg-red-50 text-red-600 p-4 rounded mb-4">
          {error}
        </div>
      )}

      {recommendation && (
        <div className="bg-white p-6 rounded-lg shadow-md">
          <h2 className="text-2xl font-semibold mb-4">Your Portfolio Recommendation</h2>
          
          <div className="mb-6">
            <h3 className="text-lg font-medium mb-2">Portfolio Allocation</h3>
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Symbol</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Weight (%)</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Amount ($)</th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {recommendation.data.allocation.map((item, index) => (
                    <tr key={index}>
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">{item.symbol}</td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{(item.weight * 100).toFixed(2)}%</td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">${item.amount.toLocaleString()}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div className="text-right">
            <p className="text-lg font-semibold">
              Total Investment: ${recommendation.data.total_amount.toLocaleString()}
            </p>
          </div>
        </div>
      )}
    </div>
  );
} 