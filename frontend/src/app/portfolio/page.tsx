'use client';

import { useState } from 'react';
import { LineChart, PieChart, ArrowRight } from 'lucide-react';

interface PortfolioRecommendation {
  portfolio: {
    recommendations: {
      allocation_summary: {
        total_investment: number;
        total_stocks: number;
        total_sectors: number;
      };
      stock_recommendations: {
        [sector: string]: Array<{
          symbol: string;
          weight: number;
          amount: number;
          suggested_shares: number;
          risk_level: string;
        }>;
      };
    };
  };
  analysis: string;
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
    <main className="min-h-screen bg-white">
      <section className="relative min-h-screen overflow-hidden">
        {/* Background Effects */}
        <div className="absolute inset-0">
          <div className="absolute inset-0 bg-gradient-to-b from-gray-50 to-white"></div>
          <div className="absolute inset-0 bg-[url('/grid-pattern.svg')] opacity-[0.03]"></div>
        </div>

        <div className="relative mx-auto max-w-7xl px-6 py-24 lg:px-8">
          {/* Badge */}
          <div className="flex justify-center">
            <div className="inline-flex items-center px-4 py-2 rounded-full bg-gray-100
                          border border-gray-200 mb-8 animate-fade-up">
              <span className="text-[15px] font-medium text-gray-800">
                Portfolio Generator
              </span>
            </div>
          </div>

          {/* Page Title */}
          <div className="text-center">
            <h1 className="text-5xl sm:text-6xl font-bold text-gray-900 tracking-tight mb-8 animate-fade-up">
              Smart Portfolio
              <span className="block mt-1 text-gray-800">Recommendations</span>
            </h1>
            
            <p className="mt-8 text-lg sm:text-xl text-gray-600 mx-auto max-w-3xl leading-relaxed animate-fade-up">
              Get personalized portfolio recommendations based on your investment preferences and risk appetite.
            </p>
          </div>

          {/* Form Section */}
          <div className="mt-12 max-w-3xl mx-auto">
            <form onSubmit={handleSubmit} className="space-y-8 bg-white p-8 rounded-2xl border border-gray-200 shadow-lg">
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
                type="submit"
                disabled={loading}
                className="w-full px-8 py-4 text-base font-medium rounded-xl 
                          bg-gray-900 text-white
                          hover:bg-gray-800 transition-all duration-200
                          shadow-lg disabled:bg-gray-300 disabled:cursor-not-allowed
                          flex items-center justify-center"
              >
                {loading ? 'Generating Portfolio...' : 'Get Portfolio Recommendation'}
                {!loading && <ArrowRight className="ml-2 w-5 h-5" />}
              </button>
            </form>

            {error && (
              <div className="mt-6 p-4 rounded-xl bg-red-50 border border-red-200">
                <p className="text-red-600">{error}</p>
              </div>
            )}

            {recommendation && (
              <div className="mt-12 bg-white p-8 rounded-2xl border border-gray-200 shadow-lg animate-fade-up">
                <div className="flex items-center justify-between mb-6">
                  <h2 className="text-2xl font-bold text-gray-900">Your Portfolio Recommendation</h2>
                  <div className="text-right">
                    <p className="text-sm text-gray-600">Total Investment</p>
                    <p className="text-2xl font-bold text-gray-900">
                      ${recommendation.portfolio.recommendations.allocation_summary.total_investment.toLocaleString()}
                    </p>
                  </div>
                </div>
                
                <div className="overflow-x-auto">
                  <table className="min-w-full divide-y divide-gray-200">
                    <thead>
                      <tr>
                        <th className="px-6 py-3 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">Symbol</th>
                        <th className="px-6 py-3 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">Weight</th>
                        <th className="px-6 py-3 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">Amount</th>
                        <th className="px-6 py-3 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">Shares</th>
                        <th className="px-6 py-3 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">Risk Level</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-200">
                      {Object.entries(recommendation.portfolio.recommendations.stock_recommendations).map(([sector, stocks]) => (
                        <>
                          <tr className="bg-gray-50">
                            <td colSpan={5} className="px-6 py-3 text-sm font-semibold text-gray-700">{sector}</td>
                          </tr>
                          {stocks.map((stock, index) => (
                            <tr key={`${sector}-${index}`} className="hover:bg-gray-50 transition-colors">
                              <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">{stock.symbol}</td>
                              <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-600">{(stock.weight * 100).toFixed(2)}%</td>
                              <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-600">${stock.amount.toLocaleString()}</td>
                              <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-600">{stock.suggested_shares}</td>
                              <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-600">{stock.risk_level}</td>
                            </tr>
                          ))}
                        </>
                      ))}
                    </tbody>
                  </table>
                </div>

                {recommendation.analysis && (
                  <div className="mt-8 p-6 bg-gray-50 rounded-xl">
                    <h3 className="text-lg font-semibold text-gray-900 mb-4">Analysis</h3>
                    <p className="text-gray-700 whitespace-pre-line">{recommendation.analysis}</p>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </section>
    </main>
  );
} 