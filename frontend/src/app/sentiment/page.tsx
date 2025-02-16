'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';

interface SentimentAnalysis {
  symbol: string;
  company_name: string;
  sentiment_analysis: string;
  news_count: number;
  analyzed_articles: number;
  period_days: number;
  sources: string[];
}

export default function SentimentPage() {
  const router = useRouter();
  const [symbol, setSymbol] = useState('');
  const [days, setDays] = useState('7');
  const [loading, setLoading] = useState(false);
  const [analysis, setAnalysis] = useState<SentimentAnalysis | null>(null);
  const [error, setError] = useState('');

  const handleAnalyze = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!symbol) {
      setError('Please enter a stock symbol');
      return;
    }

    setLoading(true);
    setError('');
    setAnalysis(null);

    try {
      const healthCheck = await fetch('http://localhost:8001/health').catch(() => null);
      if (!healthCheck) {
        throw new Error('Backend server is not running. Please start the server and try again.');
      }

      const response = await fetch('http://localhost:8001/analyze/sentiment', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        },
        body: JSON.stringify({
          symbol: symbol.toUpperCase(),
          days: parseInt(days)
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to analyze sentiment');
      }

      const data = await response.json();
      setAnalysis(data);
    } catch (err) {
      console.error('Sentiment analysis error:', err);
      const errorMessage = err instanceof Error 
        ? err.message 
        : 'Failed to analyze sentiment. Please try again.';
      setError(errorMessage);

      if (errorMessage.includes('Failed to fetch')) {
        setError('Unable to connect to the backend server. Please ensure it is running on port 8001.');
      }
    } finally {
      setLoading(false);
    }
  };

  const handleRetry = () => {
    setError('');
    setAnalysis(null);
  };

  const commonStocks = [
    { symbol: 'AAPL', name: 'Apple' },
    { symbol: 'MSFT', name: 'Microsoft' },
    { symbol: 'GOOGL', name: 'Google' },
    { symbol: 'AMZN', name: 'Amazon' },
    { symbol: 'META', name: 'Meta' },
  ];

  return (
    <div className="section-container animate-fadeUp">
      <h1 className="section-title">Market Sentiment Analysis</h1>
      <p className="section-description">
        Analyze market sentiment and news coverage for any publicly traded company
      </p>

      <div className="card-modern p-8 mb-8">
        <form onSubmit={handleAnalyze} className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="md:col-span-2">
              <label htmlFor="symbol" className="block text-sm font-medium text-[#0A2540] mb-2">
                Stock Symbol
              </label>
              <input
                type="text"
                id="symbol"
                value={symbol}
                onChange={(e) => setSymbol(e.target.value.toUpperCase())}
                className="input-modern"
                placeholder="Enter stock symbol (e.g., AAPL)"
                required
                pattern="[A-Za-z]+"
                title="Please enter a valid stock symbol (letters only)"
              />
              <div className="mt-3">
                <p className="text-sm font-medium text-[#486284] mb-2">Popular stocks:</p>
                <div className="flex flex-wrap gap-2">
                  {commonStocks.map((stock) => (
                    <button
                      key={stock.symbol}
                      type="button"
                      onClick={() => setSymbol(stock.symbol)}
                      className="badge-primary hover:bg-blue-200 transition-colors duration-200"
                    >
                      {stock.symbol}
                    </button>
                  ))}
                </div>
              </div>
            </div>
            <div>
              <label htmlFor="days" className="block text-sm font-medium text-[#0A2540] mb-2">
                Analysis Period
              </label>
              <select
                id="days"
                value={days}
                onChange={(e) => setDays(e.target.value)}
                className="select-modern"
              >
                <option value="1">Last 24 Hours</option>
                <option value="3">Last 3 Days</option>
                <option value="7">Last Week</option>
                <option value="14">Last 2 Weeks</option>
                <option value="30">Last Month</option>
              </select>
            </div>
          </div>

          <div className="flex gap-4">
            <button
              type="submit"
              disabled={loading}
              className="btn-primary"
            >
              {loading ? (
                <div className="flex items-center">
                  <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Analyzing Sentiment...
                </div>
              ) : (
                'Analyze Sentiment'
              )}
            </button>
            {error && (
              <button
                type="button"
                onClick={handleRetry}
                className="btn-secondary"
              >
                Try Again
              </button>
            )}
          </div>
        </form>
      </div>

      {error && (
        <div className="card-modern p-6 border-l-4 border-red-500 mb-8">
          <div className="flex">
            <div className="flex-shrink-0">
              <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
              </svg>
            </div>
            <div className="ml-3">
              <p className="text-sm text-red-700">{error}</p>
              {error.includes('backend server') && (
                <div className="mt-2 p-3 bg-red-50 rounded-lg">
                  <p className="text-sm text-red-600 mb-2">
                    Please start the backend server using:
                  </p>
                  <code className="block p-2 bg-red-100 rounded text-red-800 text-xs font-mono">
                    cd backend && uvicorn app.main:app --reload --port 8001
                  </code>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {analysis && (
        <div className="space-y-8 animate-fadeUp">
          <div className="card-modern p-8">
            <h2 className="heading-2 mb-6">
              Market Sentiment: {analysis.company_name} ({analysis.symbol})
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="stat-card">
                <p className="stat-label">Analysis Period</p>
                <p className="stat-value">Last {analysis.period_days} days</p>
              </div>
              <div className="stat-card">
                <p className="stat-label">Articles Found</p>
                <p className="stat-value">{analysis.news_count}</p>
                <p className="text-sm text-[#486284]">news articles</p>
              </div>
              <div className="stat-card">
                <p className="stat-label">Articles Analyzed</p>
                <p className="stat-value">{analysis.analyzed_articles}</p>
                <p className="text-sm text-[#486284]">in-depth analysis</p>
              </div>
            </div>
          </div>

          {analysis.sources.length > 0 && (
            <div className="card-modern p-8">
              <h3 className="heading-3 mb-6">News Sources</h3>
              <div className="flex flex-wrap gap-3">
                {analysis.sources.map((source, index) => (
                  <span
                    key={index}
                    className="badge-primary px-4 py-2 text-base"
                  >
                    {source}
                  </span>
                ))}
              </div>
            </div>
          )}

          <div className="card-modern p-8">
            <h3 className="heading-3 mb-6">Sentiment Analysis</h3>
            <div className="prose max-w-none">
              <p className="text-[#486284] whitespace-pre-line leading-relaxed">
                {analysis.sentiment_analysis}
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
} 