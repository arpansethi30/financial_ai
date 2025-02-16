'use client';

import { useState } from 'react';
import { toast } from 'react-hot-toast';

interface StockAnalysis {
  symbol: string;
  company_name: string;
  sector: string;
  industry: string;
  current_price: number;
  price_change: number;
  market_cap: number;
  pe_ratio: number | null;
  fifty_two_week: {
    high: number;
    low: number;
  };
  technical_indicators: {
    trend: string;
    rsi: number;
    macd: number;
    sma20: number;
    sma50: number;
    volatility: number;
    average_volume: number;
  };
  analysis: string;
}

export default function StockAnalysisPage() {
  const [symbol, setSymbol] = useState('');
  const [period, setPeriod] = useState('1y');
  const [loading, setLoading] = useState(false);
  const [analysis, setAnalysis] = useState<StockAnalysis | null>(null);
  const [error, setError] = useState('');

  const handleAnalyze = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!symbol) {
      setError('Please enter a stock symbol');
      return;
    }

    const cleanSymbol = symbol.trim().toUpperCase();
    if (!/^[A-Z]{1,5}$/.test(cleanSymbol)) {
      setError('Please enter a valid stock symbol (1-5 letters)');
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

      const response = await fetch('http://localhost:8001/analyze/stock', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        },
        body: JSON.stringify({
          symbol: cleanSymbol,
          period: period
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        if (data.detail?.includes('No data found for symbol')) {
          throw new Error(`No data found for symbol "${cleanSymbol}". Please verify the stock symbol is correct.`);
        }
        throw new Error(data.detail || 'Failed to analyze stock');
      }

      setAnalysis(data);
      toast.success('Analysis completed successfully');
    } catch (err) {
      console.error('Stock analysis error:', err);
      const errorMessage = err instanceof Error 
        ? err.message 
        : 'Failed to analyze stock. Please try again.';
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
      <h1 className="section-title">Stock Analysis</h1>
      <p className="section-description">
        Get detailed insights and AI-powered analysis for any publicly traded company
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
                maxLength={5}
                pattern="[A-Za-z]+"
                title="Please enter a valid stock symbol (1-5 letters)"
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
              <label htmlFor="period" className="block text-sm font-medium text-[#0A2540] mb-2">
                Time Period
              </label>
              <select
                id="period"
                value={period}
                onChange={(e) => setPeriod(e.target.value)}
                className="select-modern"
              >
                <option value="1mo">1 Month</option>
                <option value="3mo">3 Months</option>
                <option value="6mo">6 Months</option>
                <option value="1y">1 Year</option>
                <option value="2y">2 Years</option>
                <option value="5y">5 Years</option>
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
                  Analyzing...
                </div>
              ) : (
                'Analyze Stock'
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
            <div className="flex items-center justify-between mb-6">
              <h2 className="heading-2">
                {analysis.company_name} ({analysis.symbol})
              </h2>
              <span className={`badge ${analysis.price_change >= 0 ? 'badge-success' : 'badge-error'}`}>
                {analysis.price_change >= 0 ? '+' : ''}{analysis.price_change.toFixed(2)}%
              </span>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
              <div className="stat-card">
                <p className="stat-label">Current Price</p>
                <p className="stat-value">${analysis.current_price.toFixed(2)}</p>
              </div>
              <div className="stat-card">
                <p className="stat-label">Market Cap</p>
                <p className="stat-value">${(analysis.market_cap / 1e9).toFixed(2)}B</p>
              </div>
              <div className="stat-card">
                <p className="stat-label">Sector</p>
                <p className="stat-value text-xl">{analysis.sector}</p>
              </div>
              <div className="stat-card">
                <p className="stat-label">Industry</p>
                <p className="stat-value text-xl">{analysis.industry}</p>
              </div>
            </div>
          </div>

          <div className="card-modern p-8">
            <h3 className="heading-3 mb-6">Technical Indicators</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
              <div className="stat-card">
                <p className="stat-label">Trend</p>
                <p className={`stat-value ${analysis.technical_indicators.trend === 'Bullish' ? 'text-green-600' : 'text-red-600'}`}>
                  {analysis.technical_indicators.trend}
                </p>
              </div>
              <div className="stat-card">
                <p className="stat-label">RSI</p>
                <p className="stat-value">{analysis.technical_indicators.rsi.toFixed(2)}</p>
              </div>
              <div className="stat-card">
                <p className="stat-label">MACD</p>
                <p className="stat-value">{analysis.technical_indicators.macd.toFixed(2)}</p>
              </div>
              <div className="stat-card">
                <p className="stat-label">Volatility</p>
                <p className="stat-value">{(analysis.technical_indicators.volatility * 100).toFixed(2)}%</p>
              </div>
              <div className="stat-card">
                <p className="stat-label">20-day SMA</p>
                <p className="stat-value">${analysis.technical_indicators.sma20.toFixed(2)}</p>
              </div>
              <div className="stat-card">
                <p className="stat-label">50-day SMA</p>
                <p className="stat-value">${analysis.technical_indicators.sma50.toFixed(2)}</p>
              </div>
              <div className="stat-card">
                <p className="stat-label">Average Volume</p>
                <p className="stat-value">{analysis.technical_indicators.average_volume.toLocaleString()}</p>
              </div>
              <div className="stat-card">
                <p className="stat-label">P/E Ratio</p>
                <p className="stat-value">{analysis.pe_ratio?.toFixed(2) || 'N/A'}</p>
              </div>
            </div>
          </div>

          <div className="card-modern p-8">
            <h3 className="heading-3 mb-6">AI Analysis</h3>
            <div className="prose max-w-none">
              <p className="text-[#486284] whitespace-pre-line leading-relaxed">{analysis.analysis}</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
} 