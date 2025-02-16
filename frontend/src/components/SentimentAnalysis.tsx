'use client';

import React, { useState } from 'react';
import {
  Box,
  Button,
  Card,
  CardContent,
  CircularProgress,
  Container,
  TextField,
  Typography,
  Link,
  Divider,
} from '@mui/material';
import { TrendingUp, TrendingDown, TrendingFlat } from '@mui/icons-material';

interface NewsItem {
  title: string;
  source: string;
  url: string;
}

interface SentimentData {
  analysis: string;
  news: NewsItem[];
}

const SentimentAnalysis: React.FC = () => {
  const [ticker, setTicker] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [sentimentData, setSentimentData] = useState<SentimentData | null>(null);

  const getSentimentIcon = (analysis: string) => {
    if (analysis.toLowerCase().includes('bullish')) return <TrendingUp className="text-green-500 w-8 h-8" />;
    if (analysis.toLowerCase().includes('bearish')) return <TrendingDown className="text-red-500 w-8 h-8" />;
    return <TrendingFlat className="text-blue-500 w-8 h-8" />;
  };

  const analyzeSentiment = async () => {
    if (!ticker) {
      setError('Please enter a stock ticker');
      return;
    }

    setLoading(true);
    setError('');
    setSentimentData(null);

    try {
      const response = await fetch(`http://localhost:8001/api/sentiment-analysis?ticker=${ticker.toUpperCase()}`, {
        method: 'GET',
        headers: {
          'Accept': 'application/json',
        }
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to analyze sentiment');
      }

      const data = await response.json();
      if (data.status === 'success') {
        setSentimentData(data.data);
      } else {
        setError(data.detail || 'Failed to analyze sentiment');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch sentiment analysis');
      console.error('Sentiment analysis error:', err);
    } finally {
      setLoading(false);
    }
  };

  const formatAnalysis = (analysis: string) => {
    return analysis.split('\n').map((line, index) => (
      <p key={index} className="text-[#486284] mb-2">
        {line.trim()}
      </p>
    ));
  };

  return (
    <div className="section-container animate-fadeUp">
      <h1 className="section-title">Market Sentiment Analysis</h1>
      <p className="section-description">
        Analyze market sentiment and news coverage for any publicly traded company
      </p>

      <div className="card-modern p-8 mb-8">
        <div className="flex flex-col md:flex-row gap-4 mb-6">
          <div className="flex-grow">
            <input
              type="text"
              value={ticker}
              onChange={(e) => setTicker(e.target.value.toUpperCase())}
              placeholder="Enter stock ticker (e.g., AAPL)"
              className="input-modern w-full"
            />
          </div>
          <button
            onClick={analyzeSentiment}
            disabled={loading}
            className="btn-primary min-w-[120px]"
          >
            {loading ? (
              <div className="flex items-center justify-center">
                <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
              </div>
            ) : (
              'Analyze'
            )}
          </button>
        </div>

        {error && (
          <div className="bg-red-50 border-l-4 border-red-500 p-4 mb-6">
            <div className="flex">
              <div className="flex-shrink-0">
                <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                </svg>
              </div>
              <div className="ml-3">
                <p className="text-sm text-red-700">{error}</p>
              </div>
            </div>
          </div>
        )}

        {sentimentData && (
          <div className="space-y-6 animate-fadeUp">
            <div className="card-modern p-6">
              <div className="flex items-center gap-3 mb-4">
                {getSentimentIcon(sentimentData.analysis)}
                <h2 className="heading-2">Sentiment Analysis for {ticker}</h2>
              </div>
              <div className="prose max-w-none">
                {formatAnalysis(sentimentData.analysis)}
              </div>
            </div>

            <div className="card-modern p-6">
              <h3 className="heading-3 mb-4">Recent News Headlines</h3>
              <div className="space-y-4">
                {sentimentData.news.map((item, index) => (
                  <div key={index} className="p-4 bg-white/50 rounded-xl hover:bg-white/80 transition-colors duration-200">
                    <a
                      href={item.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-[#0A2540] hover:text-blue-600 font-medium block mb-2"
                    >
                      {item.title}
                    </a>
                    <p className="text-sm text-[#486284]">Source: {item.source}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default SentimentAnalysis; 