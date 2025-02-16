import React from 'react';
import Link from 'next/link';
import { TrendingUp, LineChart, BarChart2, PieChart, ArrowRight, Sparkles, Shield, Globe } from 'lucide-react';

export default function Home() {
  return (
    <main className="min-h-screen bg-white">
      {/* Hero Section */}
      <section className="relative min-h-screen flex items-center justify-center overflow-hidden">
        {/* Background Effects */}
        <div className="absolute inset-0">
          <div className="absolute inset-0 bg-gradient-to-b from-gray-50 to-white"></div>
          <div className="absolute inset-0 bg-[url('/grid-pattern.svg')] opacity-[0.03]"></div>
        </div>

        <div className="relative mx-auto max-w-7xl px-6 py-24 lg:px-8 text-center">
          {/* Badge */}
          <div className="inline-flex items-center px-4 py-2 rounded-full bg-gray-100
                        border border-gray-200 mb-8 animate-fade-up">
            <span className="text-[15px] font-medium text-gray-800">
              Intelligent Investing Platform
            </span>
          </div>

          {/* Hero Content */}
          <h1 className="text-5xl sm:text-6xl lg:text-7xl font-bold text-gray-900 tracking-tight mb-8 animate-fade-up">
            Financial Intelligence
            <span className="block mt-1 text-gray-800">Reimagined</span>
          </h1>
          
          <p className="mt-8 text-lg sm:text-xl text-gray-600 max-w-3xl mx-auto leading-relaxed animate-fade-up">
            Make smarter investment decisions with our advanced AI analysis platform.
            Get real-time insights, portfolio recommendations, and market analysis.
          </p>

          <div className="mt-10 flex flex-col sm:flex-row items-center justify-center gap-6 animate-fade-up">
            <a href="/get-started" 
               className="w-full sm:w-auto px-8 py-4 text-base font-medium rounded-xl 
                        bg-gray-900 text-white
                        hover:bg-gray-800 transition-all duration-200
                        shadow-lg">
              Get Started Free
            </a>
            <a href="/demo" 
               className="w-full sm:w-auto px-8 py-4 text-base font-medium rounded-xl 
                        bg-white text-gray-900 border border-gray-200
                        hover:bg-gray-50 transition-all duration-200">
              Watch Demo
            </a>
          </div>

          {/* Feature Cards */}
          <div className="mt-32 grid grid-cols-1 md:grid-cols-3 gap-8">
            {/* Market Analysis Card */}
            <div className="group">
              <div className="p-8 rounded-2xl bg-white border border-gray-200 transition duration-300
                            hover:shadow-xl hover:border-gray-300">
                <div className="h-14 w-14 rounded-xl bg-gray-100
                              flex items-center justify-center mb-6">
                  <Globe className="w-7 h-7 text-gray-700" />
                </div>
                <h3 className="text-xl font-semibold text-gray-900 mb-3">Global Market Analysis</h3>
                <p className="text-gray-600 leading-relaxed">
                  Access real-time market data and AI-powered insights from markets worldwide.
                </p>
              </div>
            </div>

            {/* Risk Management Card */}
            <div className="group">
              <div className="p-8 rounded-2xl bg-white border border-gray-200 transition duration-300
                            hover:shadow-xl hover:border-gray-300">
                <div className="h-14 w-14 rounded-xl bg-gray-100
                              flex items-center justify-center mb-6">
                  <Shield className="w-7 h-7 text-gray-700" />
                </div>
                <h3 className="text-xl font-semibold text-gray-900 mb-3">Risk Management</h3>
                <p className="text-gray-600 leading-relaxed">
                  Advanced risk assessment and portfolio protection strategies.
                </p>
              </div>
            </div>

            {/* AI Predictions Card */}
            <div className="group">
              <div className="p-8 rounded-2xl bg-white border border-gray-200 transition duration-300
                            hover:shadow-xl hover:border-gray-300">
                <div className="h-14 w-14 rounded-xl bg-gray-100
                              flex items-center justify-center mb-6">
                  <Sparkles className="w-7 h-7 text-gray-700" />
                </div>
                <h3 className="text-xl font-semibold text-gray-900 mb-3">AI Predictions</h3>
                <p className="text-gray-600 leading-relaxed">
                  Machine learning algorithms predict market trends and opportunities.
                </p>
              </div>
            </div>
          </div>

          {/* Stats Section */}
          <div className="mt-32 py-16 border-y border-gray-200">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
              <div className="text-center">
                <div className="text-4xl font-bold text-gray-900">500K+</div>
                <div className="mt-2 text-gray-600">Daily Analyses</div>
              </div>
              <div className="text-center">
                <div className="text-4xl font-bold text-gray-900">99.9%</div>
                <div className="mt-2 text-gray-600">Uptime</div>
              </div>
              <div className="text-center">
                <div className="text-4xl font-bold text-gray-900">24/7</div>
                <div className="mt-2 text-gray-600">Support</div>
              </div>
              <div className="text-center">
                <div className="text-4xl font-bold text-gray-900">150+</div>
                <div className="mt-2 text-gray-600">Global Markets</div>
              </div>
            </div>
          </div>

          {/* CTA Section */}
          <div className="mt-32 text-center">
            <h2 className="text-3xl sm:text-4xl font-bold text-gray-900 mb-6">
              Ready to transform your investment strategy?
            </h2>
            <p className="text-gray-600 mb-10 max-w-2xl mx-auto">
              Join thousands of investors using our platform to make smarter, data-driven decisions.
            </p>
            <a href="/get-started" 
               className="inline-flex items-center px-8 py-4 text-base font-medium rounded-xl 
                        bg-gray-900 text-white
                        hover:bg-gray-800 transition-all duration-200
                        shadow-lg">
              Get Started Free
              <ArrowRight className="ml-2 w-5 h-5" />
            </a>
          </div>
        </div>
      </section>
    </main>
  );
}