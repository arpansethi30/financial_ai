'use client';

import ComprehensivePortfolio from '@/components/ComprehensivePortfolio';

export default function PortfolioPage() {
  return (
    <main className="min-h-screen bg-white">
      <section className="relative min-h-screen overflow-hidden">
        {/* Background Effects */}
        <div className="absolute inset-0">
          <div className="absolute inset-0 bg-gradient-to-b from-gray-50 to-white"></div>
          <div className="absolute inset-0 bg-[url('/grid-pattern.svg')] opacity-[0.03]"></div>
        </div>

        {/* Content */}
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

          {/* Portfolio Component */}
          <div className="mt-12">
            <ComprehensivePortfolio />
          </div>
        </div>
      </section>
    </main>
  );
} 