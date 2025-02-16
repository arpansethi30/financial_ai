'use client';

import React from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';

export default function Navigation() {
  const pathname = usePathname();
  const isActive = (path: string) => pathname === path;

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 bg-white border-b border-gray-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          <div className="flex items-center">
            <Link href="/" className="flex items-center">
              <span className="text-xl font-bold text-gray-900">Financial AI</span>
            </Link>
            <div className="hidden sm:ml-10 sm:flex sm:space-x-8">
              <Link
                href="/analysis"
                className={`${
                  isActive('/analysis')
                    ? 'text-gray-900'
                    : 'text-gray-500 hover:text-gray-900'
                } inline-flex items-center px-3 py-2 text-sm font-medium transition-colors duration-200`}
              >
                Analysis
              </Link>
              <Link
                href="/portfolio"
                className={`${
                  isActive('/portfolio')
                    ? 'text-gray-900'
                    : 'text-gray-500 hover:text-gray-900'
                } inline-flex items-center px-3 py-2 text-sm font-medium transition-colors duration-200`}
              >
                Portfolio
              </Link>
              <Link
                href="/sentiment"
                className={`${
                  isActive('/sentiment')
                    ? 'text-gray-900'
                    : 'text-gray-500 hover:text-gray-900'
                } inline-flex items-center px-3 py-2 text-sm font-medium transition-colors duration-200`}
              >
                Sentiment
              </Link>
            </div>
          </div>
          <div className="flex items-center space-x-4">
            <Link
              href="/sign-in"
              className="text-sm font-medium text-gray-500 hover:text-gray-900 transition-colors duration-200"
            >
              Sign In
            </Link>
            <Link
              href="/get-started"
              className="inline-flex items-center px-4 py-2 text-sm font-medium rounded-lg
                       bg-gray-900 text-white hover:bg-gray-800 transition-all duration-200"
            >
              Get Started
            </Link>
          </div>
        </div>
      </div>
    </nav>
  );
} 