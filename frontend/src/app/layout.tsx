import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import Link from "next/link";
import Footer from "@/components/Footer";
import { BarChart, TrendingUp } from '@mui/icons-material';

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Financial AI",
  description: "AI-powered financial analysis and portfolio management",
};

const navigation = [
  { name: 'Dashboard', href: '/', icon: BarChart },
  { name: 'Portfolio Analysis', href: '/portfolio', icon: BarChart },
  { name: 'Sentiment Analysis', href: '/sentiment', icon: TrendingUp },
  // ... other navigation items ...
];

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${inter.className} min-h-screen flex flex-col`}>
        {/* Navigation */}
        <nav className="fixed top-0 left-0 right-0 z-50 bg-white/80 backdrop-blur-md border-b border-gray-200">
          <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
            <div className="flex h-16 justify-between items-center">
              <div className="flex items-center">
                <Link href="/" className="text-xl font-bold text-gray-900">
                  Financial AI
                </Link>
              </div>
              
              <div className="hidden md:flex md:items-center md:space-x-8">
                <Link href="/analysis" className="text-gray-600 hover:text-gray-900">
                  Analysis
                </Link>
                <Link href="/portfolio" className="text-gray-600 hover:text-gray-900">
                  Portfolio
                </Link>
                <Link href="/sentiment" className="text-gray-600 hover:text-gray-900">
                  Sentiment
                </Link>
              </div>

              <div className="flex items-center space-x-4">
                <Link
                  href="/sign-in"
                  className="text-gray-600 hover:text-gray-900"
                >
                  Sign In
                </Link>
                <Link
                  href="/get-started"
                  className="inline-flex items-center justify-center px-4 py-2 border border-transparent text-sm font-medium rounded-xl text-white bg-gray-900 hover:bg-gray-800"
                >
                  Get Started
                </Link>
              </div>
            </div>
          </div>
        </nav>

        {/* Main Content */}
        <div className="pt-16 flex-grow">
          {children}
        </div>

        {/* Footer */}
        <Footer />
      </body>
    </html>
  );
}
