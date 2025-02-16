import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Financial AI | Intelligent Investment Analysis",
  description: "Advanced AI-powered financial analysis and investment insights",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="h-full">
      <body className={`${inter.className} h-full antialiased bg-white`}>
        <nav className="fixed w-full z-50 bg-white/90 backdrop-blur-xl border-b border-gray-200">
          <div className="max-w-[1400px] mx-auto">
            <div className="flex h-16 items-center justify-between px-6 lg:px-8">
              <div className="flex items-center gap-10">
                <a href="/" className="text-lg font-semibold text-gray-900">
                  Financial AI
                </a>
                <div className="hidden md:flex items-center gap-6">
                  <a href="/stock-analysis" 
                     className="text-[15px] font-medium text-gray-600
                              hover:text-gray-900 transition-colors duration-200">
                    Analysis
                  </a>
                  <a href="/portfolio" 
                     className="text-[15px] font-medium text-gray-600
                              hover:text-gray-900 transition-colors duration-200">
                    Portfolio
                  </a>
                  <a href="/sentiment" 
                     className="text-[15px] font-medium text-gray-600
                              hover:text-gray-900 transition-colors duration-200">
                    Sentiment
                  </a>
                </div>
              </div>
              <div className="flex items-center gap-6">
                <a href="/sign-in" 
                   className="text-[15px] font-medium text-gray-600
                            hover:text-gray-900 transition-colors duration-200">
                  Sign In
                </a>
                <a href="/get-started" 
                   className="px-4 py-2 text-[15px] font-medium text-white rounded-full
                            bg-gray-900 hover:bg-gray-800 transition-all duration-200">
                  Get Started
                </a>
              </div>
            </div>
          </div>
        </nav>
        <main>
          {children}
        </main>
      </body>
    </html>
  );
}
