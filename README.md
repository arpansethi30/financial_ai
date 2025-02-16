# Financial Product AI

An intelligent financial product that helps users analyze stocks and create optimized investment portfolios based on their risk appetite and investment goals.

## Features

- **Stock Analysis**: Detailed analysis of individual stocks with technical indicators, fundamental data, and AI-generated insights
- **Portfolio Generation**: Create personalized investment portfolios based on:
  - Investment amount
  - Risk appetite (conservative, moderate, aggressive)
  - Investment period
- **Real-time Data**: Live stock data using yfinance
- **AI-Powered Analysis**: Intelligent insights using Google's Gemini AI
- **Interactive Charts**: Visual representation of portfolio allocation and performance

## Tech Stack

### Frontend
- Next.js 14
- React
- TypeScript
- Tailwind CSS
- Recharts for data visualization

### Backend
- FastAPI
- Python
- yfinance for stock data
- ta library for technical analysis
- Google Gemini AI for analysis
- NewsAPI for sentiment analysis

## Setup

1. Clone the repository:
```bash
git clone [repository-url]
cd financial_product
```

2. Set up the backend:
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
pip install -r requirements.txt
```

3. Create a `.env` file in the backend directory with your API keys:
```
GOOGLE_AI_API_KEY=your_gemini_api_key
NEWS_API_KEY=your_newsapi_key
```

4. Set up the frontend:
```bash
cd agent
npm install
```

5. Start the development servers:

Backend:
```bash
cd backend
uvicorn app.main:app --reload
```

Frontend:
```bash
cd agent
npm run dev
```

6. Visit http://localhost:3000 to access the application

## Usage

1. **Stock Analysis**
   - Enter a stock symbol
   - View technical indicators, charts, and AI analysis

2. **Portfolio Generation**
   - Enter investment amount
   - Select risk appetite
   - Specify investment period
   - Get personalized portfolio recommendations

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 