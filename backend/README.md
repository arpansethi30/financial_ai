# Financial Product Backend

This is the backend service for the Financial Product application, built with FastAPI.

## Setup

The backend uses Python 3.11 with Conda environment. Follow these steps to set up:

1. Ensure you have Conda installed
2. Create and activate the Conda environment:
```bash
conda create -n finagent python=3.11
conda activate finagent
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Backend

To run the backend server:

```bash
cd backend
uvicorn app.main:app --reload
```

The server will start at `http://localhost:8000`

## API Documentation

Once the server is running, you can access:
- API documentation at `http://localhost:8000/docs`
- Alternative documentation at `http://localhost:8000/redoc`

## Available Endpoints

- `GET /`: Welcome message
- `GET /health`: Health check endpoint 