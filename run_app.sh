#!/bin/bash

# 🚀 TRUTH LENSE - One-Click Reliable Startup
# This script fixes "[Errno 48] Address already in use" and connectivity issues.

echo "🧹 Cleaning up existing processes..."
# Kill any processes on port 8000 and 8080
lsof -ti :8000 | xargs kill -9 2>/dev/null
lsof -ti :8080 | xargs kill -9 2>/dev/null

echo "📡 Starting Backend (FastAPI) on port 8000..."
# Start backend in background, binding to 0.0.0.0 for universal access
source venv/bin/activate
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload > backend.log 2>&1 &
BACKEND_PID=$!

echo "🎨 Starting Frontend (HTTP Server) on port 8080..."
# Start frontend in background
cd frontend
python3 -m http.server 8080 > ../frontend.log 2>&1 &
FRONTEND_PID=$!
cd ..

echo "------------------------------------------------"
echo "✅ SUCCESS: Your app is now running!"
echo "🔗 Local Site:    http://localhost:8080"
echo "🔗 API Docs:      http://localhost:8000/docs"
echo "------------------------------------------------"
echo "Press Ctrl+C to stop both servers safely."

# Trap Ctrl+C to kill both background processes
trap "kill $BACKEND_PID $FRONTEND_PID; echo '🛑 Servers Stopped.'; exit" INT
wait
