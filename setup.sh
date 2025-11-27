#!/bin/bash

echo "====================================="
echo "SmartVisionQA - Setup Script"
echo "====================================="

# Check Python version
echo "Checking Python version..."
python --version

# Create virtual environment
echo "Creating virtual environment..."
python -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate || . venv/Scripts/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating directories..."
mkdir -p uploads temp chroma_db

# Copy environment file
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cp .env.example .env
fi

# Setup frontend
echo "Setting up frontend..."
cd frontend
npm install
cd ..

echo "====================================="
echo "Setup complete!"
echo "====================================="
echo "To start the backend:"
echo "  source venv/bin/activate  # or venv\\Scripts\\activate on Windows"
echo "  uvicorn backend.main:app --reload"
echo ""
echo "To start the frontend:"
echo "  cd frontend"
echo "  npm run dev"
echo "====================================="
