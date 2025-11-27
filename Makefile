# Makefile for SmartVisionQA project

.PHONY: help install dev-backend dev-frontend test lint format clean docker-up docker-down

help:
	@echo "SmartVisionQA - Available commands:"
	@echo "  make install       - Install all dependencies"
	@echo "  make dev-backend   - Run backend development server"
	@echo "  make dev-frontend  - Run frontend development server"
	@echo "  make test          - Run all tests"
	@echo "  make lint          - Run linters"
	@echo "  make format        - Format code"
	@echo "  make clean         - Clean temporary files"
	@echo "  make docker-up     - Start Docker containers"
	@echo "  make docker-down   - Stop Docker containers"

install:
	@echo "Installing backend dependencies..."
	cd backend && pip install -r ../requirements.txt
	@echo "Installing frontend dependencies..."
	cd frontend && npm install

dev-backend:
	@echo "Starting backend server..."
	cd backend && python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000

dev-frontend:
	@echo "Starting frontend development server..."
	cd frontend && npm run dev

test:
	@echo "Running tests..."
	pytest backend/tests -v

lint:
	@echo "Running Python linters..."
	flake8 backend
	@echo "Running JavaScript linters..."
	cd frontend && npm run lint

format:
	@echo "Formatting Python code..."
	black backend
	isort backend
	@echo "Formatting JavaScript code..."
	cd frontend && npm run format

clean:
	@echo "Cleaning temporary files..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name "node_modules" -exec rm -rf {} +

docker-up:
	@echo "Starting Docker containers..."
	docker-compose up -d

docker-down:
	@echo "Stopping Docker containers..."
	docker-compose down
