# SmartDiet ML API - Makefile
# Quick commands for common operations

.PHONY: help build up down logs clean rebuild test shell local

# Default target
help:
	@echo "SmartDiet ML API - Available Commands"
	@echo "======================================"
	@echo "make build          - Build Docker images with cache"
	@echo "make up             - Start all services"
	@echo "make down           - Stop all services"
	@echo "make logs           - Show logs (Ctrl+C to exit)"
	@echo "make logs-api       - Show API logs only"
	@echo "make rebuild        - Force rebuild without cache"
	@echo "make clean          - Remove containers and volumes"
	@echo "make clean-cache    - Clear Docker build cache"
	@echo "make shell-api      - Open shell in API container"
	@echo "make test           - Run tests (if available)"
	@echo "make local          - Run API locally (requires uv)"
	@echo "make local-config   - Switch to local config"
	@echo ""

# Build with cache (FAST after first build)
build:
	@echo "🔨 Building Docker images with BuildKit cache..."
	DOCKER_BUILDKIT=1 docker-compose build

# Start services
up:
	@echo "🚀 Starting services..."
	docker-compose up -d
	@echo "✅ Services started!"
	@echo "   API: http://localhost:8000/docs"
	@echo "   Web: http://localhost:7860"

# Stop services
down:
	@echo "🛑 Stopping services..."
	docker-compose down

# View logs
logs:
	docker-compose logs -f

logs-api:
	docker-compose logs -f api

# Force rebuild without cache (use sparingly)
rebuild:
	@echo "⚠️  Force rebuilding without cache (this will be slow)..."
	DOCKER_BUILDKIT=1 docker-compose build --no-cache

# Clean up containers and volumes
clean:
	@echo "🧹 Cleaning up containers and volumes..."
	docker-compose down -v
	@echo "✅ Cleaned up!"

# Clear Docker build cache
clean-cache:
	@echo "⚠️  Clearing Docker build cache..."
	docker buildx prune -f
	@echo "✅ Cache cleared!"

# Development: shell access
shell-api:
	docker-compose exec api /bin/bash

# Run tests (customize as needed)
test:
	@echo "🧪 Running tests..."
	docker-compose exec api pytest tests/ -v

# Check cache usage
cache-info:
	@echo "📊 Docker build cache usage:"
	docker buildx du

# Quick restart
restart: down up

# Build and start in one command
start: build up

# Development mode with code mounting (add to docker-compose.override.yml)
dev:
	@echo "🔧 Starting in development mode..."
	docker-compose -f docker-compose.yaml -f docker-compose.dev.yaml up

# Run API locally without Docker (requires uv)
local:
	@echo "🚀 Starting API locally..."
	cd app && uv run uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# Switch to local config for development
local-config:
	@echo "📝 Switching to local config..."
	cp app/configs/configs.local.yaml app/configs/configs.yaml
	@echo "✅ Config switched! Run 'make local' to start the API."

# Switch back to Docker config
docker-config:
	@echo "📝 Switching to Docker config..."
	@echo "Please restore app/configs/configs.yaml with Docker paths"
