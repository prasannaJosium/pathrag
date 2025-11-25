#!/bin/bash
# Start script for PathRAG API only

# Colors for terminal output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting PathRAG API...${NC}"

# Install backend dependencies
echo -e "${BLUE}Installing backend dependencies...${NC}"
poetry install
echo -e "${GREEN}Backend dependencies installed.${NC}"

# Start backend API
echo -e "${BLUE}Starting backend API on port 8000...${NC}"
cd "$(dirname "$0")"
poetry run uvicorn main:app --host 0.0.0.0 --port 8000

echo -e "${GREEN}API server stopped.${NC}"
