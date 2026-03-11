#!/usr/bin/env bash
set -euo pipefail

# CreditScope Development Server Launcher
# Starts all services for local development

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Load environment
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

# Default ports
BACKEND_PORT=${BACKEND_PORT:-8080}
FRONTEND_PORT=${FRONTEND_PORT:-3000}
SGLANG_PORT=${SGLANG_PORT:-8000}

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

cleanup() {
    echo -e "\n${YELLOW}Shutting down services...${NC}"
    kill $(jobs -p) 2>/dev/null || true
    exit 0
}

trap cleanup SIGINT SIGTERM

echo -e "${GREEN}🚀 Starting CreditScope Development Servers${NC}"
echo ""

# Activate virtual environment
if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
fi

# Start backend
echo -e "${GREEN}Starting Backend (port $BACKEND_PORT)...${NC}"
cd "$PROJECT_ROOT"
PYTHONPATH="$PROJECT_ROOT" uvicorn backend.main:app \
    --host 0.0.0.0 \
    --port "$BACKEND_PORT" \
    --reload \
    --reload-dir backend &

sleep 2

# Start frontend
echo -e "${GREEN}Starting Frontend (port $FRONTEND_PORT)...${NC}"
cd "$PROJECT_ROOT/frontend"
npm run dev -- --port "$FRONTEND_PORT" &

echo ""
echo -e "${GREEN}✅ Development servers started!${NC}"
echo ""
echo "Services:"
echo -e "  Backend:  ${YELLOW}http://localhost:$BACKEND_PORT${NC}"
echo -e "  Frontend: ${YELLOW}http://localhost:$FRONTEND_PORT${NC}"
echo -e "  API Docs: ${YELLOW}http://localhost:$BACKEND_PORT/docs${NC}"
echo ""
echo -e "Press ${RED}Ctrl+C${NC} to stop all services"
echo ""

# Wait for all background processes
wait
