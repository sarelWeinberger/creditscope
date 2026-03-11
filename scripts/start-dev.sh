#!/usr/bin/env bash
set -euo pipefail

# CreditScope Development Server Launcher
# Starts backend + frontend for local development

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Load environment
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

# Default ports
INFERENCE_PORT=${SGLANG_PORT:-8000}
BACKEND_PORT=${BACKEND_PORT:-8080}
FRONTEND_PORT=${FRONTEND_PORT:-3000}
START_INFERENCE=${START_INFERENCE:-true}

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

usage() {
    echo "Usage: $(basename "$0") [--inference | --no-inference]"
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --inference)
            START_INFERENCE=true
            ;;
        --no-inference)
            START_INFERENCE=false
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            usage
            exit 1
            ;;
    esac
    shift
done

PIDS=()
SHUTTING_DOWN=0

cleanup() {
    if [ "$SHUTTING_DOWN" -eq 1 ]; then
        return 0
    fi
    SHUTTING_DOWN=1
    trap - SIGINT SIGTERM EXIT

    echo -e "\n${YELLOW}Shutting down services...${NC}"
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
    echo -e "${GREEN}All services stopped.${NC}"
}

trap cleanup SIGINT SIGTERM EXIT

# Kill anything already on our ports
PORTS_TO_CLEAN=("$BACKEND_PORT" "$FRONTEND_PORT")
if [ "$START_INFERENCE" = "true" ]; then
    PORTS_TO_CLEAN+=("$INFERENCE_PORT")
fi

for port in "${PORTS_TO_CLEAN[@]}"; do
    pid=$(lsof -ti ":$port" 2>/dev/null || true)
    if [ -n "$pid" ]; then
        echo -e "${YELLOW}Killing existing process on port $port (PID $pid)${NC}"
        kill "$pid" 2>/dev/null || true
        sleep 1
    fi
done

echo -e "${GREEN}Starting CreditScope Development Servers${NC}"
echo ""

# Activate virtual environment
VENV_ACTIVATE="$PROJECT_ROOT/.venv/bin/activate"
if [ -f "$VENV_ACTIVATE" ]; then
    source "$VENV_ACTIVATE"
    echo -e "  ${GREEN}Virtual environment activated${NC}"
else
    echo -e "  ${YELLOW}Warning: No .venv found, using system Python${NC}"
fi

# Ensure PYTHONPATH includes the project root so backend/inference modules resolve
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:$PYTHONPATH}"

# ── Start Inference ───────────────────────────────────────────────────────────
if [ "$START_INFERENCE" = "true" ]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        echo -e "${GREEN}Starting Inference on port $INFERENCE_PORT...${NC}"
        cd "$PROJECT_ROOT"
        python -m inference.server &
        INFERENCE_PID=$!
        PIDS+=($INFERENCE_PID)

        echo -n "  Waiting for inference..."
        for i in $(seq 1 180); do
            if curl -sf "http://localhost:$INFERENCE_PORT/health" > /dev/null 2>&1 || \
               curl -sf "http://localhost:$INFERENCE_PORT/model_info" > /dev/null 2>&1; then
                echo -e " ${GREEN}ready${NC}"
                break
            fi
            if ! kill -0 "$INFERENCE_PID" 2>/dev/null; then
                echo -e " ${RED}failed${NC}"
                cleanup
                exit 1
            fi
            if [ "$i" -eq 180 ]; then
                echo -e " ${RED}timeout${NC}"
                cleanup
                exit 1
            fi
            sleep 2
        done
    else
        echo -e "  ${YELLOW}Skipping inference startup because no NVIDIA GPU was detected${NC}"
    fi
fi

# ── Start Backend ──────────────────────────────────────────────────────────────
echo -e "${GREEN}Starting Backend on port $BACKEND_PORT...${NC}"
cd "$PROJECT_ROOT"
uvicorn backend.main:app \
    --host 0.0.0.0 \
    --port "$BACKEND_PORT" \
    --reload \
    --reload-dir backend &
BACKEND_PID=$!
PIDS+=($BACKEND_PID)

# Wait for backend to be ready
echo -n "  Waiting for backend..."
for i in $(seq 1 15); do
    if curl -sf "http://localhost:$BACKEND_PORT/health" > /dev/null 2>&1; then
        echo -e " ${GREEN}ready${NC}"
        break
    fi
    if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
        echo -e " ${RED}failed${NC}"
        cleanup
        exit 1
    fi
    if [ "$i" -eq 15 ]; then
        echo -e " ${RED}timeout${NC}"
        cleanup
        exit 1
    fi
    sleep 1
done

# ── Start Frontend ─────────────────────────────────────────────────────────────
echo -e "${GREEN}Starting Frontend on port $FRONTEND_PORT...${NC}"
cd "$PROJECT_ROOT/frontend"

# Install deps if node_modules missing
if [ ! -d "node_modules" ]; then
    echo -e "  ${YELLOW}Installing frontend dependencies...${NC}"
    npm install
fi

npx vite --host 0.0.0.0 --port "$FRONTEND_PORT" &
PIDS+=($!)

# Return to project root
cd "$PROJECT_ROOT"

echo ""
echo -e "${GREEN}Development servers started!${NC}"
echo ""
echo "Services:"
if [ "$START_INFERENCE" = "true" ]; then
    echo -e "  Inference: ${YELLOW}http://localhost:$INFERENCE_PORT${NC}"
fi
echo -e "  Backend:  ${YELLOW}http://localhost:$BACKEND_PORT${NC}"
echo -e "  Frontend: ${YELLOW}http://localhost:$FRONTEND_PORT${NC}"
echo -e "  API Docs: ${YELLOW}http://localhost:$BACKEND_PORT/docs${NC}"
echo ""
echo -e "Press ${RED}Ctrl+C${NC} to stop all services"
echo ""

# Wait for any background process to exit
wait
