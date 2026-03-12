#!/usr/bin/env bash
set -euo pipefail

# CreditScope Development Server Launcher
# Starts backend + frontend for local development

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
RUN_DIR="$PROJECT_ROOT/.run"
PID_DIR="$RUN_DIR/pids"
LOG_DIR="$RUN_DIR/logs"

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
INFERENCE_PROFILE=${INFERENCE_PROFILE:-stable}
DETACH_MODE=true
STATUS_ONLY=false
STOP_ONLY=false
PROFILE_EXPLICIT=false

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

PIDS=()
SHUTTING_DOWN=0

usage() {
    echo "Usage: $(basename "$0") [--inference | --no-inference] [--detach | --foreground] [--status] [--stop] [--profile stable|fast]"
}

mkdir -p "$PID_DIR" "$LOG_DIR"

pid_file_for() {
    local service="$1"
    echo "$PID_DIR/$service.pid"
}

log_file_for() {
    local service="$1"
    echo "$LOG_DIR/$service.log"
}

is_pid_running() {
    local pid="$1"
    [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null
}

read_pid() {
    local service="$1"
    local pid_file

    pid_file=$(pid_file_for "$service")
    if [ -f "$pid_file" ]; then
        tr -d '[:space:]' < "$pid_file"
    fi
}

write_pid() {
    local service="$1"
    local pid="$2"

    echo "$pid" > "$(pid_file_for "$service")"
}

remove_pid_file() {
    local service="$1"
    rm -f "$(pid_file_for "$service")"
}

cleanup_stale_pid_files() {
    local service
    local pid

    for service in inference backend frontend; do
        pid=$(read_pid "$service")
        if [ -n "$pid" ] && ! is_pid_running "$pid"; then
            remove_pid_file "$service"
        fi
    done
}

kill_pid() {
    local pid="$1"

    if ! is_pid_running "$pid"; then
        return 0
    fi

    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 10); do
        if ! is_pid_running "$pid"; then
            return 0
        fi
        sleep 1
    done

    kill -9 "$pid" 2>/dev/null || true
}

stop_service() {
    local service="$1"
    local pid

    pid=$(read_pid "$service")
    if [ -z "$pid" ]; then
        remove_pid_file "$service"
        return 0
    fi

    if is_pid_running "$pid"; then
        echo -e "  ${YELLOW}Stopping $service (PID $pid)${NC}"
        kill_pid "$pid"
    fi

    remove_pid_file "$service"
}

print_status() {
    local service
    local pid
    local any_running=0

    cleanup_stale_pid_files

    for service in inference backend frontend; do
        pid=$(read_pid "$service")
        if [ -n "$pid" ] && is_pid_running "$pid"; then
            any_running=1
            echo -e "${GREEN}$service${NC}: running (PID $pid)"
            echo "  log: $(log_file_for "$service")"
        else
            echo -e "${YELLOW}$service${NC}: stopped"
        fi
    done

    if [ "$any_running" -eq 0 ]; then
        echo -e "${YELLOW}No detached CreditScope services are running.${NC}"
    fi
}

stop_detached_services() {
    local services=(frontend backend inference)
    local service

    cleanup_stale_pid_files
    echo -e "${GREEN}Stopping detached CreditScope services...${NC}"
    for service in "${services[@]}"; do
        stop_service "$service"
    done
    echo -e "${GREEN}Detached services stopped.${NC}"
}

start_background_service() {
    local service="$1"
    shift
    local log_file
    local executable
    local pid

    log_file=$(log_file_for "$service")

    if [ "$#" -eq 0 ]; then
        echo -e "${RED}No command configured for $service${NC}"
        return 1
    fi

    executable="$1"
    if [[ "$executable" != */* ]] && ! command -v "$executable" >/dev/null 2>&1; then
        echo -e "${RED}Unable to start $service: command '$executable' was not found${NC}"
        return 1
    fi

    : > "$log_file"
    nohup "$@" >> "$log_file" 2>&1 < /dev/null &
    pid="$!"
    write_pid "$service" "$pid"
    PIDS+=("$pid")

    sleep 1
    if ! is_pid_running "$pid"; then
        echo -e "${RED}$service exited immediately after launch${NC}"
        if [ -s "$log_file" ]; then
            echo -e "${YELLOW}Recent $service log output:${NC}"
            tail -n 20 "$log_file"
        fi
        remove_pid_file "$service"
        return 1
    fi
}

wait_for_http() {
    local service="$1"
    local pid="$2"
    local attempts="$3"
    shift 3
    local endpoint
    local attempt

    echo -n "  Waiting for $service..."
    for attempt in $(seq 1 "$attempts"); do
        for endpoint in "$@"; do
            if curl -sf "$endpoint" > /dev/null 2>&1; then
                echo -e " ${GREEN}ready${NC}"
                return 0
            fi
        done
        if ! is_pid_running "$pid"; then
            echo -e " ${RED}failed${NC}"
            return 1
        fi
        if [ "$attempt" -eq "$attempts" ]; then
            echo -e " ${RED}timeout${NC}"
            return 1
        fi
        sleep 1
    done
}

apply_inference_profile() {
    local stable_args_default="--max-mamba-cache-size 16 --disable-cuda-graph --chunked-prefill-size 512 --max-running-requests 2 --max-total-tokens 65536 --attention-backend triton --skip-server-warmup --fp8-gemm-backend triton"
    local fast_args_default="$stable_args_default"
    local use_profile_overrides=false

    case "$INFERENCE_PROFILE" in
        stable|fast)
            ;;
        *)
            echo -e "${RED}Unknown inference profile: $INFERENCE_PROFILE${NC}"
            echo "Supported profiles: stable, fast"
            exit 1
            ;;
    esac

    if [ "$PROFILE_EXPLICIT" = "true" ]; then
        use_profile_overrides=true
    fi

    if [ "$INFERENCE_PROFILE" = "stable" ]; then
        if [ -n "${SGLANG_EXTRA_ARGS_STABLE:-}" ] || [ -n "${CHAT_MAX_TOKENS_STABLE:-}" ] || [ -n "${DEFAULT_THINKING_BUDGET_STABLE:-}" ]; then
            use_profile_overrides=true
        fi
        if [ "$use_profile_overrides" = "true" ]; then
            export SGLANG_EXTRA_ARGS="${SGLANG_EXTRA_ARGS_STABLE:-${SGLANG_EXTRA_ARGS:-$stable_args_default}}"
            export CHAT_MAX_TOKENS="${CHAT_MAX_TOKENS_STABLE:-${CHAT_MAX_TOKENS:-512}}"
            export DEFAULT_THINKING_BUDGET="${DEFAULT_THINKING_BUDGET_STABLE:-${DEFAULT_THINKING_BUDGET:-standard}}"
        fi
    else
        if [ -n "${SGLANG_EXTRA_ARGS_FAST:-}" ] || [ -n "${CHAT_MAX_TOKENS_FAST:-}" ] || [ -n "${DEFAULT_THINKING_BUDGET_FAST:-}" ]; then
            use_profile_overrides=true
        fi
        if [ "$use_profile_overrides" = "true" ]; then
            export SGLANG_EXTRA_ARGS="${SGLANG_EXTRA_ARGS_FAST:-${SGLANG_EXTRA_ARGS:-$fast_args_default}}"
            export CHAT_MAX_TOKENS="${CHAT_MAX_TOKENS_FAST:-${CHAT_MAX_TOKENS:-256}}"
            export DEFAULT_THINKING_BUDGET="${DEFAULT_THINKING_BUDGET_FAST:-${DEFAULT_THINKING_BUDGET:-short}}"
        fi
    fi
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --inference)
            START_INFERENCE=true
            ;;
        --no-inference)
            START_INFERENCE=false
            ;;
        --detach)
            DETACH_MODE=true
            ;;
        --foreground)
            DETACH_MODE=false
            ;;
        --profile)
            if [ "$#" -lt 2 ]; then
                echo -e "${RED}--profile requires a value${NC}"
                usage
                exit 1
            fi
            INFERENCE_PROFILE="$2"
            PROFILE_EXPLICIT=true
            shift
            ;;
        --fast)
            INFERENCE_PROFILE="fast"
            PROFILE_EXPLICIT=true
            ;;
        --stable)
            INFERENCE_PROFILE="stable"
            PROFILE_EXPLICIT=true
            ;;
        --status)
            STATUS_ONLY=true
            ;;
        --stop)
            STOP_ONLY=true
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

if [ "$STATUS_ONLY" = "true" ] && [ "$STOP_ONLY" = "true" ]; then
    echo -e "${RED}--status cannot be combined with --stop${NC}"
    exit 1
fi

if [ "$STOP_ONLY" = "true" ] && [ "$DETACH_MODE" = "false" ]; then
    echo -e "${RED}--stop cannot be combined with --foreground${NC}"
    exit 1
fi

if [ "$STATUS_ONLY" = "true" ]; then
    print_status
    exit 0
fi

if [ "$STOP_ONLY" = "true" ]; then
    stop_detached_services
    exit 0
fi

apply_inference_profile

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

cleanup_stale_pid_files

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
echo -e "  ${GREEN}Inference profile:${NC} ${YELLOW}$INFERENCE_PROFILE${NC}"

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
export SGLANG_URL="${SGLANG_URL:-http://127.0.0.1:$INFERENCE_PORT}"
export PROMETHEUS_MULTIPROC_DIR="${PROMETHEUS_MULTIPROC_DIR:-/tmp/prometheus}"
mkdir -p "$PROMETHEUS_MULTIPROC_DIR"

# ── Start Inference ───────────────────────────────────────────────────────────
if [ "$START_INFERENCE" = "true" ]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        echo -e "${GREEN}Starting Inference on port $INFERENCE_PORT...${NC}"
        cd "$PROJECT_ROOT"
        if [ "$DETACH_MODE" = "true" ]; then
            start_background_service inference python -m inference.server
            INFERENCE_PID=$(read_pid inference)
        else
            python -m inference.server &
            INFERENCE_PID=$!
            PIDS+=("$INFERENCE_PID")
        fi

        if ! wait_for_http \
            inference \
            "$INFERENCE_PID" \
            180 \
            "http://localhost:$INFERENCE_PORT/health" \
            "http://localhost:$INFERENCE_PORT/model_info"; then
            cleanup
            exit 1
        fi
    else
        echo -e "  ${YELLOW}Skipping inference startup because no NVIDIA GPU was detected${NC}"
        remove_pid_file inference
    fi
else
    remove_pid_file inference
fi

# ── Start Backend ──────────────────────────────────────────────────────────────
echo -e "${GREEN}Starting Backend on port $BACKEND_PORT...${NC}"
cd "$PROJECT_ROOT"
if [ "$DETACH_MODE" = "true" ]; then
    start_background_service \
        backend \
        uvicorn \
        backend.main:app \
        --host 0.0.0.0 \
        --port "$BACKEND_PORT" \
        --reload \
        --reload-dir backend
    BACKEND_PID=$(read_pid backend)
else
    uvicorn backend.main:app \
        --host 0.0.0.0 \
        --port "$BACKEND_PORT" \
        --reload \
        --reload-dir backend &
    BACKEND_PID=$!
    PIDS+=("$BACKEND_PID")
fi

# Wait for backend to be ready
if ! wait_for_http backend "$BACKEND_PID" 15 "http://localhost:$BACKEND_PORT/health"; then
    cleanup
    exit 1
fi

# ── Start Frontend ─────────────────────────────────────────────────────────────
echo -e "${GREEN}Starting Frontend on port $FRONTEND_PORT...${NC}"
cd "$PROJECT_ROOT/frontend"

# Install deps if node_modules missing
if [ ! -d "node_modules" ]; then
    echo -e "  ${YELLOW}Installing frontend dependencies...${NC}"
    npm install
fi

if [ "$DETACH_MODE" = "true" ]; then
    start_background_service frontend npx vite --host 0.0.0.0 --port "$FRONTEND_PORT"
    FRONTEND_PID=$(read_pid frontend)
else
    npx vite --host 0.0.0.0 --port "$FRONTEND_PORT" &
    FRONTEND_PID=$!
    PIDS+=("$FRONTEND_PID")
fi

if ! wait_for_http frontend "$FRONTEND_PID" 30 "http://localhost:$FRONTEND_PORT"; then
    cleanup
    exit 1
fi

# Return to project root
cd "$PROJECT_ROOT"

echo ""
if [ "$DETACH_MODE" = "true" ]; then
    trap - SIGINT SIGTERM EXIT
    SHUTTING_DOWN=1
    echo -e "${GREEN}Detached development services started.${NC}"
else
    echo -e "${GREEN}Development servers started!${NC}"
fi
echo ""
echo "Services:"
if [ "$START_INFERENCE" = "true" ]; then
    echo -e "  Inference: ${YELLOW}http://localhost:$INFERENCE_PORT${NC}"
fi
echo -e "  Backend:  ${YELLOW}http://localhost:$BACKEND_PORT${NC}"
echo -e "  Frontend: ${YELLOW}http://localhost:$FRONTEND_PORT${NC}"
echo -e "  API Docs: ${YELLOW}http://localhost:$BACKEND_PORT/docs${NC}"
echo ""
if [ "$DETACH_MODE" = "true" ]; then
    echo "Logs:"
    if [ "$START_INFERENCE" = "true" ] && [ -f "$(log_file_for inference)" ]; then
        echo "  Inference: $(log_file_for inference)"
    fi
    echo "  Backend:  $(log_file_for backend)"
    echo "  Frontend: $(log_file_for frontend)"
    echo ""
    echo "Manage detached services with:"
    echo "  ./scripts/start-dev.sh --status"
    echo "  ./scripts/start-dev.sh --stop"
    exit 0
fi

echo -e "Press ${RED}Ctrl+C${NC} to stop all services"
echo ""

# Wait for any background process to exit
wait
