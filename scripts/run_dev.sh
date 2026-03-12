#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

BACKEND_PORT=${BACKEND_PORT:-8080}
FRONTEND_PORT=${FRONTEND_PORT:-3000}
START_DEV_ARGS=()

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

kill_pid() {
	local pid="$1"

	if ! kill -0 "$pid" 2>/dev/null; then
		return 0
	fi

	kill "$pid" 2>/dev/null || true

	for _ in $(seq 1 10); do
		if ! kill -0 "$pid" 2>/dev/null; then
			return 0
		fi
		sleep 1
	done

	kill -9 "$pid" 2>/dev/null || true
}

kill_matching_processes() {
	local pattern="$1"
	local description="$2"
	local pids

	pids=$(pgrep -f "$pattern" 2>/dev/null || true)
	if [ -z "$pids" ]; then
		return 0
	fi

	echo -e "${YELLOW}Stopping $description${NC}"
	while IFS= read -r pid; do
		[ -n "$pid" ] || continue
		if [ "$pid" = "$$" ] || [ "$pid" = "$PPID" ]; then
			continue
		fi
		echo -e "  ${YELLOW}Killing PID $pid${NC}"
		kill_pid "$pid"
	done <<< "$pids"
}

kill_port_listeners() {
	local port="$1"
	local pids

	pids=$(lsof -ti TCP:"$port" -sTCP:LISTEN 2>/dev/null || true)
	if [ -z "$pids" ]; then
		return 0
	fi

	echo -e "${YELLOW}Freeing port $port${NC}"
	while IFS= read -r pid; do
		[ -n "$pid" ] || continue
		echo -e "  ${YELLOW}Killing PID $pid${NC}"
		kill_pid "$pid"
	done <<< "$pids"
}

kill_inference_processes() {
	local patterns=(
		"python -m sglang.launch_server"
		"sglang.launch_server"
		"sglang::scheduler"
		"sglang::detokenizer"
		"vllm"
		"torchrun"
	)
	local pattern

	for pattern in "${patterns[@]}"; do
		kill_matching_processes "$pattern" "inference processes"
	done
}

release_project_cuda() {
	if ! command -v nvidia-smi >/dev/null 2>&1; then
		return 0
	fi

	local gpu_pids
	gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)
	if [ -z "$gpu_pids" ]; then
		return 0
	fi

	echo -e "${YELLOW}Releasing project CUDA processes${NC}"
	while IFS= read -r pid; do
		local cwd
		local cmdline

		[ -n "$pid" ] || continue
		[ -d "/proc/$pid" ] || continue

		cwd=$(readlink -f "/proc/$pid/cwd" 2>/dev/null || true)
		cmdline=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)

		if [[ "$cwd" == "$PROJECT_ROOT"* ]] || [[ "$cmdline" == *"$PROJECT_ROOT"* ]] || [[ "$cmdline" == *"creditscope"* ]]; then
			echo -e "  ${YELLOW}Killing CUDA PID $pid${NC}"
			kill_pid "$pid"
		fi
	done <<< "$gpu_pids"
}

reset_gpu_if_stuck() {
	if ! command -v nvidia-smi >/dev/null 2>&1; then
		return 0
	fi

	local used_memory
	used_memory=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d ' ')
	if [ -z "$used_memory" ] || [ "$used_memory" -lt 2048 ]; then
		return 0
	fi

	if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -q '[0-9]'; then
		return 0
	fi

	if ps -eo cmd | grep -E 'sglang|vllm|torchrun' | grep -v grep >/dev/null 2>&1; then
		return 0
	fi

	echo -e "${YELLOW}CUDA memory still allocated without active owners. Attempting GPU reset...${NC}"
	nvidia-smi --gpu-reset -i 0 >/dev/null 2>&1 || \
		echo -e "  ${RED}GPU reset was not permitted or failed${NC}"
}

echo -e "${GREEN}Cleaning existing CreditScope dev processes...${NC}"

while [ "$#" -gt 0 ]; do
	case "$1" in
		--inference|--no-inference|--detach|--foreground)
			START_DEV_ARGS+=("$1")
			;;
		-h|--help)
			echo "Usage: $(basename "$0") [--inference | --no-inference] [--detach | --foreground]"
			exit 0
			;;
		*)
			echo -e "${RED}Unknown option: $1${NC}"
			echo "Usage: $(basename "$0") [--inference | --no-inference] [--detach | --foreground]"
			exit 1
			;;
	esac
	shift
done

kill_matching_processes "$PROJECT_ROOT/.venv/bin/python .*uvicorn backend.main:app|uvicorn backend.main:app" "backend processes"
kill_matching_processes "$PROJECT_ROOT/frontend/node_modules/.bin/vite|vite --host 0.0.0.0 --port $FRONTEND_PORT" "frontend processes"
kill_inference_processes
kill_matching_processes "$PROJECT_ROOT" "project-scoped processes"

kill_port_listeners "$BACKEND_PORT"
kill_port_listeners "$FRONTEND_PORT"
release_project_cuda
reset_gpu_if_stuck

PROMETHEUS_MULTIPROC_DIR=${PROMETHEUS_MULTIPROC_DIR:-/tmp/prometheus}
mkdir -p "$PROMETHEUS_MULTIPROC_DIR"
rm -f "$PROMETHEUS_MULTIPROC_DIR"/*.db "$PROMETHEUS_MULTIPROC_DIR"/*.tmp 2>/dev/null || true

echo -e "${GREEN}Starting backend and frontend...${NC}"
exec "$SCRIPT_DIR/start-dev.sh" "${START_DEV_ARGS[@]}"
