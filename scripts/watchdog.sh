#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
RUN_DIR="$PROJECT_ROOT/.run"
LOG_DIR="$RUN_DIR/logs"
LOCK_FILE="$RUN_DIR/watchdog.lock"
LOG_FILE="$LOG_DIR/watchdog.log"

mkdir -p "$LOG_DIR"

if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

BACKEND_URL="${WATCHDOG_BACKEND_URL:-http://127.0.0.1:${BACKEND_PORT:-8080}/health}"
INFERENCE_URL="${WATCHDOG_INFERENCE_URL:-http://127.0.0.1:${SGLANG_PORT:-8000}/model_info}"
BACKEND_TIMEOUT="${WATCHDOG_BACKEND_TIMEOUT_SECONDS:-10}"
INFERENCE_TIMEOUT="${WATCHDOG_INFERENCE_TIMEOUT_SECONDS:-15}"
RESTART_COOLDOWN_SECONDS="${WATCHDOG_RESTART_COOLDOWN_SECONDS:-120}"
STATE_FILE="$RUN_DIR/watchdog.last_restart"

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    exit 0
fi

timestamp() {
    date -u +%Y-%m-%dT%H:%M:%SZ
}

log_line() {
    printf '[%s] %s\n' "$(timestamp)" "$*" | tee -a "$LOG_FILE"
}

check_url() {
    local url="$1"
    local timeout="$2"
    curl -fsS -m "$timeout" "$url" >/dev/null 2>&1
}

cooldown_active() {
    if [ ! -f "$STATE_FILE" ]; then
        return 1
    fi

    local last_restart now elapsed
    last_restart="$(cat "$STATE_FILE" 2>/dev/null || echo 0)"
    now="$(date +%s)"
    elapsed=$((now - last_restart))
    [ "$elapsed" -lt "$RESTART_COOLDOWN_SECONDS" ]
}

restart_stack() {
    if cooldown_active; then
        log_line "watchdog detected unhealthy services but cooldown is still active"
        return 0
    fi

    log_line "watchdog restarting stack"
    printf '%s' "$(date +%s)" > "$STATE_FILE"

    (
        cd "$PROJECT_ROOT"
        ./scripts/start-dev.sh --stop
        ./scripts/run_dev.sh
    ) >> "$LOG_FILE" 2>&1

    log_line "watchdog restart completed"
}

main() {
    local backend_ok=true
    local inference_ok=true

    if ! check_url "$BACKEND_URL" "$BACKEND_TIMEOUT"; then
        backend_ok=false
        log_line "backend health check failed: $BACKEND_URL"
    fi

    if ! check_url "$INFERENCE_URL" "$INFERENCE_TIMEOUT"; then
        inference_ok=false
        log_line "inference health check failed: $INFERENCE_URL"
    fi

    if [ "$backend_ok" = true ] && [ "$inference_ok" = true ]; then
        log_line "watchdog check passed"
        exit 0
    fi

    restart_stack
}

main "$@"