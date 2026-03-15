#!/usr/bin/env bash
set -euo pipefail

# CreditScope Development Setup Script
# Safe to run multiple times

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_ROOT/.venv"

find_system_python() {
    if command -v python3 >/dev/null 2>&1; then
        command -v python3
        return 0
    fi

    if command -v python >/dev/null 2>&1; then
        command -v python
        return 0
    fi

    echo "❌ Python is required but not installed."
    exit 1
}

echo "🚀 Setting up CreditScope development environment..."

command -v node >/dev/null 2>&1 || { echo "❌ Node.js is required but not installed."; exit 1; }
command -v npm >/dev/null 2>&1 || { echo "❌ npm is required but not installed."; exit 1; }

SYSTEM_PYTHON="$(find_system_python)"

cd "$PROJECT_ROOT"

if [ ! -d "$VENV_DIR" ]; then
    echo "📦 Creating Python virtual environment..."
    "$SYSTEM_PYTHON" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

if [ ! -x "$VENV_DIR/bin/python" ] && [ -x "$VENV_DIR/bin/python3" ]; then
    ln -sf "$VENV_DIR/bin/python3" "$VENV_DIR/bin/python"
fi

echo "📦 Installing Python dependencies..."
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

if [ -d "$PROJECT_ROOT/frontend" ] && [ -f "$PROJECT_ROOT/frontend/package.json" ]; then
    echo "📦 Installing frontend dependencies..."
    cd "$PROJECT_ROOT/frontend"
    npm install
    cd "$PROJECT_ROOT"
fi

if [ ! -f "$PROJECT_ROOT/.env" ] && [ -f "$PROJECT_ROOT/.env.example" ]; then
    echo "📝 Creating .env file from template..."
    cp "$PROJECT_ROOT/.env.example" "$PROJECT_ROOT/.env"
    echo "⚠️  Please edit .env with your configuration"
fi

mkdir -p "$PROJECT_ROOT/data"

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Edit .env with your configuration"
echo "  2. Run './scripts/run_dev.sh'"
echo ""