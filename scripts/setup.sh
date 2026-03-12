#!/usr/bin/env bash
set -euo pipefail

# CreditScope Development Setup Script
# Run this once to set up your local development environment

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "🚀 Setting up CreditScope development environment..."

# Check prerequisites
command -v python3 >/dev/null 2>&1 || { echo "❌ Python 3 is required but not installed."; exit 1; }
command -v node >/dev/null 2>&1 || { echo "❌ Node.js is required but not installed."; exit 1; }
command -v npm >/dev/null 2>&1 || { echo "❌ npm is required but not installed."; exit 1; }

# Create Python virtual environment
echo "📦 Creating Python virtual environment..."
cd "$PROJECT_ROOT"
python3 -m venv .venv
source .venv/bin/activate

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install -e ".[dev]"

# Install frontend dependencies
echo "📦 Installing frontend dependencies..."
cd "$PROJECT_ROOT/frontend"
npm install

# Copy environment template if doesn't exist
if [ ! -f "$PROJECT_ROOT/.env" ]; then
    echo "📝 Creating .env file from template..."
    cp "$PROJECT_ROOT/.env.example" "$PROJECT_ROOT/.env"
    echo "⚠️  Please edit .env with your configuration"
fi

# Create data directory
mkdir -p "$PROJECT_ROOT/data"

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Edit .env with your configuration"
echo "  2. Run './scripts/start-dev.sh' to start development servers"
echo "     Detached mode is now the default; use './scripts/start-dev.sh --foreground' for an interactive session"
echo ""
