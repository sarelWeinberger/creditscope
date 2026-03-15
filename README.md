# CreditScope

**Agentic Credit Scoring with MoE Observability**

CreditScope is an AI-powered credit scoring assistant that leverages the Qwen3.5-35B-A3B mixture-of-experts model with full observability into expert routing, chain-of-thought reasoning, and tool execution.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)
![React](https://img.shields.io/badge/react-18.3-blue.svg)

## Features

- 🤖 **Agentic Credit Analysis**: AI agent with specialized credit scoring tools
- 🧠 **MoE Observability**: Real-time visualization of expert routing across 64 experts
- 💭 **Chain-of-Thought Control**: Adjustable thinking budget with live reasoning display
- 📊 **Credit Scoring Tools**: DTI calculation, payment history analysis, collateral evaluation
- 📷 **Multimodal Support**: Process uploaded documents (pay stubs, tax returns) via OCR
- 📈 **Full Observability**: Prometheus metrics + Grafana dashboards

## Architecture

```
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│   React Frontend│◄────►│  FastAPI Backend│◄_───►│  SGLang Server  │
│   (TypeScript)  │      │   (Python 3.11) │      │  (Qwen3.5-35B)  │
└─────────────────┘      └─────────────────┘      └─────────────────┘
         │                        │                        │
         │                        │                        │
         ▼                        ▼                        ▼
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│     Vite +      │      │    SQLite DB +  │      │   MoE Hooks +   │
│    Tailwind     │      │   Credit Tools  │      │   Observability │
└─────────────────┘      └─────────────────┘      └─────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- npm 9+
- NVIDIA GPU with 24GB+ VRAM (for inference; optional for frontend/backend only)
- Docker & Docker Compose (optional, for containerized deployment)

### Development Setup

```bash
# Clone the repository
git clone https://github.com/sarelWeinberger/creditscope.git
cd creditscope

# Run setup (creates venv, installs Python + frontend deps, copies .env)
chmod +x scripts/setup.sh
./scripts/setup.sh

# Edit .env with your settings
nano .env

# Start the full dev stack (cleans old processes first)
./scripts/run_dev.sh

# Start without the inference server (no GPU required)
./scripts/run_dev.sh --no-inference

# Start with the lower-latency preset
./scripts/run_dev.sh --profile fast

# Start in the foreground (logs to terminal)
./scripts/run_dev.sh --foreground

# Check or stop detached dev services
./scripts/start-dev.sh --status
./scripts/start-dev.sh --stop

# Run a one-shot watchdog check
./scripts/watchdog.sh
```

### Expose on Public IP with nginx

After the dev servers are running, serve the app on port 80 using the included nginx setup:

```bash
./scripts/setup_nginx_http.sh
```

This installs nginx (if needed), generates a self-signed SSL certificate, and configures a reverse proxy that routes:

- `/` to the Vite frontend on port 3000
- `/api/` to the FastAPI backend on port 8080
- `/api/chat/ws` WebSocket connections to the backend

The app will be available at `http://YOUR_SERVER_IP/`.

To add your server's public IP to the allowed CORS origins, edit `.env`:

```
CORS_ORIGINS=http://localhost:3000,http://localhost:5173,http://YOUR_SERVER_IP
```

Then restart the backend:

```bash
./scripts/start-dev.sh --stop
./scripts/run_dev.sh --no-inference
```

### Server Setup From Scratch (Native)

Use this path when provisioning a fresh Ubuntu server without Docker.

#### 1. Host requirements

- Ubuntu 22.04 or 24.04
- Python 3.11+
- Node.js 18+ and npm 9+
- NVIDIA GPU with 24GB+ VRAM (optional, only for inference)
- Ports `80` and `443` open for public access

#### 2. Install system dependencies

```bash
sudo apt-get update
sudo apt-get install -y python3 python3-pip python3-venv git curl nginx openssl lsof
```

Install Node.js (if not present):

```bash
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs
```

#### 3. Clone and configure

```bash
git clone https://github.com/sarelWeinberger/creditscope.git
cd creditscope
cp .env.example .env
```

Edit `.env` and set at least:

- `AUTH_USERS`, `AUTH_PASSWORD`, and `AUTH_SECRET_KEY` for application login
- `CORS_ORIGINS` — add your server's public IP (e.g. `http://54.166.246.53`)
- `HUGGING_FACE_HUB_TOKEN` if the model requires authentication
- `BASIC_AUTH_USERS` and `BASIC_AUTH_PASSWORD` for frontend basic auth

#### 4. Run setup and start

```bash
# Install all dependencies (Python venv + npm)
./scripts/setup.sh

# Start the dev servers (without inference if no GPU)
./scripts/run_dev.sh --no-inference

# Expose on port 80 via nginx
./scripts/setup_nginx_http.sh
```

#### 5. Verify

```bash
# Check service status
./scripts/start-dev.sh --status

# Test health
curl -f http://localhost:8080/health
curl -f http://localhost:3000/

# Test public access
curl -f http://YOUR_SERVER_IP/
```

### Docker Deployment

For containerized deployment with GPU inference:

```bash
cp .env.example .env
# Edit .env with your settings

# Start all services
docker compose up -d --build

# View logs
docker compose logs -f

# Stop services
docker compose down
```

#### Docker prerequisites

Install Docker and (if using GPU) the NVIDIA Container Toolkit:

```bash
# Docker
sudo apt-get update
sudo apt-get install -y docker.io docker-compose-v2
sudo usermod -aG docker "$USER"
newgrp docker

# NVIDIA Container Toolkit (GPU only)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

#### Docker access points

- Frontend: `http://SERVER_IP/`
- Prometheus: `http://SERVER_IP:9090/`
- Grafana: `http://SERVER_IP:3001/`

#### Common issues

- `docker: command not found` — Docker is not installed.
- `could not select device driver` — NVIDIA Container Toolkit not configured.
- `401` / model download failures — set `HUGGING_FACE_HUB_TOKEN` in `.env`.
- `inference` keeps restarting — GPU lacks VRAM or `MEM_FRACTION_STATIC` is too high.
- Port conflicts — stop the conflicting service or change ports in `docker-compose.yml`.

### Production Hardening

If you are running the app with the repo scripts instead of Docker, add a watchdog so the stack restarts automatically when the model stops responding:

```bash
* * * * * cd /home/ubuntu/creditscope && ./scripts/watchdog.sh
```

Useful watchdog environment variables:

- `WATCHDOG_BACKEND_URL` defaults to `http://127.0.0.1:8080/health`
- `WATCHDOG_INFERENCE_URL` defaults to `http://127.0.0.1:8000/model_info`
- `WATCHDOG_RESTART_COOLDOWN_SECONDS` defaults to `120`

If you are using Docker in production, the compose file now includes:

- `restart: unless-stopped` for `inference`, `backend`, and `frontend`
- inference health checks against `/model_info` instead of only `/health`

For the most robust production setup, run the app behind a process supervisor such as `systemd` or Docker with health checks and automatic restarts.

## Services

| Service | Port | Description |
|---------|------|-------------|
| Frontend | 80 | Public React UI served by nginx in Docker |
| Backend | internal | FastAPI server on the compose network |
| Inference | internal | SGLang server on the compose network |
| Prometheus | 9090 | Metrics endpoint |
| Grafana | 3001 | Dashboards |

## Credit Scoring Tools

| Tool | Description |
|------|-------------|
| `calculate_credit_score` | Compute weighted credit score from factors |
| `calculate_dti` | Debt-to-income ratio calculation |
| `analyze_payment_history` | Analyze payment patterns and delinquencies |
| `evaluate_collateral` | Assess collateral value and LTV |
| `structure_loan` | Generate loan terms and amortization |
| `apply_risk_adjustments` | Apply regulatory and risk adjustments |

## API Endpoints

### Chat
- `POST /api/chat` - Process chat message with credit analysis
- `WS /api/chat/ws` - WebSocket for streaming responses

### Customers
- `GET /api/customers` - List all customers
- `GET /api/customers/{id}` - Get customer details
- `GET /api/customers/{id}/credit-report` - Get full credit report

### Observability
- `GET /api/observability/moe/current` - Current MoE expert activations
- `GET /api/observability/moe/history` - Historical expert data
- `GET /api/observability/thinking/sessions` - Thinking session data

### Configuration
- `GET /api/thinking/budgets` - Available thinking budget presets
- `POST /api/thinking/config` - Update thinking configuration

## Thinking Budget Presets

| Preset | Tokens | Use Case |
|--------|--------|----------|
| none | 0 | Direct responses only |
| minimal | 128 | Simple lookups |
| short | 512 | Quick calculations |
| standard | 2048 | Normal analysis |
| extended | 8192 | Complex reasoning |
| deep | 32768 | Thorough investigation |
| unlimited | -1 | No limit |

## Project Structure

```
creditscope/
├── backend/
│   ├── agent/          # Agent orchestration
│   ├── db/             # Database models & queries
│   ├── routers/        # API endpoints
│   ├── schemas/        # Pydantic models
│   └── tools/          # Credit scoring tools
├── frontend/
│   └── src/
│       ├── components/ # React components
│       ├── hooks/      # Custom hooks
│       └── types/      # TypeScript types
├── inference/
│   ├── server.py       # SGLang server launcher
│   ├── moe_hooks.py    # Expert routing capture
│   └── cot_controller.py # Thinking budget control
├── grafana/
│   ├── dashboards/     # Dashboard JSON
│   └── provisioning/   # Auto-provisioning
└── scripts/
    ├── setup.sh        # Development setup
    └── start-dev.sh    # Development launcher
```

## Environment Variables

See [.env.example](.env.example) for all configuration options.

Key variables:
- `MODEL_PATH` - HuggingFace model path
- `TP_SIZE` - Tensor parallelism (GPUs)
- `CONTEXT_LENGTH` - Max context window
- `DEFAULT_THINKING_BUDGET` - Default CoT budget
- `INFERENCE_PROFILE` - Startup preset (`stable` or `fast`)

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Lint code
ruff check .

# Type check
mypy backend inference
```

## License

MIT License - see [LICENSE](LICENSE) for details.
