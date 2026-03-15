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
- Node.js 20+
- NVIDIA GPU with 24GB+ VRAM (for inference)
- Docker & Docker Compose (optional)

### Development Setup

```bash
# Clone the repository
git clone https://github.com/sarelWeinberger/creditscope.git
cd creditscope

# Run setup script
chmod +x scripts/setup.sh
./scripts/setup.sh

# Configure environment
cp .env.example .env
# Edit .env with your settings

# Start development servers
./scripts/start-dev.sh

# Start development servers with the lower-latency preset
./scripts/start-dev.sh --profile fast

# Start development servers in the foreground
./scripts/start-dev.sh --foreground

# Start frontend + backend only
./scripts/start-dev.sh --no-inference

# Clean old processes, then start full dev stack
./scripts/run_dev.sh

# Clean old processes, then start with the lower-latency preset
./scripts/run_dev.sh --profile fast

# Clean old processes, then start the stack in the foreground
./scripts/run_dev.sh --foreground

# Run a one-shot watchdog check that restarts the stack if backend or inference is hung
./scripts/watchdog.sh

# Check or stop detached dev services
./scripts/start-dev.sh --status
./scripts/start-dev.sh --stop

# Put the app on public port 80 using nginx
./scripts/setup_nginx_http.sh
```

### Docker Deployment

```bash
# Create runtime configuration
cp .env.example .env
# Edit .env before first start

# Start all services
docker compose up -d --build

# View logs
docker compose logs -f

# Stop services
docker compose down
```

### Server Setup From Scratch

Use this path when provisioning a fresh Ubuntu server for the Docker deployment.

#### 1. Host requirements

- Ubuntu 22.04 or 24.04
- NVIDIA GPU with 24GB+ VRAM for the full inference stack
- NVIDIA driver installed and working (`nvidia-smi` must succeed on the host)
- 40GB+ free disk if the model will be pulled locally
- Ports `80`, `9090`, and `3001` open if you need external access

If you do not have a compatible GPU, do not use the Docker inference service. Use the repo scripts with `--no-inference`, or point the backend at an external SGLang endpoint.

#### 2. Install Docker

On Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg
sudo apt-get install -y docker.io docker-compose-v2
sudo usermod -aG docker "$USER"
newgrp docker
docker --version
docker compose version
```

If your host uses `systemd`, start and enable Docker:

```bash
sudo systemctl enable --now docker
```

#### 3. Install NVIDIA container support

The `inference` container uses GPU access. Install the NVIDIA Container Toolkit on the host:

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
```

If your host uses `systemd`, restart Docker after configuring the runtime:

```bash
sudo systemctl restart docker
```

Validate GPU access:

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.4.0-runtime-ubuntu22.04 nvidia-smi
```

#### 4. Clone and configure the app

```bash
git clone https://github.com/sarelWeinberger/creditscope.git
cd creditscope
cp .env.example .env
```

Edit `.env` and set at least these values:

- `HUGGING_FACE_HUB_TOKEN` if the selected model requires authentication
- `MODEL_PATH` if you are not using the default Qwen model
- `BASIC_AUTH_USERS` and `BASIC_AUTH_PASSWORD` before exposing the frontend publicly
- `AUTH_USERS`, `AUTH_PASSWORD`, and `AUTH_SECRET_KEY` for application login
- `GRAFANA_PASSWORD` for dashboard access
- `MODEL_CACHE_DIR` if you want model weights stored outside the default home cache

Create the local data directories used by bind mounts:

```bash
mkdir -p data
mkdir -p "$HOME/.cache/huggingface"
```

#### 5. Start the stack

```bash
docker compose up -d --build
```

Check status:

```bash
docker compose ps
docker compose logs -f inference
docker compose logs -f backend
docker compose logs -f frontend
```

Expected access points after startup:

- Frontend: `http://SERVER_IP/`
- Prometheus: `http://SERVER_IP:9090/`
- Grafana: `http://SERVER_IP:3001/`

The backend and inference services are intentionally only exposed inside the Docker network by default.

#### 6. Verify health

```bash
docker compose ps
curl -I http://127.0.0.1/
docker compose exec backend curl -f http://localhost:8080/health
docker compose exec inference curl -f http://localhost:8000/model_info
```

#### 7. Stop or update the stack

```bash
# Stop containers but keep volumes
docker compose down

# Rebuild after code or image changes
docker compose up -d --build

# Pull newer base images before rebuilding
docker compose pull
docker compose up -d --build
```

#### 8. Common first-boot issues

- `docker: command not found`: Docker is not installed on the host.
- `could not select device driver` or missing GPU in containers: the NVIDIA Container Toolkit is not configured correctly.
- `401` or model download failures: set `HUGGING_FACE_HUB_TOKEN` in `.env` if the model is gated.
- `inference` keeps restarting: the GPU likely does not have enough VRAM for the configured model or `MEM_FRACTION_STATIC` is too high.
- Frontend prompts for credentials: this is expected when `BASIC_AUTH_USERS` and `BASIC_AUTH_PASSWORD` are set.
- Port conflicts on `80`, `9090`, or `3001`: stop the conflicting service or change the published ports in [docker-compose.yml](/home/ubuntu/creditscope/docker-compose.yml).

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
