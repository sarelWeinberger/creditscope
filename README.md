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
```

### Docker Deployment

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## Services

| Service | Port | Description |
|---------|------|-------------|
| Frontend | 3000 | React UI |
| Backend | 8080 | FastAPI server |
| Inference | 8000 | SGLang server |
| Prometheus | 9090 | Metrics |
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

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

---

Built with ❤️ using Qwen3.5-35B-A3B, SGLang, FastAPI, and React
