# CoexistAI

CoexistAI is a modular, developer-friendly research assistant framework. It enables you to build, search, summarize, and automate research workflows using LLMs, web search, Reddit, YouTube, git and mapping tools—all with simple API calls or Python functions.

## 📁 Project Structure

This project has been reorganized for better maintainability:

```
CoexistAI/
├── app.py              # Main FastAPI application
├── utils/              # Core utility modules
├── config/             # Configuration files
│   ├── model_config.py # Model configuration
│   ├── docker-compose.yml # Docker composition
│   ├── Dockerfile      # Docker image definition
│   ├── entrypoint.sh   # Container entrypoint script
│   ├── requirements.txt # Python dependencies
│   └── searxng/        # SearxNG search engine config
├── docs/               # Documentation
│   ├── README.md       # Main documentation
│   ├── README_MCP.md   # MCP integration guide
│   ├── DOCKER_12FACTOR.md # Docker 12-factor guide
│   ├── ENV.md          # Environment variables reference
├── scripts/            # Utility scripts
│   ├── quick_setup.sh  # Quick setup script
│   ├── health_check.py # Health check script
│   └── system_prompt.py # System prompt script
├── tests/              # Test and verification scripts
├── notebooks/          # Jupyter notebooks
└── output/             # Output directory
```

## 🚀 Quick Start

See the main documentation in [`docs/README.md`](docs/README.md) for complete setup instructions.

For Docker-based deployment, see [`docs/DOCKER_12FACTOR.md`](docs/DOCKER_12FACTOR.md).

For environment variable configuration, see [`docs/ENV.md`](docs/ENV.md).

## 🔧 Configuration

Main configuration is in [`config/model_config.py`](config/model_config.py).

Docker setup uses [`config/docker-compose.yml`](config/docker-compose.yml).

## 🧪 Testing

Run verification tests:
```bash
cd tests/
python verify_config.py
python test_dynamic_config.py
python verify_coverage.py
```

## 📚 Documentation

- **Main Docs**: [`docs/README.md`](docs/README.md)
- **MCP Integration**: [`docs/README_MCP.md`](docs/README_MCP.md)
- **Docker Guide**: [`docs/DOCKER_12FACTOR.md`](docs/DOCKER_12FACTOR.md)
- **Environment Variables**: [`docs/ENV.md`](docs/ENV.md)

## 🛠️ Development

Quick setup:
```bash
# Run the quick setup script
./scripts/quick_setup.sh

# Or manually with Docker
docker-compose -f config/docker-compose.yml up -d
```

## 📄 License

See [`LICENSE`](LICENSE) for license information.