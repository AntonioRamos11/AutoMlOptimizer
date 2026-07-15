# MLOptimizer

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Optuna](https://img.shields.io/badge/Optuna-TPE-brightgreen.svg)](https://optuna.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

AutoML framework with Bayesian hyperparameter optimization (Optuna TPE), distributed master/slave architecture, and real-time hardware performance monitoring.

## Features

- **Bayesian Hyperparameter Optimization** — Optuna TPE sampler with custom `RepeatPruner` to avoid redundant trials
- **Two-Phase Training** — Exploration phase rapidly evaluates diverse architectures, then a Hall of Fame phase fully trains the top candidates
- **Automated Model Generation** — Dynamically builds CNN, Inception, MLP, and LSTM architectures for image classification, regression, and time series
- **Distributed Architecture** — Master/slave nodes communicate via RabbitMQ; supports multi-GPU and cloud deployments
- **Hardware Monitoring** — Real-time GPU metrics (utilization, memory, temperature, power, clocks, PCIe), CPU metrics (per-core, RAM, swap), and latency measurements
- **Fault Tolerance** — Queue deduplication, state persistence with resume capability, crash recovery via subprocess isolation
- **Slack & Google Drive Integration** — Optional notifications and results upload
- **State Resume** — Periodically saves optimization state so interrupted runs can continue

## Architecture

```
┌──────────────┐    parameters queue    ┌──────────────┐
│   Master     │───────────────────────▶│    Slave     │
│   Node       │                        │   Node(s)    │
│  (Optuna)    │◀───────────────────────│  (Training)  │
└──────────────┘    results queue       └──────────────┘
```

| Component | Role |
|-----------|------|
| **Master Node** | Runs Optuna TPE optimization, generates architectures, coordinates exploration/HoF phases, persists state |
| **Slave Node** | Receives architecture parameters, trains models in isolated subprocess, collects hardware metrics |
| **RabbitMQ** | Async message broker between master and slaves |
| **GPUMetricsCollector** | Captures GPU/CPU/latency metrics in real time |
| **HardwarePerformanceLogger** | Saves portable JSON logs with model architecture, training info, and hardware data |

## Requirements

- Python 3.10+
- TensorFlow 2.x (CUDA-compatible GPU recommended)
- RabbitMQ server (local or cloud-tunneled)

## Installation

```bash
git clone https://github.com/yourusername/mloptimizer.git
cd mloptimizer

# Conda (recommended)
conda create -n mlopt python=3.10
conda activate mlopt
pip install -r requirements.txt

# Or use the unified runner
python run.py --install

# Start RabbitMQ
sudo systemctl start rabbitmq-server
```

## Usage

### Quick Start (local master + slave)

```bash
python run.py
```

### Separate Nodes

```bash
# Terminal 1 — master
python run.py --master

# Terminal 2 — slave
python run.py --slave --dataset=cifar10 --gpu=0
```

### Cloud Deployment

```bash
python run.py --master --host="your.ngrok.io" --port=12345 --cloud-mode=1
```

### Programmatic API

```python
from app.init_nodes import InitNodes
from system_parameters import SystemParameters as SP

SP.DATASET_NAME = 'cifar10'
SP.DATASET_TYPE = 1   # 1=Image, 2=Regression, 3=TimeSeries
SP.TRIALS = 20

# Start master
InitNodes().master()

# Start slave (separate process)
InitNodes().slave()
```

## Configuration

Edit `system_parameters.py` to control:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DATASET_NAME` | `cifar10` | Dataset to use |
| `DATASET_TYPE` | `1` | 1=Image, 2=Regression, 3=TimeSeries |
| `TRIALS` | `10` | Total Optuna trials |
| `EXPLORATION_SIZE` | `5` | Models in exploration phase |
| `EXPLORATION_EPOCHS` | `5` | Epochs per exploration model |
| `HALL_OF_FAME_SIZE` | `3` | Top models promoted to deep training |
| `HALL_OF_FAME_EPOCHS` | `6` | Epochs for deep training |

## Tests

```bash
python -m unittest tests/test_architecture.py -v
python -m unittest tests/test_system.py -v
```

## License

MIT
