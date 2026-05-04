# Nightshade Quantitative System

Nightshade is an institutional-grade, low-latency market data pipeline built for ingesting and observing real-time WebSocket tick data. The system features bitemporal QuestDB storage, an ultra-fast Redis stream pipeline, automated gap filling, and a comprehensive observability dashboard.

## 🚀 Quick Start Guide

The pipeline has been rebuilt into modular, robust layers. Follow these exact steps to start ingesting and observing market data on your local machine.

### 1. Prerequisites

Before you start, ensure you have the following installed on your machine:
- **Python 3.11 or higher**
- **Docker Desktop** (required to run QuestDB and Redis locally)
  - **Windows Users:** Docker Desktop requires **WSL 2** (Windows Subsystem for Linux) to function correctly.
  - *If you do not have Docker installed:* Download it from [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop/). When installing on Windows, ensure the "Use WSL 2 instead of Hyper-V" option is checked.
  - *If you do not have WSL installed:* Open PowerShell as Administrator and run `wsl --install`. Restart your computer if prompted.
- **Git**

### 2. Environment Setup

Copy and paste the following commands into your terminal:

```bash
# Clone the repository and enter the directory
git clone https://github.com/notnamansinha/Nightshade.git
cd Nightshade

# Create a clean Python virtual environment
python -m venv .venv

# Activate the virtual environment:
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Windows (CMD):
.venv\Scripts\activate.bat
# Linux / macOS:
source .venv/bin/activate

# Install strictly the required dependencies
pip install -r requirements.txt

# Troubleshooting Windows Installation Issues:
# If you encounter an error building 'questdb' (e.g., "dataframe.pxi not found"), 
# force install the pre-compiled wheel instead of building from source:
# pip install questdb

# If you encounter a missing module error for 'psycopg2', manually install the binary:
# pip install psycopg2-binary
```

### 3. Start the Infrastructure Services (Docker)

Nightshade relies on **QuestDB** (port 9000) for tick persistence and **Redis** (port 6379) for caching and streaming.

```bash
# Start both services in the background
docker-compose up -d

# Wait a few seconds to let them fully boot
```

### 4. Setup Your Secure Vault & API Keys

Nightshade uses AES-GCM encryption for all secrets. The first time you run this command, it will ask you to create a **Master Password**. You will need this password every time you start the live ingestor.

*Note: Databento is optional. If you do not have a Databento key, the system will gracefully skip it and rely on Polygon.*

```bash
# Register your primary data provider key (Required)
python -m layer0.secrets set polygon.api_key <YOUR_POLYGON_KEY>

# Register a backup data provider key (Optional)
python -m layer0.secrets set databento.api_key <YOUR_DATABENTO_KEY>
```

### 5. Bootstrap the Identity & Data Lakes

Before the ingestion engine can start, it needs to know what stocks to track and how to structure the database.

```bash
# 1. Initialize the Security Master (creates the list of 30 US Tech stocks)
# It will prompt for your Master Password.
python -m layer1a.cli bootstrap

# 2. Create the bitemporal tables in QuestDB (Bronze, Silver, Gold schemas)
python -m layer1b.ingestor_cli schema --create
```

---

## 💻 Running the Live Pipeline

Nightshade separates live data ingestion and system observability into independent CLI modules. Open two different terminal windows to run them side-by-side.

### Terminal 1: Start Live Ingestion (Layer 1C)

This command connects to the Polygon WebSockets, handles automated token-bucket rate limiting for gaps, normalizes the data, and pipes everything into your QuestDB instance. It will ask for your master password to unlock the secure API keys.

```bash
# Ensure your virtual environment is activated, then run:
python -m layer1c.live_cli start
```

### Terminal 2: Start the Observability Dashboard (Layer 2)

This command boots a rich, terminal-based UI that provides sub-second monitoring of the ingestion pipeline's health, Redis buffer status, database write metrics, and missing-sequence gaps.

```bash
# Ensure your virtual environment is activated, then run:
python -m layer2.obs_cli dashboard
```

---

## 🏗 System Architecture

The Nightshade pipeline is meticulously engineered using a strictly compartmentalized micro-architecture. This ensures bitemporal data integrity, isolated fault domains, and zero-latency cross-contamination. The architecture is segregated into specialized "Layers," each handling a distinct lifecycle phase of market data.

### 1. High-Level Macro Architecture

The following diagram illustrates the overarching flow of data and telemetry across the Nightshade ecosystem:

```mermaid
flowchart TD
    %% Define layers
    subgraph L1C [Layer 1C: Live Data Ingestion]
        direction TB
        POLY["Polygon WebSocket\nAdapter (Primary)"]
        DB["Databento Adapter\n(Fallback)"]
        GAPF["Token-Bucket Gap\nOrchestrator"]
        NORM["Bitemporal Tick\nNormalizer"]
    end

    subgraph L1B [Layer 1B: The Data Lake]
        direction LR
        REDIS[("Redis Stream Buffer\n(L1 Cache & Pipe)")]
        QDB[("QuestDB Persistence\n(Bitemporal TSDB)")]
    end

    subgraph L2 [Layer 2: Observability & Telemetry]
        direction TB
        HEALTH["State-Machine\nHealth Checker"]
        METRIC["Thread-Safe\nMetrics Emitter"]
        OBS["Curses-Based\nLive Dashboard"]
    end

    subgraph L0 [Layer 0: Immutable Foundation]
        direction LR
        SEC["AES-GCM\nSecrets Manager"]
        CFG["Config Registry\n(Strict Validation)"]
        ALRT["Alert Manager\n(PagerDuty/Email)"]
    end

    %% Data flow connections
    POLY -- "Raw Tick Stream" --> NORM
    DB -- "Raw Tick Stream" --> NORM
    NORM -- "Normalized Ticks" --> REDIS
    REDIS -- "Batch Writes (100ms)" --> QDB
    GAPF -- "Historical Fill" --> QDB

    %% Observability connections
    L1C -. "Health/Latency Metrics" .-> METRIC
    L1B -. "I/O & DB Metrics" .-> METRIC
    METRIC -. "Aggregated Telegraf" .-> REDIS
    REDIS -. "Pub/Sub Telemetry" .-> OBS
    HEALTH -. "Component Status" .-> OBS

    %% Foundation dependencies
    L0 -. "Provides config/keys/alerts" .-> L1C
    L0 -. "Provides config/keys/alerts" .-> L1B
    L0 -. "Provides config/keys/alerts" .-> L2

    %% Styling
    classDef default fill:#f9f9f9,stroke:#333,stroke-width:2px;
    classDef storage fill:#e1f5fe,stroke:#0288d1,stroke-width:2px,color:#000;
    classDef foundation fill:#f3e5f5,stroke:#8e24aa,stroke-width:2px,color:#000;
    classDef ingest fill:#e8f5e9,stroke:#388e3c,stroke-width:2px,color:#000;
    classDef observe fill:#fff3e0,stroke:#f57c00,stroke-width:2px,color:#000;

    class REDIS,QDB storage;
    class SEC,CFG,ALRT foundation;
    class POLY,DB,GAPF,NORM ingest;
    class HEALTH,METRIC,OBS observe;
```

### 2. Deep-Dive Component Breakdown & Feature Analysis

#### Layer 0: Immutable Foundation & Security
The bedrock of the Nightshade system, ensuring operational security and deterministic configuration across all upper layers.
*   **AES-GCM Secrets Manager**: Uses Fernet symmetric encryption with PBKDF2HMAC key derivation (480,000 iterations). API keys are never stored in plaintext; a master password decrypts the vault purely in-memory at runtime.
*   **Strict Config Registry**: A singleton configuration parser that guarantees no system component boots with missing or malformed parameters, preventing cascading failures.
*   **Alert Manager**: An asynchronous, queue-based alerting subsystem that integrates with external PagerDuty or SMTP protocols for critical operational incidents.

#### Layer 1A: Security Master & Identity
Resolves external vendor symbology (e.g., Polygon tickers vs. Databento conventions) into an internal, immutable `nightshade_id`.
*   **Universe Tracker**: Automatically maintains the `NIGHTSHADE_US_TECH` universe, tracking corporate actions, symbol changes, and dynamic composition without pipeline restarts.

#### Layer 1B: Bitemporal Data Lake
The dual-engine storage architecture optimized for both sub-millisecond live piping and petabyte-scale historical quantitative research.

```mermaid
sequenceDiagram
    participant L1C as Layer 1C (Ingestor)
    participant Redis as Redis Stream Buffer
    participant QDB as QuestDB TSDB
    
    L1C->>Redis: XADD: Normalized Tick (Memory)
    Redis-->>L1C: Ack (Microsecond latency)
    
    loop Every 100ms or 500 ticks
        Redis->>QDB: ILP Batch Write (Disk)
        QDB-->>Redis: Commit Ack
        Redis->>Redis: XTRIM (Evict processed ticks)
    end
```

*   **Redis Stream Buffer (L1 Cache)**: Acts as a high-throughput shock absorber. Ticks are ingested via `XADD` into Redis streams, completely decoupling the WebSocket ingestion speed from disk write I/O.
*   **QuestDB Persistence (Bitemporal TSDB)**: Ingests data via the ultra-fast Influx Line Protocol (ILP). Uses a bitemporal schema (`ts_event` and `ts_recorded`) to perfectly reconstruct causality and prevent lookahead bias during quantitative backtesting.

#### Layer 1C: Live Data Ingestion
A highly concurrent ingestion engine utilizing protocol-based structural subtyping.
*   **WebSocket Adapters**: Maintains persistent, authenticated connections to primary (Polygon) and secondary data providers.
*   **Token-Bucket Gap Orchestrator**: Actively monitors sequence numbers. If a UDP/TCP drop occurs, it utilizes a strict Token-Bucket rate-limited algorithm to fetch historical REST data, backfilling the exact missing sequence range into QuestDB without violating vendor API limits.

#### Layer 2: Real-Time Observability Stack
Provides a "single pane of glass" terminal UI for quantitative researchers to monitor system health without overhead.
*   **Thread-Safe Metrics Emitter**: Operates on a dedicated GIL-released thread, sampling CPU, RAM, and internal queue sizes.
*   **State-Machine Health Checker**: Implements rigorous state-transition logic (`STARTING` → `HEALTHY` → `DEGRADED` → `FAILED`), preventing flapping alerts and providing deterministic subsystem restarts.

---

## 🧪 Testing

Nightshade has **100% unit test coverage** (61/61 tests) across its live ingestion and observability pipelines. All network components are rigorously mocked to run without needing real API keys.

To run the entire test suite locally:

```bash
# Ensure you are inside your virtual environment
pytest tests/layer1c tests/layer2 --cov=layer1c --cov=layer2
```
