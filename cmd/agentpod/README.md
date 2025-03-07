# Agentpod Dashboard

This is the command-line interface to get Agentpod dashboard up and running.

## Installation

You can install the CLI using Go:

```bash
# Build and install the CLI to your GOPATH/bin
make install
```

Or you can build it locally:

```bash
# Just build the binary in the current directory
make build
```

## Usage

The CLI currently supports the `start` command:

```bash
# Start the dashboard server with required PostgreSQL connection
agentpod start --postgres "postgresql://username:password@localhost:5432/database"

# Initialize the database before starting
agentpod start --init --postgres "postgresql://username:password@localhost:5432/database"

# Specify a custom port (default is 55501)
agentpod start --postgres "postgresql://username:password@localhost:5432/database" --port 8080

# Specify a custom asset URL (default is https://dash-assets.agentpod.ai)
agentpod start --postgres "postgresql://username:password@localhost:5432/database" --asset "https://custom-assets.example.com"
```

### Command Options

- `--init`: Initialize the database before starting the server
- `--postgres`: PostgreSQL connection URI (required)
- `--port`: Port to run the dashboard server on (default: 55501)
- `--asset`: URL for dashboard assets (default: https://dash-assets.agentpod.ai) 