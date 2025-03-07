# AgentPod

AgentPod was started as a simple agent framework in python which is being rewritten in Go for the sole purpose of using it in [Agent Berlin](https://agentberlin.ai). The upside of that is, it's been used in production handling scale and is battle tested. The downside is that it's very strict in the design and only has the features we needed for Berlin to work. This might change in the future but that's what it is now.

## Using AgentPod

AgentPod can be used in two ways:

1. **As a framework**: Import and use the AgentPod framework in your Go project
2. **As a dashboard**: Use the CLI to start the dashboard for no-code users

### CLI

AgentPod includes a CLI for starting the dashboard:

```bash
# Start the dashboard with a PostgreSQL connection
agentpod start --postgres "postgresql://username:password@localhost:5432/database"
```

For more details on the CLI, see the [CLI documentation](cmd/agentpod/README.md).
