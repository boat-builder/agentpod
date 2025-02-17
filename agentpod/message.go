package agentpod

import "time"

// AgentMessage represents a message exchanged by an agent.
type AgentMessage struct {
	ID        string
	Content   string
	Timestamp time.Time
	// add additional fields as needed
}
