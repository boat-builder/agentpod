// Package session provides the Session struct for per-conversation state,
// along with methods for handling user messages and producing agent outputs.
package session

import (
	"github.com/boat-builder/agentpod/agent"
	"github.com/boat-builder/agentpod/memory"
	"github.com/openai/openai-go"
)

// Session holds ephemeral conversation data & references to global resources.
type Session struct {
	userID    string
	sessionID string

	llmClient openai.Client
	mem       memory.Memory
	agent     *agent.Agent

	// Fields for conversation history, ephemeral context, partial results, etc.
}

// NewSession constructs a session with references to shared LLM & memory, but isolated state.
func NewSession(userID, sessionID string, llmClient openai.Client, mem memory.Memory, ag *agent.Agent) *Session {
	return &Session{
		userID:    userID,
		sessionID: sessionID,
		llmClient: llmClient,
		mem:       mem,
		agent:     ag,
	}
}

// In processes incoming user messages. Could queue or immediately handle them.
func (s *Session) In(userMessage string) {
	// Implementation: store message in ephemeral convo history, or pass to agent
}

// Out retrieves the next message from the session (blocking or polling).
// Could represent a final or intermediate response from the agent.
func (s *Session) Out() *Message {
	// Stub: return next queued or computed response
	return nil
}

// Close ends the session lifecycle and releases any resources if needed.
func (s *Session) Close() {
	// Implementation: flush conversation history, finalize logs, etc.
}
