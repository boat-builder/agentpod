package agentpod

import (
	"github.com/boat-builder/agentpod/agent"
	"github.com/boat-builder/agentpod/agentpod/session"
	"github.com/boat-builder/agentpod/memory"
	"github.com/openai/openai-go"
)

type Agent = agent.Agent
type Skill = agent.Skill
type Tool = agent.Tool
type LLM = openai.Client
type Memory = memory.Memory

type Pod struct {
	LLM LLM
	Mem Memory
	AI  *Agent
	// Additional fields: config, logging, etc.
}

// NewPod constructs a new Pod with the given resources.
func NewPod(llmClient LLM, mem Memory, ai *Agent) *Pod {
	return &Pod{
		LLM: llmClient,
		Mem: mem,
		AI:  ai,
	}
}

// NewSession creates a new conversation session for a given user & session ID.
// The session can use references to the LLM & memory in a thread-safe manner.
func (p *Pod) NewSession(userID, sessionID string) *session.Session {
	// Create session with ephemeral conversation state (but referencing global LLM/mem).
	return session.NewSession(userID, sessionID, p.LLM, p.Mem, p.AI)
}
