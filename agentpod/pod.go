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
type LLM = openai.Client // we might add other LLM providers in the future
type Memory = memory.Memory

type Pod struct {
	LLM LLM
	Mem Memory
	AI  *Agent
}

// NewPod constructs a new Pod with the given resources.
func NewPod(llmClient LLM, mem Memory, ai *Agent) *Pod {
	return &Pod{
		LLM: llmClient,
		Mem: mem,
		AI:  ai,
	}
}

// NewSession creates a new conversation session for a given user and session ID.
// A session handles a single user message and maintains the internal state of the agents
// as they interact to generate a response.
func (p *Pod) NewSession(userID, sessionID string) *session.Session {
	return session.NewSession(userID, sessionID, p.LLM, p.Mem, p.AI)
}
