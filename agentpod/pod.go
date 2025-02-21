package agentpod

import (
	"context"

	"github.com/boat-builder/agentpod/agent"
	"github.com/boat-builder/agentpod/agentpod/session"
	"github.com/boat-builder/agentpod/llm"
	"github.com/boat-builder/agentpod/memory"
)

type Agent = agent.Agent
type Skill = agent.Skill
type Tool = agent.Tool
type LLMConfig = llm.LLMConfig
type Memory = memory.Memory

type Pod struct {
	llmConfig *LLMConfig
	Mem       Memory
	Agent     *Agent
}

// NewPod constructs a new Pod with the given resources.
func NewPod(llmConfig *LLMConfig, mem Memory, ag *Agent) *Pod {
	return &Pod{
		llmConfig: llmConfig,
		Mem:       mem,
		Agent:     ag,
	}
}

// NewSession creates a new conversation session for a given user and session ID.
// A session handles a single user message and maintains the internal state of the agents
// as they interact to generate a response.
func (p *Pod) NewSession(ctx context.Context, userID, sessionID string) *session.Session {
	return session.NewSession(ctx, userID, sessionID, *p.llmConfig, p.Mem, p.Agent)
}

// NewAgent constructs a new Agent with the given LLM client and skills.
func NewAgent(prompt string, skills []Skill) *Agent {
	return agent.NewAgent(prompt, skills)
}

// NewSkill constructs a new Skill with the given name, description, and tools.
func NewSkill(name, description string, tools []Tool) *Skill {
	return agent.NewSkill(name, description, tools)
}

// NewMemory constructs a new Zep memory implementation.
func NewMemory() Memory {
	return memory.NewZep()
}
