package agentpod

import (
	"github.com/boat-builder/agentpod/agent"
	"github.com/boat-builder/agentpod/agentpod/session"
	"github.com/boat-builder/agentpod/llm"
	"github.com/boat-builder/agentpod/memory"
	"github.com/openai/openai-go"
)

type Agent = agent.Agent
type Skill = agent.Skill
type Tool = agent.Tool
type LLM = llm.LLM
type Memory = memory.Memory

type Pod struct {
	llm *LLM
	Mem Memory
	AI  *Agent
}

// NewPod constructs a new Pod with the given resources.
func NewPod(llm *LLM, mem Memory, ai *Agent) *Pod {
	return &Pod{
		llm: llm,
		Mem: mem,
		AI:  ai,
	}
}

// NewSession creates a new conversation session for a given user and session ID.
// A session handles a single user message and maintains the internal state of the agents
// as they interact to generate a response.
func (p *Pod) NewSession(userID, sessionID string) *session.Session {
	return session.NewSession(userID, sessionID, p.llm, p.Mem, p.AI)
}

// NewAgent constructs a new Agent with the given LLM client and skills.
func NewAgent(llmClient *openai.Client, skills []Skill) *Agent {
	return agent.NewAgent(llmClient, skills)
}

// NewSkill constructs a new Skill with the given name, description, and tools.
func NewSkill(name, description string, tools []Tool) *Skill {
	return agent.NewSkill(name, description, tools)
}

// NewTool constructs a new BasicTool with the given tool name.
func NewTool(toolName string) Tool {
	return agent.NewBasicTool(toolName)
}

// NewMemory constructs a new Zep memory implementation.
func NewMemory() Memory {
	return memory.NewZep()
}
