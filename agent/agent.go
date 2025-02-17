// Package agent provides the main Agent orchestrator, which uses LLM & Skills to process data.
package agent

import (
	"github.com/boat-builder/agentpod/llm"
)

// Agent orchestrates calls to the LLM, uses Skills/Tools, and determines how to respond.
type Agent struct {
	llmClient llm.LLM
	skills    []Skill
	// Additional config or system prompts, etc.
}

// NewAgent creates an Agent with the given LLM and optional skill set.
func NewAgent(llmClient llm.LLM, skills []Skill) *Agent {
	return &Agent{
		llmClient: llmClient,
		skills:    skills,
	}
}

// Process is a stub method that might handle a single chunk of context/input
// and produce a response, possibly invoking Skills/Tools.
func (a *Agent) Process(context string) (string, error) {
	// Implementation stub
	return "", nil
}
