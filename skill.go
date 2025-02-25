// Package agent - skill.go
// Defines the Skill structure, grouping Tools and domain-specific logic.

package agentpod

import (
	"fmt"

	"github.com/openai/openai-go"
)

// Skill holds a set of tools and a domain-specific prompt/description.
type Skill struct {
	Name          string
	Description   string
	SystemPrompt  string
	StatusMessage string
	Tools         []Tool
}

// NewSkill constructs a skill with name, description, and a set of tools.
func NewSkill(name, description string, tools []Tool) *Skill {
	return &Skill{
		Name:        name,
		Description: description,
		Tools:       tools,
	}
}

func (s *Skill) GetTools() []openai.ChatCompletionToolParam {
	tools := []openai.ChatCompletionToolParam{}
	for _, tool := range s.Tools {
		tools = append(tools, tool.OpenAI()...)
	}
	return tools
}

func (s *Skill) GetTool(name string) (Tool, error) {
	for _, tool := range s.Tools {
		if tool.Name() == name {
			return tool, nil
		}
	}
	return nil, fmt.Errorf("tool %s not found", name)
}
