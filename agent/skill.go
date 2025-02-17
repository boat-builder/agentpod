// Package agent - skill.go
// Defines the Skill structure, grouping Tools and domain-specific logic.

package agent

// Skill holds a set of tools and a domain-specific prompt/description.
type Skill struct {
	Name        string
	Description string // could act as a system/context prompt
	Tools       []Tool
}

// NewSkill constructs a skill with name, description, and a set of tools.
func NewSkill(name, description string, tools []Tool) *Skill {
	return &Skill{
		Name:        name,
		Description: description,
		Tools:       tools,
	}
}
