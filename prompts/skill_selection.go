package prompts

import (
	"strings"
)

// SkillSelectionPromptData contains data for the system prompt template.
type SkillSelectionPromptData struct {
	MainAgentSystemPrompt string
	SkillFunctions        []string
	MemoryBlocks          string
}

// SkillSelectionPromptTemplate is the template for skill selection prompts.
const SkillSelectionPromptTemplate = `
{{ .MainAgentSystemPrompt }}

You should use skill functions to attend to the user's request. Skill functions are intelligent to understand instructions but be very succinct and clear about instructions you pass to it.

{{ formatSkillFunctions .SkillFunctions }}

All the memory learned from user's previous interactions are provided below. Use it as the context to answer the user's question.

{{ .MemoryBlocks }}`

// SkillSelectionPrompt creates the skill selection prompt by applying the provided data.
func SkillSelectionPrompt(data SkillSelectionPromptData) (string, error) {
	return generateFromTemplate(SkillSelectionPromptTemplate, data)
}

// formatSkillFunctions formats the skill functions as a comma-separated string.
func formatSkillFunctions(skillFunctions []string) string {
	return strings.Join(skillFunctions, ", ")
}
