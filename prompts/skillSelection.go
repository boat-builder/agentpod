package prompts

import (
	"strings"
)

// SkillSelectionPromptData contains data for the system prompt template.
type SkillSelectionPromptData struct {
	UserSystemPrompt string
	MemoryBlocks     map[string]string
	SkillFunctions   []string
}

// SkillSelectionPromptTemplate is the template for skill selection prompts.
const SkillSelectionPromptTemplate = `
{{ .UserSystemPrompt }}

You can use skill functions {{ formatSkillFunctions .SkillFunctions }} to help you answer user's question. Do not call same skill more than once. Skill functions can internally handle the need for multiple instances of the same skill.


{{ formatMemoryBlocks .MemoryBlocks }}`

// SkillSelectionPrompt creates the skill selection prompt by applying the provided data.
func SkillSelectionPrompt(data SkillSelectionPromptData) (string, error) {
	return generateFromTemplate(SkillSelectionPromptTemplate, data)
}

// formatSkillFunctions formats the skill functions as a comma-separated string.
func formatSkillFunctions(skillFunctions []string) string {
	return strings.Join(skillFunctions, ", ")
}
