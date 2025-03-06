package prompts

// SystemPromptData contains data for the system prompt template.
type SkillContextRunnerPromptData struct {
	UserSystemPrompt  string
	SkillSystemPrompt string
	MemoryBlocks      map[string]string
}

// SkillSelectionPromptTemplate is the template for skill selection prompts.
const SkillContextRunnerPromptTemplate = `
{{ .UserSystemPrompt }}

{{ .SkillSystemPrompt }}


{{ formatMemoryBlocks .MemoryBlocks }}`

// SkillSelectionPrompt creates the skill selection prompt by applying the provided data.
func SkillContextRunnerPrompt(data SkillContextRunnerPromptData) (string, error) {
	return generateFromTemplate(SkillContextRunnerPromptTemplate, data)
}
