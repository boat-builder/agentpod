package prompts

import (
	"bytes"
	"fmt"
	"strings"
	"text/template"
)

// SystemPromptData contains data for the system prompt template.
type SystemPromptData struct {
	UserPrompt     string
	MemoryBlocks   map[string]string
	SkillFunctions []string
}

// SkillSelectionPromptTemplate is the template for skill selection prompts.
const SkillSelectionPromptTemplate = `
{{ .UserInstructions }}

You can use skill functions {{ formatSkillFunctions .SkillFunctions }} to help you answer user's question. Do not choose same skill more than once. Skill functions can internally handle the need for multiple instances of the same skill.


{{ formatMemoryBlocks .MemoryBlocks }}`

// SkillSelectionPrompt creates the skill selection prompt by applying the provided data.
func SkillSelectionPrompt(data SystemPromptData) (string, error) {
	return generateFromTemplate(SkillSelectionPromptTemplate, data)
}

// GenerateFromTemplate is a generic function that generates a prompt from any template and data.
func generateFromTemplate[T any](templateString string, data T) (string, error) {
	funcMap := template.FuncMap{
		"formatMemoryBlocks":   formatMemoryBlocks,
		"formatSkillFunctions": formatSkillFunctions,
	}

	tmpl, err := template.New("prompt").Funcs(funcMap).Parse(templateString)
	if err != nil {
		return "", err
	}
	var prompt bytes.Buffer
	if err := tmpl.Execute(&prompt, data); err != nil {
		return "", err
	}
	return prompt.String(), nil
}

// formatMemoryBlocks formats the memory blocks as key-value pairs within UserDetails tags.
func formatMemoryBlocks(memoryBlocks map[string]string) string {
	if len(memoryBlocks) == 0 {
		return ""
	}

	var builder strings.Builder
	builder.WriteString("<UserDetails>\n")

	for key, value := range memoryBlocks {
		builder.WriteString(fmt.Sprintf("%s: %s\n", key, value))
	}

	builder.WriteString("</UserDetails>")
	return builder.String()
}

// formatSkillFunctions formats the skill functions as a comma-separated string.
func formatSkillFunctions(skillFunctions []string) string {
	return strings.Join(skillFunctions, ", ")
}
