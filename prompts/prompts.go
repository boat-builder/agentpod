package prompts

import (
	"bytes"
	"fmt"
	"strings"
	"text/template"
)

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
