// Package agent - tool.go
// Defines the Tool interface and basic stubs for tool usage.
package agentpod

import "github.com/openai/openai-go"

type Tool interface {
	Name() string
	Description() string
	OpenAI() []openai.ChatCompletionToolParam
	Execute(args map[string]interface{}) (string, error)
}
