package agentpod

import (
	"context"
	"fmt"
	"strings"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/param"
)

// GetMessageText extracts the plain text content from an OpenAI chat message
// of any type (user, assistant, or developer message)
func GetMessageText(message openai.ChatCompletionMessageParamUnion) (string, error) {
	switch {
	case message.OfUser != nil:
		m := message.OfUser
		content := m.Content
		if param.IsOmitted(content.OfString) && len(content.OfArrayOfContentParts) == 0 {
			return "", fmt.Errorf("user message content is empty")
		}
		if !param.IsOmitted(content.OfString) {
			return content.OfString.Value, nil
		}
		var builder strings.Builder
		for _, part := range content.OfArrayOfContentParts {
			builder.WriteString(*part.GetText())
		}
		return builder.String(), nil

	case message.OfAssistant != nil:
		m := message.OfAssistant
		content := m.Content
		if param.IsOmitted(content.OfString) && len(content.OfArrayOfContentParts) == 0 {
			return "", fmt.Errorf("assistant message content is empty")
		}
		if !param.IsOmitted(content.OfString) {
			return content.OfString.Value, nil
		}
		var builder strings.Builder
		for _, part := range content.OfArrayOfContentParts {
			builder.WriteString(*part.GetText())
		}
		return builder.String(), nil

	case message.OfTool != nil:
		m := message.OfTool
		content := m.Content
		if param.IsOmitted(content.OfString) && len(content.OfArrayOfContentParts) == 0 {
			return "", fmt.Errorf("tool message content is empty")
		}
		if !param.IsOmitted(content.OfString) {
			return content.OfString.Value, nil
		}
		var builder strings.Builder
		for _, part := range content.OfArrayOfContentParts {
			builder.WriteString(part.Text)
		}
		return builder.String(), nil

	case message.OfDeveloper != nil:
		m := message.OfDeveloper
		content := m.Content
		if param.IsOmitted(content.OfString) && len(content.OfArrayOfContentParts) == 0 {
			return "", fmt.Errorf("developer message content is empty")
		}
		if !param.IsOmitted(content.OfString) {
			return content.OfString.Value, nil
		}
		var builder strings.Builder
		for _, part := range content.OfArrayOfContentParts {
			builder.WriteString(part.Text)
		}
		return builder.String(), nil

	default:
		return "", fmt.Errorf("unsupported message type")
	}
}

// CompileConversationHistory builds the message history for the LLM request
// now it fetches the last 5 messages but in the future, we'lll do smart things here like old message summarization etc
func CompileConversationHistory(ctx context.Context, storage Storage) (*MessageList, error) {
	return storage.GetConversations(ctx, 5, 0)
}
