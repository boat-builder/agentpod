// Start of Selection
package llm

import "github.com/openai/openai-go"

type LLMConfig struct {
	APIKey  string
	BaseURL string
	Model   string
}

// LLMMessage is a shared interface for any message type we might have.
type LLMMessage interface {
	OpenAIMessage() openai.ChatCompletionMessageParamUnion
}

// DeveloperMessage, SystemMessage, and UserMessage all implement LLMMessage.

// DeveloperMessage represents a developer-level instruction or content.
type DeveloperMessage struct {
	Content string
}

func (d *DeveloperMessage) OpenAIMessage() openai.ChatCompletionMessageParamUnion {
	return openai.ChatCompletionDeveloperMessageParam{
		Role: openai.F(openai.ChatCompletionDeveloperMessageParamRoleDeveloper),
		Content: openai.F([]openai.ChatCompletionContentPartTextParam{
			openai.TextPart(d.Content),
		}),
	}
}

// SystemMessage represents a system-level instruction or content.
type SystemMessage struct {
	Content string
}

func (s *SystemMessage) OpenAIMessage() openai.ChatCompletionMessageParamUnion {
	return openai.SystemMessage(s.Content)
}

// UserMessage represents a user-level instruction or content.
type UserMessage struct {
	Content string
}

func (u *UserMessage) OpenAIMessage() openai.ChatCompletionMessageParamUnion {
	return openai.UserMessage(u.Content)
}

// MessageList holds an ordered collection of LLMMessage to preserve the history.
type MessageList struct {
	Messages []LLMMessage
}

// Add appends a new message to the MessageList in a FIFO order.
func (ml *MessageList) Add(msg LLMMessage) {
	ml.Messages = append(ml.Messages, msg)
}

func (ml *MessageList) OpenAIMessages() []openai.ChatCompletionMessageParamUnion {
	messages := make([]openai.ChatCompletionMessageParamUnion, len(ml.Messages))
	for i, msg := range ml.Messages {
		messages[i] = msg.OpenAIMessage()
	}
	return messages
}
