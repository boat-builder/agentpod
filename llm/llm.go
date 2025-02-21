// Start of Selection
package llm

import "github.com/openai/openai-go"

type LLMConfig struct {
	APIKey  string
	BaseURL string
	Model   string
}

// We have custom UserMessage/AssistantMessage/DeveloperMessage because openai go sdk currently have openai.DeveloperMessage()
func UserMessage(content string) openai.ChatCompletionMessageParamUnion {
	return openai.UserMessage(content)
}

func AssistantMessage(content string) openai.ChatCompletionMessageParamUnion {
	return openai.AssistantMessage(content)
}

func DeveloperMessage(content string) openai.ChatCompletionMessageParamUnion {
	return openai.ChatCompletionDeveloperMessageParam{
		Role: openai.F(openai.ChatCompletionDeveloperMessageParamRoleDeveloper),
		Content: openai.F([]openai.ChatCompletionContentPartTextParam{
			openai.TextPart(content),
		}),
	}
}

// MessageList holds an ordered collection of LLMMessage to preserve the history.
type MessageList struct {
	Messages []openai.ChatCompletionMessageParamUnion
}

func NewMessageList() *MessageList {
	return &MessageList{
		Messages: []openai.ChatCompletionMessageParamUnion{},
	}
}

// Add appends a new message to the MessageList in a FIFO order.
func (ml *MessageList) Add(msg openai.ChatCompletionMessageParamUnion) {
	ml.Messages = append(ml.Messages, msg)
}

func (ml *MessageList) All() []openai.ChatCompletionMessageParamUnion {
	return ml.Messages
}

func (ml *MessageList) Clone() *MessageList {
	return &MessageList{
		Messages: append([]openai.ChatCompletionMessageParamUnion{}, ml.Messages...),
	}
}
