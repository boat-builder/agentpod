// Start of Selection
package agentpod

import (
	"fmt"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/param"
)

// TODO Remove all three and use openai functions directly
func UserMessage(content string) openai.ChatCompletionMessageParamUnion {
	return openai.UserMessage(content)
}

func AssistantMessage(content string) openai.ChatCompletionMessageParamUnion {
	return openai.AssistantMessage(content)
}

func DeveloperMessage(content string) openai.ChatCompletionMessageParamUnion {
	return openai.DeveloperMessage(content)
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

func (ml *MessageList) Len() int {
	return len(ml.Messages)
}

// Add appends one or more new messages to the MessageList in a FIFO order.
func (ml *MessageList) Add(msgs ...openai.ChatCompletionMessageParamUnion) {
	ml.Messages = append(ml.Messages, msgs...)
}

// AddFirstDeveloperMessage prepends a developer message to the message list.
// It panics if the provided message is not a developer message.
func (ml *MessageList) AddFirstDeveloperMessage(msg openai.ChatCompletionMessageParamUnion) {
	if msg.OfDeveloper == nil {
		panic("AddFirstDeveloperMessage expects a DeveloperMessage")
	}
	ml.Messages = append([]openai.ChatCompletionMessageParamUnion{msg}, ml.Messages...)
}

func (ml *MessageList) ReplaceAt(index int, newMsg openai.ChatCompletionMessageParamUnion) error {
	if index < 0 || index >= len(ml.Messages) {
		return fmt.Errorf("index out of range")
	}
	ml.Messages[index] = newMsg
	return nil
}

func (ml *MessageList) All() []openai.ChatCompletionMessageParamUnion {
	return ml.Messages
}

// CloneWithoutDeveloperMessages returns a copy of the MessageList that
// excludes any developer or system messages, preserving the original order of
// the remaining messages. This is useful when sending conversation history
// back to the LLM, where developer/system prompts should not be repeated.
func (ml *MessageList) CloneWithoutDeveloperMessages() *MessageList {
	filtered := make([]openai.ChatCompletionMessageParamUnion, 0, len(ml.Messages))
	for _, msg := range ml.Messages {
		if msg.OfDeveloper == nil && msg.OfSystem == nil {
			filtered = append(filtered, msg)
		}
	}
	return &MessageList{Messages: filtered}
}

func (ml *MessageList) Clear() {
	ml.Messages = []openai.ChatCompletionMessageParamUnion{}
}

// PrintMessages is for debugging purposes
func (ml *MessageList) PrintMessages() {
	for _, msg := range ml.Messages {
		role := "unknown"
		content := ""

		switch {
		case msg.OfUser != nil:
			role = "user"
			if !param.IsOmitted(msg.OfUser.Content.OfString) {
				content = msg.OfUser.Content.OfString.Value
			}
		case msg.OfAssistant != nil:
			role = "assistant"
			if !param.IsOmitted(msg.OfAssistant.Content.OfString) {
				content = msg.OfAssistant.Content.OfString.Value
			}
			// Print tool calls if they exist
			if len(msg.OfAssistant.ToolCalls) > 0 {
				content += "\nTool Calls:"
				for _, toolCall := range msg.OfAssistant.ToolCalls {
					content += fmt.Sprintf("\n- Function: %s", toolCall.Function.Name)
					content += fmt.Sprintf("\n  Arguments: %s", toolCall.Function.Arguments)
				}
			}
		case msg.OfDeveloper != nil:
			role = "developer"
			if !param.IsOmitted(msg.OfDeveloper.Content.OfString) {
				content = msg.OfDeveloper.Content.OfString.Value
			}
		case msg.OfTool != nil:
			role = "tool"
			if !param.IsOmitted(msg.OfTool.Content.OfString) {
				content = msg.OfTool.Content.OfString.Value
			}
		}

		fmt.Printf("Role: %s\nContent: %s\n\n", role, content)
	}
}
