package agentpod

import (
	"context"
	"fmt"
	"strings"

	"github.com/openai/openai-go"
)

// conversationContext uses LLM to look at the conversation history and extract the context
// for the conversation with only the needed information
func ConversationContext(getConversationHistory func(ctx context.Context, session *Session, limit int, offset int) (MessageList, error), currentMessage openai.ChatCompletionMessageParamUnion, llmClient *LLM, modelName string, session *Session) string {

	developerMessage := DeveloperMessage(`You are tasked to look at a conversation history and extract the context for the conversation for the given user message. 
If you think the conversation history doesn't have enough information, you can call getConversationHistory tool with right 
parameters to get more history.`)

	previous5, err := getConversationHistory(context.Background(), session, 6, 1)
	if err != nil {
		return ""
	}

	// Get information about the previous5 to satisfy the linter
	msgCount := previous5.Len()
	if msgCount == 0 {
		return getMessageContentAsString(currentMessage)
	}
	var builder strings.Builder
	for _, msg := range previous5.All() {
		builder.WriteString(getMessageContentAsString(msg))
		builder.WriteString("\n")
	}
	userMessage := UserMessage(
		fmt.Sprintf(
			"Get all relevant informattion for the user message from the conversation history.\n\n<ConversationHistory>%s</ConversationHistory>\n\n<UserMessage>%s</UserMessage>",
			builder.String(),
			getMessageContentAsString(currentMessage),
		),
	)
	completion, err := llmClient.New(context.Background(), openai.ChatCompletionNewParams{
		Messages: openai.F([]openai.ChatCompletionMessageParamUnion{developerMessage, userMessage}),
		Model:    openai.F(modelName),
	})
	if err != nil {
		return ""
	}

	return completion.Choices[0].Message.Content

}

// getMessageContentAsString extracts the content from a message as a string
func getMessageContentAsString(message openai.ChatCompletionMessageParamUnion) string {
	switch m := message.(type) {
	case openai.ChatCompletionUserMessageParam:
		// Try to extract text content from user message
		if len(m.Content.Value) > 0 {
			return fmt.Sprintf("User1: %s", extractUserMessageText(m))
		}
	case openai.ChatCompletionAssistantMessageParam:
		// For assistant message, it's likely a string
		if len(m.Content.Value) > 0 {
			return fmt.Sprintf("User2: %s", extractAssistantMessageText(m))
		}
	}
	return ""
}

// Helper functions to extract text from different message types
func extractUserMessageText(message openai.ChatCompletionUserMessageParam) string {
	for _, part := range message.Content.Value {
		if textPart, ok := part.(openai.ChatCompletionContentPartTextParam); ok {
			return textPart.Text.Value
		}
	}
	return ""
}

func extractAssistantMessageText(message openai.ChatCompletionAssistantMessageParam) string {
	var builder strings.Builder
	for _, part := range message.Content.Value {
		if textPart, ok := part.(openai.ChatCompletionContentPartTextParam); ok {
			builder.WriteString(textPart.Text.Value)
		}
	}
	return builder.String()
}
