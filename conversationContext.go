package agentpod

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/openai/openai-go"
)

const PromptFindRelevantMessages = `Identify the IDs of messages from the message history that provide context for the user's latest message. Strictly adhere to identifying only directly or indirectly referred messages in the latest interaction. If the latest message does not directly reference any prior conversation, return an empty array.

- **Input Data**: 
  - <ConversationHistory>: A series of conversations, each wrapped in <Conversation ID=X></Conversation> tags, where X is the conversation ID.
  - <LatestMessage>: The current message from one user.
- **Goal**: List the IDs of past conversations relevant for understanding the latest message without assuming relevance based on similarity.

# Output Format
- **Format**: JSON object
  - messageIDs: A list of relevant conversation IDs, e.g., ["2"] or [] if no relevant conversations are found.

# Examples

**Example 1**

- **Input**:  
  <ConversationHistory>
  <Conversation ID=1>
  Human: Hi, how are you?
  Assistant: I'm doing well, thank you. How can I help you today?
  </Conversation>

  <Conversation ID=2>
  Human: What's the delivery time for the order?
  Assistant: Delivery typically takes 3-5 business days. Do you have a specific order you're inquiring about?
  </Conversation>

  <Conversation ID=3>
  Human: Did you receive my payment?
  Assistant: Yes, we've received your payment. Thank you for confirming.
  </Conversation>
  </ConversationHistory>
  
  <LatestMessage>
  Human: When will my order arrive?
  </LatestMessage>
  

- **Output**:
  - messageIDs: ["2"]

**Example 2**

- **Input**:  
  <ConversationHistory>
  <Conversation ID=1>
  Human: Do you have any pets?
  Assistant: No, I don't have any pets. I'm an AI assistant.
  </Conversation>

  <Conversation ID=2>
  Human: Where are you from?
  Assistant: I was created by Anthropic, but I don't have a physical location.
  </Conversation>
  </ConversationHistory>
  
  <LatestMessage>
  Human: Let's meet for coffee!
  </LatestMessage>
  

- **Output**: 
  - messageIDs: []

# Notes
- Focus on relevance strictly based on explicit references in the latest message.
- Avoid assumptions based on conversational similarity without explicit links.
`

type RelevantMessageIDs struct {
	MessageIDs []string `json:"messageIDs"`
}

// GetMessageText extracts the plain text content from an OpenAI chat message
// of any type (user, assistant, or developer message)
func GetMessageText(message openai.ChatCompletionMessageParamUnion) (string, error) {
	switch m := message.(type) {
	case openai.ChatCompletionUserMessageParam:
		// User message - extract text from content parts
		if len(m.Content.Value) == 0 {
			return "", fmt.Errorf("user message content is empty")
		}

		var builder strings.Builder
		for _, part := range m.Content.Value {
			if textPart, ok := part.(openai.ChatCompletionContentPartTextParam); ok {
				builder.WriteString(textPart.Text.Value)
			}
		}
		return builder.String(), nil

	case openai.ChatCompletionAssistantMessageParam:
		// Assistant message - extract text from content parts
		if len(m.Content.Value) == 0 {
			return "", fmt.Errorf("assistant message content is empty")
		}

		var builder strings.Builder
		for _, part := range m.Content.Value {
			if textPart, ok := part.(openai.ChatCompletionContentPartTextParam); ok {
				builder.WriteString(textPart.Text.Value)
			}
		}
		return builder.String(), nil

	case openai.ChatCompletionToolMessageParam:
		// Tool message - extract text from content parts
		if len(m.Content.Value) == 0 {
			return "", fmt.Errorf("tool message content is empty")
		}

		var builder strings.Builder
		for _, part := range m.Content.Value {
			builder.WriteString(part.Text.Value)
		}
		return builder.String(), nil

	case openai.ChatCompletionMessage:
		// Direct Message type - may have direct content
		return m.Content, nil

	default:
		return "", fmt.Errorf("unsupported message type: %T", message)
	}
}

// FormatMessageList formats all messages in a MessageList into a structured string format
// showing conversations where each user message and subsequent assistant message(s) are
// wrapped in <Conversation ID=X> tags, and the entire history is wrapped in <ConversationHistory> tags.
// After the history, the current message is included within <LatestMessage> tags
func FormatMessageList(messages *MessageList, currentMessage openai.ChatCompletionMessageParamUnion) (string, error) {
	var result strings.Builder

	// Start the conversation history tag
	fmt.Fprintf(&result, "<ConversationHistory>\n")

	conversationID := 1
	inConversation := false

	for i := 0; i < len(messages.Messages); i++ {
		msg := messages.Messages[i]

		// Determine the user type based on message type
		userType := ""
		switch msg.(type) {
		case openai.ChatCompletionUserMessageParam:
			userType = "Human"
		case openai.ChatCompletionAssistantMessageParam:
			userType = "Assistant"
		default:
			userType = "Unknown"
		}

		// Get message content
		content, err := GetMessageText(msg)
		if err != nil {
			return "", fmt.Errorf("error getting message %d content: %w", i, err)
		}

		// Start a new conversation when we find a human message
		if userType == "Human" && !inConversation {
			fmt.Fprintf(&result, "<Conversation ID=%d>\n", conversationID)
			inConversation = true
		}

		// Format the message
		fmt.Fprintf(&result, "%s: %s\n", userType, content)

		// Close the conversation if this is an assistant message and the next message (if any) is from a human
		if userType == "Assistant" && inConversation {
			// Check if next message exists and is from a human
			if i+1 < len(messages.Messages) {
				nextMsg := messages.Messages[i+1]
				_, isUserMessage := nextMsg.(openai.ChatCompletionUserMessageParam)

				if isUserMessage {
					// Close current conversation and increment ID
					fmt.Fprintf(&result, "</Conversation>\n\n")
					inConversation = false
					conversationID++
				}
			} else {
				// This is the last message in the list, close the conversation
				fmt.Fprintf(&result, "</Conversation>\n\n")
				inConversation = false
			}
		}
	}

	// Ensure we close any open conversation at the end
	if inConversation {
		fmt.Fprintf(&result, "</Conversation>\n\n")
	}

	// Close the conversation history tag
	fmt.Fprintf(&result, "</ConversationHistory>\n")

	// Add the current message in a LatestMessage tag
	if currentMessage != nil {
		fmt.Fprintf(&result, "\n<LatestMessage>\n")

		// Determine the user type for the current message
		userType := ""
		switch currentMessage.(type) {
		case openai.ChatCompletionUserMessageParam:
			userType = "Human"
		case openai.ChatCompletionAssistantMessageParam:
			userType = "Assistant"
		default:
			userType = "Unknown"
		}

		// Get the content of the current message
		content, err := GetMessageText(currentMessage)
		if err != nil {
			return "", fmt.Errorf("error getting current message content: %w", err)
		}

		// Format the current message
		fmt.Fprintf(&result, "%s: %s\n", userType, content)
		fmt.Fprintf(&result, "</LatestMessage>\n")
	}

	return result.String(), nil
}

// BuildRelevantMessageHistory uses an LLM to identify and extract the most relevant portion
// of a conversation history for the current context.
//
// The function works by:
// 1. Formatting the entire message history into a structured text format
// 2. Sending this history to the LLM with a prompt to identify relevant message IDs
// 3. Finding the oldest (smallest ID) message among the identified relevant messages
// 4. Creating a new MessageList containing all messages from that oldest relevant message onwards
//
// This approach helps maintain conversation coherence while reducing context length by
// removing older irrelevant messages, which optimizes token usage and improves response quality
// by focusing the model on the most pertinent information.
func BuildRelevantMessageHistory(ctx context.Context, messages MessageList, currentMessage openai.ChatCompletionMessageParamUnion, llmClient *LLM, modelName string) (MessageList, error) {
	mergedHistoryString, err := FormatMessageList(&messages, currentMessage)
	if err != nil {
		return MessageList{}, err
	}

	schemaParam := openai.ResponseFormatJSONSchemaJSONSchemaParam{
		Name:        openai.F("relevantMessageIDs"),
		Description: openai.F("List of message IDs that are relevant to the current conversation"),
		Schema:      openai.F(GenerateSchema[RelevantMessageIDs]()),
		Strict:      openai.Bool(true),
	}

	completion, err := llmClient.New(ctx, openai.ChatCompletionNewParams{
		Messages: openai.F([]openai.ChatCompletionMessageParamUnion{
			DeveloperMessage(PromptFindRelevantMessages),
			UserMessage(mergedHistoryString),
		}),
		Model: openai.F(modelName),
		ResponseFormat: openai.F[openai.ChatCompletionNewParamsResponseFormatUnion](
			openai.ResponseFormatJSONSchemaParam{
				Type:       openai.F(openai.ResponseFormatJSONSchemaTypeJSONSchema),
				JSONSchema: openai.F(schemaParam),
			},
		),
	})
	if err != nil {
		return MessageList{}, err
	}

	relevantMessageIDs := RelevantMessageIDs{}
	err = json.Unmarshal([]byte(completion.Choices[0].Message.Content), &relevantMessageIDs)
	if err != nil {
		return MessageList{}, err
	}

	// Find the smallest (oldest) message ID from relevantMessageIDs
	if len(relevantMessageIDs.MessageIDs) == 0 {
		// No relevant messages found, return empty list
		return MessageList{}, nil
	}

	// Convert string IDs to integers
	var messageIndices []int
	for _, idStr := range relevantMessageIDs.MessageIDs {
		var id int
		_, err := fmt.Sscanf(idStr, "%d", &id)
		if err != nil {
			return MessageList{}, fmt.Errorf("error parsing message ID '%s': %w", idStr, err)
		}
		messageIndices = append(messageIndices, id)
	}

	// Find the smallest index
	smallestIndex := messageIndices[0]
	for _, idx := range messageIndices {
		if idx < smallestIndex {
			smallestIndex = idx
		}
	}

	// Create a new MessageList with all messages from smallestIndex onwards
	result := MessageList{
		Messages: make([]openai.ChatCompletionMessageParamUnion, 0),
	}

	// Include all messages from smallestIndex to the end
	if smallestIndex >= 0 && smallestIndex < len(messages.Messages) {
		result.Messages = messages.Messages[smallestIndex:]
	}

	return result, nil
}

// CompileConversationHistory builds the message history for the LLM request
// now it fetches the last 5 messages but in the future, we'lll do smart things here like old message summarization etc
func CompileConversationHistory(meta Meta, storage Storage) (*MessageList, error) {
	return storage.GetConversations(meta, 5, 0)
}
