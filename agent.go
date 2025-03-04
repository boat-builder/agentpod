// Package agent provides the main Agent orchestrator, which uses LLM & Skills to process data.
package agentpod

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"strings"
	"sync"
	"time"

	gonanoid "github.com/matoous/go-nanoid/v2"
	"github.com/openai/openai-go"
)

var ignErr *IgnorableError
var retErr *RetryableError

// Agent orchestrates calls to the LLM, uses Skills/Tools, and determines how to respond.
type Agent struct {
	prompt string
	skills []Skill
	logger *slog.Logger
}

// NewAgent creates an Agent by adding the prompt as a DeveloperMessage.
func NewAgent(prompt string, skills []Skill) *Agent {
	return &Agent{
		prompt: prompt,
		skills: skills,
		logger: slog.Default(),
	}
}

func (a *Agent) GetLogger() *slog.Logger {
	return a.logger
}

func (a *Agent) SetLogger(logger *slog.Logger) {
	a.logger = logger
}

func (a *Agent) GetSkill(name string) (*Skill, error) {
	for _, skill := range a.skills {
		if skill.Name == name {
			return &skill, nil
		}
	}
	return nil, fmt.Errorf("skill %s not found", name)
}

// buildDeveloperMessage takes the user given prompt and add the user information to it.
func (a *Agent) buildDeveloperMessage(prompt string, userInfo UserInfo) string {
	prompt = prompt + fmt.Sprintf("\n\nYou are talking to %s.", userInfo.Name)
	if len(userInfo.CustomMeta) > 0 {
		prompt += "\nHere is some information about them:\n"
		for key, value := range userInfo.CustomMeta {
			prompt += fmt.Sprintf("%s: %s\n", key, value)
		}
	}
	return prompt
}

// Run processes a user message through the LLM, executes any requested skills,
// and returns a channel of completion chunks.
func (a *Agent) Run(
	ctx context.Context,
	session *Session,
	llmClient *LLM,
	modelName string,
	send_status_func func(msg string),
	storage Storage,
) (chan openai.ChatCompletionChunk, error) {
	if a.logger == nil {
		panic("logger is not set")
	}
	outAgentChannel := make(chan openai.ChatCompletionChunk)

	// Validate session state
	if err := a.validateSessionState(session); err != nil {
		return nil, err
	}

	// Prepare session message history
	if err := a.prepareMessageHistory(ctx, session, storage); err != nil {
		return nil, err
	}

	// Process the request asynchronously
	go a.processRequest(ctx, session, llmClient, modelName, outAgentChannel, send_status_func)

	return outAgentChannel, nil
}

// validateSessionState ensures the session contains exactly one user message
func (a *Agent) validateSessionState(session *Session) error {
	// at this point, the conversation history can only contain one user message
	if session.State.MessageHistory.Len() != 1 {
		a.logger.Error("Conversation history can only contain one user message")
		return fmt.Errorf("conversation history can only contain one user message")
	}
	userMessage := session.State.MessageHistory.All()[0]
	if _, ok := userMessage.(openai.ChatCompletionUserMessageParam); !ok {
		a.logger.Error("Conversation history can only contain one user message")
		return fmt.Errorf("conversation history can only contain one user message")
	}
	return nil
}

// prepareMessageHistory builds the message history for the LLM request
func (a *Agent) prepareMessageHistory(
	ctx context.Context,
	session *Session,
	storage Storage,
) error {
	// Get the user message before clearing
	userMessage := session.State.MessageHistory.All()[0]

	// Clear the history and rebuild it
	session.State.MessageHistory.Clear()

	// Add the prompt to the message history
	userInfo, err := storage.GetUserInfo(session)
	if err != nil {
		a.logger.Error("Error getting user info", "error", err)
		return err
	}
	session.State.MessageHistory.AddFirst(a.buildDeveloperMessage(a.prompt, userInfo))

	// add the last 5 messages to the conversation history
	conversationHistory, err := storage.GetConversation(session, 5, 1)
	if err != nil {
		a.logger.Error("Error getting conversation history", "error", err)
		return err
	}
	for _, msg := range conversationHistory.All() {
		session.State.MessageHistory.Add(msg)
	}

	// adding the user message as the last message
	session.State.MessageHistory.Add(userMessage)
	return nil
}

// processRequest handles the main logic for processing a request
func (a *Agent) processRequest(
	ctx context.Context,
	session *Session,
	llmClient *LLM,
	modelName string,
	outAgentChannel chan openai.ChatCompletionChunk,
	send_status_func func(msg string),
) {
	defer close(outAgentChannel)

	// Clone the messages before appending the latest assistant message
	clonedMessages := session.State.MessageHistory.Clone()

	// Get initial response from LLM
	completion, err := a.getInitialLLMResponse(ctx, session, llmClient, modelName, outAgentChannel)
	if err != nil {
		a.handleLLMError(err, modelName, outAgentChannel)
		return
	}

	// If no tool calls were requested, we're done
	if completion.Choices[0].Message.ToolCalls == nil {
		return
	}

	// Process tool calls if requested
	results := a.processToolCalls(ctx, completion, clonedMessages, llmClient, modelName, outAgentChannel, send_status_func)

	// Handle results based on number of tool calls
	a.handleToolCallResults(completion, results, session, llmClient, modelName, outAgentChannel)
}

// getInitialLLMResponse gets the initial response from the LLM
func (a *Agent) getInitialLLMResponse(
	ctx context.Context,
	session *Session,
	llmClient *LLM,
	modelName string,
	outAgentChannel chan openai.ChatCompletionChunk,
) (openai.ChatCompletionAccumulator, error) {
	params := openai.ChatCompletionNewParams{
		Messages: openai.F(session.State.MessageHistory.All()),
		Model:    openai.F(modelName),
		StreamOptions: openai.F(openai.ChatCompletionStreamOptionsParam{
			IncludeUsage: openai.F(true),
		}),
	}
	if len(a.ConvertSkillsToTools()) > 0 {
		params.Tools = openai.F(a.ConvertSkillsToTools())
	}

	stream := llmClient.NewStreaming(ctx, params)
	defer stream.Close()

	completion := openai.ChatCompletionAccumulator{}
	for stream.Next() {
		chunk := stream.Current()
		if chunk.Choices[0].Delta.Content != "" {
			outAgentChannel <- chunk
		}
		completion.AddChunk(chunk)
		if _, finished := completion.JustFinishedContent(); finished {
			break
		}
	}

	if stream.Err() != nil {
		return completion, stream.Err()
	}

	if len(completion.Choices) == 0 {
		a.logger.Error("No completion choices")
		return completion, fmt.Errorf("no completion choices")
	}

	// Check if both tool call and content are non-empty
	bothToolCallAndContent := completion.Choices[0].Message.ToolCalls != nil && completion.Choices[0].Message.Content != ""
	if bothToolCallAndContent {
		a.logger.Error("Expectation is that both tool call and content are both non-empty")
	}

	// Append the message to our messages
	session.State.MessageHistory.Add(completion.Choices[0].Message)

	return completion, nil
}

// handleLLMError handles errors from LLM API calls
func (a *Agent) handleLLMError(err error, modelName string, outAgentChannel chan openai.ChatCompletionChunk) {
	content := "Error occurred!"
	a.logger.Error("Error streaming", "error", err)
	if strings.Contains(err.Error(), "ContentPolicyViolationError") {
		a.logger.Error("Content policy violation!", "error", err)
		content = "Content policy violation! If this was a mistake, please reach out to the support. Consecutive violations may result in a temporary/permanent ban."
	}
	id, _ := gonanoid.New()
	outAgentChannel <- openai.ChatCompletionChunk{
		ID:          id,
		Created:     time.Now().Unix(),
		Model:       modelName,
		Object:      "chat.completion.chunk",
		ServiceTier: "standard",
		Choices: []openai.ChatCompletionChunkChoice{
			{
				Index: 0,
				Delta: openai.ChatCompletionChunkChoicesDelta{
					Content: content,
				},
				FinishReason: openai.ChatCompletionChunkChoicesFinishReasonStop,
			},
		},
	}
}

// processToolCalls processes any tool calls requested by the LLM
func (a *Agent) processToolCalls(
	ctx context.Context,
	completion openai.ChatCompletionAccumulator,
	clonedMessages *MessageList,
	llmClient *LLM,
	modelName string,
	outAgentChannel chan openai.ChatCompletionChunk,
	send_status_func func(msg string),
) map[string]openai.ChatCompletionMessageParamUnion {
	results := make(map[string]openai.ChatCompletionMessageParamUnion)

	if completion.Choices[0].Message.ToolCalls == nil {
		return results
	}

	toolsToCall := completion.Choices[0].Message.ToolCalls
	var wg sync.WaitGroup
	var mu sync.Mutex

	for _, tool := range toolsToCall {
		skill, err := a.GetSkill(tool.Function.Name)
		if err != nil {
			a.logger.Error("Error getting skill", "error", err)
			continue
		}

		if skill.StatusMessage != "" {
			send_status_func(skill.StatusMessage)
		}

		wg.Add(1)
		go func(skill *Skill, toolID string) {
			defer wg.Done()
			// Clone the messages again so all goroutines get different message history
			result, err := a.SkillContextRunner(ctx, skill, toolID, clonedMessages.Clone(), llmClient, modelName, outAgentChannel, send_status_func)
			if err != nil {
				a.logger.Error("Error running skill", "error", err)
				return
			}

			mu.Lock()
			results[toolID] = result
			mu.Unlock()
		}(skill, tool.ID)
	}

	wg.Wait()
	return results
}

// handleToolCallResults handles the results of tool calls
func (a *Agent) handleToolCallResults(
	completion openai.ChatCompletionAccumulator,
	results map[string]openai.ChatCompletionMessageParamUnion,
	session *Session,
	llmClient *LLM,
	modelName string,
	outAgentChannel chan openai.ChatCompletionChunk,
) {
	// If multiple skills were called, summarize the results
	if len(completion.Choices[0].Message.ToolCalls) > 1 {
		a.summarizeMultipleToolResults(results, session, llmClient, modelName, outAgentChannel)
	} else if len(completion.Choices[0].Message.ToolCalls) == 1 {
		// If only one skill was called, return the result directly
		a.returnSingleToolResult(completion, results, session, modelName, outAgentChannel)
	}
}

// summarizeMultipleToolResults summarizes results when multiple tools were called
func (a *Agent) summarizeMultipleToolResults(
	results map[string]openai.ChatCompletionMessageParamUnion,
	session *Session,
	llmClient *LLM,
	modelName string,
	outAgentChannel chan openai.ChatCompletionChunk,
) {
	// Prepare the results for the OpenAI API call
	for _, result := range results {
		session.State.MessageHistory.Add(result)
	}

	params := openai.ChatCompletionNewParams{
		Messages: openai.F(session.State.MessageHistory.All()),
		Model:    openai.F(modelName),
		StreamOptions: openai.F(openai.ChatCompletionStreamOptionsParam{
			IncludeUsage: openai.F(true),
		}),
	}

	stream := llmClient.NewStreaming(context.Background(), params)
	defer stream.Close()

	for stream.Next() {
		chunk := stream.Current()
		if chunk.Choices[0].Delta.Content != "" {
			outAgentChannel <- chunk
		}
	}

	if stream.Err() != nil {
		a.logger.Error("Error streaming", "error", stream.Err())
		id, _ := gonanoid.New()
		outAgentChannel <- openai.ChatCompletionChunk{
			ID:      id,
			Created: time.Now().Unix(),
			Model:   modelName,
			Object:  "chat.completion.chunk",
			Choices: []openai.ChatCompletionChunkChoice{
				{
					Index: 0,
					Delta: openai.ChatCompletionChunkChoicesDelta{
						Content: "Error occurred while streaming",
					},
					FinishReason: openai.ChatCompletionChunkChoicesFinishReasonStop,
				},
			},
		}
	}
}

// returnSingleToolResult returns the result when only one tool was called
func (a *Agent) returnSingleToolResult(
	completion openai.ChatCompletionAccumulator,
	results map[string]openai.ChatCompletionMessageParamUnion,
	session *Session,
	modelName string,
	outAgentChannel chan openai.ChatCompletionChunk,
) {
	// If only one skill is called, return the result directly
	toolCallID := completion.Choices[0].Message.ToolCalls[0].ID
	session.State.MessageHistory.Add(results[toolCallID])
	resp := results[toolCallID]

	id, _ := gonanoid.New()
	if message, ok := resp.(openai.ChatCompletionToolMessageParam); ok {
		outAgentChannel <- openai.ChatCompletionChunk{
			ID:      id,
			Created: time.Now().Unix(),
			Model:   modelName,
			Object:  "chat.completion.chunk",
			Choices: []openai.ChatCompletionChunkChoice{
				{
					Index: 0,
					Delta: openai.ChatCompletionChunkChoicesDelta{
						Content: message.Content.Value[0].Text.Value,
					},
				},
			},
		}
	} else {
		a.logger.Error("Unexpected message type")
	}
}

// TODO - we probably need to have a custom made description for the tool that uses skill.description
// TODO - Enable Strict mode for the functions
func (a *Agent) ConvertSkillsToTools() []openai.ChatCompletionToolParam {
	tools := []openai.ChatCompletionToolParam{}
	for _, skill := range a.skills {
		tools = append(tools, openai.ChatCompletionToolParam{
			Function: openai.F(openai.FunctionDefinitionParam{
				Name:        openai.F(skill.Name),
				Description: openai.F(skill.Description),
				Parameters:  openai.F(openai.FunctionParameters{}),
			}),
			Type: openai.F(openai.ChatCompletionToolTypeFunction),
		})
	}
	return tools
}

func (a *Agent) GenerateSummary(ctx context.Context, messages *MessageList, llmClient *LLM, modelName string) (string, error) {
	a.logger.Info("Summarizing conversation")
	lastUserMessage := messages.LastUserMessageString()
	if lastUserMessage == "" {
		return "", fmt.Errorf("no user message found")
	}

	summaryPrompt := fmt.Sprintf("Based on the conversation history, answer my original question.\nQuestion:%s", lastUserMessage)
	messages.Add(DeveloperMessage(summaryPrompt))

	completion, err := llmClient.New(
		ctx,
		openai.ChatCompletionNewParams{
			Messages: openai.F(messages.All()),
			Model:    openai.F(modelName),
		})
	if err != nil {
		return "", err
	}

	return completion.Choices[0].Message.Content, nil
}

func MessageWhenToolError(toolCallID string) openai.ChatCompletionToolMessageParam {
	return openai.ChatCompletionToolMessageParam{
		Role:       openai.F(openai.ChatCompletionToolMessageParamRoleTool),
		Content:    openai.F([]openai.ChatCompletionContentPartTextParam{{Type: openai.F(openai.ChatCompletionContentPartTextTypeText), Text: openai.F("Error occurred while running. Do not retry")}}),
		ToolCallID: openai.F(toolCallID),
	}
}

func MessageWhenToolErrorWithRetry(errorString string, toolCallID string) openai.ChatCompletionToolMessageParam {
	return openai.ChatCompletionToolMessageParam{
		Role:       openai.F(openai.ChatCompletionToolMessageParamRoleTool),
		Content:    openai.F([]openai.ChatCompletionContentPartTextParam{{Type: openai.F(openai.ChatCompletionContentPartTextTypeText), Text: openai.F(fmt.Sprintf("Error: %s.\nRetry", errorString))}}),
		ToolCallID: openai.F(toolCallID),
	}
}

func (a *Agent) SkillContextRunner(ctx context.Context, skill *Skill, skillToolCallID string, clonedMessages *MessageList, llmClient *LLM, modelName string, outAgentChannel chan openai.ChatCompletionChunk, send_status_func func(msg string)) (openai.ChatCompletionMessageParamUnion, error) {
	a.logger.Info("Running skill", "skill", skill.Name)
	// TODO - we need to have some sort of hard limit for the number iterations possible
	// take the first developer message and append the skill.SystemPrompt at the end of it and then call clonedMessages.ReplaceAt(0, DeveloperMessage(skill.SystemPrompt))
	firstMessage := clonedMessages.All()[0]
	developerMsg, ok := firstMessage.(openai.ChatCompletionDeveloperMessageParam)
	if !ok {
		a.logger.Error("First message is not a developer message")
		return nil, fmt.Errorf("first message is not a developer message")
	}

	// Create a new developer message that includes the skill system prompt
	// Use the DeveloperMessage function to create a fresh message with combined content
	combinedContent := ""
	// Extract text from the original developer message
	for _, part := range developerMsg.Content.Value {
		combinedContent += part.Text.Value
	}
	combinedContent += "\n" + skill.SystemPrompt + "\n" + "Do not hallucinate or make up any information. Only use the tools and the data provided by the tools"

	// Replace the first message with the new combined developer message
	clonedMessages.ReplaceAt(0, DeveloperMessage(combinedContent))

	// Initial call to have LLM think step by step before executing tools
	var toolNames []string
	for _, tool := range skill.Tools {
		toolNames = append(toolNames, tool.Name())
	}

	toolsInfo := "Available tools:\n"
	for _, name := range toolNames {
		toolsInfo += fmt.Sprintf("- %s\n", name)
	}

	thinkingPromptText := fmt.Sprintf("Think step by step about how to approach this task. What information do you need? What steps will you take? Please plan your approach carefully. You only have below tools available to you:\n\n%s", toolsInfo)
	thinkingPrompt := DeveloperMessage(thinkingPromptText)
	clonedMessages.Add(thinkingPrompt)

	thinkingParams := openai.ChatCompletionNewParams{
		Messages: openai.F(clonedMessages.All()),
		Model:    openai.F(modelName),
	}

	thinkingCompletion, err := llmClient.New(ctx, thinkingParams)
	if err != nil {
		a.logger.Error("Error calling LLM for step-by-step thinking", "error", err)
		return MessageWhenToolErrorWithRetry("Network error", skillToolCallID), err
	}

	// Add the LLM's thinking to the message history
	clonedMessages.Add(thinkingCompletion.Choices[0].Message)

	for {
		// We add a developer message just for this loop (not to the history) to ensure the LLM doesn't output anything except the necessary tool to be called.
		params := openai.ChatCompletionNewParams{
			Messages: openai.F(clonedMessages.All()),
			Model:    openai.F(modelName),
		}
		a.logger.Info("Running skill", "skill", skill.Name, "tools", skill.Tools)
		if len(skill.GetTools()) > 0 {
			params.Tools = openai.F(skill.GetTools())
		}
		completion, err := llmClient.New(ctx, params)
		if err != nil {
			a.logger.Error("Error calling LLM while running skill", "error", err)
			return MessageWhenToolErrorWithRetry("Network error", skillToolCallID), err
		}
		clonedMessages.Add(completion.Choices[0].Message)

		// Check if both tool call and content are non-empty
		// this won't affect the flow currently but this is not the expectation
		bothToolCallAndContent := completion.Choices[0].Message.ToolCalls != nil && completion.Choices[0].Message.Content != ""
		if bothToolCallAndContent {
			a.logger.Error("Expectation is that both tool call and content are both non-empty")
		}

		// if there is no tool call, break
		if completion.Choices[0].Message.ToolCalls == nil {
			break
		}

		for _, toolCall := range completion.Choices[0].Message.ToolCalls {
			tool, err := skill.GetTool(toolCall.Function.Name)
			if err != nil {
				a.logger.Error("Error getting tool", "error", err)
				clonedMessages.Add(MessageWhenToolError(toolCall.ID))
				continue
			}
			if tool.StatusMessage() != "" {
				send_status_func(tool.StatusMessage())
			}
			a.logger.Info("Tool", "tool", tool.Name(), "arguments", toolCall.Function.Arguments)
			arguments := map[string]interface{}{}
			err = json.Unmarshal([]byte(toolCall.Function.Arguments), &arguments)
			if err != nil {
				a.logger.Error("Error unmarshalling tool arguments", "error", err)
				clonedMessages.Add(MessageWhenToolErrorWithRetry(err.Error(), skillToolCallID))
				continue
			}
			// TODO - model doesn't always generate valid JSON, so we need to validate the arguments and ask LLM to fix if there are errors
			output, err := tool.Execute(arguments)
			if err != nil {
				a.logger.Error("Error executing tool", "error", err)

				switch {
				case errors.As(err, &ignErr):
					// It's an IgnorableError
					clonedMessages.Add(MessageWhenToolError(toolCall.ID))
					continue
				case errors.As(err, &retErr):
					// It's a RetryableError
					clonedMessages.Add(MessageWhenToolErrorWithRetry(err.Error(), toolCall.ID))

				default:
					// Some other error
					clonedMessages.Add(MessageWhenToolError(toolCall.ID))
					continue
				}
			} else {
				clonedMessages.Add(openai.ChatCompletionToolMessageParam{
					Role:       openai.F(openai.ChatCompletionToolMessageParamRoleTool),
					Content:    openai.F([]openai.ChatCompletionContentPartTextParam{{Type: openai.F(openai.ChatCompletionContentPartTextTypeText), Text: openai.F(output)}}),
					ToolCallID: openai.F(toolCall.ID),
				})
			}
		}
	}
	allMessages := clonedMessages.All()
	lastMessage := allMessages[len(allMessages)-1]
	// If it's a ChatCompletionMessage, convert it to a tool message
	if chatMsg, ok := lastMessage.(openai.ChatCompletionMessage); ok {
		content := chatMsg.Content

		return openai.ChatCompletionToolMessageParam{
			Role:       openai.F(openai.ChatCompletionToolMessageParamRoleTool),
			Content:    openai.F([]openai.ChatCompletionContentPartTextParam{{Type: openai.F(openai.ChatCompletionContentPartTextTypeText), Text: openai.F(content)}}),
			ToolCallID: openai.F(skillToolCallID),
		}, nil
	} else {
		a.logger.Error("Unexpected message type in SkillContextRunner result", "type", fmt.Sprintf("%T", lastMessage))
		return openai.ChatCompletionToolMessageParam{
			Role:       openai.F(openai.ChatCompletionToolMessageParamRoleTool),
			Content:    openai.F([]openai.ChatCompletionContentPartTextParam{{Type: openai.F(openai.ChatCompletionContentPartTextTypeText), Text: openai.F("Error: The skill execution did not produce a valid response")}}),
			ToolCallID: openai.F(skillToolCallID),
		}, nil
	}
}
