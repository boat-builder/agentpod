// Package agent provides the main Agent orchestrator, which uses LLM & Skills to process data.
package agentpod

import (
	"context"
	"fmt"
	"log/slog"
	"strings"
	"sync"

	"github.com/boat-builder/agentpod/prompts"
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
	// Validate that all skills have both Description and SystemPrompt set
	for _, skill := range skills {
		if skill.Description == "" {
			panic(fmt.Sprintf("skill '%s' is missing a Description", skill.Name))
		}
		if skill.SystemPrompt == "" {
			panic(fmt.Sprintf("skill '%s' is missing a SystemPrompt", skill.Name))
		}
	}

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

// summarizeMultipleToolResults summarizes results when multiple tools were called
func (a *Agent) summarizeMultipleToolResults(
	results map[string]openai.ChatCompletionMessageParamUnion,
	clonedMessages *MessageList,
	llmClient *LLM,
	modelName string,
	outAgentChannel chan string,
) {
	// Prepare the results for the OpenAI API call
	for _, result := range results {
		clonedMessages.Add(result)
	}

	params := openai.ChatCompletionNewParams{
		Messages: openai.F(clonedMessages.All()),
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
			outAgentChannel <- chunk.Choices[0].Delta.Content
		}
	}

	if stream.Err() != nil {
		a.logger.Error("Error streaming", "error", stream.Err())
		outAgentChannel <- "Error occurred while streaming"
	}
}

// returnSingleToolResult returns the result when only one tool was called
func (a *Agent) returnSingleToolResult(
	completion *openai.ChatCompletion,
	results map[string]openai.ChatCompletionMessageParamUnion,
	outAgentChannel chan string,
) {
	// If only one skill is called, return the result directly
	toolCallID := completion.Choices[0].Message.ToolCalls[0].ID
	resp := results[toolCallID]

	if message, ok := resp.(openai.ChatCompletionToolMessageParam); ok {
		outAgentChannel <- message.Content.Value[0].Text.Value
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

// chooseSkills gets the initial response from the LLM that chooses the skills
func (a *Agent) chooseSkills(
	ctx context.Context,
	clonedMessages *MessageList,
	userInfo UserInfo,
	llmClient *LLM,
	modelName string,
) (*openai.ChatCompletion, error) {
	memoryBlocks := make(map[string]string)
	memoryBlocks["UserName"] = userInfo.Name
	for key, value := range userInfo.Meta {
		memoryBlocks[key] = value
	}

	skillFunctions := make([]string, len(a.skills))
	for i, skill := range a.skills {
		skillFunctions[i] = skill.Name
	}

	systemPromptData := prompts.SkillSelectionPromptData{
		UserSystemPrompt: a.prompt,
		MemoryBlocks:     memoryBlocks,
		SkillFunctions:   skillFunctions,
	}
	systemPrompt, err := prompts.SkillSelectionPrompt(systemPromptData)
	if err != nil {
		a.logger.Error("Error getting system prompt", "error", err)
		return nil, err
	}

	clonedMessages.AddFirst(systemPrompt)

	params := openai.ChatCompletionNewParams{
		Messages: openai.F(clonedMessages.All()),
		Model:    openai.F(modelName),
	}
	if len(a.ConvertSkillsToTools()) > 0 {
		params.Tools = openai.F(a.ConvertSkillsToTools())
	}

	completion, err := llmClient.New(ctx, params)
	if err != nil {
		a.logger.Error("Error getting initial response", "error", err)
		return nil, err
	}

	if len(completion.Choices) == 0 {
		a.logger.Error("No completion choices")
		return completion, fmt.Errorf("no completion choices")
	}

	// Check if both tool call and content are non-empty
	bothToolCallAndContent := completion.Choices[0].Message.ToolCalls != nil && completion.Choices[0].Message.Content != ""
	if bothToolCallAndContent {
		a.logger.Error("Expectation is that tool call and content shouldn't both be non-empty", "message", completion.Choices[0].Message)
	}

	// Check for duplicate skills in tool calls
	if len(completion.Choices[0].Message.ToolCalls) > 1 {
		// Create a map to track seen skill names
		seenSkills := make(map[string]bool)
		var uniqueToolCalls []openai.ChatCompletionMessageToolCall

		// Keep only the first occurrence of each skill
		for _, toolCall := range completion.Choices[0].Message.ToolCalls {
			skillName := toolCall.Function.Name
			if !seenSkills[skillName] {
				seenSkills[skillName] = true
				uniqueToolCalls = append(uniqueToolCalls, toolCall)
			} else {
				a.logger.Warn("Removing duplicate skill from completion", "skill", skillName)
			}
		}

		// If duplicates were found, update the tool calls in the completion object
		if len(uniqueToolCalls) < len(completion.Choices[0].Message.ToolCalls) {
			completion.Choices[0].Message.ToolCalls = uniqueToolCalls
		}
	}

	return completion, nil
}

// collateSkillCallResults handles the results of tool calls
func (a *Agent) collateSkillCallResults(
	completion *openai.ChatCompletion,
	results map[string]openai.ChatCompletionMessageParamUnion,
	clonedMessages *MessageList,
	llmClient *LLM,
	modelName string,
	outAgentChannel chan string,
) {
	// If multiple skills were called, summarize the results
	if len(completion.Choices[0].Message.ToolCalls) > 1 {
		a.summarizeMultipleToolResults(results, clonedMessages, llmClient, modelName, outAgentChannel)
	} else if len(completion.Choices[0].Message.ToolCalls) == 1 {
		// If only one skill was called, return the result directly
		a.returnSingleToolResult(completion, results, outAgentChannel)
	}
}

// handleLLMError handles errors from LLM API calls
func (a *Agent) handleLLMError(err error, outAgentChannel chan string) {
	content := "Error occurred!"
	a.logger.Error("Error streaming", "error", err)
	if strings.Contains(err.Error(), "ContentPolicyViolationError") {
		a.logger.Error("Content policy violation!", "error", err)
		content = "Content policy violation! If this was a mistake, please reach out to the support. Consecutive violations may result in a temporary/permanent ban."
	}
	outAgentChannel <- content
}

// processRequest handles the main logic for processing a request
func (a *Agent) processRequest(
	ctx context.Context,
	session *Session,
	llmClient *LLM,
	modelName string,
	storage Storage,
	outChan chan string,
	send_status_func func(msg string),
) {
	defer close(outChan)

	userInfo, err := storage.GetUserInfo(session)
	if err != nil {
		a.logger.Error("Error getting user info", "error", err)
		return
	}

	// Clone the messages before appending the latest assistant message
	// Get initial response from LLM
	completion, err := a.chooseSkills(ctx, session.State.MessageHistory.Clone(), userInfo, llmClient, modelName)
	if err != nil {
		a.handleLLMError(err, outChan)
		return
	}
	if completion.Choices[0].Message.Content != "" {
		outChan <- completion.Choices[0].Message.Content
	}

	// If no tool calls were requested, we're done
	if completion.Choices[0].Message.ToolCalls == nil {
		return
	}

	// Offload the skill call to SkillContextRunner
	results := make(map[string]openai.ChatCompletionMessageParamUnion)
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
			result, err := a.SkillContextRunner(ctx, skill, toolID, session.State.MessageHistory.Clone(), llmClient, modelName, outChan, send_status_func, userInfo)
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

	// creating a new cloned message that doesn't have anything from skill context runner but has the tool calls
	clonedMessages := session.State.MessageHistory.Clone()
	clonedMessages.Add(completion.Choices[0].Message)
	// Handle results based on number of tool calls
	a.collateSkillCallResults(completion, results, clonedMessages, llmClient, modelName, outChan)
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
) (chan string, error) {
	if a.logger == nil {
		panic("logger is not set")
	}
	outAgentChannel := make(chan string)

	// Prepare session message history and validate state
	if err := CompileConversationHistory(session, storage); err != nil {
		a.logger.Error(err.Error())
		return nil, err
	}

	// Process the request asynchronously
	go a.processRequest(ctx, session, llmClient, modelName, storage, outAgentChannel, send_status_func)

	return outAgentChannel, nil
}
