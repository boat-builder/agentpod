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
func (a *Agent) summarizeMultipleToolResults(results map[string]openai.ChatCompletionMessageParamUnion, clonedMessages *MessageList, llm *LLM) (string, error) {
	// Prepare the results for the OpenAI API call
	for _, result := range results {
		clonedMessages.Add(result)
	}

	params := openai.ChatCompletionNewParams{
		Messages: openai.F(clonedMessages.All()),
		Model:    openai.F(llm.GenerationModel),
		StreamOptions: openai.F(openai.ChatCompletionStreamOptionsParam{
			IncludeUsage: openai.F(true),
		}),
	}

	stream := llm.NewStreaming(context.Background(), params)
	defer stream.Close()

	var fullResponse strings.Builder

	for stream.Next() {
		chunk := stream.Current()
		if chunk.Choices[0].Delta.Content != "" {
			fullResponse.WriteString(chunk.Choices[0].Delta.Content)
		}
	}

	if stream.Err() != nil {
		a.logger.Error("Error streaming", "error", stream.Err())
		return "", stream.Err()
	}

	return fullResponse.String(), nil
}

// returnSingleToolResult returns the result when only one tool was called
func (a *Agent) returnSingleToolResult(
	completion *openai.ChatCompletion,
	results map[string]openai.ChatCompletionMessageParamUnion,
) (string, error) {
	// If only one skill is called, return the result directly
	toolCallID := completion.Choices[0].Message.ToolCalls[0].ID
	resp := results[toolCallID]

	if message, ok := resp.(openai.ChatCompletionToolMessageParam); ok {
		return message.Content.Value[0].Text.Value, nil
	} else {
		a.logger.Error("Unexpected message type")
		return "", fmt.Errorf("unexpected message type")
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
func (a *Agent) chooseSkills(ctx context.Context, llm *LLM, clonedMessages *MessageList, memoryBlock *MemoryBlock) (*openai.ChatCompletion, error) {
	skillFunctions := make([]string, len(a.skills))
	for i, skill := range a.skills {
		skillFunctions[i] = skill.Name
	}

	systemPromptData := prompts.SkillSelectionPromptData{
		MainAgentSystemPrompt: a.prompt,
		MemoryBlocks:          memoryBlock.Parse(),
		SkillFunctions:        skillFunctions,
	}
	systemPrompt, err := prompts.SkillSelectionPrompt(systemPromptData)
	if err != nil {
		a.logger.Error("Error getting system prompt", "error", err)
		return nil, err
	}

	clonedMessages.AddFirst(systemPrompt)

	params := openai.ChatCompletionNewParams{
		Messages: openai.F(clonedMessages.All()),
		Model:    openai.F(llm.GenerationModel),
	}
	if len(a.ConvertSkillsToTools()) > 0 {
		params.Tools = openai.F(a.ConvertSkillsToTools())
	}

	completion, err := llm.New(ctx, params)
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
	messageHistory *MessageList,
	llm *LLM,
) (string, error) {
	// If multiple skills were called, summarize the results
	if len(completion.Choices[0].Message.ToolCalls) > 1 {
		return a.summarizeMultipleToolResults(results, messageHistory, llm)
	} else if len(completion.Choices[0].Message.ToolCalls) == 1 {
		// If only one skill was called, return the result directly
		return a.returnSingleToolResult(completion, results)
	}

	return "", fmt.Errorf("no tool calls found in completion")
}

// handleLLMError handles errors from LLM API calls
func (a *Agent) handleLLMError(err error, outUserChannel chan Response) {
	content := "Error occurred!"
	a.logger.Error("Error streaming", "error", err)
	if strings.Contains(err.Error(), "ContentPolicyViolationError") {
		a.logger.Error("Content policy violation!", "error", err)
		content = "Content policy violation! If this was a mistake, please reach out to the support. Consecutive violations may result in a temporary/permanent ban."
	}
	outUserChannel <- Response{
		Content: content,
		Type:    ResponseTypeError,
	}
}

// sendThoughtsAboutSkills generates "thinking" messages to keep the user engaged while skills are processing
func (a *Agent) sendThoughtsAboutSkills(ctx context.Context, llm *LLM, messageHistory *MessageList, toolsToCall []openai.ChatCompletionMessageToolCall, outUserChannel chan Response) {
	if len(toolsToCall) == 0 {
		return
	}

	allSpecSystemPrompt := `You have these tools available for you to use. But first you need to send a response to the user about what you are planning to do. Make sure to strategize in details.
	
	Notes:
	- Do not mention about the tools or details about the tools like SQL, Python API etc. 
	- You can mention about what you are trying to achieve by mentioning what these tools enable you to do. For example, if an SQL table enable you to get latest whether, you can say "I am getting whether data" instead of "I'll look at the SQL database for whether data".
	- Make it very detailed.
	- Strictly do not answer the question. You are just planning.

	Here are the details about the tools:
	`
	for _, tool := range toolsToCall {
		skill, err := a.GetSkill(tool.Function.Name)
		if err != nil {
			a.logger.Error("Error getting skill", "error", err)
			continue
		}
		allSpecSystemPrompt += fmt.Sprintf("\n%s\n", skill.Spec())
	}

	outUserChannel <- Response{
		Type: ResponseTypeThinkingStart,
	}
	// making sure we send the end response when the agent is done
	defer func() {
		defer func() {
			if r := recover(); r != nil {
				a.logger.Error("Panic when sending end response", "error", r)
			}
		}()
		outUserChannel <- Response{
			Type: ResponseTypeThinkingEnd,
		}
	}()

	messageHistory.AddFirst(allSpecSystemPrompt)
	stream := llm.NewStreaming(ctx, openai.ChatCompletionNewParams{
		Messages: openai.F(messageHistory.All()),
		Model:    openai.F(llm.SmallGenerationModel),
	})
	defer stream.Close()

	for stream.Next() {
		chunk := stream.Current()
		if chunk.Choices[0].Delta.Content != "" {
			outUserChannel <- Response{
				Content: chunk.Choices[0].Delta.Content,
				Type:    ResponseTypeThinking,
			}
		}
	}

}

// sendThoughtsAboutSkills generates "thinking" messages to keep the user engaged while skills are processing
func (a *Agent) sendThoughtsAboutTools(ctx context.Context, llm *LLM, messageHistory *MessageList, toolsToCall []openai.ChatCompletionMessageToolCall, outUserChannel chan Response) {
	if len(toolsToCall) == 0 {
		return
	}

	systemPrompt := `Assistant has recommended to run a few functions. Now you need send status update to the user before you executing the request from assistant.
	
	Notes:
	- Do not mention tools/functions in details (like what it is going to do technically like using SQL, Python API etc)
	- You should mention about what you are assistant is trying to achieve by executing the functions.
	- You should not mention about the "assistant" at all. Only focus on what assistant has asked to do.
	`

	outUserChannel <- Response{
		Type: ResponseTypeThinkingStart,
	}
	// making sure we send the end response when the agent is done
	defer func() {
		defer func() {
			if r := recover(); r != nil {
				a.logger.Error("Panic when sending end response", "error", r)
			}
		}()
		outUserChannel <- Response{
			Type: ResponseTypeThinkingEnd,
		}
	}()

	messageHistory.AddFirst(systemPrompt)
	assistantMessage := "Execute below functions and get me the results so I can answer you better.\n\n"
	for _, tool := range toolsToCall {
		assistantMessage += fmt.Sprintf("Function Name: %s\nFunction Args: %s\n\n", tool.Function.Name, tool.Function.Arguments)
	}
	messageHistory.Add(AssistantMessage(assistantMessage))

	stream := llm.NewStreaming(ctx, openai.ChatCompletionNewParams{
		Messages: openai.F(messageHistory.All()),
		Model:    openai.F(llm.SmallGenerationModel),
	})
	defer stream.Close()

	for stream.Next() {
		chunk := stream.Current()
		if chunk.Choices[0].Delta.Content != "" {
			outUserChannel <- Response{
				Content: chunk.Choices[0].Delta.Content,
				Type:    ResponseTypeThinking,
			}
		}
	}

}

// Run processes a user message through the LLM, executes any requested skills. It returns only after the agent is done.
// The intermediary messages are sent to the outUserChannel.
func (a *Agent) Run(ctx context.Context, llm *LLM, messageHistory *MessageList, memoryBlock *MemoryBlock, outUserChannel chan Response) {
	if a.logger == nil {
		panic("logger is not set")
	}

	// making sure we send the end response when the agent is done
	defer func() {
		defer func() {
			if r := recover(); r != nil {
				a.logger.Error("Panic when sending end response", "error", r)
			}
		}()
		outUserChannel <- Response{
			Type: ResponseTypeEnd,
		}
	}()

	completion, err := a.chooseSkills(ctx, llm, messageHistory.Clone(), memoryBlock)
	if err != nil {
		a.handleLLMError(err, outUserChannel)
		return
	}
	if completion.Choices[0].Message.Content != "" {
		outUserChannel <- Response{
			Content: completion.Choices[0].Message.Content,
			Type:    ResponseTypePartialText,
		}
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

	// sending fake thoughts to the user to keep the user engaged
	go a.sendThoughtsAboutSkills(ctx, llm, messageHistory.Clone(), toolsToCall, outUserChannel)

	for _, tool := range toolsToCall {
		skill, err := a.GetSkill(tool.Function.Name)
		if err != nil {
			a.logger.Error("Error getting skill", "error", err)
			continue
		}

		wg.Add(1)
		go func(skill *Skill, toolID string) {
			defer wg.Done()
			// Clone the messages again so all goroutines get different message history
			result, err := a.SkillContextRunner(ctx, messageHistory.Clone(), llm, outUserChannel, memoryBlock, skill, tool.ID)
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
	messageHistory.Add(completion.Choices[0].Message)
	// Handle results based on number of tool calls
	collatedResult, err := a.collateSkillCallResults(completion, results, messageHistory, llm)
	if err != nil {
		a.handleLLMError(err, outUserChannel)
		return
	}

	outUserChannel <- Response{
		Content: collatedResult,
		Type:    ResponseTypePartialText,
	}
}
