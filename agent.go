// Package agent provides the main Agent orchestrator, which uses LLM & Skills to process data.
package agentpod

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"strings"
	"sync"

	"github.com/boat-builder/agentpod/prompts"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/param"
)

const maxAgentLoops = 25

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

func (a *Agent) StopTool() openai.ChatCompletionToolParam {
	return openai.ChatCompletionToolParam{
		Function: openai.FunctionDefinitionParam{
			Name: "stop",
			Description: param.Opt[string]{
				Value: `Call this tool when you are ready to finish the task or can't do anything more. Pass the final assistant reply for the user in the "response" argument.`,
			},
			Parameters: openai.FunctionParameters{
				"type": "object",
				"properties": map[string]interface{}{
					"response": map[string]interface{}{
						"type":        "string",
						"description": "The final response that should be shown to the user.",
					},
				},
				"required": []string{"response"},
			},
		},
	}
}

// TODO - we probably need to have a custom made description for the tool that uses skill.description
func (a *Agent) ConvertSkillsToTools() []openai.ChatCompletionToolParam {
	tools := []openai.ChatCompletionToolParam{}
	for _, skill := range a.skills {
		tools = append(tools, openai.ChatCompletionToolParam{
			Function: openai.FunctionDefinitionParam{
				Name:        skill.Name,
				Description: param.Opt[string]{Value: skill.Description},
				Parameters: openai.FunctionParameters{
					"type": "object",
					"properties": map[string]interface{}{
						"instruction": map[string]interface{}{
							"type":        "string",
							"description": "A detailed instruction on what to achieve",
						},
					},
					"required": []string{"instruction"},
				},
			},
		})
	}
	return tools
}

// decideNextAction gets the initial response from the LLM that decides whether to use skills or stop execution
func (a *Agent) decideNextAction(ctx context.Context, llm LLM, clonedMessages *MessageList, memoryBlock *MemoryBlock) (*openai.ChatCompletion, error) {
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

	tools := []openai.ChatCompletionToolParam{}
	if len(a.ConvertSkillsToTools()) > 0 {
		tools = append([]openai.ChatCompletionToolParam{a.StopTool()}, a.ConvertSkillsToTools()...)
	}
	// TODO make it strict to call the tool when the openai sdk supports passing the option 'required'
	params := openai.ChatCompletionNewParams{
		Messages:   clonedMessages.All(),
		Model:      llm.CheapModel(),
		ToolChoice: openai.ChatCompletionToolChoiceOptionUnionParam{OfAuto: param.Opt[string]{Value: "required"}},
		Tools:      tools,
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

// Run processes a user message through the LLM, executes any requested skills. It returns only after the agent is done.
// The intermediary messages are sent to the outUserChannel.
func (a *Agent) Run(ctx context.Context, llm LLM, messageHistory *MessageList, memoryBlock *MemoryBlock, outUserChannel chan Response) {
	if a.logger == nil {
		panic("logger is not set")
	}

	// Create a cancel function from the context
	ctx, cancel := context.WithCancel(ctx)

	// making sure we send the end response when the agent is done and cancel the context
	defer func() {
		defer func() {
			if r := recover(); r != nil {
				a.logger.Error("Panic when sending end response", "error", r)
			}
		}()
		cancel()
		close(outUserChannel)
	}()

	var hasStopToolCall bool

	if len(a.skills) == 0 {
		a.logger.Error("agent cannot run without skills")
		outUserChannel <- Response{
			Content: "Agent cannot run without skills.",
			Type:    ResponseTypeError,
		}
		return
	}

	for i := 0; ; i++ {
		if i >= maxAgentLoops {
			a.logger.Error("agent has reached max loop count", "count", maxAgentLoops)
			outUserChannel <- Response{
				Content: "Agent has been running for too long and has been stopped.",
				Type:    ResponseTypeError,
			}
			return
		}
		completion, err := a.decideNextAction(ctx, llm, messageHistory.Clone(), memoryBlock)
		if err != nil {
			a.handleLLMError(err, outUserChannel)
			return
		}

		// If no tool calls were requested, we're done - this doesn't happen as tool is "required"
		if completion.Choices[0].Message.ToolCalls == nil {
			break
		}

		// Separate stop tools from skill tools
		skillToolCalls := []openai.ChatCompletionMessageToolCall{}
		for _, toolCall := range completion.Choices[0].Message.ToolCalls {
			if toolCall.Function.Name == "stop" {
				hasStopToolCall = true

				// Extract the response argument if present
				if toolCall.Function.Arguments != "" {
					var args map[string]interface{}
					if err := json.Unmarshal([]byte(toolCall.Function.Arguments), &args); err == nil {
						if resp, ok := args["response"].(string); ok {
							a.logger.Info("Stop tool called with response. We don't respond this to the caller from here though", "response", resp)
						}
					}
				}
			} else {
				skillToolCalls = append(skillToolCalls, toolCall)
			}
		}

		// Execute all skill tools in the current response
		skillCallResults := make(map[string]*openai.ChatCompletionToolMessageParam)
		var wg sync.WaitGroup
		var mu sync.Mutex

		for _, tool := range skillToolCalls {
			skill, err := a.GetSkill(tool.Function.Name)
			if err != nil {
				a.logger.Error("Error getting skill", "error", err)
				continue
			}

			wg.Add(1)
			go func(skill *Skill, tool openai.ChatCompletionMessageToolCall) {
				defer wg.Done()
				// Clone the messages again so all goroutines get different message history
				result, err := a.SkillContextRunner(ctx, messageHistory.Clone(), llm, memoryBlock, skill, tool)
				if err != nil {
					a.logger.Error("Error running skill", "error", err)
					return
				}

				mu.Lock()
				skillCallResults[tool.ID] = result
				mu.Unlock()
			}(skill, tool)
		}

		wg.Wait()

		// Add the completion message to history, but filter out the stop tool call
		messageToAdd := completion.Choices[0].Message
		if messageToAdd.ToolCalls != nil {
			filteredToolCalls := []openai.ChatCompletionMessageToolCall{}
			for _, toolCall := range messageToAdd.ToolCalls {
				if toolCall.Function.Name != "stop" {
					filteredToolCalls = append(filteredToolCalls, toolCall)
				}
			}
			// Only update and add the message if there are non-stop tool calls. We have this specific condition here because
			// we tinker with the tool calls to filter out one of the call.
			if len(filteredToolCalls) > 0 {
				messageToAdd.ToolCalls = filteredToolCalls
				messageHistory.Add(messageToAdd.ToParam())
			}
		} else {
			messageHistory.Add(messageToAdd.ToParam())
		}
		// Add tool results to message history
		for _, result := range skillCallResults {
			messageHistory.Add(openai.ChatCompletionMessageParamUnion{OfTool: result})
		}

		// If stop tool was called, break the loop
		if hasStopToolCall {
			break
		}
	}
}
