// Package agent provides the main Agent orchestrator, which uses LLM & Skills to process data.
package agentpod

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"sync"

	"github.com/openai/openai-go"
)

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

// Run returns a stream of chat completion chunks. We don't do the streaming with channels like the session do
// because session is the one that tracks a session's life cycle. We still need to figure out how to route
// the intermediate input messages if "interactive=true" but the whole idea is Agent's will not have to deal
// with the lifecycle events like interactiveness with the end user which is the abstraction openai.client has
func (a *Agent) Run(ctx context.Context, session *Session, llmClient *LLM, modelName string) (chan openai.ChatCompletionChunk, error) {
	if a.logger == nil {
		panic("logger is not set")
	}
	outAgentChannel := make(chan openai.ChatCompletionChunk)

	// Add the prompt to the message history
	session.State.MessageHistory.AddFirst(a.prompt)

	go func() {
		defer close(outAgentChannel)
		for {
			// TODO every where we have tools, may be we need to ask if we need to continue before making a call with tools
			// Stream messages from the LLM
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
			// TODO We need to keep the tool info in the state as well?
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
			// TODO when there is a stream Error, the stream.Next() won't execute and hence the next line will panic. Handle Stream.Error everywhere

			// Check if both tool call and content are non-empty
			// this won't affect the flow currently but this is not the expectation
			bothToolCallAndContent := completion.Choices[0].Message.ToolCalls != nil && completion.Choices[0].Message.Content != ""
			if bothToolCallAndContent {
				a.logger.Error("Expectation is that both tool call and content are both non-empty")
			}

			// cloning the message before appending the latest assistant message
			clonedMessages := session.State.MessageHistory.Clone()

			// Append the message to our messages
			session.State.MessageHistory.Add(completion.Choices[0].Message)

			// Process tool calls if they exist
			if completion.Choices[0].Message.ToolCalls != nil {
				toolsToCall := completion.Choices[0].Message.ToolCalls

				var wg sync.WaitGroup
				var mu sync.Mutex
				results := make(map[string]openai.ChatCompletionMessageParamUnion)

				for _, tool := range toolsToCall {
					skill, err := a.GetSkill(tool.Function.Name)
					if err != nil {
						a.logger.Error("Error getting skill", "error", err)
						continue
					}

					wg.Add(1)
					go func(skill *Skill, toolID string) {
						defer wg.Done()
						result, err := a.SkillContextRunner(ctx, skill, toolID, clonedMessages, llmClient, modelName)
						if err != nil {
							a.logger.Error("Error running skill", "error", err)
							return
						}

						mu.Lock()
						// TODO - we need to handle the case where the tool call is not successful
						results[toolID] = result
						mu.Unlock()
					}(skill, tool.ID)
				}

				wg.Wait()

				// Prepare the results for the OpenAI API call
				for _, result := range results {
					session.State.MessageHistory.Add(result)
				}
			} else {
				// If no tool calls, break the loop
				break
			}
		}
	}()

	return outAgentChannel, nil
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
	lastUserMessage := messages.LastUserMessageString()
	if lastUserMessage == "" {
		return "", fmt.Errorf("no user message found")
	}

	summaryPrompt := fmt.Sprintf("Summarize the conversation as an answer to the first question: %s", lastUserMessage)
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

func (a *Agent) SkillContextRunner(ctx context.Context, skill *Skill, parentToolCallID string, clonedMessages *MessageList, llmClient *LLM, modelName string) (openai.ChatCompletionMessageParamUnion, error) {
	a.logger.Info("Running skill", "skill", skill.Name)
	for {
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
			return nil, err
		}

		// adding the assistant message to the cloned messages
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
				return nil, err
			}
			a.logger.Info("Tool", "tool", tool.Name(), "arguments", toolCall.Function.Arguments)
			arguments := map[string]interface{}{}
			err = json.Unmarshal([]byte(toolCall.Function.Arguments), &arguments)
			if err != nil {
				return nil, err
			}
			// TODO - model doesn't always generate valid JSON, so we need to validate the arguments and ask LLM to fix if there are errors
			output, err := tool.Execute(arguments)
			a.logger.Info("Tool output", "output", output)
			if err != nil {
				return nil, err
			}
			clonedMessages.Add(openai.ChatCompletionToolMessageParam{
				Role:       openai.F(openai.ChatCompletionToolMessageParamRoleTool),
				Content:    openai.F([]openai.ChatCompletionContentPartTextParam{{Type: openai.F(openai.ChatCompletionContentPartTextTypeText), Text: openai.F(output)}}),
				ToolCallID: openai.F(toolCall.ID),
			})
		}
	}

	finalResponse, err := a.GenerateSummary(ctx, clonedMessages, llmClient, modelName)
	if err != nil {
		return nil, err
	}
	return openai.ChatCompletionToolMessageParam{
		Role:       openai.F(openai.ChatCompletionToolMessageParamRoleTool),
		Content:    openai.F([]openai.ChatCompletionContentPartTextParam{{Type: openai.F(openai.ChatCompletionContentPartTextTypeText), Text: openai.F(finalResponse)}}),
		ToolCallID: openai.F(parentToolCallID),
	}, nil
}
