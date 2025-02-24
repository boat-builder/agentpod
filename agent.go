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

// Run returns a stream of chat completion chunks. We don't do the streaming with channels like the session do
// because session is the one that tracks a session's life cycle. We still need to figure out how to route
// the intermediate input messages if "interactive=true" but the whole idea is Agent's will not have to deal
// with the lifecycle events like interactiveness with the end user which is the abstraction openai.client has
func (a *Agent) Run(ctx context.Context, session *Session, llmClient *LLM, modelName string, send_status_func func(msg string)) (chan openai.ChatCompletionChunk, error) {
	if a.logger == nil {
		panic("logger is not set")
	}
	outAgentChannel := make(chan openai.ChatCompletionChunk)

	// Add the prompt to the message history
	session.State.MessageHistory.AddFirst(a.prompt)

	go func() {
		defer close(outAgentChannel)
		// cloning the message before appending the latest assistant message
		clonedMessages := session.State.MessageHistory.Clone()

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
			content := "Error occured!"
			a.logger.Error("Error streaming", "error", stream.Err())
			if strings.Contains(stream.Err().Error(), "ContentPolicyViolationError") {
				a.logger.Error("Content policy violation!", "error", stream.Err())
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
			return
		}

		if len(completion.Choices) == 0 {
			a.logger.Error("No completion choices")
			return
		}

		// Check if both tool call and content are non-empty
		// this won't affect the flow currently but this is not the expectation
		bothToolCallAndContent := completion.Choices[0].Message.ToolCalls != nil && completion.Choices[0].Message.Content != ""
		if bothToolCallAndContent {
			a.logger.Error("Expectation is that both tool call and content are both non-empty")
		}

		// Append the message to our messages
		session.State.MessageHistory.Add(completion.Choices[0].Message)

		// if no tools are called, we'd just return (content must have already sent to the client through the channel)
		if completion.Choices[0].Message.ToolCalls == nil {
			return
		}

		results := make(map[string]openai.ChatCompletionMessageParamUnion)

		// Process tool calls if they exist
		if completion.Choices[0].Message.ToolCalls != nil {
			toolsToCall := completion.Choices[0].Message.ToolCalls

			var wg sync.WaitGroup
			var mu sync.Mutex

			for _, tool := range toolsToCall {
				skill, err := a.GetSkill(tool.Function.Name)
				if skill.StatusMessage != "" {
					send_status_func(skill.StatusMessage)
				}
				if err != nil {
					a.logger.Error("Error getting skill", "error", err)
					continue
				}

				wg.Add(1)
				go func(skill *Skill, toolID string) {
					defer wg.Done()
					result, err := a.SkillContextRunner(ctx, skill, toolID, clonedMessages, llmClient, modelName, outAgentChannel, send_status_func)
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

		}

		// if multiple skills are called, we'd need to summarize the results - without passing any tools
		if len(completion.Choices[0].Message.ToolCalls) > 1 {
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
			stream := llmClient.NewStreaming(ctx, params)
			defer stream.Close()
			for stream.Next() {
				chunk := stream.Current()
				if chunk.Choices[0].Delta.Content != "" {
					outAgentChannel <- chunk
				}
			}

			if stream.Err() != nil || len(completion.Choices) == 0 {
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

		} else if len(completion.Choices[0].Message.ToolCalls) == 1 {
			// if only one skill is called, we'd need to return the result
			session.State.MessageHistory.Add(results[completion.Choices[0].Message.ToolCalls[0].ID])
			resp := results[completion.Choices[0].Message.ToolCalls[0].ID]
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
	// add the skill description to the cloned messages
	for {
		// We add a developer message just for this loop (not to the history) to ensure the LLM doesn't output anything except the necessary tool to be called.
		history := append(clonedMessages.All(), DeveloperMessage("Do not output anything except the necessary tool to be called. If no tools are needed, just say 'No tools needed'"))
		params := openai.ChatCompletionNewParams{
			Messages: openai.F(history),
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

		// adding the assistant message to the cloned messages - done after the tool existance check to avoid having the "no tools needed" message
		clonedMessages.Add(completion.Choices[0].Message)

		for _, toolCall := range completion.Choices[0].Message.ToolCalls {
			tool, err := skill.GetTool(toolCall.Function.Name)
			if err != nil {
				a.logger.Error("Error getting tool", "error", err)
				clonedMessages.Add(MessageWhenToolError(toolCall.ID))
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
			}
			// TODO - model doesn't always generate valid JSON, so we need to validate the arguments and ask LLM to fix if there are errors
			output, err := tool.Execute(arguments)
			if err != nil {
				a.logger.Error("Error executing tool", "error", err)

				switch {
				case errors.As(err, &ignErr):
					// It's an IgnorableError
					clonedMessages.Add(MessageWhenToolError(toolCall.ID))

				case errors.As(err, &retErr):
					// It's a RetryableError
					clonedMessages.Add(MessageWhenToolErrorWithRetry(err.Error(), skillToolCallID))

				default:
					// Some other error
					clonedMessages.Add(MessageWhenToolError(toolCall.ID))
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

	// TODO - when some sort of error occurs, we should retry but with not the error message. Also, once the retry is success, we don't really
	// need the error message in the history
	finalResponse, err := a.GenerateSummary(ctx, clonedMessages, llmClient, modelName)
	if err != nil {
		a.logger.Error("Error generating summary", "error", err)
		return MessageWhenToolErrorWithRetry("Error generating summary", skillToolCallID), err
	}
	return openai.ChatCompletionToolMessageParam{
		Role:       openai.F(openai.ChatCompletionToolMessageParamRoleTool),
		Content:    openai.F([]openai.ChatCompletionContentPartTextParam{{Type: openai.F(openai.ChatCompletionContentPartTextTypeText), Text: openai.F(finalResponse)}}),
		ToolCallID: openai.F(skillToolCallID),
	}, nil
}
