package agentpod

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/boat-builder/agentpod/prompts"
	"github.com/openai/openai-go"
)

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

func (a *Agent) SkillContextRunner(ctx context.Context, messageHistory *MessageList, llm *LLM, outChan chan Response, memoryBlock *MemoryBlock, skill *Skill, skillToolCallID string) (openai.ChatCompletionMessageParamUnion, error) {
	a.logger.Info("Running skill", "skill", skill.Name)

	promptData := prompts.SkillContextRunnerPromptData{
		MainAgentSystemPrompt: a.prompt,
		SkillSystemPrompt:     skill.SystemPrompt,
		MemoryBlocks:          memoryBlock.Parse(),
	}
	systemPrompt, err := prompts.SkillContextRunnerPrompt(promptData)
	if err != nil {
		a.logger.Error("Error getting system prompt", "error", err)
		return nil, err
	}
	messageHistory.AddFirst(systemPrompt)

	isFirstIteration := true
	for {
		// First iteration is when the main planning happens - use the bigger model.
		modelToUse := llm.SmallReasoningModel
		if isFirstIteration {
			modelToUse = llm.ReasoningModel
			isFirstIteration = false
		}

		params := openai.ChatCompletionNewParams{
			Messages:        openai.F(messageHistory.All()),
			Model:           openai.F(modelToUse),
			ReasoningEffort: openai.F(openai.ChatCompletionReasoningEffortHigh),
		}
		a.logger.Info("Running skill", "skill", skill.Name, "tools", skill.Tools)
		if len(skill.GetTools()) > 0 {
			params.Tools = openai.F(skill.GetTools())
		}

		// we need this because we need to send thoughts to the user. The thoughts sending go routine
		// doesn't get the tool calls from here tool calls but instead as an assistant message
		messageHistoryBeforeLLMCall := messageHistory.Clone()

		completion, err := llm.New(ctx, params)
		if err != nil {
			a.logger.Error("Error calling LLM while running skill", "error", err)
			return MessageWhenToolErrorWithRetry("Network error", skillToolCallID), err
		}
		messageHistory.Add(completion.Choices[0].Message)

		// Check if both tool call and content are non-empty
		bothToolCallAndContent := completion.Choices[0].Message.ToolCalls != nil && completion.Choices[0].Message.Content != ""
		if bothToolCallAndContent {
			a.logger.Error("Expectation is that tool call and content shouldn't both be non-empty", "message", completion.Choices[0].Message)
		}

		// if there is no tool call, break
		if completion.Choices[0].Message.ToolCalls == nil {
			break
		}

		// sending fake thoughts to the user to keep the user engaged
		toolsToCall := completion.Choices[0].Message.ToolCalls
		go a.sendThoughtsAboutTools(ctx, llm, messageHistoryBeforeLLMCall, toolsToCall, outChan)

		for _, toolCall := range completion.Choices[0].Message.ToolCalls {
			tool, err := skill.GetTool(toolCall.Function.Name)
			if err != nil {
				a.logger.Error("Error getting tool", "error", err)
				messageHistory.Add(MessageWhenToolError(toolCall.ID))
				continue
			}
			if tool.StatusMessage() != "" {
				outChan <- Response{
					Content: tool.StatusMessage(),
					Type:    ResponseTypeStatus,
				}
			}
			a.logger.Info("Tool", "tool", tool.Name(), "arguments", toolCall.Function.Arguments)
			arguments := map[string]interface{}{}
			err = json.Unmarshal([]byte(toolCall.Function.Arguments), &arguments)
			if err != nil {
				a.logger.Error("Error unmarshalling tool arguments", "error", err)
				messageHistory.Add(MessageWhenToolErrorWithRetry(err.Error(), skillToolCallID))
				continue
			}
			// TODO - model doesn't always generate valid JSON, so we need to validate the arguments and ask LLM to fix if there are errors
			output, err := tool.Execute(arguments)
			if err != nil {
				a.logger.Error("Error executing tool", "error", err)

				switch {
				case errors.As(err, &ignErr):
					// It's an IgnorableError
					messageHistory.Add(MessageWhenToolError(toolCall.ID))
					continue
				case errors.As(err, &retErr):
					// It's a RetryableError
					messageHistory.Add(MessageWhenToolErrorWithRetry(err.Error(), toolCall.ID))

				default:
					// Some other error
					messageHistory.Add(MessageWhenToolError(toolCall.ID))
					continue
				}
			} else {
				messageHistory.Add(openai.ChatCompletionToolMessageParam{
					Role:       openai.F(openai.ChatCompletionToolMessageParamRoleTool),
					Content:    openai.F([]openai.ChatCompletionContentPartTextParam{{Type: openai.F(openai.ChatCompletionContentPartTextTypeText), Text: openai.F(output)}}),
					ToolCallID: openai.F(toolCall.ID),
				})
			}
		}
	}
	allMessages := messageHistory.All()
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
