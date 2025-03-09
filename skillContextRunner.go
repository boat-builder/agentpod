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

func (a *Agent) SkillContextRunner(ctx context.Context, skill *Skill, skillToolCallID string, clonedMessages *MessageList, llmClient *LLM, modelName string, outAgentChannel chan string, send_status_func func(msg string), userInfo UserInfo) (openai.ChatCompletionMessageParamUnion, error) {
	a.logger.Info("Running skill", "skill", skill.Name)
	memoryBlocks := make(map[string]string)
	memoryBlocks["UserName"] = userInfo.Name
	for key, value := range userInfo.Meta {
		memoryBlocks[key] = value
	}

	systemPromptData := prompts.SkillContextRunnerPromptData{
		UserSystemPrompt:  a.prompt,
		SkillSystemPrompt: skill.SystemPrompt,
		MemoryBlocks:      memoryBlocks,
	}
	systemPrompt, err := prompts.SkillContextRunnerPrompt(systemPromptData)
	if err != nil {
		a.logger.Error("Error getting system prompt", "error", err)
		return nil, err
	}
	clonedMessages.AddFirst(systemPrompt)

	// TODO - we need to have some sort of hard limit for the number iterations possible

	// Initial call to have LLM think step by step before executing tools
	// var toolNames []string
	// for _, tool := range skill.Tools {
	// 	toolNames = append(toolNames, tool.Name())
	// }

	// toolsInfo := "Available tools:\n"
	// for _, name := range toolNames {
	// 	toolsInfo += fmt.Sprintf("- %s\n", name)
	// }

	// thinkingPromptText := fmt.Sprintf("You will have these tools available to you. Strategize how can you give the best possible answer using these tools:\n\n%s", toolsInfo)
	// thinkingPrompt := DeveloperMessage(thinkingPromptText)
	// clonedMessages.Add(thinkingPrompt)

	// thinkingParams := openai.ChatCompletionNewParams{
	// 	Messages: openai.F(clonedMessages.All()),
	// 	Model:    openai.F(modelName),
	// }

	// thinkingCompletion, err := llmClient.New(ctx, thinkingParams)
	// if err != nil {
	// 	a.logger.Error("Error calling LLM for step-by-step thinking", "error", err)
	// 	return MessageWhenToolErrorWithRetry("Network error", skillToolCallID), err
	// }

	// // Add the LLM's thinking to the message history
	// clonedMessages.Add(thinkingCompletion.Choices[0].Message)

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
		bothToolCallAndContent := completion.Choices[0].Message.ToolCalls != nil && completion.Choices[0].Message.Content != ""
		if bothToolCallAndContent {
			a.logger.Error("Expectation is that tool call and content shouldn't both be non-empty", "message", completion.Choices[0].Message)
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
