package agentpod

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"

	"github.com/boat-builder/agentpod/prompts"
	"github.com/openai/openai-go"
)

func MessageWhenToolError(toolCallID string) openai.ChatCompletionMessageParamUnion {
	return openai.ToolMessage("Error occurred while running. Do not retry", toolCallID)
}

func MessageWhenToolErrorWithRetry(errorString string, toolCallID string) openai.ChatCompletionMessageParamUnion {
	return openai.ToolMessage(fmt.Sprintf("Error: %s.\nRetry", errorString), toolCallID)
}

func (a *Agent) SkillContextRunner(ctx context.Context, messageHistory *MessageList, llm LLM, outChan chan Response, memoryBlock *MemoryBlock, skill *Skill, skillToolCallID string) (*openai.ChatCompletionToolMessageParam, error) {
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

	for {
		params := openai.ChatCompletionNewParams{
			Messages:        messageHistory.All(),
			Model:           llm.StrongModel(),
			ReasoningEffort: "high",
		}
		a.logger.Info("Running skill", "skill", skill.Name, "tools", skill.Tools)
		if len(skill.GetTools()) > 0 {
			params.Tools = skill.GetTools()
		}

		completion, err := llm.New(ctx, params)
		if err != nil {
			a.logger.Error("Error calling LLM while running skill", "error", err)
			return MessageWhenToolErrorWithRetry("Network error", skillToolCallID).OfTool, err
		}
		messageHistory.Add(completion.Choices[0].Message.ToParam())

		// Check if both tool call and content are non-empty
		bothToolCallAndContent := completion.Choices[0].Message.ToolCalls != nil && completion.Choices[0].Message.Content != ""
		if bothToolCallAndContent {
			a.logger.Error("Expectation is that tool call and content shouldn't both be non-empty", "message", completion.Choices[0].Message)
		}

		// if there is no tool call, break
		if completion.Choices[0].Message.ToolCalls == nil {
			break
		}
		toolsToCall := completion.Choices[0].Message.ToolCalls

		// Create a wait group to wait for all tool executions to complete
		var wg sync.WaitGroup
		// Create a channel to collect results from goroutines
		resultsChan := make(chan *openai.ChatCompletionToolMessageParam, len(toolsToCall))

		for _, toolCall := range toolsToCall {
			wg.Add(1)
			go func(toolCall openai.ChatCompletionMessageToolCall) {
				defer wg.Done()

				tool, err := skill.GetTool(toolCall.Function.Name)
				if err != nil {
					a.logger.Error("Error getting tool", "error", err)
					resultsChan <- nil
					return
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
					resultsChan <- nil
					return
				}

				output, err := tool.Execute(ctx, arguments)
				if err != nil {
					a.logger.Error("Error executing tool", "error", err)
					resultsChan <- nil
					return
				}

				resultsChan <- openai.ToolMessage(output, toolCall.ID).OfTool
			}(toolCall)
		}

		// Start a goroutine to close the result channel when all tools are done
		go func() {
			wg.Wait()
			close(resultsChan)
		}()

		// Process results as they come in
		for result := range resultsChan {
			if result == nil {
				continue
			}

			messageHistory.Add(openai.ChatCompletionMessageParamUnion{OfTool: result})
		}
	}

	allMessages := messageHistory.All()
	lastMessage := allMessages[len(allMessages)-1]
	// If it's a ChatCompletionMessage, convert it to a tool message
	if lastMessage.GetRole() != nil && *lastMessage.GetRole() == "assistant" {
		contentPtr := lastMessage.GetContent().AsAny().(*string)
		if contentPtr == nil {
			return openai.ToolMessage("Error: The skill execution did not produce a valid response", skillToolCallID).OfTool, nil
		}
		return openai.ToolMessage(*contentPtr, skillToolCallID).OfTool, nil
	} else {
		a.logger.Error("Unexpected message type in SkillContextRunner result", "type", fmt.Sprintf("%T", lastMessage))
		return openai.ToolMessage("Error: The skill execution did not produce a valid response", skillToolCallID).OfTool, nil
	}
}
