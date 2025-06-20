package agentpod

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"

	"github.com/boat-builder/agentpod/prompts"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/param"
)

const maxSkillLoops = 25

func MessageWhenToolError(toolCallID string) openai.ChatCompletionMessageParamUnion {
	return openai.ToolMessage("Error occurred while running. Do not retry", toolCallID)
}

func MessageWhenToolErrorWithRetry(errorString string, toolCallID string) openai.ChatCompletionMessageParamUnion {
	return openai.ToolMessage(fmt.Sprintf("Error: %s.\nRetry", errorString), toolCallID)
}

func (a *Agent) SkillContextRunner(ctx context.Context, messageHistory *MessageList, llm LLM, memoryBlock *MemoryBlock, skill *Skill, skillToolCall openai.ChatCompletionMessageToolCall) (*openai.ChatCompletionToolMessageParam, error) {
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

	// Extract the "instruction" argument from the tool call and append it as a user message so that the LLM
	// inside the skill context clearly understands the task it needs to perform.
	if skillToolCall.Function.Arguments != "" {
		var toolArgs map[string]interface{}
		if err := json.Unmarshal([]byte(skillToolCall.Function.Arguments), &toolArgs); err == nil {
			if instr, ok := toolArgs["instruction"].(string); ok && instr != "" {
				messageHistory.Add(UserMessage(instr))
			}
		} else {
			a.logger.Error("Error unmarshalling instruction from tool call arguments", "error", err)
		}
	}

	var (
		hasStopToolCall  bool
		stopToolResponse string
	)

	for i := 0; ; i++ {
		if i >= maxSkillLoops {
			a.logger.Error("skill has reached max loop count", "skill", skill.Name, "count", maxSkillLoops)
			return openai.ToolMessage("Error: The skill exceeded maximum allowed iterations and was stopped.", skillToolCall.ID).OfTool, fmt.Errorf("skill %s exceeded max loop iterations", skill.Name)
		}

		// Build the list of tools exposed to the skill-level LLM. Always include the
		// stop tool so that the model can explicitly finish execution when needed.
		tools := []openai.ChatCompletionToolParam{a.StopTool()}
		if len(skill.GetTools()) > 0 {
			tools = append(tools, skill.GetTools()...)
		}

		params := openai.ChatCompletionNewParams{
			Messages:        messageHistory.All(),
			Model:           llm.StrongModel(),
			ReasoningEffort: "high",
			ToolChoice:      openai.ChatCompletionToolChoiceOptionUnionParam{OfAuto: param.Opt[string]{Value: "required"}},
			Tools:           tools,
		}

		completion, err := llm.New(ctx, params)
		if err != nil {
			a.logger.Error("Error calling LLM while running skill", "error", err)
			return MessageWhenToolErrorWithRetry("Network error", skillToolCall.ID).OfTool, err
		}

		// Separate stop tool calls (if any) from other tool calls so that we can
		// execute only the skill tools while respecting the stop request.
		skillToolCalls := []openai.ChatCompletionMessageToolCall{}

		if completion.Choices[0].Message.ToolCalls != nil {
			for _, tc := range completion.Choices[0].Message.ToolCalls {
				if tc.Function.Name == "stop" {
					hasStopToolCall = true
					if tc.Function.Arguments != "" {
						var args map[string]interface{}
						if err := json.Unmarshal([]byte(tc.Function.Arguments), &args); err == nil {
							if resp, ok := args["response"].(string); ok {
								stopToolResponse = resp
							}
						}
					}
				} else {
					skillToolCalls = append(skillToolCalls, tc)
				}
			}
		}

		// Add the assistant message to history but filter out the stop tool call so that
		// subsequent reasoning cycles don't see it again.
		messageToAdd := completion.Choices[0].Message
		if messageToAdd.ToolCalls != nil {
			filtered := []openai.ChatCompletionMessageToolCall{}
			for _, tc := range messageToAdd.ToolCalls {
				if tc.Function.Name != "stop" {
					filtered = append(filtered, tc)
				}
			}
			if len(filtered) > 0 {
				messageToAdd.ToolCalls = filtered
				messageHistory.Add(messageToAdd.ToParam())
			}
		} else {
			messageHistory.Add(messageToAdd.ToParam())
		}

		toolsToCall := skillToolCalls

		// Create a wait group to wait for all tool executions to complete
		var wg sync.WaitGroup
		// Create a channel to collect results from goroutines
		resultsChan := make(chan *openai.ChatCompletionToolMessageParam, len(toolsToCall))

		for _, toolCall := range toolsToCall {
			a.logger.Info("Running tool for the skill", "skill", skill.Name, "tool", toolCall.Function.Name)
			wg.Add(1)
			go func(toolCall openai.ChatCompletionMessageToolCall) {
				defer wg.Done()

				tool, err := skill.GetTool(toolCall.Function.Name)
				if err != nil {
					a.logger.Error("Error getting tool", "error", err)
					resultsChan <- nil
					return
				}

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

		// If a stop tool was requested, break out of the loop after processing the
		// remaining tool calls.
		if hasStopToolCall {
			break
		}

		if completion.Choices[0].Message.ToolCalls == nil {
			// The model returned no tool calls, meaning it provided a direct answer. We can
			// exit early as there is nothing left to execute.
			break
		}
	}

	// If stop tool provided a response, return it.
	if stopToolResponse != "" {
		return openai.ToolMessage(stopToolResponse, skillToolCall.ID).OfTool, nil
	}

	a.logger.Error("Unexpected situation in SkillContextRunner result. Function is done but stop response is empty")
	return openai.ToolMessage("Error: The skill execution did not produce a valid response", skillToolCall.ID).OfTool, nil
}
