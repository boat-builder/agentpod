// Package agent provides the main Agent orchestrator, which uses LLM & Skills to process data.
package agent

import (
	"context"

	"github.com/boat-builder/agentpod/llm"
	"github.com/openai/openai-go"
)

// Agent orchestrates calls to the LLM, uses Skills/Tools, and determines how to respond.
type Agent struct {
	skills    []Skill
	messages  llm.MessageList
	llmClient *openai.Client
	modelName string
}

// NewAgent creates an Agent by adding the prompt as a DeveloperMessage.
func NewAgent(prompt string, skills []Skill) *Agent {
	devMsg := &llm.DeveloperMessage{Content: prompt}
	msgList := llm.MessageList{}
	msgList.Add(devMsg)

	return &Agent{
		skills:   skills,
		messages: msgList,
	}
}

func (a *Agent) SetLLM(llmClient *openai.Client, modelName string) {
	a.llmClient = llmClient
	a.modelName = modelName
}

// TODO - we probably need our own ssestream wrapper
// Run returns a stream of chat completion chunks. We don't do the streaming with channels like the session do
// because session is the one that tracks a session's life cycle. We still need to figure out how to route
// the intermediate input messages if "interactive=true" but the whole idea is Agent's will not have to deal
// with the lifecycle events like interactiveness with the end user which is the abstraction openai.client has
func (a *Agent) Run(userMessage string) (chan openai.ChatCompletionChunk, error) {
	if a.llmClient == nil {
		panic("llmClient is not set")
	}
	outAgentChannel := make(chan openai.ChatCompletionChunk)
	a.messages.Add(&llm.UserMessage{Content: userMessage})

	go func() {
		defer close(outAgentChannel)
		stream := a.llmClient.Chat.Completions.NewStreaming(context.Background(), openai.ChatCompletionNewParams{
			Messages: openai.F(a.messages.OpenAIMessages()),
			Model:    openai.F(a.modelName),
			StreamOptions: openai.F(openai.ChatCompletionStreamOptionsParam{
				IncludeUsage: openai.F(true),
			}),
		})
		defer stream.Close()
		for stream.Next() {
			chunk := stream.Current()
			outAgentChannel <- chunk
		}

		// TODO - If the stream ended without the final message, check once more
		// if content, finished := completion.JustFinishedContent(); finished {
		// 	s.outUserChannel <- Message{
		// 		Content: content,
		// 		Type:    MessageTypeEnd,
		// 	}
		// }

		// TODO - handle errors
		// strea.Err()
	}()

	return outAgentChannel, nil
}
