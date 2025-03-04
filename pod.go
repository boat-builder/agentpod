package agentpod

import (
	"context"
	"log/slog"

	"github.com/openai/openai-go"
)

type Pod struct {
	llmConfig *LLMConfig
	Mem       Memory
	Agent     Agent
	logger    *slog.Logger
	storage   Storage
}

type UserInfo struct {
	Name       string
	CustomMeta map[string]string
}

// NewPod constructs a new Pod with the given resources.
func NewPod(llmConfig *LLMConfig, mem Memory, ag *Agent, storage Storage) *Pod {
	return &Pod{
		llmConfig: llmConfig,
		Mem:       mem,
		Agent:     *ag,
		logger:    slog.Default(),
		storage:   storage,
	}
}

// NewSession creates a new conversation session for a given user and session ID.
// A session handles a single user message and maintains the internal state of the agents
// as they interact to generate a response.
func (p *Pod) NewSession(ctx context.Context, customerID, sessionID string, customMeta map[string]string) *Session {
	sess := newSession(ctx, customerID, sessionID, customMeta, p.llmConfig.Model)
	go p.run(sess)
	return sess
}

// run is the main loop for the session. It listens for user messages and process here. Although
// we don't support now, the idea is that session should support interactive mode which is why
// the input channel exists. Session should hold the control of how to route the messages to whichever agents
// when we support multiple agents.
// TODO - handle refusal everywhere
// TODO - handle other errors like network errors everywhere
func (p *Pod) run(sess *Session) {
	defer sess.Close()
	send_status_func := func(msg string) {
		sess.OutUserChannel <- Response{
			Content: msg,
			Type:    ResponseTypeStatus,
		}
	}
	select {
	case <-sess.Ctx.Done():
		sess.OutUserChannel <- Response{Type: ResponseTypeEnd}
	case userMessage, ok := <-sess.InUserChannel:
		if !ok {
			p.logger.Error("Session input channel closed")
			sess.OutUserChannel <- Response{Type: ResponseTypeEnd}
			return
		}
		completion := openai.ChatCompletionAccumulator{}

		outAgentChannel, err := p.Agent.Run(
			sess.Ctx,
			sess.WithUserMessage(userMessage),
			p.llmConfig.NewLLMClient(),
			p.llmConfig.Model,
			send_status_func,
			p.storage,
		)
		if err != nil {
			sess.OutUserChannel <- Response{
				Content: err.Error(),
				Type:    ResponseTypeError,
			}
		}
		var openAIMessageID string
		for chunk := range outAgentChannel {
			// when chunk id is not same as the previous one, it's part of a new message. Reset everything.
			if chunk.ID != openAIMessageID {
				openAIMessageID = chunk.ID
				completion = openai.ChatCompletionAccumulator{}
			}
			completion.AddChunk(chunk)
			// We won't send the message as a "final message" because there can be other streams in progress.
			// We'll wait for the channel to close
			if len(chunk.Choices) > 0 && chunk.Choices[0].Delta.Content != "" {
				sess.OutUserChannel <- Response{
					Content: chunk.Choices[0].Delta.Content,
					Type:    ResponseTypePartialText,
				}
			}
		}

		// channel is closed, send the final message
		sess.OutUserChannel <- Response{
			Type: ResponseTypeEnd,
		}
	}
}
