// Package session provides the Session struct for per-conversation state,
// along with methods for handling user messages and producing agent outputs.
package session

import (
	"context"
	"log/slog"
	"sync"

	"github.com/boat-builder/agentpod/agent"
	"github.com/boat-builder/agentpod/llm"
	"github.com/boat-builder/agentpod/memory"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
)

// Session holds ephemeral conversation data & references to global resources.
type Session struct {
	userID    string
	sessionID string

	llm *openai.Client
	mem memory.Memory
	ag  *agent.Agent

	// Fields for conversation history, ephemeral context, partial results, etc.
	inUserChannel  chan string
	outUserChannel chan Message

	modelName string

	// New fields for graceful shutdown
	ctx       context.Context
	cancel    context.CancelFunc
	closeOnce sync.Once

	logger *slog.Logger
}

// NewSession constructs a session with references to shared LLM & memory, but isolated state.
func NewSession(ctx context.Context, userID, sessionID string, llmConfig llm.LLMConfig, mem memory.Memory, ag *agent.Agent) *Session {
	var llmClient *openai.Client
	if llmConfig.BaseURL != "" {
		llmClient = openai.NewClient(option.WithBaseURL(llmConfig.BaseURL), option.WithAPIKey(llmConfig.APIKey))
	} else {
		llmClient = openai.NewClient(option.WithAPIKey(llmConfig.APIKey))
	}
	ag.SetLLM(llmClient, llmConfig.Model)
	ctx, cancel := context.WithCancel(ctx)
	s := &Session{
		userID:         userID,
		sessionID:      sessionID,
		llm:            llmClient,
		mem:            mem,
		ag:             ag,
		inUserChannel:  make(chan string),
		outUserChannel: make(chan Message),
		modelName:      llmConfig.Model,
		ctx:            ctx,
		cancel:         cancel,
		logger:         slog.Default(),
	}
	go s.run()
	return s
}

// In processes incoming user messages. Could queue or immediately handle them.
func (s *Session) In(userMessage string) {
	s.inUserChannel <- userMessage
}

// Out retrieves the next message from the output channel, blocking until a message is available.
func (s *Session) Out() Message {
	return <-s.outUserChannel
}

// Close ends the session lifecycle and releases any resources if needed.
func (s *Session) Close() {
	s.closeOnce.Do(func() {
		s.cancel()
		close(s.inUserChannel)
	})
}

// run is the main loop for the session. It listens for user messages and process here. Although
// we don't support now, the idea is that session should support interactive mode which is why
// the input channel exists. Session should hold the control of how to route the messages to whichever agents
// when we support multiple agents.
// TODO - handle refusal everywhere
// TODO - handle other errors like network errors everywhere
func (s *Session) run() {
	defer s.Close()
	select {
	case <-s.ctx.Done():
		s.outUserChannel <- Message{Type: MessageTypeEnd}
	case userMessage, ok := <-s.inUserChannel:
		if !ok {
			s.logger.Error("Session input channel closed")
			s.outUserChannel <- Message{Type: MessageTypeEnd}
			return
		}
		completion := openai.ChatCompletionAccumulator{}
		outAgentChannel, err := s.ag.Run(userMessage)
		if err != nil {
			s.outUserChannel <- Message{
				Content: err.Error(),
				Type:    MessageTypeError,
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
				s.outUserChannel <- Message{
					Content: chunk.Choices[0].Delta.Content,
					Type:    MessageTypePartialText,
				}
			}
		}

		// TODO - Handle any errors from the stream

		// channel is closed, send the final message
		s.outUserChannel <- Message{
			Type: MessageTypeEnd,
		}
	}
}

// TokenRates defines cost per million tokens for input and output
type TokenRates struct {
	Input  float64
	Output float64
}

// Pricing constants for GPT-4o and GPT-4o-mini (in dollars per million tokens)
const (
	GPT4oInputRate      = 2.5
	GPT4oOutputRate     = 10.0
	GPT4oMiniInputRate  = 0.15
	GPT4oMiniOutputRate = 0.60
)

// ModelPricings is a map of model names to their pricing information
var ModelPricings = map[string]TokenRates{
	"gpt-4o": {
		Input:  GPT4oInputRate,
		Output: GPT4oOutputRate,
	},
	"azure/gpt-4o": {
		Input:  GPT4oInputRate,
		Output: GPT4oOutputRate,
	},
	"gpt-4o-mini": {
		Input:  GPT4oMiniInputRate,
		Output: GPT4oMiniOutputRate,
	},
	"azure/gpt-4o-mini": {
		Input:  GPT4oMiniInputRate,
		Output: GPT4oMiniOutputRate,
	},
}

// CostDetails represents detailed cost information for a session
type CostDetails struct {
	InputTokens  int64
	OutputTokens int64
	TotalCost    float64
}

// Cost returns the accumulated cost of the session.
// It calculates the cost based on the total input and output tokens and the pricing for the session's model.
// TODO
func (s *Session) Cost() (*CostDetails, bool) {
	pricing, exists := ModelPricings[s.modelName]
	if !exists {
		return nil, false
	}

	inputCost := float64(0) * pricing.Input / 1000000
	outputCost := float64(0) * pricing.Output / 1000000
	totalCost := inputCost + outputCost

	return &CostDetails{
		InputTokens:  0,
		OutputTokens: 0,
		TotalCost:    totalCost,
	}, true
}
