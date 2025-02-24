// Package session provides the Session struct for per-conversation state,
// along with methods for handling user messages and producing agent outputs.
package agentpod

import (
	"context"
	"log/slog"
	"sync"
)

// Session holds ephemeral conversation data & references to global resources.
type Session struct {
	Ctx       context.Context
	Cancel    context.CancelFunc
	CloseOnce sync.Once
	// Fields for conversation history, ephemeral context, partial results, etc.
	InUserChannel  chan string
	OutUserChannel chan Message
	State          *SessionState

	logger    *slog.Logger
	modelName string
}

// NewSession constructs a session with references to shared LLM & memory, but isolated state.
// TODO - make sure the context is properly managed, propagated
func newSession(ctx context.Context, customerID, sessionID string, customMeta map[string]string, modelName string) *Session {
	state := NewSessionState()
	ctx, cancel := context.WithCancel(ctx)
	ctx = context.WithValue(ctx, ContextKey("customerID"), customerID)
	ctx = context.WithValue(ctx, ContextKey("sessionID"), sessionID)
	ctx = context.WithValue(ctx, ContextKey("customMeta"), customMeta)
	s := &Session{
		InUserChannel:  make(chan string),
		OutUserChannel: make(chan Message),
		Ctx:            ctx,
		Cancel:         cancel,
		logger:         slog.Default(),
		State:          state,
		modelName:      modelName,
	}
	return s
}

// In processes incoming user messages. Could queue or immediately handle them.
func (s *Session) In(userMessage string) {
	s.InUserChannel <- userMessage
}

// Out retrieves the next message from the output channel, blocking until a message is available.
func (s *Session) Out() Message {
	return <-s.OutUserChannel
}

// Close ends the session lifecycle and releases any resources if needed.
func (s *Session) Close() {
	s.CloseOnce.Do(func() {
		s.Cancel()
		close(s.InUserChannel)
	})
}

func (s *Session) WithUserMessage(userMessage string) *Session {
	// if message history has atleast one message, this should panic
	if s.State.MessageHistory.Len() > 0 {
		panic("message history has atleast one message")
	}
	s.State.MessageHistory.Add(UserMessage(userMessage))
	return s
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
