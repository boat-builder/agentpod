// Package session provides the Session struct for per-conversation state,
// along with methods for handling user messages and producing agent outputs.
package session

import (
	"context"
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
	ai  *agent.Agent

	// Fields for conversation history, ephemeral context, partial results, etc.
	inChannel  chan string
	outChannel chan Message

	// New fields for tracking cost information
	accumulatedInputTokens  int64
	accumulatedOutputTokens int64
	modelName               string

	// New fields for graceful shutdown
	ctx       context.Context
	cancel    context.CancelFunc
	closeOnce sync.Once
}

// NewSession constructs a session with references to shared LLM & memory, but isolated state.
func NewSession(ctx context.Context, userID, sessionID string, llmConfig llm.LLMConfig, mem memory.Memory, ag *agent.Agent) *Session {
	var llmClient *openai.Client
	if llmConfig.BaseURL != "" {
		llmClient = openai.NewClient(option.WithBaseURL(llmConfig.BaseURL), option.WithAPIKey(llmConfig.APIKey))
	} else {
		llmClient = openai.NewClient(option.WithAPIKey(llmConfig.APIKey))
	}
	ctx, cancel := context.WithCancel(ctx)
	s := &Session{
		userID:     userID,
		sessionID:  sessionID,
		llm:        llmClient,
		mem:        mem,
		ai:         ag,
		inChannel:  make(chan string),
		outChannel: make(chan Message),
		modelName:  llmConfig.Model,
		ctx:        ctx,
		cancel:     cancel,
	}
	// Initialize cost tracking fields
	s.accumulatedInputTokens = 0
	s.accumulatedOutputTokens = 0
	go s.run()
	return s
}

// In processes incoming user messages. Could queue or immediately handle them.
func (s *Session) In(userMessage string) {
	s.inChannel <- userMessage
}

// Out retrieves the next message from the output channel, blocking until a message is available.
func (s *Session) Out() Message {
	return <-s.outChannel
}

// Close ends the session lifecycle and releases any resources if needed.
func (s *Session) Close() {
	s.closeOnce.Do(func() {
		s.cancel()
		close(s.inChannel)
	})
}

// Run processes messages from the input channel, performs chat completion, and sends results to the output channel.
func (s *Session) run() {
	defer s.Close()

	select {
	case <-s.ctx.Done():
		s.outChannel <- Message{Type: MessageTypeEnd}
	case userMessage, ok := <-s.inChannel:
		if !ok {
			s.outChannel <- Message{Type: MessageTypeEnd}
			close(s.outChannel)
			return
		}

		// Process the user message
		stream := s.llm.Chat.Completions.NewStreaming(context.Background(), openai.ChatCompletionNewParams{
			Messages: openai.F([]openai.ChatCompletionMessageParamUnion{
				openai.UserMessage(userMessage),
			}),
			Model: openai.F(s.modelName),
			StreamOptions: openai.F(openai.ChatCompletionStreamOptionsParam{
				IncludeUsage: openai.F(true),
			}),
		})

		completion := openai.ChatCompletionAccumulator{}
		for stream.Next() {
			chunk := stream.Current()
			completion.AddChunk(chunk)
			s.accumulatedInputTokens += chunk.Usage.PromptTokens
			s.accumulatedOutputTokens += chunk.Usage.CompletionTokens

			// Check if the accumulator indicates the content is complete
			if content, finished := completion.JustFinishedContent(); finished {
				s.outChannel <- Message{
					Content: content,
					Type:    MessageTypeEnd,
				}
				break
			}

			// Only send partial message if there is non-empty content
			if len(chunk.Choices) > 0 && chunk.Choices[0].Delta.Content != "" {
				s.outChannel <- Message{
					Content: chunk.Choices[0].Delta.Content,
					Type:    MessageTypePartialText,
				}
			}
		}

		// If the stream ended without the final message, check once more
		if content, finished := completion.JustFinishedContent(); finished {
			s.outChannel <- Message{
				Content: content,
				Type:    MessageTypeEnd,
			}
		}

		// Handle any errors from the stream
		if err := stream.Err(); err != nil {
			s.outChannel <- Message{
				Content: err.Error(),
				Type:    MessageTypeError,
			}
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
func (s *Session) Cost() (*CostDetails, bool) {
	pricing, exists := ModelPricings[s.modelName]
	if !exists {
		return nil, false
	}

	inputCost := float64(s.accumulatedInputTokens) * pricing.Input / 1000000
	outputCost := float64(s.accumulatedOutputTokens) * pricing.Output / 1000000
	totalCost := inputCost + outputCost

	return &CostDetails{
		InputTokens:  s.accumulatedInputTokens,
		OutputTokens: s.accumulatedOutputTokens,
		TotalCost:    totalCost,
	}, true
}
