// Package session provides the Session struct for per-conversation state,
// along with methods for handling user messages and producing agent outputs.
package agentpod

import (
	"context"
	"log/slog"
	"sync"

	gonanoid "github.com/matoous/go-nanoid/v2"
)

// Session holds ephemeral conversation data & references to global resources.
type Session struct {
	ctx       context.Context
	cancel    context.CancelFunc
	closeOnce sync.Once

	inUserChannel  chan string
	outUserChannel chan Response

	llm    LLM
	memory Memory
	agent  *Agent

	logger *slog.Logger
}

// NewSession constructs a session with references to shared LLM & memory, but isolated state.
func NewSession(ctx context.Context, llm LLM, mem Memory, ag *Agent) *Session {
	sessionID, err := gonanoid.New()
	if err != nil {
		panic(err)
	}
	ctx, cancel := context.WithCancel(ctx)
	ctx = context.WithValue(ctx, ContextKey("sessionID"), sessionID)
	s := &Session{
		ctx:       ctx,
		cancel:    cancel,
		closeOnce: sync.Once{},

		inUserChannel:  make(chan string),
		outUserChannel: make(chan Response),

		llm:    llm,
		memory: mem,
		agent:  ag,

		logger: slog.Default(),
	}
	go s.run()
	return s
}

func (s *Session) ID() string {
	return s.ctx.Value(ContextKey("sessionID")).(string)
}

// In processes incoming user messages. Could queue or immediately handle them.
func (s *Session) In(userMessage string) {
	s.inUserChannel <- userMessage
}

// Out retrieves the next message from the output channel, blocking until a message is available.
func (s *Session) Out() Response {
	response := <-s.outUserChannel
	return response
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
	s.logger.Info("Session started", "sessionID", s.ctx.Value(ContextKey("sessionID")))
	defer s.Close()
	storage := NewInMemoryStorage()
	select {
	case <-s.ctx.Done():
		s.outUserChannel <- Response{Type: ResponseTypeEnd}
	case userMessage, ok := <-s.inUserChannel:
		if !ok {
			s.logger.Error("Session input channel closed")
			s.outUserChannel <- Response{Type: ResponseTypeEnd}
			return
		}
		err := storage.AddUserMessage(s.ctx, userMessage)
		if err != nil {
			s.logger.Error("Error creating conversation", "error", err)
		}

		// Prepare session message history and validate state
		messageHistory, err := CompileConversationHistory(s.ctx, storage)
		if err != nil {
			s.logger.Error("Error compiling conversation history", "error", err)
			return
		}

		memoryBlock, err := s.memory.Retrieve(s.ctx)
		if err != nil {
			s.logger.Error("Error getting user info", "error", err)
			return
		}

		// We use a two-channel approach to ensure proper message aggregation:
		// 1. An internal channel receives all agent responses
		// 2. These responses are processed sequentially in this goroutine
		// 3. Messages are aggregated here before being sent to storage
		// This prevents race conditions between aggregation and storage operations
		internalChannel := make(chan Response)
		var aggregatedResponse string

		// Ensure channel is closed when we're done with it
		defer close(internalChannel)

		go s.agent.Run(s.ctx, s.llm, messageHistory, memoryBlock, internalChannel)

		for response := range internalChannel {
			s.outUserChannel <- response
			if response.Type == ResponseTypePartialText {
				aggregatedResponse += response.Content
			}
			if response.Type == ResponseTypeEnd {
				break
			}
		}

		// Finish the conversation in the store with the fully aggregated response
		err = storage.AddAssistantMessage(s.ctx, aggregatedResponse)
		if err != nil {
			s.logger.Error("Error finishing conversation", "error", err)
		}

		// Run method is done, send the final message
		s.outUserChannel <- Response{
			Type: ResponseTypeEnd,
		}
	}
}
