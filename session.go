// Package session provides the Session struct for per-conversation state,
// along with methods for handling user messages and producing agent outputs.
package agentpod

import (
	"context"
	"log/slog"
	"sync"
)

type Meta struct {
	CustomerID string
	SessionID  string
	Extra      map[string]string
}

type UserInfo struct {
	Name string
	Meta map[string]string
}

// Session holds ephemeral conversation data & references to global resources.
type Session struct {
	ctx       context.Context
	cancel    context.CancelFunc
	closeOnce sync.Once

	inUserChannel  chan string
	outUserChannel chan Response

	llm     *LLM
	memory  Memory
	agent   *Agent
	storage Storage

	meta Meta

	logger *slog.Logger
}

// NewSession constructs a session with references to shared LLM & memory, but isolated state.
func NewSession(ctx context.Context, llm *LLM, mem Memory, ag *Agent, storage Storage, meta Meta) *Session {
	ctx, cancel := context.WithCancel(ctx)
	ctx = context.WithValue(ctx, ContextKey("customerID"), meta.CustomerID)
	ctx = context.WithValue(ctx, ContextKey("sessionID"), meta.SessionID)
	ctx = context.WithValue(ctx, ContextKey("extra"), meta.Extra)
	s := &Session{
		ctx:       ctx,
		cancel:    cancel,
		closeOnce: sync.Once{},

		inUserChannel:  make(chan string),
		outUserChannel: make(chan Response),

		llm:     llm,
		memory:  mem,
		agent:   ag,
		storage: storage,

		meta: meta,

		logger: slog.Default(),
	}
	go s.run()
	return s
}

// In processes incoming user messages. Could queue or immediately handle them.
func (s *Session) In(userMessage string) {
	s.inUserChannel <- userMessage
}

// Out retrieves the next message from the output channel, blocking until a message is available.
func (s *Session) Out() Response {
	return <-s.outUserChannel
}

// Close ends the session lifecycle and releases any resources if needed.
func (s *Session) Close() {
	s.closeOnce.Do(func() {
		s.cancel()
		close(s.inUserChannel)
	})
}

func (s *Session) sendStatus(msg string) {
	s.outUserChannel <- Response{
		Content: msg,
		Type:    ResponseTypeStatus,
	}
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
		s.outUserChannel <- Response{Type: ResponseTypeEnd}
	case userMessage, ok := <-s.inUserChannel:
		if !ok {
			s.logger.Error("Session input channel closed")
			s.outUserChannel <- Response{Type: ResponseTypeEnd}
			return
		}
		err := s.storage.CreateConversation(s.meta, userMessage)
		if err != nil {
			s.logger.Error("Error creating conversation", "error", err)
		}

		outAgentChannel, err := s.agent.Run(s.ctx, s.llm, s.sendStatus, s.storage)
		if err != nil {
			s.outUserChannel <- Response{
				Content: err.Error(),
				Type:    ResponseTypeError,
			}
		}
		aggregated := ""
		for chunk := range outAgentChannel {
			// We'll wait for the channel to close
			if len(chunk) > 0 {
				aggregated += chunk
				s.outUserChannel <- Response{
					Content: chunk,
					Type:    ResponseTypePartialText,
				}
			}
		}

		// Finish the conversation in the store
		err = s.storage.FinishConversation(s.meta, aggregated)
		if err != nil {
			s.logger.Error("Error finishing conversation", "error", err)
		}

		// channel is closed, send the final message
		s.outUserChannel <- Response{
			Type: ResponseTypeEnd,
		}
	}
}
