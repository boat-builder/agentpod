// Package session provides the Session struct for per-conversation state,
// along with methods for handling user messages and producing agent outputs.
package session

import (
	"context"
	"log"

	"github.com/boat-builder/agentpod/agent"
	"github.com/boat-builder/agentpod/memory"
	"github.com/openai/openai-go"
)

// Session holds ephemeral conversation data & references to global resources.
type Session struct {
	userID    string
	sessionID string

	llm   openai.Client
	mem   memory.Memory
	agent *agent.Agent

	// Fields for conversation history, ephemeral context, partial results, etc.
	inChannel  chan string
	outChannel chan Message
}

// NewSession constructs a session with references to shared LLM & memory, but isolated state.
func NewSession(userID, sessionID string, llmClient openai.Client, mem memory.Memory, ag *agent.Agent) *Session {
	s := &Session{
		userID:     userID,
		sessionID:  sessionID,
		llm:        llmClient,
		mem:        mem,
		agent:      ag,
		inChannel:  make(chan string),
		outChannel: make(chan Message),
	}
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
	// Implementation: flush conversation history, finalize logs, etc.
}

// Run processes messages from the input channel, performs chat completion, and sends results to the output channel.
func (s *Session) run() {
	for userMessage := range s.inChannel {
		completion, err := s.llm.Chat.Completions.New(context.Background(), openai.ChatCompletionNewParams{
			Messages: openai.F([]openai.ChatCompletionMessageParamUnion{
				openai.UserMessage(userMessage),
			}),
			Model: openai.F(openai.ChatModelGPT4o),
		})
		if err != nil {
			log.Printf("Error during chat completion: %v", err)
			continue
		}
		msg := Message{
			Content: completion.Choices[0].Message.Content,
			Type:    "output", // TODO: Replace with an appropriate MessageType constant if available
		}
		s.outChannel <- msg
	}
}
