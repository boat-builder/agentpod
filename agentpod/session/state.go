package session

import "github.com/boat-builder/agentpod/llm"

type SessionState struct {
	MessageHistory *llm.MessageList
}

func NewSessionState() *SessionState {
	return &SessionState{
		MessageHistory: llm.NewMessageList(),
	}
}
