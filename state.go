package agentpod

type SessionState struct {
	MessageHistory *MessageList
}

func NewSessionState() *SessionState {
	return &SessionState{
		MessageHistory: NewMessageList(),
	}
}
