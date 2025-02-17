package agentpod

// Session represents an active session for the agent.
type Session struct {
	ID     string
	Agent  AgentInterface
	Active bool
	// add additional session details if needed
}

// NewSession creates a new session with the provided ID and agent.
func NewSession(id string, agent AgentInterface) *Session {
	return &Session{
		ID:     id,
		Agent:  agent,
		Active: true,
	}
}

// End terminates the session.
func (s *Session) End() {
	s.Active = false
	// add further cleanup if necessary
}

// IsActive returns whether the session is currently active.
func (s *Session) IsActive() bool {
	return s.Active
}
