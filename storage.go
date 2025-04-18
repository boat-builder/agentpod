package agentpod

import "sync"

// Storage is an interface that abstracts the user storage layer. For agentpod, a conversation is a pair of
// user messages and assistant messages.
type Storage interface {
	// conversation related
	// GetConversations should return the conversations in the order of creating them.
	// The first message in the returned list must be older than the second message in the list.
	// Be careful on applying limit and offset. If the limit is 10 and offset is 5, it means
	// we'll do the offset from the end of the conversation (i.e., skip the last 5 conversations
	// in the whole chat history) and then take the 10 messages from that point backwards and
	// return a list of those 10 messages arranged in the described order.
	GetConversations(meta Meta, limit int, offset int) (*MessageList, error)
	CreateConversation(meta Meta, userMessage string) error
	FinishConversation(meta Meta, assistantMessage string) error
}

// InMemoryStorage implements the Storage interface using in-memory data structures
type InMemoryStorage struct {
	mu            sync.RWMutex
	conversations []*conversation
}

type conversation struct {
	sessionID        string
	userMessage      string
	assistantMessage string
}

// NewInMemoryStorage creates a new instance of InMemoryStorage
func NewInMemoryStorage() *InMemoryStorage {
	return &InMemoryStorage{
		conversations: make([]*conversation, 0),
	}
}

// GetConversations returns the conversations in the order they were created
func (s *InMemoryStorage) GetConversations(meta Meta, limit int, offset int) (*MessageList, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	messageList := &MessageList{}

	// Calculate the start and end indices for the conversations we want to return
	start := len(s.conversations) - offset - limit
	if start < 0 {
		start = 0
	}
	end := len(s.conversations) - offset
	if end < 0 {
		end = 0
	}

	// Iterate through the conversations in reverse order
	for i := end - 1; i >= start; i-- {
		conv := s.conversations[i]
		if conv.userMessage != "" {
			messageList.Add(UserMessage(conv.userMessage))
		}
		if conv.assistantMessage != "" {
			messageList.Add(AssistantMessage(conv.assistantMessage))
		}
	}

	return messageList, nil
}

// CreateConversation creates a new conversation with the user message
func (s *InMemoryStorage) CreateConversation(meta Meta, userMessage string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Check if conversation already exists
	for _, conv := range s.conversations {
		if conv.sessionID == meta.SessionID {
			conv.userMessage = userMessage
			return nil
		}
	}

	// Create new conversation
	s.conversations = append(s.conversations, &conversation{
		sessionID:   meta.SessionID,
		userMessage: userMessage,
	})

	return nil
}

// FinishConversation adds the assistant message to the existing conversation
func (s *InMemoryStorage) FinishConversation(meta Meta, assistantMessage string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Find the conversation and add the assistant message
	for _, conv := range s.conversations {
		if conv.sessionID == meta.SessionID {
			conv.assistantMessage = assistantMessage
			return nil
		}
	}

	// If conversation doesn't exist, create it with just the assistant message
	s.conversations = append(s.conversations, &conversation{
		sessionID:        meta.SessionID,
		assistantMessage: assistantMessage,
	})

	return nil
}
