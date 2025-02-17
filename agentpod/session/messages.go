// Package session - messages.go
// Defines data structures for messages exchanged during a session.

package session

type MessageType string

const (
	MessageTypeStatus       MessageType = "status"
	MessageTypeResult       MessageType = "result"
	MessageTypeInputRequest MessageType = "input-request"
	MessageTypeError        MessageType = "error"
)

// Message represents a communication unit from the Session/Agent to the caller/UI.
type Message struct {
	Content string
	Type    MessageType
	// Additional fields: e.g. isFinal, timestamps, etc.
}
