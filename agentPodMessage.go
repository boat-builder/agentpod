package agentpod

type MessageType string

const (
	MessageTypeStatus       MessageType = "status"
	MessageTypePartialText  MessageType = "partial-text"
	MessageTypeEnd          MessageType = "end"
	MessageTypeInputRequest MessageType = "input-request"
	MessageTypeError        MessageType = "error"
)

// Message represents a communication unit from the Session/Agent to the caller/UI.
type Message struct {
	Content string
	Type    MessageType
}
