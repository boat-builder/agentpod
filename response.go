package agentpod

type ResponseType string

const (
	ResponseTypeEnd   ResponseType = "end"
	ResponseTypeError ResponseType = "error"
)

// Response represents a communication unit from the Agent to the caller/UI.
type Response struct {
	Content string
	Type    ResponseType
}
