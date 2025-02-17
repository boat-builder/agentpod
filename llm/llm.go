// Package llm provides interfaces and stubs for language model integrations.
package llm

// LLM is an interface for sending prompts to a language model and getting responses.
type LLM interface {
	ProcessPrompt(prompt string) (string, error)
}

// MockLLM is a stub for testing or demonstration.
type MockLLM struct{}

func NewMockLLM() *MockLLM {
	return &MockLLM{}
}

func (m *MockLLM) ProcessPrompt(prompt string) (string, error) {
	// Implementation stub
	return "mock response", nil
}
