package agentpod

import "fmt"

// Memory interface defines the behavior for memory management.
type Memory interface {
	// Save stores a value in memory.
	Save(key string, value interface{}) error
	// Load retrieves a value from memory.
	Load(key string) (interface{}, error)
	// Delete removes a value from memory.
	Delete(key string) error
}

// DefaultMemory is a simple in-memory implementation of the Memory interface.
type DefaultMemory struct {
	storage map[string]interface{}
}

// NewDefaultMemory creates a new instance of DefaultMemory.
func NewDefaultMemory() *DefaultMemory {
	return &DefaultMemory{
		storage: make(map[string]interface{}),
	}
}

// Save stores a value in DefaultMemory.
func (m *DefaultMemory) Save(key string, value interface{}) error {
	m.storage[key] = value
	return nil
}

// Load retrieves a value from DefaultMemory.
func (m *DefaultMemory) Load(key string) (interface{}, error) {
	if val, exists := m.storage[key]; exists {
		return val, nil
	}
	return nil, fmt.Errorf("key %s not found", key)
}

// Delete removes a value from DefaultMemory.
func (m *DefaultMemory) Delete(key string) error {
	delete(m.storage, key)
	return nil
}
