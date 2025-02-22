// Package memory provides an interface for storing and retrieving conversation data.
package agentpod

// Memory is an interface for reading/writing conversation data or other context.
type Memory interface {
	Retrieve(userID, sessionID, key string) (string, error)
	Update(userID, sessionID, key, value string) error
}

// ZepMemory is a placeholder for a database-based memory implementation.
type Zep struct{}

func NewZep() *Zep {
	return &Zep{}
}

func (z *Zep) Retrieve(userID, sessionID, key string) (string, error) {
	// Implementation stub: query DB for row matching userID/sessionID/key
	return "", nil
}

func (z *Zep) Update(userID, sessionID, key, value string) error {
	// Implementation stub: upsert or update DB record
	return nil
}
