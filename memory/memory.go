// Package memory provides an interface for storing and retrieving conversation data.
package memory

// Memory is an interface for reading/writing conversation data or other context.
type Memory interface {
	Retrieve(userID, sessionID, key string) (string, error)
	Update(userID, sessionID, key, value string) error
}

// DBMemory is a placeholder for a database-based memory implementation.
type DBMemory struct {
	// Possibly store DB connections here, e.g. *sql.DB or a client
}

func NewDBMemory( /* db config */ ) *DBMemory {
	return &DBMemory{}
}

func (dbm *DBMemory) Retrieve(userID, sessionID, key string) (string, error) {
	// Implementation stub: query DB for row matching userID/sessionID/key
	return "", nil
}

func (dbm *DBMemory) Update(userID, sessionID, key, value string) error {
	// Implementation stub: upsert or update DB record
	return nil
}
