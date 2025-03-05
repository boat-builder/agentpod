package agentpod

import (
	"database/sql"
	"fmt"
	"time"

	_ "github.com/mattn/go-sqlite3"
)

var _ Storage = &SQLiteStorage{}

// SQLiteStorage implements the Storage interface using SQLite.
// It provides functionality to store and retrieve conversation data.
type SQLiteStorage struct {
	db *sql.DB
}

// NewSQLiteStorage creates a new SQLiteStorage instance with the provided database file path.
// It initializes the database schema if it doesn't exist.
func NewSQLiteStorage(dbPath string) (*SQLiteStorage, error) {
	db, err := sql.Open("sqlite3", dbPath)
	if err != nil {
		return nil, fmt.Errorf("failed to open database: %w", err)
	}

	if err := db.Ping(); err != nil {
		return nil, fmt.Errorf("failed to connect to database: %w", err)
	}

	storage := &SQLiteStorage{db: db}
	if err := storage.initDB(); err != nil {
		db.Close()
		return nil, fmt.Errorf("failed to initialize database: %w", err)
	}

	return storage, nil
}

// initDB creates the necessary tables if they don't exist.
func (s *SQLiteStorage) initDB() error {
	createTableSQL := `
	CREATE TABLE IF NOT EXISTS conversations (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		session_id TEXT NOT NULL,
		customer_id TEXT NOT NULL,
		user_id TEXT,
		user_message TEXT,
		assistant_message TEXT,
		created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
		updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
		UNIQUE(session_id)
	);`

	_, err := s.db.Exec(createTableSQL)
	if err != nil {
		return fmt.Errorf("failed to create tables: %w", err)
	}

	return nil
}

// Close closes the database connection.
func (s *SQLiteStorage) Close() error {
	return s.db.Close()
}

// GetConversations retrieves conversation history for the given session.
// It returns a MessageList containing user and assistant messages.
func (s *SQLiteStorage) GetConversations(session *Session, limit int, offset int) (MessageList, error) {
	query := `
	SELECT user_message, assistant_message
	FROM conversations
	ORDER BY created_at DESC
	LIMIT ? OFFSET ?
	`

	rows, err := s.db.Query(query, session.SessionID, limit, offset)
	if err != nil {
		return MessageList{}, fmt.Errorf("failed to query conversations: %w", err)
	}
	defer rows.Close()

	messages := NewMessageList()

	for rows.Next() {
		var userMsg, assistantMsg sql.NullString
		if err := rows.Scan(&userMsg, &assistantMsg); err != nil {
			return MessageList{}, fmt.Errorf("failed to scan row: %w", err)
		}

		if userMsg.Valid && userMsg.String != "" {
			messages.Add(UserMessage(userMsg.String))
		}

		if assistantMsg.Valid && assistantMsg.String != "" {
			messages.Add(AssistantMessage(assistantMsg.String))
		}
	}

	if err := rows.Err(); err != nil {
		return MessageList{}, fmt.Errorf("error iterating rows: %w", err)
	}

	return *messages, nil
}

// CreateConversation stores a new conversation with the user message.
// It uses the session's ID as the unique identifier for the conversation.
func (s *SQLiteStorage) CreateConversation(session *Session, userMessage string) error {
	query := `
	INSERT INTO conversations (session_id, customer_id, user_id, user_message, created_at, updated_at)
	VALUES (?, ?, ?, ?, ?, ?)
	`

	now := time.Now()
	_, err := s.db.Exec(query, session.SessionID, session.CustomerID, "", userMessage, now, now)
	if err != nil {
		return fmt.Errorf("failed to create conversation: %w", err)
	}

	return nil
}

// FinishConversation updates an existing conversation with the assistant's response.
// It looks up the conversation using the session ID.
func (s *SQLiteStorage) FinishConversation(session *Session, assistantMessage string) error {
	// First check if the conversation exists
	var exists bool
	err := s.db.QueryRow("SELECT EXISTS(SELECT 1 FROM conversations WHERE session_id = ?)", session.SessionID).Scan(&exists)
	if err != nil {
		return fmt.Errorf("failed to check conversation existence: %w", err)
	}

	if !exists {
		return fmt.Errorf("no conversation found with session_id: %s", session.SessionID)
	}

	// Update the conversation with the assistant message
	query := `
	UPDATE conversations 
	SET assistant_message = ?, updated_at = ?
	WHERE session_id = ?
	`

	now := time.Now()
	_, err = s.db.Exec(query, assistantMessage, now, session.SessionID)
	if err != nil {
		return fmt.Errorf("failed to update conversation with assistant message: %w", err)
	}

	return nil
}

// GetUserInfo provides a dummy implementation that returns an empty UserInfo.
// As per requirements, we don't need actual user info in the SQLite table.
func (s *SQLiteStorage) GetUserInfo(session *Session) (UserInfo, error) {
	return UserInfo{
		Name:       "",
		CustomMeta: make(map[string]string),
	}, nil
}
