package agentpod

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/openai/openai-go"
)

func TestSQLiteStorage(t *testing.T) {
	// Create a temporary database file
	dbFile := "./test_conversations.db"
	defer os.Remove(dbFile) // Clean up after test

	// Initialize SQLite storage
	storage, err := NewSQLiteStorage(dbFile)
	if err != nil {
		t.Fatalf("Failed to initialize SQLite storage: %v", err)
	}
	defer storage.Close()

	// Create a test session
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	session := &Session{
		Ctx:        ctx,
		Cancel:     cancel,
		CustomerID: "test-customer",
		SessionID:  "test-session-" + time.Now().Format(time.RFC3339),
		CustomMeta: map[string]string{"test": "value"},
		State:      &SessionState{},
	}

	// Test CreateConversation
	t.Run("CreateConversation", func(t *testing.T) {
		userMessage := "Hello, how can you help me?"
		err := storage.CreateConversation(session, userMessage)
		if err != nil {
			t.Fatalf("Failed to create conversation: %v", err)
		}

		// Try to create another conversation with the same session ID (should fail due to UNIQUE constraint)
		err = storage.CreateConversation(session, "Another message")
		if err == nil {
			t.Fatalf("Expected error when creating duplicate conversation, but got none")
		}
	})

	// Test FinishConversation
	t.Run("FinishConversation", func(t *testing.T) {
		assistantMessage := "I can answer questions and assist with various tasks."
		err := storage.FinishConversation(session, assistantMessage)
		if err != nil {
			t.Fatalf("Failed to finish conversation: %v", err)
		}

		// Try to finish a non-existent conversation
		nonExistentSession := &Session{
			Ctx:        ctx,
			Cancel:     cancel,
			CustomerID: "test-customer",
			SessionID:  "non-existent-session",
			CustomMeta: map[string]string{},
			State:      &SessionState{},
		}
		err = storage.FinishConversation(nonExistentSession, "This should fail")
		if err == nil {
			t.Fatalf("Expected error when finishing non-existent conversation, but got none")
		}
	})

	// Test GetConversation
	t.Run("GetConversation", func(t *testing.T) {
		messages, err := storage.GetConversations(session, 10, 0)
		if err != nil {
			t.Fatalf("Failed to get conversations: %v", err)
		}

		// We should have 2 messages (1 user, 1 assistant)
		if messages.Len() != 2 {
			t.Fatalf("Expected 2 messages, but got %d", messages.Len())
		}

		// Check the first message (user message)
		userMsg, ok := messages.Messages[0].(openai.ChatCompletionUserMessageParam)
		if !ok {
			t.Fatalf("Expected first message to be a user message")
		}
		if userMsg.Role.Value != openai.ChatCompletionUserMessageParamRoleUser {
			t.Fatalf("Expected user message role to be 'user', but got '%s'", userMsg.Role.Value)
		}
		if len(userMsg.Content.Value) == 0 {
			t.Fatalf("Expected user message content to be non-empty")
		}
		textPart, ok := userMsg.Content.Value[0].(openai.ChatCompletionContentPartTextParam)
		if !ok {
			t.Fatalf("Expected user message content to be text")
		}
		if textPart.Text.Value != "Hello, how can you help me?" {
			t.Fatalf("Expected user message content to be 'Hello, how can you help me?', but got '%s'", textPart.Text.Value)
		}

		// Check the second message (assistant message)
		assistantMsg, ok := messages.Messages[1].(openai.ChatCompletionAssistantMessageParam)
		if !ok {
			t.Fatalf("Expected second message to be an assistant message")
		}
		if assistantMsg.Role.Value != openai.ChatCompletionAssistantMessageParamRoleAssistant {
			t.Fatalf("Expected assistant message role to be 'assistant', but got '%s'", assistantMsg.Role.Value)
		}

		// For assistant message, we need to check if the content contains our expected text
		found := false
		expectedText := "I can answer questions and assist with various tasks."

		for _, content := range assistantMsg.Content.Value {
			switch c := content.(type) {
			case openai.ChatCompletionContentPartTextParam:
				if c.Text.Value == expectedText {
					found = true
				}
			}
		}

		if !found {
			t.Fatalf("Expected to find assistant message with content '%s', but it was not found", expectedText)
		}
	})

	// Test GetUserInfo (dummy implementation)
	t.Run("GetUserInfo", func(t *testing.T) {
		userInfo, err := storage.GetUserInfo(session)
		if err != nil {
			t.Fatalf("Failed to get user info: %v", err)
		}

		// Verify that the dummy implementation returns an empty UserInfo
		if userInfo.Name != "" {
			t.Fatalf("Expected empty user name, but got '%s'", userInfo.Name)
		}
		if len(userInfo.CustomMeta) != 0 {
			t.Fatalf("Expected empty custom meta, but got %v", userInfo.CustomMeta)
		}
	})
}
