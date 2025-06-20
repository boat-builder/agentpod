package tests

import (
	"context"
	"strings"
	"testing"

	"github.com/boat-builder/agentpod"
)

// MockMemory implements the Memory interface for testing
type MockMemory struct {
	RetrieveFn func(ctx context.Context) (*agentpod.MemoryBlock, error)
}

// Retrieve returns a memory block for testing
func (m *MockMemory) Retrieve(ctx context.Context) (*agentpod.MemoryBlock, error) {
	if m.RetrieveFn != nil {
		return m.RetrieveFn(ctx)
	}
	// Default implementation returns an empty memory block
	memoryBlock := agentpod.NewMemoryBlock()
	return memoryBlock, nil
}

// Default memory retrieval function that includes basic user data
func getDefaultMemory(ctx context.Context) (*agentpod.MemoryBlock, error) {
	memoryBlock := agentpod.NewMemoryBlock()
	if userID, ok := ctx.Value(agentpod.ContextKey("extra")).(map[string]string)["user_id"]; ok {
		memoryBlock.AddString("user_id", userID)
	}
	if sessionID, ok := ctx.Value(agentpod.ContextKey("sessionID")).(string); ok {
		memoryBlock.AddString("session_id", sessionID)
	}
	return memoryBlock, nil
}

// Function to create memory with country information
func getCountryMemory(ctx context.Context) (*agentpod.MemoryBlock, error) {
	memoryBlock := agentpod.NewMemoryBlock()
	userDetailsBlock := agentpod.NewMemoryBlock()
	userDetailsBlock.AddString("country", "United Kingdom")
	memoryBlock.AddBlock("UserDetails", userDetailsBlock)
	return memoryBlock, nil
}

func TestMemoryRetrieval(t *testing.T) {
	config := LoadConfig()
	if config.KeywordsAIAPIKey == "" || config.KeywordsAIEndpoint == "" {
		t.Fatal("KeywordsAIAPIKey or KeywordsAIEndpoint is not set")
	}

	llm := agentpod.NewLLM(
		config.KeywordsAIAPIKey,
		config.KeywordsAIEndpoint,
		"azure/o3-mini",
		"azure/gpt-4o-mini",
	)

	// Create mock memory with country information
	mem := &MockMemory{
		RetrieveFn: getCountryMemory,
	}

	ai := agentpod.NewAgent("You are a helpful assistant. Answer questions based on the user's information.", []agentpod.Skill{})

	ctx := context.Background()
	ctx = context.WithValue(ctx, agentpod.ContextKey("customerID"), GenerateNewTestID())
	ctx = context.WithValue(ctx, agentpod.ContextKey("extra"), map[string]string{"user_id": GenerateNewTestID()})

	convSession := agentpod.NewSession(ctx, llm, mem, ai)

	convSession.In("Which country am I from?")

	var finalContent string
	for {
		out := convSession.Out()
		if out.Type == agentpod.ResponseTypePartialText {
			finalContent += out.Content
		}
		if out.Type == agentpod.ResponseTypeEnd {
			break
		}
	}

	if !strings.Contains(strings.ToLower(finalContent), "united kingdom") {
		t.Fatal("Expected response to contain 'United Kingdom', got:", finalContent)
	}
}
