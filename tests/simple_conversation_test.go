package tests

import (
	"context"
	"testing"

	"github.com/boat-builder/agentpod/agentpod"
	"github.com/boat-builder/agentpod/llm"
	"github.com/boat-builder/agentpod/memory"
)

func TestSimpleConversation(t *testing.T) {
	config := LoadConfig()
	if config.KeywordsAIAPIKey == "" || config.KeywordsAIEndpoint == "" {
		t.Fatal("KeywordsAIAPIKey or KeywordsAIEndpoint is not set")
	}

	llmConfig := llm.LLMConfig{
		BaseURL: config.KeywordsAIEndpoint,
		APIKey:  config.KeywordsAIAPIKey,
	}
	mem := &memory.Zep{}
	ai := &agentpod.Agent{}

	pod := agentpod.NewPod(&llmConfig, mem, ai)
	ctx := context.Background()
	session := pod.NewSession(ctx, "user1", "session1")

	session.In("This is a test script. Respond with just 'test confirmed' for the test to pass.")
	out := session.Out()
	t.Log("Received response:", out.Content)
	if out.Content != "test confirmed" {
		t.Fatal("Expected 'test confirmed', got:", out.Content)
	}

}
