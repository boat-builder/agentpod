package tests

import (
	"context"
	"testing"

	"github.com/boat-builder/agentpod/agentpod"
	"github.com/boat-builder/agentpod/agentpod/session"
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
		Model:   "gpt-4o-mini",
	}
	mem := &memory.Zep{}
	ai := agentpod.NewAgent("Your a repeater. You'll repeat after whatever the user says.", []agentpod.Skill{})

	pod := agentpod.NewPod(&llmConfig, mem, ai)
	ctx := context.Background()
	convSession := pod.NewSession(ctx, "user1", "session1")

	convSession.In("test confirmed")

	var finalContent string
	for {
		out := convSession.Out()
		finalContent += out.Content
		if out.Type == session.MessageTypeEnd {
			break
		}
	}

	if finalContent != "test confirmed" {
		t.Fatal("Expected 'test confirmed', got:", finalContent)
	}

}
