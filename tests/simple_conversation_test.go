package tests

import (
	"testing"

	"github.com/boat-builder/agentpod/agentpod"
	"github.com/boat-builder/agentpod/memory"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
)

func TestSimpleConversation(t *testing.T) {
	config := LoadConfig()

	llmClient := *openai.NewClient(option.WithAPIKey(config.OpenAIAPIKey))
	mem := &memory.Zep{}
	ai := &agentpod.Agent{}

	pod := agentpod.NewPod(&llmClient, mem, ai)
	session := pod.NewSession("user1", "session1")

	session.In("This is a test script. Respond with just 'test confirmed' for the test to pass.")
	out := session.Out()
	t.Log("Received response:", out.Content)
	if out.Content != "test confirmed" {
		t.Fatal("Expected 'test confirmed', got:", out.Content)
	}

}
