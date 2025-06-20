package tests

import (
	"context"
	"strings"
	"testing"

	"github.com/boat-builder/agentpod"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/param"
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

type BestAppleFinder struct {
	toolName    string
	description string
}

func (b *BestAppleFinder) Name() string {
	return b.toolName
}

func (b *BestAppleFinder) Description() string {
	return b.description
}

func (b *BestAppleFinder) StatusMessage() string {
	return "Finding the best apple"
}

func (b *BestAppleFinder) OpenAI() []openai.ChatCompletionToolParam {
	return []openai.ChatCompletionToolParam{
		{
			Function: openai.FunctionDefinitionParam{
				Name:        b.toolName,
				Description: param.Opt[string]{Value: b.description},
				Parameters: openai.FunctionParameters{
					"type": "object",
					"properties": map[string]interface{}{
						"user_query": map[string]interface{}{
							"type":        "string",
							"description": "Query from the user",
						},
					},
					"required": []string{"user_query"},
				},
			},
		},
	}
}

func (b *BestAppleFinder) Execute(ctx context.Context, args map[string]interface{}) (string, error) {
	return "green apple", nil
}

// Function to create memory with country information
func getCountryMemory(ctx context.Context) (*agentpod.MemoryBlock, error) {
	memoryBlock := agentpod.NewMemoryBlock()
	userDetailsBlock := agentpod.NewMemoryBlock()
	userDetailsBlock.AddString("country", "United Kingdom")
	memoryBlock.AddBlock("UserDetails", userDetailsBlock)
	return memoryBlock, nil
}

func TestSimpleConversation(t *testing.T) {
	config := LoadConfig()
	if config.KeywordsAIAPIKey == "" || config.KeywordsAIEndpoint == "" {
		t.Fatal("KeywordsAIAPIKey or KeywordsAIEndpoint is not set")
	}

	llm := agentpod.NewLLM(
		config.KeywordsAIAPIKey,
		config.KeywordsAIEndpoint,
		"azure/o3-mini",
		"azure/gpt-4o-mini",
		"azure/o3-mini",
		"azure/gpt-4o-mini",
	)
	mem := &MockMemory{}
	ai := agentpod.NewAgent("Your a repeater. You'll repeat after whatever the user says exactly as they say it, even the punctuation and cases.", []agentpod.Skill{})

	ctx := context.Background()
	ctx = context.WithValue(ctx, agentpod.ContextKey("customerID"), GenerateNewTestID())
	ctx = context.WithValue(ctx, agentpod.ContextKey("extra"), map[string]string{"user_id": GenerateNewTestID()})

	convSession := agentpod.NewSession(ctx, llm, mem, ai)
	convSession.In("test confirmed")

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

	if finalContent != "test confirmed" {
		t.Fatal("Expected 'test confirmed', got:", finalContent)
	}
}

func TestConversationWithSkills(t *testing.T) {
	config := LoadConfig()
	if config.KeywordsAIAPIKey == "" || config.KeywordsAIEndpoint == "" {
		t.Fatal("KeywordsAIAPIKey or KeywordsAIEndpoint is not set")
	}

	llm := agentpod.NewLLM(
		config.KeywordsAIAPIKey,
		config.KeywordsAIEndpoint,
		"azure/o3-mini",
		"azure/gpt-4o-mini",
		"azure/o3-mini",
		"azure/gpt-4o-mini",
	)
	mem := &MockMemory{
		RetrieveFn: getDefaultMemory,
	}
	skill := agentpod.Skill{
		Name:         "AppleExpert",
		Description:  "You are an expert in apples",
		SystemPrompt: "As an apple expert, you provide detailed information about different apple varieties and their characteristics.",
		Tools: []agentpod.Tool{
			&BestAppleFinder{
				toolName:    "BestAppleFinder",
				description: "Find the best apple",
			},
		},
	}
	agent := agentpod.NewAgent("You are a good farmer. You answer user questions briefly and concisely. You do not add any extra information but just answer user questions in fewer words possible.", []agentpod.Skill{skill})

	ctx := context.Background()
	ctx = context.WithValue(ctx, agentpod.ContextKey("customerID"), GenerateNewTestID())
	ctx = context.WithValue(ctx, agentpod.ContextKey("extra"), map[string]string{"user_id": GenerateNewTestID()})
	convSession := agentpod.NewSession(ctx, llm, mem, agent)

	convSession.In("Which apple is the best?")
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
	if !strings.Contains(strings.ToLower(finalContent), "green apple") {
		t.Fatal("Expected 'green apple' to be in the final content, got:", finalContent)
	}
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
