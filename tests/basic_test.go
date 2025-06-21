package tests

import (
	"context"
	"fmt"
	"strings"
	"sync"
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

// Function to create memory with country information
func getCountryMemory(ctx context.Context) (*agentpod.MemoryBlock, error) {
	memoryBlock := agentpod.NewMemoryBlock()
	userDetailsBlock := agentpod.NewMemoryBlock()
	userDetailsBlock.AddString("country", "United Kingdom")
	memoryBlock.AddBlock("UserDetails", userDetailsBlock)
	return memoryBlock, nil
}

// PopulationTool is a mock tool for testing purposes.
type PopulationTool struct {
	mu         sync.Mutex
	WasCalled  bool
	CountryArg string
}

func (t *PopulationTool) Name() string { return "get_country_population" }
func (t *PopulationTool) Description() string {
	return "Gets the population for a given country."
}

func (t *PopulationTool) OpenAI() []openai.ChatCompletionToolParam {
	return []openai.ChatCompletionToolParam{
		{
			Function: openai.FunctionDefinitionParam{
				Name:        t.Name(),
				Description: param.Opt[string]{Value: t.Description()},
				Parameters: openai.FunctionParameters{
					"type": "object",
					"properties": map[string]interface{}{
						"country": map[string]interface{}{
							"type":        "string",
							"description": "The country to get the population for.",
						},
					},
					"required": []string{"country"},
				},
			},
		},
	}
}

func (t *PopulationTool) Execute(ctx context.Context, args map[string]interface{}) (string, error) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if country, ok := args["country"].(string); ok {
		t.WasCalled = true
		t.CountryArg = country
		return "The population is 50 million", nil
	}
	return "", fmt.Errorf("country argument is missing or not a string")
}

func TestSkillWithMemory(t *testing.T) {
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

	populationTool := &PopulationTool{}

	censusSkill := agentpod.Skill{
		Name:            "CensusSkill",
		ToolDescription: "This skill can provide population data for different countries.",
		SystemPrompt:    "You are a census expert. You can provide population data.",
		Tools:           []agentpod.Tool{populationTool},
	}

	ai := agentpod.NewAgent("You are a helpful assistant.", []agentpod.Skill{censusSkill})

	ctx := context.Background()
	ctx = context.WithValue(ctx, agentpod.ContextKey("customerID"), GenerateNewTestID())
	ctx = context.WithValue(ctx, agentpod.ContextKey("extra"), map[string]string{"user_id": GenerateNewTestID()})

	convSession := agentpod.NewSession(ctx, llm, mem, ai)

	convSession.In("What is the population of my country?")

	for {
		out := convSession.Out()
		if out.Type == agentpod.ResponseTypeEnd {
			break
		}
	}

	populationTool.mu.Lock()
	defer populationTool.mu.Unlock()
	if !populationTool.WasCalled {
		t.Fatal("Expected population tool to be called, but it was not.")
	}

	if strings.ToLower(populationTool.CountryArg) != "united kingdom" {
		t.Fatalf("Expected population tool to be called with 'United Kingdom', but got '%s'", populationTool.CountryArg)
	}
}
