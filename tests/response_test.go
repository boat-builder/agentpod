package tests

import (
	"context"
	"strings"
	"testing"

	"github.com/boat-builder/agentpod"
	"github.com/openai/openai-go/packages/param"
	"github.com/openai/openai-go/responses"
	"github.com/openai/openai-go/shared"
)

func TestNewResponse(t *testing.T) {
	config := LoadConfig()
	if config.KeywordsAIAPIKey == "" || config.KeywordsAIEndpoint == "" {
		t.Skip("Skipping test because Keywords AI credentials are not set")
	}

	// Create a new LLM client with Keywords AI configuration
	llm := agentpod.NewLLM(
		config.KeywordsAIAPIKey,
		config.KeywordsAIEndpoint,
		"azure/o3-mini",
		"azure/gpt-4o-mini",
		"azure/o3-mini",
		"azure/gpt-4o-mini",
	)

	// Create a context with metadata
	ctx := context.WithValue(context.Background(), agentpod.ContextKey("sessionID"), "test-session-123")
	ctx = context.WithValue(ctx, agentpod.ContextKey("customerID"), "test-customer-456")
	ctx = context.WithValue(ctx, agentpod.ContextKey("extra"), map[string]string{
		"user_id":  "test-user-789",
		"test_key": "test_value",
	})

	// Create test parameters for the Response API
	params := responses.ResponseNewParams{
		Model: shared.ResponsesModel("gpt-4"),
		Input: responses.ResponseNewParamsInputUnion{
			OfString: param.Opt[string]{Value: "What is the capital of France?"},
		},
	}

	// Call the NewResponse function
	response, err := llm.NewResponse(ctx, params)
	if err != nil {
		t.Fatalf("Failed to create response: %v", err)
	}

	// Basic validation of the response
	if response == nil {
		t.Fatal("Expected non-nil response")
	}

	// Check if the response contains the expected content
	// Note: The actual content will depend on the model's response
	if response.OutputText() == "" {
		t.Error("Expected non-empty response content")
	}

	// Verify the response contains the correct answer
	if !strings.Contains(strings.ToLower(response.OutputText()), "paris") {
		t.Errorf("Expected response to contain 'Paris', got: %s", response.OutputText())
	}

}
