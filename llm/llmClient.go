package llm

import (
	"context"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/openai/openai-go/packages/ssestream"
)

// Define a custom type for context keys
type ContextKey string

type LLMConfig struct {
	APIKey  string
	BaseURL string
	Model   string
}

// LLM is a wrapper around the openai client, just to inject the extra metadata for now
type LLM struct {
	client *openai.Client
}

func (config *LLMConfig) NewLLMClient() *LLM {
	var client *openai.Client
	if config.BaseURL != "" {
		client = openai.NewClient(option.WithBaseURL(config.BaseURL), option.WithAPIKey(config.APIKey))
	} else {
		client = openai.NewClient(option.WithAPIKey(config.APIKey))
	}
	return &LLM{
		client: client,
	}
}

func injectIdentifiers(ctx context.Context, opts []option.RequestOption) []option.RequestOption {
	if sessionID, ok := ctx.Value(ContextKey("sessionID")).(string); ok {
		opts = append(opts, option.WithJSONSet("custom_identifier", sessionID))
	}

	if userID, ok := ctx.Value(ContextKey("userID")).(string); ok {
		opts = append(opts, option.WithJSONSet("customer_identifier", userID))
	}

	return opts
}

func (c *LLM) New(ctx context.Context, params openai.ChatCompletionNewParams) (*openai.ChatCompletion, error) {
	opts := []option.RequestOption{}
	opts = injectIdentifiers(ctx, opts)
	return c.client.Chat.Completions.New(ctx, params, opts...)
}

func (c *LLM) NewStreaming(ctx context.Context, params openai.ChatCompletionNewParams) *ssestream.Stream[openai.ChatCompletionChunk] {
	opts := []option.RequestOption{}
	opts = injectIdentifiers(ctx, opts)
	return c.client.Chat.Completions.NewStreaming(ctx, params, opts...)
}
