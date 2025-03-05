package agentpod

import (
	"context"

	"github.com/invopop/jsonschema"
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

func optsWithIds(ctx context.Context, opts []option.RequestOption) []option.RequestOption {
	if sessionID, ok := ctx.Value(ContextKey("sessionID")).(string); ok {
		opts = append(opts, option.WithJSONSet("custom_identifier", sessionID))
	}

	if customerID, ok := ctx.Value(ContextKey("customerID")).(string); ok {
		opts = append(opts, option.WithJSONSet("customer_identifier", customerID))
	}

	if customMeta, ok := ctx.Value(ContextKey("customMeta")).(map[string]string); ok {
		for key, value := range customMeta {
			opts = append(opts, option.WithJSONSet(key, value))
		}
	}

	return opts
}

// TODO failures like too long, non-processable etc from the LLM needs to be handled
func (c *LLM) New(ctx context.Context, params openai.ChatCompletionNewParams) (*openai.ChatCompletion, error) {
	opts := []option.RequestOption{}
	opts = optsWithIds(ctx, opts)
	return c.client.Chat.Completions.New(ctx, params, opts...)
}

func (c *LLM) NewStreaming(ctx context.Context, params openai.ChatCompletionNewParams) *ssestream.Stream[openai.ChatCompletionChunk] {
	opts := []option.RequestOption{}
	opts = optsWithIds(ctx, opts)
	return c.client.Chat.Completions.NewStreaming(ctx, params, opts...)
}

func GenerateSchema[T any]() interface{} {
	reflector := jsonschema.Reflector{
		AllowAdditionalProperties: false,
		DoNotReference:            true,
	}
	var v T
	schema := reflector.Reflect(v)
	return schema
}
