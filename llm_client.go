package agentpod

import (
	"context"
	"encoding/base64"
	"encoding/json"

	"github.com/invopop/jsonschema"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/openai/openai-go/packages/ssestream"
	"github.com/openai/openai-go/responses"
)

// Define a custom type for context keys
type ContextKey string

type KeywordsAIClient struct {
	APIKey       string
	BaseURL      string
	_strongModel string
	_cheapModel  string
	client       openai.Client
}

func NewKeywordsAIClient(apiKey string, baseURL string, strongModel string, cheapModel string) *KeywordsAIClient {
	var client openai.Client
	if baseURL != "" {
		client = openai.NewClient(option.WithBaseURL(baseURL), option.WithAPIKey(apiKey))
	} else {
		client = openai.NewClient(option.WithAPIKey(apiKey))
	}
	return &KeywordsAIClient{
		APIKey:       apiKey,
		BaseURL:      baseURL,
		_strongModel: strongModel,
		_cheapModel:  cheapModel,
		client:       client,
	}
}

func optsWithIds(ctx context.Context, opts []option.RequestOption, useHeader bool) []option.RequestOption {
	identifiers := make(map[string]interface{})

	if sessionID, ok := ctx.Value(ContextKey("sessionID")).(string); ok {
		if useHeader {
			identifiers["custom_identifier"] = sessionID
		} else {
			opts = append(opts, option.WithJSONSet("custom_identifier", sessionID))
		}
	}

	if customerID, ok := ctx.Value(ContextKey("customerID")).(string); ok {
		if useHeader {
			identifiers["customer_identifier"] = customerID
		} else {
			opts = append(opts, option.WithJSONSet("customer_identifier", customerID))
		}
	}

	if extraMeta, ok := ctx.Value(ContextKey("extra")).(map[string]string); ok {
		if useHeader {
			identifiers["metadata"] = extraMeta
		} else {
			metadata := make(map[string]string)
			for key, value := range extraMeta {
				metadata[key] = value
			}
			opts = append(opts, option.WithJSONSet("metadata", metadata))
		}
	} else if useHeader {
		identifiers["metadata"] = make(map[string]string)
	}

	if useHeader && len(identifiers) > 0 {
		jsonData, err := json.Marshal(identifiers)
		if err != nil {
			// If marshaling fails, we'll just return the opts without the header
			return opts
		}
		opts = append(opts, option.WithHeader("X-Data-Keywordsai-Params", base64.StdEncoding.EncodeToString(jsonData)))
	}

	return opts
}

// TODO failures like too long, non-processable etc from the LLM needs to be handled
func (c *KeywordsAIClient) New(ctx context.Context, params openai.ChatCompletionNewParams) (*openai.ChatCompletion, error) {
	opts := []option.RequestOption{}
	opts = optsWithIds(ctx, opts, false)
	return c.client.Chat.Completions.New(ctx, params, opts...)
}

func (c *KeywordsAIClient) NewStreaming(ctx context.Context, params openai.ChatCompletionNewParams) *ssestream.Stream[openai.ChatCompletionChunk] {
	opts := []option.RequestOption{}
	opts = optsWithIds(ctx, opts, false)
	return c.client.Chat.Completions.NewStreaming(ctx, params, opts...)
}

// NewResponse creates a new response using OpenAI's Response API
func (c *KeywordsAIClient) NewResponse(ctx context.Context, params responses.ResponseNewParams) (*responses.Response, error) {
	opts := []option.RequestOption{}
	opts = optsWithIds(ctx, opts, true)
	return c.client.Responses.New(ctx, params, opts...)
}

func (c *KeywordsAIClient) CheapModel() string {
	return c._cheapModel
}

func (c *KeywordsAIClient) StrongModel() string {
	return c._strongModel
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

// Ensure KeywordsAIClient satisfies the LLM interface.
var _ LLM = (*KeywordsAIClient)(nil)
