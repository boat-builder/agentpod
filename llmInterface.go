package agentpod

import (
	"context"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/ssestream"
)

// LLM defines the minimal contract required by the agent runtime to
// interact with a language-model provider. Implementations may add
// additional helper methods but only the operations below are relied
// upon by the rest of the codebase.
type LLM interface {
	// New issues a non-streaming chat completion request.
	New(ctx context.Context, params openai.ChatCompletionNewParams) (*openai.ChatCompletion, error)

	// NewStreaming issues a streaming chat completion request, returning
	// an ssestream.Stream to consume the chunks.
	NewStreaming(ctx context.Context, params openai.ChatCompletionNewParams) *ssestream.Stream[openai.ChatCompletionChunk]
}
