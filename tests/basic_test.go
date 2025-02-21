package tests

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/boat-builder/agentpod/agentpod"
	"github.com/boat-builder/agentpod/agentpod/session"
	"github.com/boat-builder/agentpod/llm"
	"github.com/boat-builder/agentpod/memory"
	"github.com/openai/openai-go"
)

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

func (b *BestAppleFinder) OpenAI() []openai.ChatCompletionToolParam {
	return []openai.ChatCompletionToolParam{
		{
			Type: openai.F(openai.ChatCompletionToolTypeFunction),
			Function: openai.F(openai.FunctionDefinitionParam{
				Name:        openai.F(b.toolName),
				Description: openai.F(b.description),
				Parameters: openai.F(openai.FunctionParameters{
					"type": "object",
					"properties": map[string]interface{}{
						"user_query": map[string]interface{}{
							"type":        "string",
							"description": "Query from the user",
						},
					},
					"required": []string{"user_query"},
				}),
			}),
		},
	}
}

func (b *BestAppleFinder) Execute(args map[string]interface{}) (string, error) {
	userQuery := args["user_query"].(string)
	if userQuery != "Which apple is the best?" {
		return "", fmt.Errorf("invalid user query")
	}
	return "green apple", nil
}

func TestSimpleConversation(t *testing.T) {
	config := LoadConfig()
	if config.KeywordsAIAPIKey == "" || config.KeywordsAIEndpoint == "" {
		t.Fatal("KeywordsAIAPIKey or KeywordsAIEndpoint is not set")
	}

	llmConfig := llm.LLMConfig{
		BaseURL: config.KeywordsAIEndpoint,
		APIKey:  config.KeywordsAIAPIKey,
		Model:   "azure/gpt-4o-mini",
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

func TestConversationWithSkills(t *testing.T) {
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
	skill := agentpod.NewSkill("AppleExpert", "You are an expert in apples", []agentpod.Tool{
		&BestAppleFinder{
			toolName:    "BestAppleFinder",
			description: "Find the best apple",
		},
	})
	agent := agentpod.NewAgent("You are a good farmer", []agentpod.Skill{*skill})

	pod := agentpod.NewPod(&llmConfig, mem, agent)
	ctx := context.Background()
	convSession := pod.NewSession(ctx, "user1", "session1")

	convSession.In("Which apple is the best?")
	var finalContent string
	for {
		out := convSession.Out()
		finalContent += out.Content
		if out.Type == session.MessageTypeEnd {
			break
		}
	}
	if !strings.Contains(strings.ToLower(finalContent), "green apple") {
		t.Fatal("Expected 'green apple' to be in the final content, got:", finalContent)
	}
}
