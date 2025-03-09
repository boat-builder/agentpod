package tests

import (
	"context"
	"strings"
	"testing"

	"github.com/boat-builder/agentpod"
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

func (b *BestAppleFinder) StatusMessage() string {
	return "Finding the best apple"
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
	return "green apple", nil
}

// MockStorage implements the Storage interface for testing
type MockStorage struct {
	ConversationFn func(*agentpod.Session, int, int) (agentpod.MessageList, error)
	UserInfoFn     func(*agentpod.Session) (agentpod.UserInfo, error)
}

// GetConversations returns the conversation history
func (m *MockStorage) GetConversations(session *agentpod.Session, limit int, offset int) (agentpod.MessageList, error) {
	return m.ConversationFn(session, limit, offset)
}

// SaveConversation is a no-op for testing
func (m *MockStorage) CreateConversation(session *agentpod.Session, userMessage string) error {
	return nil
}

func (m *MockStorage) FinishConversation(session *agentpod.Session, assistantMessage string) error {
	return nil
}

// GetUserInfo returns user information
func (m *MockStorage) GetUserInfo(session *agentpod.Session) (agentpod.UserInfo, error) {
	return m.UserInfoFn(session)
}

// Default empty conversation history function
func getEmptyConversationHistory(session *agentpod.Session, limit int, offset int) (agentpod.MessageList, error) {
	return agentpod.MessageList{}, nil
}

// Default user info function
func getDefaultUserInfo(session *agentpod.Session) (agentpod.UserInfo, error) {
	return agentpod.UserInfo{
		Name: "John Doe",
	}, nil
}

func TestSimpleConversation(t *testing.T) {
	config := LoadConfig()
	if config.KeywordsAIAPIKey == "" || config.KeywordsAIEndpoint == "" {
		t.Fatal("KeywordsAIAPIKey or KeywordsAIEndpoint is not set")
	}

	llmConfig := agentpod.LLMConfig{
		BaseURL: config.KeywordsAIEndpoint,
		APIKey:  config.KeywordsAIAPIKey,
		Model:   "azure/o3-mini",
	}
	mem := &agentpod.Zep{}
	ai := agentpod.NewAgent("Your a repeater. You'll repeat after whatever the user says.", []agentpod.Skill{})

	// Create a mock storage with empty conversation history
	storage := &MockStorage{
		ConversationFn: getEmptyConversationHistory,
		UserInfoFn:     getDefaultUserInfo,
	}

	pod := agentpod.NewPod(&llmConfig, mem, ai, storage)
	ctx := context.Background()
	orgID := GenerateNewTestID()
	sessionID := GenerateNewTestID()
	userID := GenerateNewTestID()
	convSession := pod.NewSession(ctx, orgID, sessionID, map[string]string{"user_id": userID})

	convSession.In("test confirmed")

	var finalContent string
	for {
		out := convSession.Out()
		finalContent += out.Content
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

	llmConfig := agentpod.LLMConfig{
		BaseURL: config.KeywordsAIEndpoint,
		APIKey:  config.KeywordsAIAPIKey,
		Model:   "azure/o3-mini",
	}
	mem := &agentpod.Zep{}
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

	// Create a mock storage with empty conversation history
	storage := &MockStorage{
		ConversationFn: getEmptyConversationHistory,
		UserInfoFn:     getDefaultUserInfo,
	}

	pod := agentpod.NewPod(&llmConfig, mem, agent, storage)
	ctx := context.Background()
	orgID := GenerateNewTestID()
	sessionID := GenerateNewTestID()
	userID := GenerateNewTestID()
	convSession := pod.NewSession(ctx, orgID, sessionID, map[string]string{"user_id": userID})

	convSession.In("Which apple is the best?")
	var finalContent string
	for {
		out := convSession.Out()
		finalContent += out.Content
		if out.Type == agentpod.ResponseTypeEnd {
			break
		}
	}
	if !strings.Contains(strings.ToLower(finalContent), "green apple") {
		t.Fatal("Expected 'green apple' to be in the final content, got:", finalContent)
	}
}

// Function for non-empty conversation history
func getNonEmptyConversationHistory(session *agentpod.Session, limit int, offset int) (agentpod.MessageList, error) {
	messages := agentpod.MessageList{}
	messages.Add(
		agentpod.UserMessage("Can you tell me which color is apple?"),
		agentpod.AssistantMessage("The apple is generally red"),
	)
	return messages, nil
}

func TestConversationWithHistory(t *testing.T) {
	config := LoadConfig()
	if config.KeywordsAIAPIKey == "" || config.KeywordsAIEndpoint == "" {
		t.Fatal("KeywordsAIAPIKey or KeywordsAIEndpoint is not set")
	}

	llmConfig := agentpod.LLMConfig{
		BaseURL: config.KeywordsAIEndpoint,
		APIKey:  config.KeywordsAIAPIKey,
		Model:   "azure/o3-mini",
	}
	mem := &agentpod.Zep{}
	ai := agentpod.NewAgent("You are an assistant!", []agentpod.Skill{})

	// Create a mock storage with non-empty conversation history
	storage := &MockStorage{
		ConversationFn: getNonEmptyConversationHistory,
		UserInfoFn:     getDefaultUserInfo,
	}

	pod := agentpod.NewPod(&llmConfig, mem, ai, storage)
	ctx := context.Background()
	orgID := GenerateNewTestID()
	sessionID := GenerateNewTestID()
	userID := GenerateNewTestID()
	convSession := pod.NewSession(ctx, orgID, sessionID, map[string]string{"user_id": userID})

	convSession.In("is it a fruit or a vegetable? Answer in one word without extra punctuation.")

	var finalContent string
	for {
		out := convSession.Out()
		finalContent += out.Content
		if out.Type == agentpod.ResponseTypeEnd {
			break
		}
	}

	if strings.ToLower(finalContent) != "fruit" {
		t.Fatal("Expected 'fruit', got:", finalContent)
	}
}
