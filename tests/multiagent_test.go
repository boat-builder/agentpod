package tests

import (
	"context"
	"strings"
	"testing"

	"github.com/boat-builder/agentpod"
	"github.com/openai/openai-go"
)

// RestaurantTool implements the Tool interface for restaurant recommendations
type RestaurantTool struct {
	toolName    string
	description string
	restaurants map[string]Restaurant
}

type Restaurant struct {
	Name     string
	Cuisine  string
	Location string
}

func NewRestaurantTool() *RestaurantTool {
	return &RestaurantTool{
		toolName:    "RestaurantDatabase",
		description: "Provides information about restaurants in a specific location",
		restaurants: map[string]Restaurant{
			"Pasta Paradise": {
				Name:     "Pasta Paradise",
				Cuisine:  "Italian",
				Location: "Downtown",
			},
			"Sushi Master": {
				Name:     "Sushi Master",
				Cuisine:  "Japanese",
				Location: "Uptown",
			},
			"Taco Fiesta": {
				Name:     "Taco Fiesta",
				Cuisine:  "Mexican",
				Location: "Midtown",
			},
		},
	}
}

func (r *RestaurantTool) Name() string {
	return r.toolName
}

func (r *RestaurantTool) Description() string {
	return r.description
}

func (r *RestaurantTool) StatusMessage() string {
	return "Finding the perfect restaurant for you"
}

func (r *RestaurantTool) OpenAI() []openai.ChatCompletionToolParam {
	return []openai.ChatCompletionToolParam{
		{
			Type: openai.F(openai.ChatCompletionToolTypeFunction),
			Function: openai.F(openai.FunctionDefinitionParam{
				Name:        openai.F(r.toolName),
				Description: openai.F(r.description),
				Parameters: openai.F(openai.FunctionParameters{
					"type": "object",
					"properties": map[string]interface{}{
						"location": map[string]interface{}{
							"type":        "string",
							"description": "User's location",
						},
						"cuisine": map[string]interface{}{
							"type":        "string",
							"description": "Preferred cuisine",
						},
					},
					"required": []string{"location", "cuisine"},
				}),
			}),
		},
	}
}

func (r *RestaurantTool) Execute(args map[string]interface{}) (string, error) {
	location := args["location"].(string)
	cuisine := args["cuisine"].(string)

	for _, restaurant := range r.restaurants {
		if restaurant.Location == location && restaurant.Cuisine == cuisine {
			return restaurant.Name, nil
		}
	}
	return "No matching restaurant found", nil
}

// CuisineTool implements the Tool interface for cuisine recommendations
type CuisineTool struct {
	toolName    string
	description string
	dishes      map[string][]string
}

func NewCuisineTool() *CuisineTool {
	return &CuisineTool{
		toolName:    "CuisineDatabase",
		description: "Database of all the available dishes in all the restaurants",
		dishes: map[string][]string{
			"Pasta Paradise": {"Carbonara", "Lasagna", "Risotto"},
			"Sushi Master":   {"Dragon Roll", "Sashimi Platter", "Tempura"},
			"Taco Fiesta":    {"Street Tacos", "Burrito Bowl", "Quesadilla"},
		},
	}
}

func (c *CuisineTool) Name() string {
	return c.toolName
}

func (c *CuisineTool) Description() string {
	return c.description
}

func (c *CuisineTool) StatusMessage() string {
	return "Finding the perfect dishes for you"
}

func (c *CuisineTool) OpenAI() []openai.ChatCompletionToolParam {
	return []openai.ChatCompletionToolParam{
		{
			Type: openai.F(openai.ChatCompletionToolTypeFunction),
			Function: openai.F(openai.FunctionDefinitionParam{
				Name:        openai.F(c.toolName),
				Description: openai.F(c.description),
				Parameters: openai.F(openai.FunctionParameters{
					"type": "object",
					"properties": map[string]interface{}{
						"restaurant": map[string]interface{}{
							"type":        "string",
							"description": "Restaurant name",
						},
					},
					"required": []string{"restaurant"},
				}),
			}),
		},
	}
}

func (c *CuisineTool) Execute(args map[string]interface{}) (string, error) {
	restaurant := args["restaurant"].(string)
	if dishes, ok := c.dishes[restaurant]; ok {
		return strings.Join(dishes, ", "), nil
	}
	return "No dishes found for this restaurant", nil
}

// Function to create memory with user preferences
func getUserPreferencesMemory(meta *agentpod.Meta) (*agentpod.MemoryBlock, error) {
	memoryBlock := agentpod.NewMemoryBlock()
	userDetailsBlock := agentpod.NewMemoryBlock()
	userDetailsBlock.AddString("location", "Downtown")
	userDetailsBlock.AddString("favorite_cuisines", "Italian,Japanese,Mexican")
	memoryBlock.AddBlock("UserDetails", userDetailsBlock)
	return memoryBlock, nil
}

func TestMultiAgentRestaurantRecommendation(t *testing.T) {
	config := LoadConfig()
	if config.KeywordsAIAPIKey == "" || config.KeywordsAIEndpoint == "" {
		t.Fatal("KeywordsAIAPIKey or KeywordsAIEndpoint is not set")
	}

	llm := agentpod.NewLLM(
		config.KeywordsAIAPIKey,
		config.KeywordsAIEndpoint,
		"azure/gpt-4o",
		"azure/gpt-4o",
		"azure/gpt-4o-mini",
		"azure/gpt-4o-mini",
	)

	// Create mock memory with user preferences
	mem := &MockMemory{
		RetrieveFn: getUserPreferencesMemory,
	}

	// Create restaurant agent with restaurant recommendation tool
	restaurantTool := NewRestaurantTool()
	restaurantAgent := agentpod.NewAgent(
		"You are a restaurant recommendation expert. You help users find the perfect restaurant based on their location and cuisine preferences. Be concise and direct in your recommendations.",
		[]agentpod.Skill{{
			Name:         "RestaurantExpert",
			Description:  "Expert in restaurant recommendations",
			SystemPrompt: "As a restaurant expert, you provide personalized restaurant recommendations based on location and cuisine preferences.",
			Tools:        []agentpod.Tool{restaurantTool},
		}},
	)

	// Create cuisine agent with cuisine recommendation tool
	cuisineTool := NewCuisineTool()
	cuisineAgent := agentpod.NewAgent(
		"You are a cuisine expert. You help users discover delicious dishes at recommended restaurants. Be concise and direct in your recommendations.",
		[]agentpod.Skill{{
			Name:         "CuisineExpert",
			Description:  "Expert in cuisine and dishes",
			SystemPrompt: "As a cuisine expert, you provide dish recommendations for specific restaurants.",
			Tools:        []agentpod.Tool{cuisineTool},
		}},
	)

	// Create a mock storage with empty conversation history
	storage := &MockStorage{}
	storage.ConversationFn = getEmptyConversationHistory(storage)

	orgID := GenerateNewTestID()
	sessionID := GenerateNewTestID()
	userID := GenerateNewTestID()

	// Create session with restaurant agent
	restaurantSession := agentpod.NewSession(context.Background(), llm, mem, restaurantAgent, storage, agentpod.Meta{
		CustomerID: orgID,
		SessionID:  sessionID,
		Extra:      map[string]string{"user_id": userID},
	})

	// Create session with cuisine agent
	cuisineSession := agentpod.NewSession(context.Background(), llm, mem, cuisineAgent, storage, agentpod.Meta{
		CustomerID: orgID,
		SessionID:  sessionID,
		Extra:      map[string]string{"user_id": userID},
	})

	// Test restaurant recommendation
	restaurantSession.In("What's a good restaurant for me?")
	var restaurantResponse string
	for {
		out := restaurantSession.Out()
		restaurantResponse += out.Content
		if out.Type == agentpod.ResponseTypeEnd {
			break
		}
	}

	// Verify restaurant recommendation
	if !strings.Contains(strings.ToLower(restaurantResponse), "pasta paradise") {
		t.Fatal("Expected 'Pasta Paradise' to be in the restaurant recommendation, got:", restaurantResponse)
	}

	// Test cuisine recommendation
	cuisineSession.In("What dishes do they have?")
	var cuisineResponse string
	for {
		out := cuisineSession.Out()
		cuisineResponse += out.Content
		if out.Type == agentpod.ResponseTypeEnd {
			break
		}
	}

	// Verify cuisine recommendation
	expectedDishes := []string{"carbonara", "lasagna", "risotto"}
	foundDish := false
	for _, dish := range expectedDishes {
		if strings.Contains(strings.ToLower(cuisineResponse), dish) {
			foundDish = true
			break
		}
	}
	if !foundDish {
		t.Fatal("Expected at least one of the dishes to be in the cuisine recommendation, got:", cuisineResponse)
	}

	// Verify CreateConversation was called with the correct messages
	if !storage.WasCreateConversationCalled() {
		t.Fatal("Expected CreateConversation to be called")
	}
	if storage.GetUserMessage(sessionID) != "What dishes do they have?" {
		t.Fatalf("Expected user message to match, got: %s", storage.GetUserMessage(sessionID))
	}
}
