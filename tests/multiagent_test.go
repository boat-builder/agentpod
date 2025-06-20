package tests

import (
	"context"
	"strings"
	"testing"

	"github.com/boat-builder/agentpod"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/param"
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
			Function: openai.FunctionDefinitionParam{
				Name:        r.toolName,
				Description: param.Opt[string]{Value: r.description},
				Parameters: openai.FunctionParameters{
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
				},
			},
		},
	}
}

func (r *RestaurantTool) Execute(ctx context.Context, args map[string]interface{}) (string, error) {
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
			Function: openai.FunctionDefinitionParam{
				Name:        c.toolName,
				Description: param.Opt[string]{Value: c.description},
				Parameters: openai.FunctionParameters{
					"type": "object",
					"properties": map[string]interface{}{
						"restaurant": map[string]interface{}{
							"type":        "string",
							"description": "Restaurant name",
						},
					},
					"required": []string{"restaurant"},
				},
			},
		},
	}
}

func (c *CuisineTool) Execute(ctx context.Context, args map[string]interface{}) (string, error) {
	restaurant := args["restaurant"].(string)
	if dishes, ok := c.dishes[restaurant]; ok {
		return strings.Join(dishes, ", "), nil
	}
	return "No dishes found for this restaurant", nil
}

// Function to create memory with user preferences
func getUserPreferencesMemory(ctx context.Context) (*agentpod.MemoryBlock, error) {
	memoryBlock := agentpod.NewMemoryBlock()
	userDetailsBlock := agentpod.NewMemoryBlock()
	userDetailsBlock.AddString("location", "Downtown")
	userDetailsBlock.AddString("favorite_cuisines", "Italian")
	memoryBlock.AddBlock("UserDetails", userDetailsBlock)
	return memoryBlock, nil
}

const mainPrompt = `You are a restaurant recommendation expert tasked with helping users find the perfect restaurant based on their location and cuisine preferences. Provide concise and direct recommendations using the available data from authorized tools.

- Focus on the user's location and specified cuisine preferences.
- Avoid making assumptions about restaurants that are not readily available through your tools.
- Ensure recommendations are based solely on the data you can access.
- Clearly communicate the recommendation and justify the choice with relevant details that enhance the user's decision-making process.

(Note: Ensure all relevant data is provided and realistic for actual recommendations.)`

func testRestaurantRecommendation(t *testing.T, prompt string) {
	config := LoadConfig()
	if config.KeywordsAIAPIKey == "" || config.KeywordsAIEndpoint == "" {
		t.Fatal("KeywordsAIAPIKey or KeywordsAIEndpoint is not set")
	}

	llm := agentpod.NewLLM(
		config.KeywordsAIAPIKey,
		config.KeywordsAIEndpoint,
		"azure/o3-mini",
		"azure/gpt-4o",
		"azure/o3-mini",
		"azure/gpt-4o-mini",
	)

	// Create mock memory with user preferences
	mem := &MockMemory{
		RetrieveFn: getUserPreferencesMemory,
	}

	// Create restaurant agent with restaurant recommendation tool
	restaurantTool := NewRestaurantTool()
	cuisineTool := NewCuisineTool()
	restaurantAgent := agentpod.NewAgent(
		mainPrompt,
		[]agentpod.Skill{
			{
				Name:         "RestaurantExpert",
				Description:  "Expert in restaurant recommendations. You cannot make cusine recommendations. We have a cuisine expert for that.",
				SystemPrompt: "As a restaurant expert, you provide personalized restaurant recommendations. Do not make any recommendations on dishes. We have cusines expert for that.",
				Tools:        []agentpod.Tool{restaurantTool},
			},
			{
				Name:         "CuisineExpert",
				Description:  "Expert in cuisine and dishes, you provide dish recommendations for restaurants found by RestaurantExpert. Should not be called before restaurant expert made the restaurant recommendation.",
				SystemPrompt: "As a cuisine expert, you provide dish recommendations for restaurants found by RestaurantExpert. You should only do recommendations on cusines for the restaurants you have access to. You should not assume the existance of any restaurants that you don't have access to",
				Tools:        []agentpod.Tool{cuisineTool},
			},
		},
	)

	ctx := context.Background()
	ctx = context.WithValue(ctx, agentpod.ContextKey("customerID"), GenerateNewTestID())
	ctx = context.WithValue(ctx, agentpod.ContextKey("extra"), map[string]string{"user_id": GenerateNewTestID(), "domain": "test"})

	// Create session with restaurant agent
	restaurantSession := agentpod.NewSession(ctx, llm, mem, restaurantAgent)

	restaurantSession.In(prompt)
	var response string
	for {
		out := restaurantSession.Out()
		if out.Type == agentpod.ResponseTypePartialText {
			response += out.Content
		}
		if out.Type == agentpod.ResponseTypeEnd {
			break
		}
	}

	// Verify restaurant recommendation
	if !strings.Contains(strings.ToLower(response), "pasta paradise") {
		t.Fatal("Expected 'Pasta Paradise' to be in the restaurant recommendation, got:", response)
	}

	// Verify cuisine recommendation
	dishes := []string{"carbonara", "lasagna", "risotto"}
	foundDish := false
	for _, dish := range dishes {
		if strings.Contains(strings.ToLower(response), dish) {
			foundDish = true
			break
		}
	}
	if !foundDish {
		t.Fatal("Expected at least one of the dishes to be in the cuisine recommendation, got:", response)
	}
}

func TestMultiAgentRestaurantRecommendationWithSummarizer(t *testing.T) {
	testRestaurantRecommendation(t, "Can you recommend a good restaurant for me?")
}

func TestMultiAgentRestaurantRecommendationWithoutSummarizer(t *testing.T) {
	testRestaurantRecommendation(t, "I am looking for an Italian restaurant in Downtown. Can you suggest one? After that, can you recommend me some dishes there?")
}
