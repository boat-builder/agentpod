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

// MockDB simulates a database for orders and inventory, and tracks tool execution.
type MockDB struct {
	mu          sync.Mutex
	orders      map[string]map[string]int
	orderStatus map[string]string
	inventory   map[string]int
	trace       []string
}

func NewMockDB() *MockDB {
	return &MockDB{
		orders: map[string]map[string]int{
			"order-123": {"item-abc": 2, "item-def": 1},
		},
		orderStatus: make(map[string]string),
		inventory: map[string]int{
			"item-abc": 10,
			"item-def": 5,
		},
		trace: []string{},
	}
}

func (db *MockDB) addTrace(event string) {
	db.mu.Lock()
	defer db.mu.Unlock()
	db.trace = append(db.trace, event)
}

// --- OrderManagementSkill Tools ---

type GetOrderDetailsTool struct {
	db *MockDB
}

func (t *GetOrderDetailsTool) Name() string        { return "GetOrderDetails" }
func (t *GetOrderDetailsTool) Description() string { return "Gets the details of an order by its ID." }
func (t *GetOrderDetailsTool) StatusMessage() string {
	return "Getting order details..."
}
func (t *GetOrderDetailsTool) OpenAI() []openai.ChatCompletionToolParam {
	return []openai.ChatCompletionToolParam{
		{
			Function: openai.FunctionDefinitionParam{
				Name:        t.Name(),
				Description: param.Opt[string]{Value: t.Description()},
				Parameters: openai.FunctionParameters{
					"type": "object",
					"properties": map[string]interface{}{
						"orderID": map[string]interface{}{
							"type":        "string",
							"description": "The ID of the order to retrieve.",
						},
					},
					"required": []string{"orderID"},
				},
			},
		},
	}
}
func (t *GetOrderDetailsTool) Execute(ctx context.Context, args map[string]interface{}) (string, error) {
	t.db.addTrace(t.Name())
	orderID := args["orderID"].(string)
	t.db.mu.Lock()
	defer t.db.mu.Unlock()
	details, ok := t.db.orders[orderID]
	if !ok {
		return "", fmt.Errorf("order %s not found", orderID)
	}
	return fmt.Sprintf("Order %s contains: %v", orderID, details), nil
}

type UpdateOrderStatusTool struct {
	db *MockDB
}

func (t *UpdateOrderStatusTool) Name() string { return "UpdateOrderStatus" }
func (t *UpdateOrderStatusTool) Description() string {
	return "Updates the status of an order."
}
func (t *UpdateOrderStatusTool) StatusMessage() string { return "Updating order status..." }
func (t *UpdateOrderStatusTool) OpenAI() []openai.ChatCompletionToolParam {
	return []openai.ChatCompletionToolParam{
		{
			Function: openai.FunctionDefinitionParam{
				Name:        t.Name(),
				Description: param.Opt[string]{Value: t.Description()},
				Parameters: openai.FunctionParameters{
					"type": "object",
					"properties": map[string]interface{}{
						"orderID": map[string]interface{}{
							"type":        "string",
							"description": "The ID of the order to update.",
						},
						"status": map[string]interface{}{
							"type":        "string",
							"description": "The new status for the order.",
						},
					},
					"required": []string{"orderID", "status"},
				},
			},
		},
	}
}
func (t *UpdateOrderStatusTool) Execute(ctx context.Context, args map[string]interface{}) (string, error) {
	t.db.addTrace(t.Name())
	orderID := args["orderID"].(string)
	status := args["status"].(string)
	t.db.mu.Lock()
	defer t.db.mu.Unlock()
	t.db.orderStatus[orderID] = status
	return fmt.Sprintf("Order %s status updated to %s", orderID, status), nil
}

// --- InventoryManagementSkill Tools ---

type CheckStockTool struct {
	db *MockDB
}

func (t *CheckStockTool) Name() string        { return "CheckStock" }
func (t *CheckStockTool) Description() string { return "Checks the stock level for a given item ID." }
func (t *CheckStockTool) StatusMessage() string {
	return "Checking stock..."
}
func (t *CheckStockTool) OpenAI() []openai.ChatCompletionToolParam {
	return []openai.ChatCompletionToolParam{
		{
			Function: openai.FunctionDefinitionParam{
				Name:        t.Name(),
				Description: param.Opt[string]{Value: t.Description()},
				Parameters: openai.FunctionParameters{
					"type": "object",
					"properties": map[string]interface{}{
						"itemID": map[string]interface{}{
							"type":        "string",
							"description": "The ID of the item to check.",
						},
					},
					"required": []string{"itemID"},
				},
			},
		},
	}
}
func (t *CheckStockTool) Execute(ctx context.Context, args map[string]interface{}) (string, error) {
	t.db.addTrace(t.Name())
	itemID := args["itemID"].(string)
	t.db.mu.Lock()
	defer t.db.mu.Unlock()
	stock, ok := t.db.inventory[itemID]
	if !ok {
		return "", fmt.Errorf("item %s not found in inventory", itemID)
	}
	return fmt.Sprintf("Item %s has %d units in stock.", itemID, stock), nil
}

type UpdateStockTool struct {
	db *MockDB
}

func (t *UpdateStockTool) Name() string        { return "UpdateStock" }
func (t *UpdateStockTool) Description() string { return "Updates the stock level for an item." }
func (t *UpdateStockTool) StatusMessage() string {
	return "Updating stock..."
}
func (t *UpdateStockTool) OpenAI() []openai.ChatCompletionToolParam {
	return []openai.ChatCompletionToolParam{
		{
			Function: openai.FunctionDefinitionParam{
				Name:        t.Name(),
				Description: param.Opt[string]{Value: t.Description()},
				Parameters: openai.FunctionParameters{
					"type": "object",
					"properties": map[string]interface{}{
						"itemID": map[string]interface{}{
							"type":        "string",
							"description": "The ID of the item to update.",
						},
						"quantity": map[string]interface{}{
							"type":        "integer",
							"description": "The quantity to adjust the stock by (negative to decrease).",
						},
					},
					"required": []string{"itemID", "quantity"},
				},
			},
		},
	}
}
func (t *UpdateStockTool) Execute(ctx context.Context, args map[string]interface{}) (string, error) {
	t.db.addTrace(t.Name())
	itemID := args["itemID"].(string)
	quantity := int(args["quantity"].(float64))
	t.db.mu.Lock()
	defer t.db.mu.Unlock()
	if _, ok := t.db.inventory[itemID]; !ok {
		return "", fmt.Errorf("item %s not found in inventory", itemID)
	}
	t.db.inventory[itemID] += quantity
	return fmt.Sprintf("Stock for item %s updated by %d.", itemID, quantity), nil
}

func TestECommerceOrderFulfillment(t *testing.T) {
	config := LoadConfig()
	if config.KeywordsAIAPIKey == "" || config.KeywordsAIEndpoint == "" {
		t.Skip("Skipping test: KeywordsAIAPIKey or KeywordsAIEndpoint is not set")
	}

	llm := agentpod.NewLLM(
		config.KeywordsAIAPIKey,
		config.KeywordsAIEndpoint,
		"azure/o3-mini",
		"azure/gpt-4o-mini",
	)

	db := NewMockDB()
	mem := &MockMemory{RetrieveFn: getDefaultMemory}

	orderSkill := agentpod.Skill{
		Name:         "OrderManagementSkill",
		Description:  "Manages customer orders, including retrieving details (such as order Id, item details etc) and updating status.",
		SystemPrompt: "You are an order management specialist.",
		Tools: []agentpod.Tool{
			&GetOrderDetailsTool{db: db},
			&UpdateOrderStatusTool{db: db},
		},
	}

	inventorySkill := agentpod.Skill{
		Name:         "InventoryManagementSkill",
		Description:  "Manages warehouse inventory, including checking and updating stock levels.",
		SystemPrompt: "You are an inventory management specialist. You are dependent on the Item ID to do any operations on the inventory.",
		Tools: []agentpod.Tool{
			&CheckStockTool{db: db},
			&UpdateStockTool{db: db},
		},
	}

	agent := agentpod.NewAgent("You are an e-commerce order fulfillment agent. Your goal is to process new orders by checking and updating inventory, and then updating the order status.", []agentpod.Skill{orderSkill, inventorySkill})

	ctx := context.Background()
	ctx = context.WithValue(ctx, agentpod.ContextKey("customerID"), GenerateNewTestID())
	ctx = context.WithValue(ctx, agentpod.ContextKey("extra"), map[string]string{"user_id": GenerateNewTestID()})

	convSession := agentpod.NewSession(ctx, llm, mem, agent)
	convSession.In("Process order order-123")

	var finalContent string
	for {
		out := convSession.Out()
		switch out.Type {
		case agentpod.ResponseTypePartialText:
			finalContent += out.Content
		case agentpod.ResponseTypeError:
			t.Fatalf("Received an unexpected error: %s", out.Content)
		case agentpod.ResponseTypeEnd:
			goto endLoop
		}
	}
endLoop:

	// Assert final state
	db.mu.Lock()
	defer db.mu.Unlock()

	if db.inventory["item-abc"] != 8 {
		t.Errorf("Expected inventory for item-abc to be 8, got %d", db.inventory["item-abc"])
	}
	if db.inventory["item-def"] != 4 {
		t.Errorf("Expected inventory for item-def to be 4, got %d", db.inventory["item-def"])
	}
	if db.orderStatus["order-123"] != "PROCESSED" {
		t.Errorf("Expected order status for order-123 to be 'PROCESSED', got '%s'", db.orderStatus["order-123"])
	}

	// Assert tool execution trace
	expectedTrace := []string{
		"GetOrderDetails",
		"CheckStock",
		"CheckStock",
		"UpdateStock",
		"UpdateStock",
		"UpdateOrderStatus",
	}

	if len(db.trace) != len(expectedTrace) {
		t.Fatalf("Expected %d tool calls, but got %d. Trace: %v", len(expectedTrace), len(db.trace), db.trace)
	}

	for i, expected := range expectedTrace {
		if db.trace[i] != expected {
			t.Errorf("Expected tool call #%d to be '%s', but got '%s'. Trace: %v", i+1, expected, db.trace[i], db.trace)
		}
	}

	if !strings.Contains(strings.ToLower(finalContent), "processed") {
		t.Logf("Final content: %s", finalContent)
	}

}
