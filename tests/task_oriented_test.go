package tests

import (
	"context"
	"encoding/json"
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
	orders      map[string]Order
	orderStatus map[string]string
	inventory   map[string]*InventoryItem
	trace       []string
}

// Define richer data models for orders and inventory
// These models provide explicit IDs, names, and counts so that LLM tooling
// can reference both identifiers and human-readable names.
//
// An individual item that belongs to an order.
// It contains both the machine ID and a human-friendly name.
//
// JSON tags use camelCase to align with typical API payload formats.
type OrderItem struct {
	ItemID   string `json:"itemId"`
	ItemName string `json:"itemName"`
	Quantity int    `json:"quantity"`
}

// A customer order, identified by its orderId and a set of items.
type Order struct {
	OrderID string      `json:"orderId"`
	Items   []OrderItem `json:"items"`
}

// An item stored in the warehouse inventory.
type InventoryItem struct {
	ItemID     string `json:"itemId"`
	ItemName   string `json:"itemName"`
	StoreCount int    `json:"storeCount"`
}

func NewMockDB() *MockDB {
	return &MockDB{
		orders: map[string]Order{
			"order-123": {
				OrderID: "order-123",
				Items: []OrderItem{
					{ItemID: "item-abc", ItemName: "item-abc", Quantity: 2},
					{ItemID: "item-def", ItemName: "item-def", Quantity: 1},
				},
			},
		},
		orderStatus: make(map[string]string),
		inventory: map[string]*InventoryItem{
			"item-abc": {ItemID: "item-abc", ItemName: "item-abc", StoreCount: 10},
			"item-def": {ItemID: "item-def", ItemName: "item-def", StoreCount: 5},
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

func (t *GetOrderDetailsTool) Name() string { return "GetOrderDetails" }
func (t *GetOrderDetailsTool) Description() string {
	return "Gets the details of an order by its ID and returns a JSON payload with the list of items (itemID and quantity)."
}
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
	orderID, _ := args["orderID"].(string)
	t.db.mu.Lock()
	defer t.db.mu.Unlock()
	order, ok := t.db.orders[orderID]
	if !ok {
		return "", fmt.Errorf("order %s not found", orderID)
	}
	// Marshal the full order (including item IDs, names, and quantities) so that the LLM
	// can reference either identifier or display name.
	jsonBytes, err := json.Marshal(order)
	if err != nil {
		return "", fmt.Errorf("failed to marshal order details: %w", err)
	}
	return string(jsonBytes), nil
}

type UpdateOrderStatusTool struct {
	db *MockDB
}

func (t *UpdateOrderStatusTool) Name() string { return "UpdateOrderStatus" }
func (t *UpdateOrderStatusTool) Description() string {
	return "Updates the status of an order. Valid statuses are: PROCESSED & CANCELLED."
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
	resp := struct {
		OrderID string `json:"orderId"`
		Status  string `json:"status"`
	}{OrderID: orderID, Status: status}
	jsonBytes, err := json.Marshal(resp)
	if err != nil {
		return "", fmt.Errorf("failed to marshal status update: %w", err)
	}
	return string(jsonBytes), nil
}

// --- InventoryManagementSkill Tools ---

type CheckStockTool struct {
	db *MockDB
}

func (t *CheckStockTool) Name() string { return "CheckStock" }
func (t *CheckStockTool) Description() string {
	return "Checks the stock level for a given item ID. The itemID must exactly match one of the IDs provided in the order details JSON."
}
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
	invItem, ok := t.db.inventory[itemID]
	if !ok {
		return "", fmt.Errorf("item %s not found in inventory", itemID)
	}
	// Return a structured payload with both the item name and the current stock level.
	jsonBytes, err := json.Marshal(invItem)
	if err != nil {
		return "", fmt.Errorf("failed to marshal inventory item: %w", err)
	}
	return string(jsonBytes), nil
}

type UpdateStockTool struct {
	db *MockDB
}

func (t *UpdateStockTool) Name() string { return "UpdateStock" }
func (t *UpdateStockTool) Description() string {
	return "Updates the stock level for an item. The itemID must exactly match one of the IDs provided in the order details JSON."
}
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
	invItem, ok := t.db.inventory[itemID]
	if !ok {
		return "", fmt.Errorf("item %s not found in inventory", itemID)
	}
	invItem.StoreCount += quantity
	resp := struct {
		ItemID        string `json:"itemId"`
		ItemName      string `json:"itemName"`
		NewStoreCount int    `json:"newStoreCount"`
	}{ItemID: invItem.ItemID, ItemName: invItem.ItemName, NewStoreCount: invItem.StoreCount}
	jsonBytes, err := json.Marshal(resp)
	if err != nil {
		return "", fmt.Errorf("failed to marshal update response: %w", err)
	}
	return string(jsonBytes), nil
}

func TestECommerceOrderFulfillment(t *testing.T) {
	config := LoadConfig()
	if config.KeywordsAIAPIKey == "" || config.KeywordsAIEndpoint == "" {
		t.Skip("Skipping test: KeywordsAIAPIKey or KeywordsAIEndpoint is not set")
	}

	llm := agentpod.NewKeywordsAIClient(
		config.KeywordsAIAPIKey,
		config.KeywordsAIEndpoint,
		"azure/o1",
		"azure/gpt-4o-mini",
	)

	db := NewMockDB()
	mem := &MockMemory{RetrieveFn: getDefaultMemory}

	orderSkill := agentpod.Skill{
		Name:            "OrderManagementSkill",
		ToolDescription: "Manages customer orders, including retrieving order and item details and updating status. If you need to know the details of the items in the order, or you need to place the order and change the status, this is the tool you should use.",
		SystemPrompt:    "You are an order management specialist.",
		Tools: []agentpod.Tool{
			&GetOrderDetailsTool{db: db},
			&UpdateOrderStatusTool{db: db},
		},
	}

	inventorySkill := agentpod.Skill{
		Name:            "InventoryManagementSkill",
		ToolDescription: "Manages warehouse inventory, including checking and updating stock levels. You are dependent on the Item ID to do any operations on the inventory.",
		SystemPrompt:    "You are an inventory management specialist. You are dependent on the Item ID to do any operations on the inventory. If Item ID is not available, you should return an error message to the user. You can't process orders, you can only check and update the stock.",
		Tools: []agentpod.Tool{
			&CheckStockTool{db: db},
			&UpdateStockTool{db: db},
		},
	}
	agentPrompt := `You are an e-commerce order fulfillment agent. Your task is to process incoming orders by interacting only with the tools provided to you. Do not ask the user any questions.

For each order:
  1. Check inventory for all items in the order.
  2. If all items are in stock:
    - Deduct the ordered quantity from inventory.
    - Update the order status to “fulfilled”.
  3. If any item is out of stock:
    - Do not modify the inventory.
    - Do not update the order status.
    - Return an error message stating which items are insufficient in stock.

Always ensure inventory is validated before processing any order. You must only use tool outputs to make decisions.`

	agent := agentpod.NewAgent(agentPrompt, []agentpod.Skill{orderSkill, inventorySkill})

	ctx := context.Background()
	ctx = context.WithValue(ctx, agentpod.ContextKey("customerID"), GenerateNewTestID())
	ctx = context.WithValue(ctx, agentpod.ContextKey("extra"), map[string]string{"user_id": GenerateNewTestID()})

	convSession := agentpod.NewSession(ctx, llm, mem, agent)
	convSession.In("Process order order-123")

	var finalContent string
	for {
		out := convSession.Out()
		switch out.Type {
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

	if db.inventory["item-abc"].StoreCount != 8 {
		t.Errorf("Expected inventory for item-abc to be 8, got %d", db.inventory["item-abc"].StoreCount)
	}
	if db.inventory["item-def"].StoreCount != 4 {
		t.Errorf("Expected inventory for item-def to be 4, got %d", db.inventory["item-def"].StoreCount)
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
