package tests

import (
	"os"
)

type Config struct {
	OpenAIAPIKey    string
	AzureAIAPIKey   string
	AzureAIEndpoint string
}

func LoadConfig() *Config {
	return &Config{
		OpenAIAPIKey:    os.Getenv("OPENAI_API_KEY"),
		AzureAIAPIKey:   os.Getenv("AZURE_AI_API_KEY"),
		AzureAIEndpoint: os.Getenv("AZURE_AI_ENDPOINT"),
	}
}
