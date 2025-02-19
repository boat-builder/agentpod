package tests

import (
	"log"
	"os"

	"github.com/joho/godotenv"
)

type Config struct {
	OpenAIAPIKey       string
	AzureAIAPIKey      string
	AzureAIEndpoint    string
	KeywordsAIAPIKey   string
	KeywordsAIEndpoint string
}

func LoadConfig() *Config {
	err := godotenv.Load()
	if err != nil {
		log.Println("Error loading .env file, falling back to environment variables")
	}

	return &Config{
		OpenAIAPIKey:       getEnv("OPENAI_API_KEY", ""),
		AzureAIAPIKey:      getEnv("AZURE_AI_API_KEY", ""),
		AzureAIEndpoint:    getEnv("AZURE_AI_ENDPOINT", ""),
		KeywordsAIAPIKey:   getEnv("KEYWORDSAI_API_KEY", ""),
		KeywordsAIEndpoint: getEnv("KEYWORDSAI_ENDPOINT", ""),
	}
}

func getEnv(key, fallback string) string {
	if value, exists := os.LookupEnv(key); exists {
		return value
	}
	return fallback
}
