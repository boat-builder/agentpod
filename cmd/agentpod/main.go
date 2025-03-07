package main

import (
	"flag"
	"fmt"
	"log"
	"net/url"
	"os"
	"strings"

	"github.com/boat-builder/agentpod/dashboard"
)

const defaultPort = 55501
const defaultAssetURL = "https://dash-assets.agentpod.ai"

func main() {
	// Define command line flags
	startCmd := flag.NewFlagSet("start", flag.ExitOnError)

	// Define flags for the start command
	initDB := startCmd.Bool("init", false, "Initialize the database")
	assetURL := startCmd.String("asset", defaultAssetURL, "URL for dashboard assets")
	postgresURI := startCmd.String("postgres", "", "PostgreSQL connection URI (required)")
	port := startCmd.Int("port", defaultPort, "Port to run the dashboard server on")

	// Check if any command is provided
	if len(os.Args) < 2 {
		fmt.Println("Expected 'start' subcommand")
		os.Exit(1)
	}

	// Parse the command
	switch os.Args[1] {
	case "start":
		startCmd.Parse(os.Args[2:])
	default:
		fmt.Printf("Unknown command: %s\n", os.Args[1])
		fmt.Println("Expected 'start' subcommand")
		os.Exit(1)
	}

	// Validate required flags
	if startCmd.Parsed() {
		if *postgresURI == "" {
			fmt.Println("Error: --postgres flag is required")
			startCmd.PrintDefaults()
			os.Exit(1)
		}

		// Validate asset URL if provided
		if *assetURL != defaultAssetURL {
			_, err := url.ParseRequestURI(*assetURL)
			if err != nil {
				fmt.Printf("Error: Invalid asset URL: %s\n", err)
				os.Exit(1)
			}
		}

		// Make sure the asset URL starts with http:// or https://
		if !strings.HasPrefix(*assetURL, "http://") && !strings.HasPrefix(*assetURL, "https://") {
			fmt.Println("Error: Asset URL must start with http:// or https://")
			os.Exit(1)
		}
	}

	// Initialize storage
	storage, err := dashboard.NewPostgresStorage(*postgresURI)
	if err != nil {
		log.Fatalf("Failed to initialize storage: %v", err)
	}

	// Initialize database if requested
	if *initDB {
		log.Println("Initializing database...")
		if err := storage.InitDB(); err != nil {
			log.Fatalf("Failed to initialize database: %v", err)
		}
		log.Println("Database initialized successfully")
	}

	// Create and start the dashboard
	dash := dashboard.NewDashboard(*assetURL, storage)
	log.Printf("Starting dashboard server on port %d...", *port)
	if err := dash.Serve(*port); err != nil {
		log.Fatalf("Failed to start dashboard server: %v", err)
	}
}
