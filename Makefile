.PHONY: build clean install

# Binary name
BINARY_NAME=agentpod

# Go parameters
GOCMD=go
GOBUILD=$(GOCMD) build
GOCLEAN=$(GOCMD) clean
GOINSTALL=$(GOCMD) install

# Main build target
build:
	$(GOBUILD) -o $(BINARY_NAME) -v ./cmd/agentpod

# Install to GOPATH/bin
install:
	$(GOINSTALL) ./cmd/agentpod

# Clean up
clean:
	$(GOCLEAN)
	rm -f $(BINARY_NAME)

# Default target
all: build 