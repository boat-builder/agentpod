// Package agentpod - errors.go
// Defines session-specific errors.

package agentpod

import "errors"

var (
	ErrSessionClosed = errors.New("session has been closed")
	ErrNoMessage     = errors.New("no message available")
)
