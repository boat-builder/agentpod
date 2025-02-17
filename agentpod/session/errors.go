// Package session - errors.go
// Defines session-specific errors.

package session

import "errors"

var (
	ErrSessionClosed = errors.New("session has been closed")
	ErrNoMessage     = errors.New("no message available")
)
