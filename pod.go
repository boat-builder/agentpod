package agentpod

import (
	"context"
	"log"
	"log/slog"

	"github.com/boat-builder/agentpod/dashboard"
)

const DashboardRemoteURL = "https://dash-assets.agentpod.ai"

type Pod struct {
	llmConfig *LLMConfig
	Mem       Memory
	Agent     Agent
	logger    *slog.Logger
	storage   Storage
}

type UserInfo struct {
	Name string
	Meta map[string]string
}

// NewPod constructs a new Pod with the given resources.
func NewPod(llmConfig *LLMConfig, mem Memory, ag *Agent, storage Storage) *Pod {
	return &Pod{
		llmConfig: llmConfig,
		Mem:       mem,
		Agent:     *ag,
		logger:    slog.Default(),
		storage:   storage,
	}
}

// NewSession creates a new conversation session for a given user and session ID.
// A session handles a single user message and maintains the internal state of the agents
// as they interact to generate a response.
func (p *Pod) NewSession(ctx context.Context, customerID, sessionID string, customMeta map[string]string) *Session {
	sess := newSession(ctx, customerID, sessionID, customMeta, p.llmConfig.Model)
	go p.run(sess)
	return sess
}

// run is the main loop for the session. It listens for user messages and process here. Although
// we don't support now, the idea is that session should support interactive mode which is why
// the input channel exists. Session should hold the control of how to route the messages to whichever agents
// when we support multiple agents.
// TODO - handle refusal everywhere
// TODO - handle other errors like network errors everywhere
func (p *Pod) run(sess *Session) {
	defer sess.Close()
	send_status_func := func(msg string) {
		sess.OutUserChannel <- Response{
			Content: msg,
			Type:    ResponseTypeStatus,
		}
	}
	select {
	case <-sess.Ctx.Done():
		sess.OutUserChannel <- Response{Type: ResponseTypeEnd}
	case userMessage, ok := <-sess.InUserChannel:
		if !ok {
			p.logger.Error("Session input channel closed")
			sess.OutUserChannel <- Response{Type: ResponseTypeEnd}
			return
		}
		err := p.storage.CreateConversation(sess, userMessage)
		if err != nil {
			p.logger.Error("Error creating conversation", "error", err)
		}

		outAgentChannel, err := p.Agent.Run(
			sess.Ctx,
			sess.WithUserMessage(userMessage),
			p.llmConfig.NewLLMClient(),
			p.llmConfig.Model,
			send_status_func,
			p.storage,
		)
		if err != nil {
			sess.OutUserChannel <- Response{
				Content: err.Error(),
				Type:    ResponseTypeError,
			}
		}
		aggregated := ""
		for chunk := range outAgentChannel {
			// We'll wait for the channel to close
			if len(chunk) > 0 {
				aggregated += chunk
				sess.OutUserChannel <- Response{
					Content: chunk,
					Type:    ResponseTypePartialText,
				}
			}
		}

		// Finish the conversation in the store
		err = p.storage.FinishConversation(sess, aggregated)
		if err != nil {
			p.logger.Error("Error finishing conversation", "error", err)
		}

		// channel is closed, send the final message
		sess.OutUserChannel <- Response{
			Type: ResponseTypeEnd,
		}
	}
}

func (p *Pod) StartDashboard(port int) {
	if dashboardStorage, ok := p.storage.(dashboard.Storage); ok {
		dashboard := dashboard.NewDashboard(DashboardRemoteURL, dashboardStorage)
		dashboard.Serve(port)
	}
	log.Fatalln("Dashboard storage not implemented")
}
