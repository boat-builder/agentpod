package agentpod

import (
	"context"
	"log/slog"

	"github.com/boat-builder/agentpod/agent"
	"github.com/boat-builder/agentpod/agentMessage"
	"github.com/boat-builder/agentpod/agentpod/session"
	"github.com/boat-builder/agentpod/llm"
	"github.com/boat-builder/agentpod/memory"
	"github.com/openai/openai-go"
)

type Agent = agent.Agent
type Skill = agent.Skill
type Tool = agent.Tool
type LLMConfig = llm.LLMConfig
type Memory = memory.Memory

type Pod struct {
	llmConfig *LLMConfig
	Mem       Memory
	Agent     Agent
	logger    *slog.Logger
}

// NewPod constructs a new Pod with the given resources.
func NewPod(llmConfig *LLMConfig, mem Memory, ag *Agent) *Pod {
	return &Pod{
		llmConfig: llmConfig,
		Mem:       mem,
		Agent:     *ag,
		logger:    slog.Default(),
	}
}

// NewSession creates a new conversation session for a given user and session ID.
// A session handles a single user message and maintains the internal state of the agents
// as they interact to generate a response.
func (p *Pod) NewSession(ctx context.Context, userID, sessionID string) *session.Session {
	sess := session.NewSession(ctx, userID, sessionID, p.llmConfig.Model)
	go p.run(sess)
	return sess
}

// run is the main loop for the session. It listens for user messages and process here. Although
// we don't support now, the idea is that session should support interactive mode which is why
// the input channel exists. Session should hold the control of how to route the messages to whichever agents
// when we support multiple agents.
// TODO - handle refusal everywhere
// TODO - handle other errors like network errors everywhere
func (p *Pod) run(sess *session.Session) {
	defer sess.Close()
	select {
	case <-sess.Ctx.Done():
		sess.OutUserChannel <- agentMessage.Message{Type: agentMessage.MessageTypeEnd}
	case userMessage, ok := <-sess.InUserChannel:
		if !ok {
			p.logger.Error("Session input channel closed")
			sess.OutUserChannel <- agentMessage.Message{Type: agentMessage.MessageTypeEnd}
			return
		}
		completion := openai.ChatCompletionAccumulator{}
		outAgentChannel, err := p.Agent.Run(sess.Ctx, sess.WithUserMessage(userMessage), p.llmConfig.NewLLMClient(), p.llmConfig.Model)
		if err != nil {
			sess.OutUserChannel <- agentMessage.Message{
				Content: err.Error(),
				Type:    agentMessage.MessageTypeError,
			}
		}
		var openAIMessageID string
		for chunk := range outAgentChannel {
			// when chunk id is not same as the previous one, it's part of a new message. Reset everything.
			if chunk.ID != openAIMessageID {
				openAIMessageID = chunk.ID
				completion = openai.ChatCompletionAccumulator{}
			}
			completion.AddChunk(chunk)
			// We won't send the message as a "final message" because there can be other streams in progress.
			// We'll wait for the channel to close
			if len(chunk.Choices) > 0 && chunk.Choices[0].Delta.Content != "" {
				sess.OutUserChannel <- agentMessage.Message{
					Content: chunk.Choices[0].Delta.Content,
					Type:    agentMessage.MessageTypePartialText,
				}
			}
		}

		// channel is closed, send the final message
		sess.OutUserChannel <- agentMessage.Message{
			Type: agentMessage.MessageTypeEnd,
		}
	}

}

// NewAgent constructs a new Agent with the given LLM client and skills.
func NewAgent(prompt string, skills []Skill) *Agent {
	return agent.NewAgent(prompt, skills)
}

// NewSkill constructs a new Skill with the given name, description, and tools.
func NewSkill(name, description string, tools []Tool) *Skill {
	return agent.NewSkill(name, description, tools)
}

// NewMemory constructs a new Zep memory implementation.
func NewMemory() Memory {
	return memory.NewZep()
}
