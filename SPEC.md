# AgentPod Design Spec

## 1. Introduction

AgentPod is an AI agent framework that uses a session-based interaction model. Each user message in the chat window initiates a new session, which handles input processing, context management via memory, and agent coordination. The session ends once a final response is delivered to the user.


```go
// initialization
agent := agentpod.NewAgent(AgentMainPrompt, []agentpod.Skill{
    skills.KeywordResearchSkill(keywordsPlace),
})
memory := agentpod.NewMem0()
storage := agentpod.NewPostgresql()
llm := agentpod.OpenAI(cfg.AI.KeywordsAIAPIKey, cfg.AI.KeywordsAIBaseURL)

// create pod
pod := agentpod.NewPod(agent, memory, storage, llm)


// start the session
session := pod.NewSession(ctx, userID, sessionID, meta)
session.In("Hey there")
msg := session.Out()
```


## 2. High-Level Architecture

The system consists of three primary layers that represent both functional components and a hierarchical organization of entities:

### Session
The Session Layer handles incoming user messages and serves as the entry point for all interactions. It performs initial LLM calls to determine next actions based on user input, updates memory with relevant information, and constructs the appropriate context for processing. Though orthogonal to the agent object itself, the session takes the agent object and invokes it to process input requests while providing necessary context. Once prepared, it invokes the agent with this enriched context to generate responses. A key feature of the Session Layer is its ability to maintain session state for pause/resume capabilities. This enables the system to handle long-running operations without keeping an active session process in memory. When a task requires extended processing time, the system can save the session state and resume from that saved state when new information becomes available, allowing for efficient resource utilization.

### Agent
The Agent Layer provides the structural framework for organizing capabilities within the system. Agents are configured with system prompts and skills that define their behavior and accessible functionality. For any given task, an agent "chooses which hat to wear" by selecting the appropriate skill for the current context. The agent's high-level run method determines if a task requires one or multiple skills. When multiple skills are needed, the agent creates separate instances for each skill context, potentially passing outputs from one skill instance to another, and merges the results together. This orchestration is controlled by the session runner. Rather than being complex entities themselves, agents primarily serve as organized containers for the skills and tools hierarchy, with the ability to coordinate between different skill contexts to accomplish complex tasks.

### Skill and Tool
The Skill and Tool Layer implements a hierarchical approach to functionality organization. Skills are defined as groups of related tools with associated system prompts that specify their domain or functionality. When an agent operates within a specific skill context (wearing a particular "hat"), it only has access to the tools that belong to that skill. This design allows developers to expose only relevant tools when needed, rather than overwhelming the LLM with all available options. Instead of implementing multiple isolated agents each with a single task, the framework enables a hierarchical organization of capabilities through skills. Each skill encapsulates a specific system prompt and a collection of tools that can be grouped logically, making the system more modular and easier to extend. An agent cannot wear different skill hats simultaneously but can switch between them as orchestrated by its high-level run method.


## FAQ

> Why don't we use "send_message" or something similar to send the messages back to the user?

As all the models becoming reasoning models, internal monologue is becoming a different thing at the model level. With that, all the non-thinking tokens the model generates must sent to the user 
