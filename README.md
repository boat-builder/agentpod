# AgentPod

AgentPod is a Go package that provides an extensible framework for building agents with support for skills, tools, memory, and sessions.

## Package Structure

- go.mod: Module definition.
- README.md: Project description.
- agentpod/
  - agent.go: Defines the Agent struct and its interface.
  - skill.go: Defines Skill and related interfaces.
  - tool.go: Defines Tool and related interfaces.
  - memory.go: Defines Memory interface and default implementation.
  - session.go: Defines the Session struct and its methods.
  - message.go: Defines message-related types like AgentMessage.
  - errors.go: Centralized error definitions.

AI Agent Framework Design Document

1. Introduction

This document outlines the design for an AI agent framework that leverages a session-based interaction model. Each new message from the user in the chat window triggers a new session. The session is responsible for pre-processing the input, managing context (via memory), and then handing control over to a unified agent that uses skills and tools to generate a response. Once the agent completes its processing, the final message is delivered to the user, and the session lifecycle ends.

2. High-Level Architecture

The system is composed of three primary layers:
	•	Session Layer:
Handles incoming user messages, performs initial LLM calls to determine the next action, updates memory, and constructs the context. It then invokes the agent with the enriched context.
	•	Agent Layer:
Acts as the central orchestrator. The agent uses a predefined system prompt, the provided context, and a set of skills to manage the dialogue. It interacts with the LLM repeatedly to refine responses, triggers tool calls through skills as needed, and engages in back-and-forth communication with the user until a final response is ready.
	•	Skill and Tool Layer:
Instead of having multiple isolated agents (each with a single task), the design uses the concept of skills. Each skill is a group of tools and is associated with a system prompt that defines its domain or functionality. The agent can invoke a particular skill (and its tools) as part of its processing.

3. Components and Their Responsibilities

3.1 Session
	•	Instantiation:
Every new chat message from the user instantiates a new session object.
	•	The session receives dependencies: a memory instance, an LLM model, and the central agent (which includes its skills).
	•	Responsibilities:
	•	Initial Processing:
Immediately on creation (or via an initial Run(userPrompt) call), the session calls the LLM to evaluate the raw user message.
	•	Memory Integration:
Based on the LLM's output and any existing session memory, it determines if additional context needs to be appended or if the memory should be updated.
	•	Context Construction:
If necessary, the session builds a context using the information retrieved from memory. This context is then combined with a system prompt (if applicable) before handing control to the agent.
	•	Message Routing:
The session provides a unified message flow interface. It accepts user input via a method (e.g., In()) and delivers responses (status updates, intermediate results, input requests, final responses) through an output method (e.g., Out()).
	•	Lifecycle:
The session's life is transient—once the agent has finished processing the conversation (i.e., the final message is generated), the session is considered complete, and its resources may be cleaned up.

3.2 Agent
	•	Instantiation:
The agent is a singleton (or a shared instance) that is passed to every new session. It is constructed with:
	•	A system prompt that provides high-level instructions.
	•	A set of skills.
	•	A reference to an LLM model (which may also be shared).
	•	Responsibilities:
	•	Orchestration:
Once invoked by the session with an enriched context, the agent takes over. It uses its system prompt combined with the provided context to determine the next course of action.
	•	Dialogue Management:
The agent performs iterative back-and-forth with the LLM:
	•	It may ask for clarifications or request additional information.
	•	It manages intermediate responses, including status updates and input requests.
	•	Skill Invocation:
Based on the conversation's evolution, the agent selects an appropriate skill. Each skill comprises a group of tools tailored for a specific domain (e.g., calculations, search, summarization). The agent uses these skills to execute specialized actions.
	•	Final Response Delivery:
Once all necessary information is processed and the appropriate tools have been called (if needed), the agent generates the final response. This message is passed back to the session and eventually to the user.

3.3 Skills and Tools
	•	Skills:
	•	Definition:
A skill is a grouping of one or more tools along with a descriptive system prompt. This prompt sets the context for that skill, outlining its domain and responsibilities.
	•	Usage:
When the agent identifies a need for specialized processing, it selects a relevant skill. The system prompt associated with the skill helps guide the LLM and ensures that the appropriate tools are invoked.
	•	Tools:
	•	Definition:
Each tool is a functional unit that the agent can call to execute specific tasks. Examples include calculator functions, search APIs, or data retrieval operations.
	•	Invocation:
Tools are invoked as part of the agent's internal workflow. They may provide intermediate results that update the session's memory or refine the context for further LLM calls.

3.4 Memory
	•	Session-Level Memory:
The memory component is specific to a user session. It is designed to capture conversation history and relevant context that may influence subsequent interactions.
	•	Responsibilities:
	•	Update:
As the conversation progresses, the session updates memory with new context, tool outputs, or LLM responses.
	•	Retrieval:
When constructing a context for the agent, the session queries memory to obtain relevant information that might affect the final response.
	•	Benefits:
This design ensures that every new chat window interaction is contextually aware, without the need for a global memory shared across all sessions.

4. Message Flow and User Interaction
	•	Unified Message Pipeline:
All messages (status updates, input requests, final results, errors) flow sequentially to the chat window via a common output mechanism. Each message includes metadata (such as a type) that informs the UI how to display it.
	•	User Interaction:
	•	Initial Trigger:
On receiving a new message, a session object is instantiated.
	•	Session Processing:
The session sends the raw input to the LLM for preliminary analysis. Based on the result, it decides whether to update memory and construct a richer context.
	•	Agent Engagement:
The enriched context is passed to the agent, which then takes over. The agent interacts with the LLM and its skills, may request additional input from the user (signaled via specific message types), and continues the dialogue until a final answer is produced.
	•	Finalization:
Once the final message is ready, it is sent back to the session's output. The session then terminates, marking the end of that particular interaction.

5. Lifecycle of a Session
	1.	Initialization:
	•	User sends a new message.
	•	A new session is created with the LLM, memory, and the agent (with its skills).
	2.	Pre-Processing:
	•	The session calls the LLM to analyze the raw message.
	•	Based on the response, the session determines whether additional memory-based context is required.
	•	Memory is queried and updated as needed.
	3.	Agent Invocation:
	•	The enriched context (raw input + memory context) is passed to the agent.
	•	The agent uses its system prompt and the current context to start a dialogue.
	4.	Iterative Interaction:
	•	The agent communicates with the LLM, potentially invoking skills and their associated tools.
	•	If additional user input is required, the agent signals this via an output message (e.g., an input request).
	•	The session accepts further user messages via a dedicated method and feeds them into the agent's ongoing dialogue.
	5.	Finalization:
	•	Once the agent concludes the conversation (i.e., it has generated a final response), this message is passed back to the session.
	•	The final message is delivered to the user.
	•	The session lifecycle ends, and resources are released.

6. Error Handling and Recovery
	•	Intermediate Errors:
If errors occur during any LLM call or tool execution, error messages (with an appropriate message type) are sent to the user. The session may allow the user to retry or provide additional context.
	•	Session Termination:
In case of unrecoverable errors, the session ensures that the final error message is returned to the user before closing the session.

7. Conclusion

This design leverages a session-based interaction model that integrates LLM processing, session-level memory, and a multi-skill agent architecture. By separating the responsibilities of pre-processing (in the session) and dialogue management (in the agent), the framework offers both flexibility and clarity. Each new user message triggers a dedicated session that constructs context, delegates processing to a unified agent (with specialized skills and tools), and finally returns a coherent, sequentially delivered message to the user. This design not only simplifies multi-turn dialogues but also ensures that specialized tasks are handled via the skills mechanism—making the framework both scalable and user-centric.