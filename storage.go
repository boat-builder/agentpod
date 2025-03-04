package agentpod

type Storage interface {
	GetConversation(session *Session, limit int, offset int) (MessageList, error)
	CreateConversation(session *Session, userMessage string) error
	FinishConversation(session *Session, assistantMessage string) error

	GetUserInfo(session *Session) (UserInfo, error)
}
