package agentpod

type Storage interface {
	GetConversation(session *Session, limit int, offset int) (MessageList, error)
	SaveConversation(session *Session, messages MessageList) error

	GetUserInfo(session *Session) (UserInfo, error)
}
