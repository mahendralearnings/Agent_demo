from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import SQLChatMessageHistory
from .settings import settings

# def get_memory(session_id: str):
#     history = SQLChatMessageHistory(session_id=session_id, connection_string=f"sqlite:///{settings.sqlite_path}")
#     return ConversationBufferMemory(memory_key="history", chat_memory=history, return_messages=True)



def get_memory(session_id: str) -> ConversationBufferMemory:
    history = SQLChatMessageHistory(
        session_id=session_id,
        connection_string=f"sqlite:///{settings.sqlite_path}"
    )
    # use "chat_history" (not "history") to match MessagesPlaceholder
    return ConversationBufferMemory(
        memory_key="chat_history",
        chat_memory=history,
        return_messages=True
    )
