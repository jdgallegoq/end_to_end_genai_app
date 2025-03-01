from config.config import openai_api_key

from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from services.prompts.prompts import SYS_PROMPT


class OpenAILLM:
    def __init__(self, model="gpt-4o-mini"):
        self.llm = ChatOpenAI(
            api_key=openai_api_key, model=model, temperature=0.1, streaming=True
        )
        self.memory = ConversationBufferWindowMemory(k=20, return_messages=True)

    def define_prompt(self, sys_prompt: str = SYS_PROMPT) -> ChatPromptTemplate:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", sys_prompt),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}"),
            ]
        )

        return prompt

    def base_chain(self, sys_prompt: str = SYS_PROMPT):
        self.chain = self.define_prompt(sys_prompt=SYS_PROMPT) | self.llm

    def chain(
        self,
        sys_prompt: str = SYS_PROMPT,
        st_chat_history: StreamlitChatMessageHistory = None,
    ):
        conversation_chain = RunnableWithMessageHistory(
            self.base_chain(sys_prompt=SYS_PROMPT),
            lambda session_id: st_chat_history,
            input_messages_key="input",
            history_messages_key="history",
        )

        return conversation_chain
