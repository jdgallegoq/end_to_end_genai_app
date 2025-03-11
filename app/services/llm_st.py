from config.config import model_name, openai_api_key

from operator import itemgetter
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.callbacks.base import BaseCallbackHandler

from operator import itemgetter
from utils.utils import format_docs

from services.prompts.prompts import SYS_PROMPT, QA_RAG_PROMPT


class OpenAILLM:
    def __init__(self, qa_rag:bool = False):
        self.llm = ChatOpenAI(
            api_key=openai_api_key, model=model_name, temperature=0.1, streaming=True
        )
        self.memory = ConversationBufferWindowMemory(k=20, return_messages=True)
        self.qa_rag = qa_rag

    def define_prompt(self, sys_prompt: str = SYS_PROMPT) -> ChatPromptTemplate:
        if self.qa_rag:
            prompt = ChatPromptTemplate.from_template(QA_RAG_PROMPT)
        else:
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", sys_prompt),
                    MessagesPlaceholder(variable_name="history"),
                    ("human", "{input}"),
                ]
            )

        return prompt

    def base_chain(self, sys_prompt: str = SYS_PROMPT, retriever: any = None):
        if self.qa_rag:
            self.chain = (
                {
                    "context": itemgetter("question")
                    |
                    retriever
                    |
                    format_docs,
                    "question": itemgetter("question")
                }
                |
                self.define_prompt()
                |
                self.llm
            )
        else:
            self.chain = self.define_prompt(sys_prompt=sys_prompt) | self.memory | self.llm

    def chain(
        self,
        sys_prompt: str = SYS_PROMPT,
        st_chat_history: StreamlitChatMessageHistory = None,
    ):
        conversation_chain = RunnableWithMessageHistory(
            self.base_chain(sys_prompt=sys_prompt),
            lambda session_id: st_chat_history,
            input_messages_key="input",
            history_messages_key="history",
        )

        return conversation_chain
