from config.config import model_name, openai_api_key

from operator import itemgetter
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.runnables import RunnablePassthrough

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
        if self.qa_rag and retriever is not None:
            # Simpler chain definition using RunnablePassthrough
            def get_context(question):
                docs = retriever.get_relevant_documents(question)
                return format_docs(docs)
            
            chain = (
                RunnablePassthrough.assign(
                    context=lambda x: get_context(x["question"])
                )
                | self.define_prompt()
                | self.llm
            )
        else:
            chain = self.define_prompt(sys_prompt=sys_prompt) | self.memory | self.llm
        
        # Store the chain as an instance variable and also return it
        self.chain = chain
        return chain

    def chain(
        self,
        sys_prompt: str = SYS_PROMPT,
        st_chat_history: StreamlitChatMessageHistory = None,
        retriever: any = None,
    ):
        # Create the base chain
        base_chain = self.base_chain(sys_prompt=sys_prompt, retriever=retriever)
        
        # Set up the conversation chain with history
        # Use question as the input_messages_key to match what's being sent in app_streamlit.py
        conversation_chain = RunnableWithMessageHistory(
            base_chain,
            lambda session_id: st_chat_history,
            input_messages_key="question",
            history_messages_key="history",
        )

        return conversation_chain
