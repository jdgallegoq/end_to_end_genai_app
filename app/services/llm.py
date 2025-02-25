from config.config import openai_api_key

from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain.schema import StrOutputParser
from operator import itemgetter


class OpenAILLM:
    def __init__(self, model="gpt-4o-mini"):
        self.llm = ChatOpenAI(
            api_key=openai_api_key, model=model, temperature=0.1, streaming=True
        )
        self.memory = ConversationBufferWindowMemory(k=20, return_messages=True)

    def define_prompt(self, sys_prompt: str = None) -> ChatPromptTemplate:
        if not sys_prompt:
            sys_prompt = """
                Act as a helpful assistant and answer questions to the best of your ability.
                Do not make up answers.
                """

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", sys_prompt),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}"),
            ]
        )

        return prompt

    def chain(self, sys_prompt: str):
        conversation_chain = (
            RunnablePassthrough.assign(
                history=RunnableLambda(self.memory.load_memory_variables)
            )
            | itemgetter("history")
            | self.define_prompt()
            | self.llm
            | StrOutputParser()
        )

        return conversation_chain
