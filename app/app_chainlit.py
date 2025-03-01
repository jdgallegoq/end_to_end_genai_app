import chainlit as cl
from langchain.schema.runnable.config import RunnableConfig
from services.llm import OpenAILLM


@cl.on_chat_start
async def when_starts_chat():
    cl.user_session.set("chain", OpenAILLM().chain())
    cl.user_session.set("memory", OpenAILLM().memory)


@cl.on_message
async def on_user_message(message: cl.Message):
    chain = cl.user_session.get("chain")
    memory = cl.user_session.get("memory")

    chatgpt_message = cl.Message(content="")
    async for chunk in chain.astream(
        {"input": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await chatgpt_message.stream_token(chunk)

    await chatgpt_message.send()
    memory.save_context({"input": message.content}, {"output": chatgpt_message.content})
