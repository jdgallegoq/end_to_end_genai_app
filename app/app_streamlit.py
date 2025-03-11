import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from services.llm_st import OpenAILLM
from services.handlers.handlers import StreamHandler, PostMessageHandler
from services.retriever import configure_retriever

from config.config import QA_RAG

# Customize initial app landing page
st.set_page_config(page_title="AI Assistant", page_icon="🤖")
st.title("Welcome I am AI Assistant 🤖")


streamlit_msg_history = StreamlitChatMessageHistory(key="langchain_messages")

if __name__ == "__main__":
    # configure uploaded files
    uploaded_files = st.sidebar.file_uploader(
        label="Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        st.info("please upload PDF documents to continue.")
        st.stop()
    
    # create retriever
    retriever = configure_retriever(uploaded_files)

    # start chain
    if QA_RAG:
        conversation_chain = OpenAILLM(qa_rag=True).chain(st_chat_history=streamlit_msg_history, retriever=retriever)
    else:
        conversation_chain = OpenAILLM().chain(st_chat_history=streamlit_msg_history)

    if len(streamlit_msg_history) == 0:
        streamlit_msg_history.add_ai_message("How can I help you?")

    # render current messages from streamlit_msg_history
    for msg in streamlit_msg_history.messages:
        st.chat_message(msg.type).write(msg.content)

    # If user inputs a new prompt, display it and show the response
    if user_prompt := st.chat_input():
        st.chat_message("human").write(user_prompt)
        with st.chat_message("ai"):
            stream_handler = StreamHandler(st.empty())
            sources_container = st.write("")
            pm_handler = PostMessageHandler(sources_container)
            config = {"callbacks": [stream_handler, pm_handler]}
            response = conversation_chain.invoke({"input": user_prompt}, config)
