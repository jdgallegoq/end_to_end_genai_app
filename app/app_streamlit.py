import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from services.llm_st import OpenAILLM
from services.handlers.handlers import StreamHandler

# Customize initial app landing page
st.set_page_config(page_title="AI Assistant", page_icon="🤖")
st.title("Welcome I am AI Assistant 🤖")

streamlit_msg_history = StreamlitChatMessageHistory()

conversation_chain = OpenAILLM().chain(st_chat_history=streamlit_msg_history)

if __name__ == "__main__":
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
            config = {"configurable": {"session_id": "any"}, "callbacks": [stream_handler]}
            response = conversation_chain.invoke({"input": user_prompt}, config)
