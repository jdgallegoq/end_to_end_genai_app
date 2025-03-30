from uuid import uuid4
import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from services.llm import OpenAILLM
from services.handlers.handlers import StreamHandler, PostMessageHandler
from services.retriever import configure_retriever
from PIL import Image
import io
import base64

from config.config import TASK

# Customize initial app landing page
st.set_page_config(page_title="AI Assistant", page_icon="🤖")
st.title("Welcome I am AI Assistant 🤖")

# Generate a session id and store it in session state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid4())

streamlit_msg_history = StreamlitChatMessageHistory(key="langchain_messages")

task = TASK.get("multimodal")

# Function to encode image to base64 for storing in chat history
def encode_image_to_base64(image_file):
    if image_file is None:
        return None
    
    # Reset pointer to beginning of file
    image_file.seek(0)
    image_bytes = image_file.read()
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return encoded

if __name__ == "__main__":
    # configure uploaded files
    uploaded_files = st.sidebar.file_uploader(
        label="Upload Image file",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info("please upload Image file to continue.")
        st.stop()
    
    # Display uploaded images in the sidebar
    st.sidebar.header("Uploaded Images")
    for i, uploaded_file in enumerate(uploaded_files):
        # Open and display the image
        image = Image.open(uploaded_file)
        st.sidebar.image(image, caption=f"Image {i+1}: {uploaded_file.name}", use_container_width=True)
        # Reset file pointer to beginning after reading
        uploaded_file.seek(0)

    # start chain
    if task == "qa_rag":
        conversation_chain = OpenAILLM(task="qa_rag").chain(st_chat_history=streamlit_msg_history)
    elif task == "multimodal":
        conversation_chain = OpenAILLM(task="multimodal").chain(st_chat_history=streamlit_msg_history)
    else:
        conversation_chain = OpenAILLM(task="text").chain(st_chat_history=streamlit_msg_history)

    if len(streamlit_msg_history.messages) == 0:
        streamlit_msg_history.add_ai_message("How can I help you? You can ask me about the image you uploaded.")

    # render current messages from streamlit_msg_history
    for msg in streamlit_msg_history.messages:
        st.chat_message(msg.type).write(msg.content)

    # If user inputs a new prompt, display it and show the response
    if user_prompt := st.chat_input():
        # Select the first image for processing
        if uploaded_files:
            # Reset pointer to beginning of file
            uploaded_files[0].seek(0)
            selected_image = uploaded_files[0]
            
            # Display the user message with both text and image
            with st.chat_message("human"):
                st.write(user_prompt)
                image = Image.open(selected_image)
                st.image(image, width=300)
                
            # Store the message in a text-only format for history
            streamlit_msg_history.add_user_message(user_prompt)
                
            # Reset the file pointer again after displaying
            selected_image.seek(0)
            image_data = selected_image
        else:
            image_data = None
            st.chat_message("human").write(user_prompt)
            streamlit_msg_history.add_user_message(user_prompt)
        
        with st.chat_message("ai"):
            stream_handler = StreamHandler(st.empty())
            sources_container = st.write("")
            pm_handler = PostMessageHandler(sources_container)
            
            # Add callbacks to config
            config = {
                "callbacks": [stream_handler, pm_handler],
                "configurable": {"session_id": st.session_state.session_id}
            }
            
            # Pass the image and question without session_id in the inputs
            response = conversation_chain.invoke(
                {
                    "question": user_prompt,
                    "image": image_data  # Pass the file object
                },
                config
            )
            
            # Add the AI response to the message history
            if hasattr(response, "content"):
                streamlit_msg_history.add_ai_message(response.content)
