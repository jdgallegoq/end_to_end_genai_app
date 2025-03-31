from uuid import uuid4
import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from services.llm_st import OpenAILLM
from services.handlers.handlers import StreamHandler, PostMessageHandler
from services.retriever import configure_retriever
import logging

from config.config import QA_RAG

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Customize initial app landing page
st.set_page_config(page_title="AI Assistant", page_icon="🤖")
st.title("Welcome I am AI Assistant 🤖")

# Generate a session id and store it in session state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid4())

# Initialize chat history
streamlit_msg_history = StreamlitChatMessageHistory(key="langchain_messages")

if __name__ == "__main__":
    # Configure uploaded files
    uploaded_files = st.sidebar.file_uploader(
        label="Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info("Please upload PDF documents to continue.")
        st.stop()
    
    # Process PDFs and set up the chain
    try:
        # Create retriever
        with st.spinner("Processing documents... This might take a moment."):
            retriever = configure_retriever(uploaded_files)
            
        if retriever is None:
            st.error("Failed to process documents. Please try uploading different PDF files.")
            st.stop()
            
        # Log document processing success
        logger.info("Document retriever created successfully")
        st.success(f"Successfully processed {len(uploaded_files)} document(s)")
        
        # Create conversation chain
        if QA_RAG:
            st.info("Running in Question-Answering mode with document retrieval")
            conversation_chain = OpenAILLM(qa_rag=True).chain(
                st_chat_history=streamlit_msg_history, 
                retriever=retriever
            )
        else:
            conversation_chain = OpenAILLM().chain(
                st_chat_history=streamlit_msg_history
            )
    
        # Initialize with welcome message if history is empty
        if len(streamlit_msg_history.messages) == 0:
            welcome_msg = "How can I help you with your documents? Ask me anything about the PDFs you've uploaded."
            streamlit_msg_history.add_ai_message(welcome_msg)
    
        # Render current messages from streamlit_msg_history
        for msg in streamlit_msg_history.messages:
            st.chat_message(msg.type).write(msg.content)
    
        # Handle user input
        if user_prompt := st.chat_input():
            # Save and display user message
            st.chat_message("human").write(user_prompt)
            streamlit_msg_history.add_user_message(user_prompt)
            
            # Process response with AI
            with st.chat_message("ai"):
                # Set up handlers for streaming and post-processing
                stream_handler = StreamHandler(st.empty())
                sources_container = st.write("")
                pm_handler = PostMessageHandler(sources_container)
                
                # Configure LLM call
                config = {
                    "callbacks": [stream_handler, pm_handler],
                    "configurable": {"session_id": st.session_state.session_id}
                }
                
                # Log the query being sent
                logger.info(f"Sending query to LLM: {user_prompt[:50]}...")
                
                # Invoke the chain with the user prompt
                response = conversation_chain.invoke(
                    {"question": user_prompt}, 
                    config
                )
                
                # Store AI response in chat history
                if hasattr(response, "content"):
                    streamlit_msg_history.add_ai_message(response.content)
                    
    except Exception as e:
        logger.error(f"Application error: {str(e)}", exc_info=True)
        st.error(f"An error occurred: {str(e)}")
        st.info("Please try refreshing the page or uploading different documents.")
        
    # Display sidebar info
    st.sidebar.markdown("---")
    st.sidebar.info(
        "This AI Assistant uses OpenAI to analyze your documents. "
        "Your questions are answered based on the content of the PDFs you upload."
    )
