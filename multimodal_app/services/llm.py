from config.config import model_name, openai_api_key

from operator import itemgetter
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.messages import HumanMessage, SystemMessage
import base64
import io
from PIL import Image

from operator import itemgetter
from utils.utils import format_docs

from services.prompts.prompts import SYS_PROMPT, QA_RAG_PROMPT, MULTIMODAL_PROMPT


class OpenAILLM:
    def __init__(self, task:str="text"):
        if openai_api_key is None:
            raise ValueError("OpenAI API key is not set")
        if model_name is None:
            raise ValueError("Model name is not set")
        self.llm = ChatOpenAI(
            api_key=openai_api_key, model=model_name, temperature=0.1, streaming=True
        )
        self.task = task

    def define_prompt(self, sys_prompt: str = SYS_PROMPT) -> ChatPromptTemplate:
        if self.task == "text":
            prompt = ChatPromptTemplate.from_template(SYS_PROMPT)
        elif self.task == "qa_rag":
            prompt = ChatPromptTemplate.from_template(QA_RAG_PROMPT)
        elif self.task == "multimodal":
            prompt = ChatPromptTemplate.from_template(MULTIMODAL_PROMPT)
        else:
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", sys_prompt),
                    MessagesPlaceholder(variable_name="history"),
                    ("human", "{input}"),
                ]
            )

        return prompt
        
    def encode_image(self, image_file):
        """
        Encodes an image file to base64
        """
        if image_file is None:
            return None
            
        # Convert the file to an image
        image = Image.open(image_file)
        
        # Convert to RGB if image has an alpha channel
        if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3] if image.mode == 'RGBA' else None)
            image = background
            
        # Convert to JPEG format in memory
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        buffer.seek(0)
        
        # Encode as base64
        image_bytes = buffer.getvalue()
        encoded_image = base64.b64encode(image_bytes).decode('utf-8')
        
        return encoded_image

    def base_chain(self, sys_prompt: str = SYS_PROMPT, retriever: any = None):
        if self.task == "qa_rag":
            prompt = self.define_prompt()
            self.chain = (
                {
                    "context": itemgetter("question")
                    | retriever
                    | format_docs,
                    "question": itemgetter("question")
                }
                | prompt
                | self.llm
            )
        elif self.task == "multimodal":
            # For multimodal, we need to create a custom prompt chain that works with history
            def prepare_multimodal_messages(inputs):
                image_file = inputs.get("image")
                question = inputs.get("question")
                history = inputs.get("history", [])
                
                # Encode the image file to base64
                encoded_image = self.encode_image(image_file)
                
                # Create a system message with the multimodal prompt
                messages = [SystemMessage(content=MULTIMODAL_PROMPT)]
                
                # Add history messages if they exist
                if history:
                    messages.extend(history)
                
                # Add the new human message with both text and image
                if encoded_image:
                    human_message = HumanMessage(
                        content=[
                            {"type": "text", "text": question},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
                        ]
                    )
                else:
                    # Fallback to text-only if no image
                    human_message = HumanMessage(content=question)
                
                messages.append(human_message)
                return messages
            
            self.chain = (
                prepare_multimodal_messages
                | self.llm
            )
        else:
            prompt = self.define_prompt(sys_prompt=sys_prompt)
            self.chain = prompt | self.llm

    def chain(
        self,
        sys_prompt: str = SYS_PROMPT,
        st_chat_history: StreamlitChatMessageHistory = None,
        retriever: any = None,
    ):
        self.base_chain(sys_prompt=sys_prompt, retriever=retriever)
        
        # Properly configure RunnableWithMessageHistory with get_session_history function
        def get_session_history(session_id):
            return st_chat_history
        
        # For multimodal tasks, we need to handle the input/history differently
        if self.task == "multimodal":
            conversation_chain = RunnableWithMessageHistory(
                self.chain,
                get_session_history,
                input_messages_key="question",
                history_messages_key="history",
            )
        else:
            conversation_chain = RunnableWithMessageHistory(
                self.chain,
                get_session_history,
                input_messages_key="question",
                history_messages_key="history",
            )

        return conversation_chain
