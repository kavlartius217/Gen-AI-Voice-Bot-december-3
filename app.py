import streamlit as st
import speech_recognition as sr
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import queue
import threading
import numpy as np
import av
import pydub
from typing import Union
import logging
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from langchain.tools import Tool
from langchain.prompts import PromptTemplate
from langchain.agents import create_react_agent, AgentExecutor
from gtts import gTTS
import os
import io
import base64

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Streamlit page config
st.set_page_config(
    page_title="Restaurant Voice Assistant",
    page_icon="🎤",
    layout="wide"
)

# Initialize API keys from Streamlit secrets
def init_api_keys():
    if 'GROQ_API_KEY' not in st.secrets or 'GOOGLE_API_KEY' not in st.secrets:
        st.error("Please set GROQ_API_KEY and GOOGLE_API_KEY in Streamlit secrets")
        st.stop()
    return st.secrets["GROQ_API_KEY"], st.secrets["GOOGLE_API_KEY"]

# Initialize LLM and tools
@st.cache_resource
def initialize_agent():
    groq_key, google_key = init_api_keys()
    
    # Initialize LLM
    llm = ChatGroq(
        model_name="gemma2-9b-it",
        api_key=groq_key
    )
    
    # Initialize embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=google_key
    )
    
    # Load and process table data
    csv_loader = CSVLoader("table_data.csv")
    documents = csv_loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    docs = text_splitter.split_documents(documents)
    
    # Create vector store and retriever
    db = FAISS.from_documents(docs, embeddings)
    retriever = db.as_retriever()
    
    # Create tools
    reservation_tool = create_retriever_tool(
        retriever,
        "reservation_data_tool",
        "has the table data for the purpose of making reservations"
    )
    
    def say_hello(customer_input):
        if customer_input.lower() in ["hello", "hey"]:
            return "Hello! Welcome to LeChateau. How can I help you today?"
    
    greeting_tool = Tool.from_function(
        func=say_hello,
        name="say_hello_tool",
        description="use this tool to greet the customer after the customer has greeted you"
    )
    
    tools = [reservation_tool, greeting_tool]
    
    # Create prompt
    prompt = PromptTemplate.from_template("""You are an AI assistant managing reservations at LeChateau restaurant. When the guest greets you, you shall also greet the guest and use say_hello_tool for this task. When a guest requests a reservation, use the reservation_data tool to check available tables for the specified time and number of people. Present all available tables with their specific locations (e.g., "Table 4 by the window", "Table 7 in the garden area"). After displaying options, let the guest choose their preferred table and confirm their booking immediately.

{tools}

Follow this one-step-at-a-time format:
Question: {input}
Thought: [ONE simple thought about what to do next]
Action: [ONE tool from {tool_names}]
Action Input: [Just the input value without variable names or equals signs]
Observation: [Tool's response]
Thought: [ONE simple thought about the observation]
Final Answer: [Response to guest]

Question: {input}
Thought:{agent_scratchpad}""")
    
    # Create agent
    agent = create_react_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        llm=llm,
        handle_parsing_errors=True,
        verbose=True,
        max_iterations=15
    )

class AudioProcessor:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.transcript_queue = queue.Queue()
        self.audio_queue = queue.Queue()
        self.is_processed = threading.Event()

    def process_audio(self, frame):
        try:
            audio_data = frame.to_ndarray()
            self.audio_queue.put(audio_data)
            
            if not self.is_processed.is_set():
                self.process_accumulated_audio()
            
            return frame
        except Exception as e:
            logger.error(f"Error processing audio frame: {str(e)}")
            return frame

    def process_accumulated_audio(self):
        try:
            audio_data = []
            while not self.audio_queue.empty():
                audio_data.append(self.audio_queue.get())
            
            if not audio_data:
                return

            combined_audio = np.concatenate(audio_data)
            audio_segment = pydub.AudioSegment(
                combined_audio.tobytes(), 
                frame_rate=48000,
                sample_width=2,
                channels=1
            )

            audio_wav = audio_segment.export(format="wav").read()
            
            with sr.AudioFile(audio_wav) as source:
                audio = self.recognizer.record(source)
                text = self.recognizer.recognize_google(audio)
                if text:
                    self.transcript_queue.put(text)
                    self.is_processed.set()
        
        except sr.UnknownValueError:
            logger.info("Speech not recognized")
        except Exception as e:
            logger.error(f"Error processing accumulated audio: {str(e)}")

def get_voice_input() -> Union[str, None]:
    try:
        processor = AudioProcessor()
        
        ctx = webrtc_streamer(
            key="voice-input",
            mode=WebRtcMode.SENDONLY,
            audio_receiver_size=1024,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": False, "audio": True},
            async_processing=True,
            video_processor_factory=None,
            audio_processor_factory=lambda: processor
        )

        if ctx.audio_receiver:
            if not processor.transcript_queue.empty():
                return processor.transcript_queue.get()
    
    except Exception as e:
        st.error(f"Error getting voice input: {str(e)}")
        return None

    return None

def text_to_speech(text):
    try:
        tts = gTTS(text=text, lang='en')
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        return fp.getvalue()
    except Exception as e:
        logger.error(f"Error in text to speech conversion: {str(e)}")
        return None

def main():
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Initialize agent
    agent_executor = initialize_agent()
    
    st.title("🎤 Restaurant Voice Assistant")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message["role"] == "assistant":
                # Convert response to speech and create audio player
                audio_bytes = text_to_speech(message["content"])
                if audio_bytes:
                    st.audio(audio_bytes, format='audio/mp3')
    
    # Chat input form
    with st.form(key="chat_form", clear_on_submit=True):
        cols = st.columns([3, 1, 1])
        
        with cols[0]:
            user_input = st.text_input("Message", key="user_input", label_visibility="collapsed")
        with cols[1]:
            voice_button = st.form_submit_button("🎤 Voice", use_container_width=True)
        with cols[2]:
            submit_button = st.form_submit_button("Send", use_container_width=True)

        if voice_button:
            with st.spinner("Listening..."):
                voice_text = get_voice_input()
                if voice_text:
                    st.session_state.user_input = voice_text
                    st.rerun()

        if submit_button and (user_input or st.session_state.get('user_input')):
            final_input = user_input or st.session_state.get('user_input', '')
            
            # Add user message
            st.session_state.messages.append({"role": "user", "content": final_input})
            
            # Get response from agent
            with st.spinner("Processing..."):
                response = agent_executor.invoke({
                    "input": final_input,
                    "chat_history": st.session_state.chat_history
                })
                
                # Add assistant response
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response['output']
                })
                
                # Update chat history
                st.session_state.chat_history.extend([
                    final_input,
                    response['output']
                ])
            
            if 'user_input' in st.session_state:
                del st.session_state.user_input
            
            st.rerun()

    # Clear chat button
    if st.sidebar.button("Clear Chat"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        if 'user_input' in st.session_state:
            del st.session_state.user_input
        st.rerun()

if __name__ == "__main__":
    main()