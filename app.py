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

def load_css():
    st.markdown("""
        <style>
        /* Elegant color scheme */
        :root {
            --bg-color: #ffffff;
            --primary: #8B7355;
            --accent: #C4B6A6;
            --text: #1A1A1A;
            --border: #E5E5E5;
            --highlight: #D4C4B7;
            --error: #DC3545;
            --success: #198754;
        }
        
        .stApp {
            background-color: var(--bg-color);
        }
        
        .header {
            background: linear-gradient(to right, var(--primary), var(--accent));
            padding: 2.5rem;
            border-radius: 0;
            text-align: center;
            color: white;
            margin-bottom: 2rem;
            box-shadow: 0 2px 12px rgba(0,0,0,0.1);
        }
        
        .message {
            padding: 1.2rem 1.8rem;
            border-radius: 4px;
            margin: 0.8rem 0;
            line-height: 1.6;
            border: 1px solid var(--border);
            font-size: 1rem;
        }
        
        .user-message {
            background: #F8F8F8;
            color: var(--text);
            margin-left: 2rem;
        }
        
        .assistant-message {
            background: white;
            color: var(--text);
            margin-right: 2rem;
            border-left: 3px solid var(--primary);
        }
        
        .voice-controls {
            display: flex;
            justify-content: center;
            gap: 1rem;
            margin: 1rem 0;
        }
        
        .stButton > button {
            background: var(--primary);
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 4px;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            background: var(--accent);
            transform: translateY(-1px);
        }
        
        .record-button > button {
            background: var(--error);
        }
        
        .record-button > button:hover {
            background: #bb2d3b;
        }
        
        .input-area {
            background: white;
            padding: 1.5rem;
            border-top: 1px solid var(--border);
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            z-index: 1000;
        }
        
        .stTextInput > div > div > input {
            border: 1px solid var(--border);
            border-radius: 4px;
            padding: 0.8rem 1.2rem;
        }
        
        .stTextInput > div > div > input:focus {
            border-color: var(--primary);
            box-shadow: 0 0 0 1px var(--primary);
        }
        
        .streamlit-webrtc-container {
            display: none;
        }
        
        .stAudio {
            margin-top: 0.5rem;
        }
        
        .success-text {
            color: var(--success);
        }
        
        .error-text {
            color: var(--error);
        }
        
        div[data-testid="stVerticalBlock"] {
            gap: 0rem;
        }
        
        /* Add padding to main container to prevent content from being hidden behind fixed input area */
        .main > div {
            padding-bottom: 100px;
        }
        </style>
    """, unsafe_allow_html=True)

# Initialize Streamlit page config
st.set_page_config(
    page_title="Restaurant Voice Assistant",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def init_api_keys():
    if 'GROQ_API_KEY' not in st.secrets or 'GOOGLE_API_KEY' not in st.secrets:
        st.error("Please set GROQ_API_KEY and GOOGLE_API_KEY in Streamlit secrets")
        st.stop()
    return st.secrets["GROQ_API_KEY"], st.secrets["GOOGLE_API_KEY"]

@st.cache_resource
def initialize_agent():
    groq_key, google_key = init_api_keys()
    
    llm = ChatGroq(
        model_name="gemma2-9b-it",
        api_key=groq_key
    )
    
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=google_key
    )
    
    csv_loader = CSVLoader("table_data (1).csv")
    documents = csv_loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    docs = text_splitter.split_documents(documents)
    
    db = FAISS.from_documents(docs, embeddings)
    retriever = db.as_retriever()
    
    reservation_tool = create_retriever_tool(
        retriever,
        "reservation_data_tool",
        "has the table data for the purpose of making reservations"
    )
    
    def say_hello(customer_input):
        if customer_input.lower() in ["hello", "hey", "hi"]:
            return "Hello! Welcome to LeChateau. How can I help you today?"
    
    greeting_tool = Tool.from_function(
        func=say_hello,
        name="say_hello_tool",
        description="use this tool to greet the customer after the customer has greeted you"
    )
    
    tools = [reservation_tool, greeting_tool]
    
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
        recording_placeholder = st.empty()
        
        cols = st.columns([3, 1, 1])
        with cols[1]:
            start_recording = st.button("▶️ Start Recording", key="start")
        with cols[2]:
            stop_recording = st.button("⏹️ Stop Recording", key="stop")
        
        if start_recording:
            st.session_state.recording = True
            recording_placeholder.warning("🎤 Recording... Click Stop when finished")
            
        if stop_recording:
            st.session_state.recording = False
            recording_placeholder.info("Processing audio...")
        
        if st.session_state.get('recording', False):
            ctx = webrtc_streamer(
                key="voice-input",
                mode=WebRtcMode.SENDONLY,
                audio_receiver_size=1024,
                rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                media_stream_constraints={"video": False, "audio": True},
                async_processing=True,
                video_processor_factory=None,
                audio_processor_factory=lambda: processor,
                desired_playing_state=True
            )

            if ctx.audio_receiver and not processor.transcript_queue.empty():
                text = processor.transcript_queue.get()
                recording_placeholder.success(f"Transcribed: {text}")
                return text
    
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
    load_css()
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'recording' not in st.session_state:
        st.session_state.recording = False
    
    agent_executor = initialize_agent()
    
    st.markdown("""
        <div class="header">
            <h1>🎤 LeChateau Assistant</h1>
            <p>Your personal restaurant concierge</p>
        </div>
    """, unsafe_allow_html=True)
    
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            message_class = "user-message" if message["role"] == "user" else "assistant-message"
            st.markdown(f"""
                <div class="message {message_class}">
                    {message["content"]}
                </div>
            """, unsafe_allow_html=True)
            
            if message["role"] == "assistant":
                audio_bytes = text_to_speech(message["content"])
                if audio_bytes:
                    st.audio(audio_bytes, format='audio/mp3')
    
    st.markdown('<div class="input-area">', unsafe_allow_html=True)
    cols = st.columns([6, 2, 2])
    
    with cols[0]:
        user_input = st.text_input(
            "Message",
            key="user_input",
            label_visibility="collapsed",
            placeholder="Type your message here..."
        )
    
    with cols[1]:
        voice_input = get_voice_input()
        if voice_input:
            st.session_state.user_input = voice_input
            st.rerun()
    
    with cols[2]:
        send_button = st.button("Send 📤", key="send", use_container_width=True)
    
    if send_button and (user_input or st.session_state.get('user_input')):
        final_input = user_input or st.session_state.get('user_input', '')
        st.session_state.messages.append({"role": "user", "content": final_input})
        
        with st.spinner("💭 Thinking..."):
            response = agent_executor.invoke({
                "input": final_input,
                "chat_history": st.session_state.chat_history
            })
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": response['output']
            })
            
            st.session_state.chat_history.extend([final_input, response['output']])
        
        if 'user_input' in st.session_state:
            del st.session_state.user_input
        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    with st.sidebar:
        if st.button("🗑️ Clear Chat", key="clear"):
            st.session_state.messages = []
            st.session_state.chat_history = []
            if 'user_input' in st.session_state:
                del st.session_state.user_input
            st.rerun()

if __name__ == "__main__":
    main()
