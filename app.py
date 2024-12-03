[Full code was too long - continuing with Part 2 of 2]

Here's the first part of the complete code:

```python
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="LeChateau Concierge",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def load_css():
    st.markdown("""
        <style>
        :root {
            --bg-dark: #0C0C0C;
            --primary: #14171A;
            --gold: #D4AF37;
            --light-gold: #F4E6BA;
            --text: #FFFFFF;
            --text-dark: #A0A0A0;
            --error: #FF4444;
            --success: #50C878;
        }
        
        .stApp {
            background: linear-gradient(135deg, var(--bg-dark), var(--primary));
            color: var(--text);
        }
        
        .modern-header {
            background: linear-gradient(90deg, var(--bg-dark), var(--primary));
            padding: 2rem;
            margin: -6rem -4rem 2rem -4rem;
            border-bottom: 2px solid var(--gold);
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.3);
        }
        
        .modern-header h1 {
            font-family: 'Cinzel', serif;
            font-size: 3.5rem;
            background: linear-gradient(45deg, var(--gold), var(--light-gold));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 0.5rem;
        }
        
        .modern-header p {
            color: var(--text-dark);
            text-align: center;
            font-size: 1.2rem;
            font-weight: 300;
            letter-spacing: 2px;
        }
        
        .message {
            padding: 1.2rem 1.8rem;
            margin: 1rem auto;
            border-radius: 12px;
            max-width: 85%;
            backdrop-filter: blur(10px);
            animation: slideIn 0.3s ease;
        }
        
        .user-message {
            background: rgba(20, 23, 26, 0.8);
            border-left: 3px solid var(--gold);
            margin-left: 15%;
            color: var(--text);
        }
        
        .assistant-message {
            background: rgba(12, 12, 12, 0.8);
            border-right: 3px solid var(--light-gold);
            margin-right: 15%;
            color: var(--text);
        }
        
        .voice-button {
            position: fixed;
            right: 2rem;
            bottom: 7rem;
            width: 60px;
            height: 60px;
            border-radius: 50%;
            background: var(--primary);
            border: 2px solid var(--gold);
            color: var(--gold);
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: all 0.3s ease;
            z-index: 1000;
            font-size: 1.5rem;
        }
        
        .voice-button:hover {
            transform: scale(1.1);
            box-shadow: 0 0 20px rgba(212, 175, 55, 0.3);
        }
        
        .voice-button.recording {
            animation: pulse 1.5s infinite;
            background: var(--error);
        }
        
        .input-area {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background: rgba(12, 12, 12, 0.95);
            backdrop-filter: blur(10px);
            padding: 1.5rem 2rem;
            border-top: 1px solid var(--gold);
            z-index: 900;
        }
        
        .stTextInput input {
            background: rgba(20, 23, 26, 0.8);
            border: 1px solid var(--gold);
            color: var(--text);
            border-radius: 25px;
            padding: 1rem 1.5rem;
            font-size: 1rem;
            transition: all 0.3s ease;
        }
        
        .stTextInput input:focus {
            border-color: var(--light-gold);
            box-shadow: 0 0 15px rgba(212, 175, 55, 0.2);
        }
        
        .status-indicator {
            position: fixed;
            top: 1rem;
            right: 1rem;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            background: rgba(12, 12, 12, 0.9);
            color: var(--gold);
            font-size: 0.9rem;
            z-index: 1000;
            animation: fadeIn 0.3s ease;
        }
        
        @keyframes slideIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(255, 68, 68, 0.4); }
            70% { box-shadow: 0 0 0 20px rgba(255, 68, 68, 0); }
            100% { box-shadow: 0 0 0 0 rgba(255, 68, 68, 0); }
        }
        
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        
        .streamlit-webrtc-container {
            display: none;
        }
        
        ::-webkit-scrollbar {
            width: 6px;
        }
        
        ::-webkit-scrollbar-track {
            background: var(--bg-dark);
        }
        
        ::-webkit-scrollbar-thumb {
            background: var(--gold);
            border-radius: 3px;
        }
        
        .main > div {
            padding-bottom: 100px;
        }
        </style>
        
        <link href="https://fonts.googleapis.com/css2?family=Cinzel:wght@400;600&display=swap" rel="stylesheet">
    """, unsafe_allow_html=True)

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
        api_key=groq_key,
        temperature=0.7
    )
    
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=google_key
    )
    
    try:
        csv_loader = CSVLoader("table_data (1).csv")
        documents = csv_loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        docs = text_splitter.split_documents(documents)
        
        db = FAISS.from_documents(docs, embeddings)
        retriever = db.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
        reservation_tool = create_retriever_tool(
            retriever,
            "reservation_data_tool",
            "Searches and retrieves restaurant table availability and details"
        )
        
        def say_hello(customer_input: str) -> str:
            if any(word in customer_input.lower() for word in ["hello", "hey", "hi"]):
                return "Hello! Welcome to LeChateau. How can I assist you with your dining plans today?"
            return None
        
        greeting_tool = Tool.from_function(
            func=say_hello,
            name="say_hello_tool",
            description="Greets the customer with a warm welcome message"
        )
        
        tools = [reservation_tool, greeting_tool]
        
        prompt = PromptTemplate.from_template(
            """You are an AI assistant managing reservations at LeChateau restaurant. When the guest greets you, you shall also greet the guest and use say_hello_tool for this task. When a guest requests a reservation, use the reservation_data tool to check available tables for the specified time and number of people. Present all available tables with their specific locations (e.g., "Table 4 by the window", "Table 7 in the garden area"). After displaying options, let the guest choose their preferred table and confirm their booking immediately.

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
Thought:{agent_scratchpad}"""
        )
        
        agent = create_react_agent(llm, tools, prompt)
        
        return AgentExecutor(
            agent=agent,
            tools=tools,
            llm=llm,
            handle_parsing_errors=True,
            verbose=True,
            max_iterations=15
        )
        
    except Exception as e:
        logger.error(f"Error initializing agent: {str(e)}")
        st.error("Failed to initialize the restaurant assistant. Please try again.")
        st.stop()

class AudioProcessor:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.transcript_queue = queue.Queue()
        self.audio_queue = queue.Queue()
        self.is_processed = threading.Event()
        self.debug = True

    def process_audio(self, frame):
        try:
            audio_data = frame.to_ndarray()
            if self.debug:
                logger.info(f"Audio frame received: shape={audio_data.shape}")
            
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
            frame_count = 0
            while not self.audio_queue.empty():
                frame = self.audio_queue.get()
                audio_data.append(frame)
                frame_count += 1

            if not audio_data:
                if self.debug:
                    logger.warning("No audio data to process")
                return

            combined_audio = np.concatenate(audio_data)
            if self.debug:
                logger.info(f"Combined {frame_count} frames, shape={combined_audio.shape}")

            audio_segment = pydub.AudioSegment(
                combined_audio.tobytes(), 
                frame_rate=48000,
                sample_width=2,
                channels=1
            )

            audio_wav = io.BytesIO()
            audio_segment.export(audio_wav, format="wav")
            audio_wav.seek(0)
            
            with sr.AudioFile(audio_wav) as source:
                audio = self.recognizer.record(source)
                if self.debug:
                    logger.info("Audio captured and ready for transcription")
                
                text = self.recognizer.recognize_google(audio)
                if text:
                    if self.debug:
                        logger.info(f"Successfully transcribed: {text}")
                    self.transcript_queue.put(text)
                    self.is_processed.set()
                    
        except sr.UnknownValueError:
            logger.warning("Speech not recognized")
        except Exception as e:
            logger.error(f"Error processing audio: {str(e)}")

def get_voice_input() -> Union[str, None]:
    try:
        processor = AudioProcessor()
        status_placeholder = st.empty()
        
        if st.button("🎙️", key="voice_toggle", help="Click to start/stop recording", 
                    use_container_width=True):
            if not st.session_state.get('recording', False):
                st.session_state.recording = True
                status_placeholder.markdown(
                    """<div class="status-indicator">Recording... Speak now</div>""",
                    unsafe_allow_html=True
                )
            else:
                st.session_state.recording = False
                status_placeholder.markdown(
                    """<div class="status-indicator">Processing audio...</div>""",
                    unsafe_allow_html=True
                )
        
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
                status_placeholder.markdown(
                    f"""<div class="status-indicator">Transcribed: {text}</div>""",
                    unsafe_allow_html=True
                )
                return text
    
    except Exception as e:
        logger.error(f"Voice input error: {str(e)}")
        status_placeholder.markdown(
            """<div class="status-indicator" style="color: var(--error);">
                Recording failed
            </div>""",
            unsafe_allow_html=True
        )
        return None

    return None

def text_to_speech(text: str) -> Union[bytes, None]:
    try:
        tts = gTTS(text=text, lang='en')
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        return fp.getvalue()
    except Exception as e:
        logger.error(f"Text to speech error: {str(e)}")
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
        <div class="modern-header">
            <h1>LeChateau</h1>
            <p>Luxury Dining Concierge</p>
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
    cols = st.columns([8, 2])
    
    with cols[0]:
        user_input = st.text_input(
            "Message",
            key="user_input",
            label_visibility="collapsed",
            placeholder="How may I assist you today?"
        )
    
    with cols[1]:
        st.markdown("""
            <div class="voice-button">
                🎙️
            </div>
        """, unsafe_allow_html=True)
        voice_input = get_voice_input()
        if voice_input:
            st.session_state.user_input = voice_input
            st.rerun()
    
    if user_input or st.session_state.get('user_input'):
        final_input = user_input or st.session_state.get('user_input', '')
        st.session_state.messages.append({"role": "user", "content": final_input})
        
        with st.spinner(""):
            try:
                response = agent_executor.invoke({
                    "input": final_input,
                    "chat_history": st.session_state.chat_history
                })
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response['output']
                })
                
                st.session_state.chat_history.extend([final_input, response['output']])
                
            except Exception as e:
                logger.error(f"Error processing request: {str(e)}")
                st.error("I apologize, but I couldn't process your request. Please try again.")
        
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
        
