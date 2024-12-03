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
            --text: #FFFFFF;
            --error: #FF4444;
        }
        
        /* Simple Voice Recorder Button */
        .voice-button {
            position: fixed;
            bottom: 3rem;
            right: 2rem;
            width: 60px;
            height: 60px;
            border-radius: 50%;
            background: var(--primary);
            border: 3px solid var(--gold);
            color: var(--gold);
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
            transition: all 0.3s ease;
            z-index: 1000;
        }

        .voice-button:hover {
            transform: scale(1.1);
            background: var(--gold);
            color: var(--primary);
        }

        .voice-button.recording {
            background: var(--error);
            border-color: var(--error);
            animation: pulse 1s infinite;
        }

        /* Pulse animation for the recording state */
        @keyframes pulse {
            0% {
                transform: scale(1);
            }
            50% {
                transform: scale(1.2);
            }
            100% {
                transform: scale(1);
            }
        }

        /* Style for the microphone icon */
        .voice-button i {
            font-size: 2rem;
        }
        
        /* Optional: Styling for a small text hint when idle */
        .voice-button span {
            display: none;
        }

        .voice-button.recording span {
            display: block;
            font-size: 0.8rem;
            color: var(--text);
            text-align: center;
        }
        
        /* Simple style for the container */
        .input-area {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background: rgba(20, 23, 26, 0.95);
            padding: 1rem 2rem;
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
        }
        
        .stTextInput input:focus {
            border-color: var(--gold);
            box-shadow: 0 0 15px rgba(212, 175, 55, 0.3);
        }
        
        </style>
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
        
