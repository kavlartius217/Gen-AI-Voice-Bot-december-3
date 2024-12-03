```python
# Section 1: Imports
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

# Section 2: Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Section 3: Streamlit Configuration
st.set_page_config(
    page_title="Restaurant Voice Assistant",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Section 4: CSS and UI
[Previous CSS code remains exactly the same]

# Section 5: API Key Management
def init_api_keys():
    if 'GROQ_API_KEY' not in st.secrets:
        st.error("Missing GROQ_API_KEY in Streamlit secrets")
        st.stop()
    if 'GOOGLE_API_KEY' not in st.secrets:
        st.error("Missing GOOGLE_API_KEY in Streamlit secrets")
        st.stop()
    
    logger.info("API keys initialized successfully")
    return st.secrets["GROQ_API_KEY"], st.secrets["GOOGLE_API_KEY"]

# Section 6: LangChain Setup
@st.cache_resource
def initialize_agent():
    try:
        groq_key, google_key = init_api_keys()
        
        # Initialize LLM
        llm = ChatGroq(
            model_name="gemma2-9b-it",
            api_key=groq_key,
            temperature=0.7
        )
        
        # Initialize embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=google_key
        )
        
        # Load and process reservation data
        try:
            csv_loader = CSVLoader("table_data (1).csv")
            documents = csv_loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50
            )
            docs = text_splitter.split_documents(documents)
            
            # Create vector store
            db = FAISS.from_documents(docs, embeddings)
            retriever = db.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )
            
            # Create tools
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
            
            # Create prompt template
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
            
            logger.info("Agent initialized successfully")
            return AgentExecutor(
                agent=agent,
                tools=tools,
                llm=llm,
                handle_parsing_errors=True,
                verbose=True,
                max_iterations=15
            )
            
        except FileNotFoundError:
            logger.error("table_data (1).csv file not found")
            st.error("Restaurant data file not found. Please check the data file.")
            st.stop()
            
    except Exception as e:
        logger.error(f"Error initializing agent: {str(e)}")
        st.error("Failed to initialize the restaurant assistant. Please try again.")
        st.stop()

# Section 7: Audio Processing
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
                logger.info(f"Audio frame received: shape={audio_data.shape}, max={np.max(audio_data)}")
            
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

def verify_audio_input(audio_data) -> bool:
    """Verify if audio input is being properly processed"""
    try:
        if audio_data is None:
            logger.warning("No audio data received")
            return False
        
        if not isinstance(audio_data, np.ndarray):
            logger.warning(f"Invalid audio format: {type(audio_data)}")
            return False
        
        if audio_data.size == 0:
            logger.warning("Empty audio data")
            return False
        
        if np.max(np.abs(audio_data)) < 0.01:
            logger.warning("Audio signal too weak")
            return False
        
        logger.info("Audio verification passed")
        return True
    except Exception as e:
        logger.error(f"Audio verification failed: {str(e)}")
        return False

def get_voice_input() -> Union[str, None]:
    try:
        processor = AudioProcessor()
        status_placeholder = st.empty()
        verification_placeholder = st.empty()
        
        cols = st.columns([4, 1, 1, 4])
        with cols[1]:
            if st.button("⏺️", key="start", help="Start Recording"):
                st.session_state.recording = True
                status_placeholder.warning("Recording in progress...")
                logger.info("Recording started")
        
        with cols[2]:
            if st.button("⏹️", key="stop", help="Stop Recording"):
                st.session_state.recording = False
                status_placeholder.info("Processing audio...")
                logger.info("Recording stopped")
        
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

            if ctx.audio_receiver:
                logger.info("Audio receiver initialized")
                if not processor.transcript_queue.empty():
                    text = processor.transcript_queue.get()
                    verification_placeholder.success("✓ Audio successfully processed")
                    status_placeholder.success(f"Transcribed: {text}")
                    return text
                else:
                    verification_placeholder.info("Waiting for speech...")
    
    except Exception as e:
        logger.error(f"Error in voice input: {str(e)}")
        st.error("Error processing voice input. Please try again.")
        return None

    return None

# Section 8: Text to Speech
def text_to_speech(text: str) -> Union[bytes, None]:
    try:
        tts = gTTS(text=text, lang='en')
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        return fp.getvalue()
    except Exception as e:
        logger.error(f"Error in text to speech conversion: {str(e)}")
        return None

# Section 9: Main Application
def main():
    load_css()
    
    # Initialize session states
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'recording' not in st.session_state:
        st.session_state.recording = False
    
    # Initialize agent
    agent_executor = initialize_agent()
    
    # Luxury header
    st.markdown("""
        <div class="luxury-header">
            <h1>LeChateau Concierge</h1>
            <p>Exquisite Dining, Personalized Service</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Chat display
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            message_class = "user-message" if message["role"] == "user" else "assistant-message"
            st.markdown(f"""
                <div class="message-container {message_class}">
                    {message["content"]}
                </div>
            """, unsafe_allow_html=True)
            
            if message["role"] == "assistant":
                audio_bytes = text_to_speech(message["content"])
                if audio_bytes:
                    st.audio(audio_bytes, format='audio/mp3')
    
    # Input area
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    cols = st.columns([6, 2, 2])
    
    with cols[0]:
        user_input = st.text_input(
            "Message",
            key="user_input",
            label_visibility="collapsed",
            placeholder="How may I assist you today?"
        )
    
    with cols[1]:
        voice_input = get_voice_input()
        if voice_input:
            st.session_state.user_input = voice_input
            st.rerun()
    
    with cols[2]:
        send_button = st.button("Send 📤", key="send", use_container_width=True)
    
    # Handle input
    if send_button and (user_input or st.session_state.get('user_input')):
        final_input = user_input or st.session_state.get('user_input', '')
        st.session_state.messages.append({"role": "user", "content": final_input})
        
        with st.spinner("Processing your request..."):
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
    
    # Clear chat button
    with st.sidebar:
        if st.button("🗑️ Clear Chat", key="clear"):
            st.session_state.messages = []
            st.session_state.chat_history = []
            if 'user_input' in st.session_state:
                del st.session_state.user_input
            st.rerun()

if __name__ == "__main__":
    main()
```
