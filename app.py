import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import numpy as np
import av
from typing import List
from gtts import gTTS
import tempfile
import os
from langchain_groq import ChatGroq
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool

# Custom CSS for luxury styling
LUXURY_STYLE = """
<style>
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
    }
    
    /* Custom container for luxury content */
    .luxury-container {
        background: linear-gradient(145deg, rgba(26, 26, 26, 0.9) 0%, rgba(45, 45, 45, 0.9) 100%);
        border: 1px solid #B8860B;
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(184, 134, 11, 0.1);
    }
    
    /* Header styling */
    .luxury-header {
        background: linear-gradient(to right, #B8860B, #FFD700);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Cinzel', serif;
        font-size: 3rem;
        text-align: center;
        margin-bottom: 2rem;
        padding: 2rem 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .luxury-subheader {
        color: #DAA520;
        font-family: 'Cormorant Garamond', serif;
        font-size: 1.5rem;
        text-align: center;
        font-style: italic;
        margin-bottom: 2rem;
    }
    
    /* Recording button styling */
    .stButton button {
        background: linear-gradient(145deg, #1a1a1a, #2d2d2d);
        color: #DAA520;
        border: 2px solid #B8860B;
        border-radius: 50px;
        padding: 1rem 2rem;
        font-family: 'Cormorant Garamond', serif;
        font-size: 1.2rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(184, 134, 11, 0.2);
    }
    
    .stButton button:hover {
        background: linear-gradient(145deg, #B8860B, #DAA520);
        color: #1a1a1a;
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(184, 134, 11, 0.3);
    }
    
    /* Chat message styling */
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
        font-family: 'Cormorant Garamond', serif;
    }
    
    .user-message {
        background: linear-gradient(145deg, rgba(184, 134, 11, 0.1), rgba(218, 165, 32, 0.1));
        border-left: 3px solid #B8860B;
    }
    
    .assistant-message {
        background: linear-gradient(145deg, rgba(45, 45, 45, 0.9), rgba(26, 26, 26, 0.9));
        border-right: 3px solid #DAA520;
    }
    
    /* Audio player styling */
    .stAudio {
        background: linear-gradient(145deg, #1a1a1a, #2d2d2d);
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        border: 1px solid #B8860B;
    }
    
    /* Status indicator */
    .status-indicator {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 10px;
    }
    
    .status-active {
        background: #DAA520;
        box-shadow: 0 0 10px #B8860B;
        animation: pulse 1.5s infinite;
    }
    
    @keyframes pulse {
        0% {
            transform: scale(1);
            opacity: 1;
        }
        50% {
            transform: scale(1.2);
            opacity: 0.7;
        }
        100% {
            transform: scale(1);
            opacity: 1;
        }
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1a1a1a;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #B8860B;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #DAA520;
    }
</style>

<!-- Import Google Fonts -->
<link href="https://fonts.googleapis.com/css2?family=Cinzel:wght@400;700&family=Cormorant+Garamond:ital,wght@0,400;0,600;1,400&display=swap" rel="stylesheet">
"""

class AudioProcessor:
    def __init__(self):
        self.audio_chunks: List[np.ndarray] = []
        self.recording = False
        self.sample_rate = 16000

    def process_audio(self, frame: av.AudioFrame) -> av.AudioFrame:
        if self.recording:
            audio_array = frame.to_ndarray()
            self.audio_chunks.append(audio_array)
        return frame

def text_to_speech(text: str) -> str:
    try:
        tts = gTTS(text, lang='en')
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
            tts.save(temp_file.name)
            return temp_file.name
    except Exception as e:
        st.error(f"Text-to-speech error: {str(e)}")
        return None

def main():
    st.set_page_config(
        page_title="LeChateau Concierge",
        page_icon="👑",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Inject custom CSS
    st.markdown(LUXURY_STYLE, unsafe_allow_html=True)
    
    # Header
    st.markdown('<div class="luxury-header">LeChateau Concierge</div>', unsafe_allow_html=True)
    st.markdown('<div class="luxury-subheader">Your Personal Reservation Assistant</div>', unsafe_allow_html=True)

    # Initialize session state
    if 'audio_processor' not in st.session_state:
        st.session_state.audio_processor = AudioProcessor()
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Create main columns
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown('<div class="luxury-container">', unsafe_allow_html=True)
        
        # Minimal recording interface
        recording_col1, recording_col2 = st.columns([3, 1])
        
        with recording_col1:
            webrtc_ctx = webrtc_streamer(
                key="voice-recorder",
                mode=WebRtcMode.AUDIO_ONLY,
                audio_processor_factory=lambda: st.session_state.audio_processor,
                rtc_configuration={
                    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
                },
                media_stream_constraints={"audio": True},
            )
        
        with recording_col2:
            if webrtc_ctx.state.playing:
                st.markdown(
                    '<div style="text-align: center; color: #DAA520;">'
                    '<div class="status-indicator status-active"></div>'
                    'Recording...</div>',
                    unsafe_allow_html=True
                )
                st.session_state.audio_processor.recording = True
            else:
                st.session_state.audio_processor.recording = False

        if len(st.session_state.audio_processor.audio_chunks) > 0:
            if st.button("Process Recording", key="process_recording"):
                audio_data = np.concatenate(st.session_state.audio_processor.audio_chunks)
                st.audio(audio_data, format="audio/wav")
                st.session_state.audio_processor.audio_chunks = []
        
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="luxury-container">', unsafe_allow_html=True)
        st.markdown('<h3 style="color: #DAA520; font-family: \'Cinzel\', serif; text-align: center;">Conversation History</h3>', unsafe_allow_html=True)
        
        for message in st.session_state.messages:
            css_class = "user-message" if message['role'] == 'user' else "assistant-message"
            icon = "👤" if message['role'] == 'user' else "👑"
            
            st.markdown(
                f'<div class="chat-message {css_class}">{icon} {message["content"]}</div>',
                unsafe_allow_html=True
            )
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Footer
    st.markdown(
        '<div style="text-align: center; padding: 2rem; color: #B8860B; font-family: \'Cormorant Garamond\', serif;">'
        'LeChateau - Where Luxury Meets Innovation'
        '<br><span style="font-size: 0.8rem;">Powered by Advanced AI Technology</span>'
        '</div>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
