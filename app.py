import streamlit as st
from streamlit_webrtc import webrtc_streamer
import numpy as np
import av
from typing import List
from gtts import gTTS
import tempfile
import os
from langchain_groq import ChatGroq
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool

# Luxury styling
LUXURY_STYLE = """
<style>
    .main { background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%); }
    
    .luxury-container {
        background: linear-gradient(145deg, rgba(26, 26, 26, 0.9) 0%, rgba(45, 45, 45, 0.9) 100%);
        border: 1px solid #B8860B;
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(184, 134, 11, 0.1);
    }
    
    .luxury-header {
        background: linear-gradient(to right, #B8860B, #FFD700);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Cinzel', serif;
        font-size: 3rem;
        text-align: center;
        margin-bottom: 2rem;
        padding: 2rem 0;
    }
    
    .luxury-subheader {
        color: #DAA520;
        font-family: 'Cormorant Garamond', serif;
        font-size: 1.5rem;
        text-align: center;
        font-style: italic;
        margin-bottom: 2rem;
    }
    
    .stButton button {
        background: linear-gradient(145deg, #1a1a1a, #2d2d2d);
        color: #DAA520;
        border: 2px solid #B8860B;
        border-radius: 50px;
        padding: 1rem 2rem;
        font-family: 'Cormorant Garamond', serif;
        font-size: 1.2rem;
        transition: all 0.3s ease;
    }
    
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
        font-family: 'Cormorant Garamond', serif;
        color: #DAA520;
    }
    
    .status-indicator {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 10px;
        background: #DAA520;
        animation: pulse 1.5s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.2); opacity: 0.7; }
        100% { transform: scale(1); opacity: 1; }
    }
</style>
<link href="https://fonts.googleapis.com/css2?family=Cinzel:wght@400;700&family=Cormorant+Garamond:ital,wght@0,400;0,600;1,400&display=swap" rel="stylesheet">
"""

class AudioProcessor:
    def __init__(self):
        self.chunks = []
        self.recording = False

    def process_audio(self, frame):
        if self.recording:
            sound = frame.to_ndarray()
            self.chunks.append(sound)
        return frame

def create_audio_processor():
    return AudioProcessor()

def main():
    st.set_page_config(
        page_title="LeChateau Voice Concierge",
        page_icon="👑",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Inject custom styling
    st.markdown(LUXURY_STYLE, unsafe_allow_html=True)
    
    # Header
    st.markdown('<div class="luxury-header">LeChateau Voice Concierge</div>', unsafe_allow_html=True)
    st.markdown('<div class="luxury-subheader">Your Personal Reservation Assistant</div>', unsafe_allow_html=True)

    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Main content columns
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown('<div class="luxury-container">', unsafe_allow_html=True)
        
        # Audio recording interface
        webrtc_ctx = webrtc_streamer(
            key="voice-recorder",
            media_stream_constraints={
                "audio": True,
                "video": False
            },
            audio_processor_factory=create_audio_processor,
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            },
            translations={
                "start": "🎙️ Start Recording",
                "stop": "⏹️ Stop Recording"
            }
        )

        # Recording status indicator
        if webrtc_ctx.state.playing:
            st.markdown(
                '<div style="text-align: center; color: #DAA520;">'
                '<div class="status-indicator"></div>'
                'Recording in progress...</div>',
                unsafe_allow_html=True
            )
            if hasattr(webrtc_ctx, 'audio_processor'):
                webrtc_ctx.audio_processor.recording = True

        # Process recording button
        if st.button("✨ Process Recording", key="process"):
            if hasattr(webrtc_ctx, 'audio_processor') and webrtc_ctx.audio_processor.chunks:
                with st.spinner("Processing audio..."):
                    audio_data = np.concatenate(webrtc_ctx.audio_processor.chunks)
                    st.audio(audio_data, format="audio/wav")
                    webrtc_ctx.audio_processor.chunks = []
                st.success("Audio processed successfully!")
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Chat history column
    with col2:
        st.markdown('<div class="luxury-container">', unsafe_allow_html=True)
        st.markdown(
            '<h3 style="color: #DAA520; font-family: \'Cinzel\', serif; text-align: center;">'
            '💬 Conversation History</h3>', 
            unsafe_allow_html=True
        )
        
        if not st.session_state.messages:
            st.markdown(
                '<div style="text-align: center; color: #B8860B; font-style: italic;">'
                'Your conversation will appear here...</div>',
                unsafe_allow_html=True
            )
        
        for msg in st.session_state.messages:
            st.markdown(
                f'<div class="chat-message">{msg}</div>',
                unsafe_allow_html=True
            )
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Footer
    st.markdown(
        '<div style="text-align: center; padding: 2rem; color: #B8860B; '
        'font-family: \'Cormorant Garamond\', serif;">'
        'LeChateau - Where Elegance Meets Innovation<br>'
        '<span style="font-size: 0.8rem;">Powered by Advanced AI Technology</span>'
        '</div>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
