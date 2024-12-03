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

def load_css():
    st.markdown("""
        <style>
        /* Modern Dark Luxury Theme */
        :root {
            --black: #000000;
            --dark-gray: #1a1a1a;
            --gold: #FFD700;
            --rose-gold: #B76E79;
            --silver: #C0C0C0;
            --white: #FFFFFF;
        }
        
        /* Base Styles */
        .stApp {
            background: radial-gradient(circle at center, var(--dark-gray), var(--black));
        }
        
        /* Modern Header */
        .modern-header {
            background: linear-gradient(45deg, rgba(0,0,0,0.9), rgba(26,26,26,0.9));
            border-bottom: 2px solid var(--gold);
            padding: 2rem;
            margin: -6rem -4rem 2rem -4rem;
            backdrop-filter: blur(10px);
            position: relative;
            z-index: 100;
        }
        
        .modern-header h1 {
            font-family: 'Montserrat', sans-serif;
            font-size: 3rem;
            background: linear-gradient(to right, var(--gold), var(--rose-gold));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 0.5rem;
        }
        
        .modern-header p {
            color: var(--silver);
            text-align: center;
            font-size: 1.2rem;
            font-weight: 300;
        }
        
        /* Voice Control */
        .voice-control {
            position: fixed;
            right: 2rem;
            bottom: 6rem;
            z-index: 1000;
            width: 60px;
            height: 60px;
            border-radius: 50%;
            background: var(--dark-gray);
            border: 2px solid var(--gold);
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .voice-control:hover {
            transform: scale(1.1);
            box-shadow: 0 0 20px rgba(255, 215, 0, 0.3);
        }
        
        .voice-control.recording {
            animation: pulse 1.5s infinite;
        }
        
        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(255, 215, 0, 0.4); }
            70% { box-shadow: 0 0 0 20px rgba(255, 215, 0, 0); }
            100% { box-shadow: 0 0 0 0 rgba(255, 215, 0, 0); }
        }
        
        /* Messages */
        .message {
            padding: 1rem 1.5rem;
            margin: 1rem 0;
            border-radius: 10px;
            max-width: 80%;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            animation: slideIn 0.3s ease;
        }
        
        .user-message {
            background: linear-gradient(135deg, #2a2a2a, #1a1a1a);
            border-left: 3px solid var(--gold);
            margin-left: auto;
            color: var(--white);
        }
        
        .assistant-message {
            background: linear-gradient(135deg, #1a1a1a, #000000);
            border-right: 3px solid var(--rose-gold);
            margin-right: auto;
            color: var(--silver);
        }
        
        @keyframes slideIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        /* Input Area */
        .input-area {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background: rgba(0, 0, 0, 0.8);
            backdrop-filter: blur(10px);
            border-top: 1px solid var(--gold);
            padding: 1rem 2rem;
            z-index: 900;
        }
        
        .stTextInput > div > div > input {
            background: rgba(26, 26, 26, 0.8);
            border: 1px solid var(--gold);
            border-radius: 25px;
            color: var(--white);
            padding: 0.8rem 1.5rem;
            font-size: 1rem;
        }
        
        .stTextInput > div > div > input:focus {
            border-color: var(--rose-gold);
            box-shadow: 0 0 10px rgba(183, 110, 121, 0.3);
        }
        
        /* Status Indicators */
        .status {
            position: fixed;
            top: 1rem;
            right: 1rem;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            background: rgba(0, 0, 0, 0.8);
            color: var(--gold);
            font-size: 0.9rem;
            z-index: 1000;
            animation: fadeIn 0.3s ease;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        
        /* Hide WebRTC elements */
        .streamlit-webrtc-container {
            display: none;
        }
        
        /* Custom Scrollbar */
        ::-webkit-scrollbar {
            width: 6px;
        }
        
        ::-webkit-scrollbar-track {
            background: var(--black);
        }
        
        ::-webkit-scrollbar-thumb {
            background: var(--gold);
            border-radius: 3px;
        }
        
        /* Additional padding for fixed elements */
        .main > div {
            padding-bottom: 80px;
        }
        </style>
        
        <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;600&display=swap" rel="stylesheet">
    """, unsafe_allow_html=True)

[Previous initialization code and AudioProcessor class remain the same...]

def get_voice_input() -> Union[str, None]:
    try:
        processor = AudioProcessor()
        status_placeholder = st.empty()
        
        col1, col2, col3 = st.columns([4, 1, 4])
        with col2:
            voice_container = st.container()
            with voice_container:
                st.markdown("""
                    <div class="voice-control">
                        <span style="font-size: 1.5rem">🎤</span>
                    </div>
                """, unsafe_allow_html=True)
                
                if st.button("", key="voice_button", help="Click to start/stop recording"):
                    if not st.session_state.get('recording', False):
                        st.session_state.recording = True
                        status_placeholder.markdown("""
                            <div class="status">Recording... Speak now</div>
                        """, unsafe_allow_html=True)
                    else:
                        st.session_state.recording = False
                        status_placeholder.markdown("""
                            <div class="status">Processing audio...</div>
                        """, unsafe_allow_html=True)
        
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
                status_placeholder.markdown(f"""
                    <div class="status">Transcribed: {text}</div>
                """, unsafe_allow_html=True)
                return text
    
    except Exception as e:
        logger.error(f"Voice input error: {str(e)}")
        status_placeholder.markdown("""
            <div class="status" style="color: #ff4444;">Recording failed</div>
        """, unsafe_allow_html=True)
        return None

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
```
