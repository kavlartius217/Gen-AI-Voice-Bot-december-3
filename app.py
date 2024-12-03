import streamlit as st
from audio_recorder_streamlit import audio_recorder
from gtts import gTTS
import tempfile
import os
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from langchain.tools import Tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_openai_tools_agent, AgentExecutor
import google.generativeai as genai

# Page configuration
st.set_page_config(
    page_title="LeChateau AI Concierge",
    page_icon="👑",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for regal styling
st.markdown("""
    <style>
    .main {
        background-color: #f8f5f0;
    }
    .stApp {
        background: linear-gradient(135deg, #f8f5f0 0%, #e8e1d5 100%);
    }
    .css-1d391kg {
        background-color: #2c1810;
    }
    .stButton button {
        background-color: #8b4513;
        color: #f8f5f0;
        border: 2px solid #654321;
        border-radius: 5px;
        padding: 0.5rem 2rem;
        font-family: 'Cinzel', serif;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        background-color: #654321;
        border-color: #8b4513;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 15px rgba(0,0,0,0.1);
        font-family: 'Lora', serif;
    }
    .user-message {
        background-color: #f0e6d6;
        border-left: 5px solid #8b4513;
    }
    .bot-message {
        background-color: #fff;
        border-right: 5px solid #2c1810;
    }
    h1 {
        font-family: 'Cinzel', serif;
        color: #2c1810;
        text-align: center;
        padding: 2rem;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .subheader {
        font-family: 'Lora', serif;
        color: #5c4033;
        text-align: center;
        margin-bottom: 2rem;
        font-style: italic;
    }
    .status-message {
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
        font-family: 'Lora', serif;
    }
    .success {
        background-color: #e6efe6;
        color: #2c5530;
        border-left: 5px solid #2c5530;
    }
    .error {
        background-color: #f8d7da;
        color: #721c24;
        border-left: 5px solid #721c24;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize API keys and configurations
if 'initialized' not in st.session_state:
    st.session_state.initialized = False

def initialize_apis():
    try:
        # Initialize Gemini
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        
        # Initialize ChatGroq
        st.session_state.llm = ChatGroq(
            model_name="gemma2-9b-it",
            api_key=st.secrets["GROQ_API_KEY"]
        )
        
        # Initialize embeddings
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=st.secrets["GEMINI_API_KEY"],
            temperature=0.7
        )
        
        st.session_state.initialized = True
        return True
    except Exception as e:
        st.error(f"Initialization error: {str(e)}")
        return False

# Audio handling functions
def transcribe_audio(audio_file):
    """Transcribe audio using Gemini"""
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        result = model.generate_content([audio_file, "transcribe the audio as it is"])
        return result.text
    except Exception as e:
        st.error(f"Transcription error: {str(e)}")
        return None

def text_to_speech(text):
    """Convert text to speech using gTTS"""
    try:
        tts = gTTS(text, lang='en')
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
            tts.save(temp_audio.name)
            return temp_audio.name
    except Exception as e:
        st.error(f"Text-to-speech error: {str(e)}")
        return None

# Initialize tools and agent
def initialize_agent():
    try:
        # Load CSV data
        csv = CSVLoader("table_data.csv")
        data = csv.load()
        
        # Split documents
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = splitter.split_documents(data)
        
        # Create vector store
        db = FAISS.from_documents(docs, st.session_state.embeddings)
        retriever = db.as_retriever()
        
        # Create tools
        reservation_tool = create_retriever_tool(
            retriever,
            "reservation_data_tool",
            "has the table data for the purpose of making reservations"
        )
        
        greeting_tool = Tool.from_function(
            func=lambda x: "Hello Welcome to LeChateau! How can I help you today?" 
                         if x.lower() in ["hello", "hey", "hi"] else None,
            name="say_hello_tool",
            description="use this tool to greet the customer"
        )
        
        # Create prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a sophisticated AI concierge at LeChateau. Always maintain a formal, elegant tone."),
            ("system", "For table inquiries: present available options with detailed ambiance descriptions"),
            ("system", "For greetings: respond warmly but professionally"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        # Create agent
        agent = create_openai_tools_agent(
            st.session_state.llm,
            [reservation_tool, greeting_tool],
            prompt
        )
        
        return AgentExecutor(
            agent=agent,
            tools=[reservation_tool, greeting_tool],
            llm=st.session_state.llm,
            handle_parsing_errors=True,
            verbose=True,
            max_iterations=15
        )
    except Exception as e:
        st.error(f"Agent initialization error: {str(e)}")
        return None

# Main UI
st.markdown("<h1>👑 LeChateau AI Concierge</h1>", unsafe_allow_html=True)
st.markdown("<p class='subheader'>Your personal reservation assistant</p>", unsafe_allow_html=True)

# Initialize if needed
if not st.session_state.initialized:
    if initialize_apis():
        st.session_state.agent_executor = initialize_agent()

# Create two columns for layout
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 🎙️ Voice Interface")
    st.markdown("Speak your request into the microphone below")
    
    audio_bytes = audio_recorder(
        pause_threshold=2.0,
        sample_rate=44100
    )

    if audio_bytes:
        st.markdown("<div class='status-message success'>🎯 Audio recorded successfully!</div>", 
                   unsafe_allow_html=True)
        st.audio(audio_bytes, format="audio/wav")
        
        with st.spinner("📝 Transcribing your request..."):
            text = transcribe_audio(audio_bytes)
            if text:
                st.markdown(f"<div class='chat-message user-message'>🗣️ You: {text}</div>", 
                          unsafe_allow_html=True)
                
                with st.spinner("🤔 Processing your request..."):
                    response = st.session_state.agent_executor.invoke({
                        "input": text,
                        "chat_history": st.session_state.get('chat_history', [])
                    })
                    
                    # Update chat history
                    if 'chat_history' not in st.session_state:
                        st.session_state.chat_history = []
                    st.session_state.chat_history.append({"user": text, "bot": response['output']})
                    
                    st.markdown(f"<div class='chat-message bot-message'>👑 Concierge: {response['output']}</div>", 
                              unsafe_allow_html=True)
                    
                    # Generate and play audio response
                    with st.spinner("🔊 Generating voice response..."):
                        speech_file = text_to_speech(response['output'])
                        if speech_file:
                            st.audio(speech_file, format="audio/mp3")
                            os.unlink(speech_file)

with col2:
    st.markdown("### 📜 Conversation History")
    if st.session_state.get('chat_history'):
        for chat in st.session_state.chat_history:
            st.markdown(f"<div class='chat-message user-message'>🗣️ Guest: {chat['user']}</div>", 
                       unsafe_allow_html=True)
            st.markdown(f"<div class='chat-message bot-message'>👑 Concierge: {chat['bot']}</div>", 
                       unsafe_allow_html=True)
    else:
        st.markdown("*Your conversation history will appear here*")

# Footer
st.markdown("""
    <div style='text-align: center; color: #5c4033; padding: 2rem; font-family: "Lora", serif;'>
        <p>LeChateau - Where Elegance Meets Innovation</p>
        <p style='font-size: 0.8rem;'>Powered by Advanced AI Technology</p>
    </div>
""", unsafe_allow_html=True)
