import streamlit as st

# Page config must be first
st.set_page_config(
    page_title="LeChateau Reservation Bot",
    page_icon="🍽️",
    layout="wide"
)

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
from gtts import gTTS
import tempfile
import os

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
    }
    
    h1, h2, h3 {
        color: #FFD700 !important;
        font-family: 'Playfair Display', serif;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .subheader {
        color: #DAA520 !important;
        font-family: 'Cormorant Garamond', serif;
        font-style: italic;
    }
    
    .stTextArea {
        background: linear-gradient(145deg, #2d2d2d 0%, #1a1a1a 100%);
        border: 1px solid #FFD700;
        border-radius: 10px;
        color: #FFD700 !important;
    }
    
    .st-emotion-cache-16idsys {
        background: linear-gradient(145deg, #1a1a1a, #2d2d2d);
        border: 1px solid #DAA520;
        color: #FFD700;
    }
    
    .st-emotion-cache-16r3i8g {
        background: linear-gradient(145deg, #2d2d2d, #1a1a1a);
        border: 1px solid #FFD700;
        color: #DAA520;
    }
    
    .stButton button {
        background: linear-gradient(145deg, #FFD700, #DAA520);
        color: #1a1a1a;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        background: linear-gradient(145deg, #DAA520, #FFD700);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(218,165,32,0.3);
    }
    
    .stAudio {
        background: linear-gradient(145deg, #2d2d2d, #1a1a1a);
        border: 1px solid #DAA520;
        border-radius: 10px;
        padding: 10px;
    }
    
    hr {
        border: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, #FFD700, transparent);
        margin: 20px 0;
    }
    
    .chat-container {
        background: linear-gradient(145deg, #1a1a1a, #2d2d2d);
        border-radius: 15px;
        padding: 20px;
        border: 1px solid #DAA520;
        margin: 10px 0;
    }
    
    .stSpinner {
        border-color: #FFD700;
    }
    
    ::-webkit-scrollbar {
        width: 10px;
        background: #1a1a1a;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(145deg, #FFD700, #DAA520);
        border-radius: 5px;
    }
    
    .stTextInput input {
        background: linear-gradient(145deg, #2d2d2d, #1a1a1a);
        border: 1px solid #DAA520;
        color: #FFD700;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize API keys from Streamlit secrets
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

# Configure Google Gemini
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def text_to_speech(text):
    """Convert text to speech and save as temporary file"""
    try:
        tts = gTTS(text, lang='en')
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
            tts.save(temp_audio.name)
            return temp_audio.name
    except Exception as e:
        st.error(f"Text-to-speech error: {str(e)}")
        return None

def initialize_llm_tools():
    """Initialize LLM and tools"""
    # Initialize Groq LLM (faster than OpenAI)
    llm = ChatGroq(
        model_name="gemma2-9b-it",
        api_key=GROQ_API_KEY,
        temperature=0.7
    )
    
    # Initialize embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )
    
    # Load and process CSV data
    csv_path = "table_data(1).csv"  # Update with your path
    csv_loader = CSVLoader(csv_path)
    csv_data = csv_loader.load()
    
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(csv_data)
    
    # Create vector store
    db = FAISS.from_documents(docs, embeddings)
    retriever = db.as_retriever()
    
    # Create tools
    reservation_tool = create_retriever_tool(
        retriever,
        "reservation_data_tool",
        "has the table data for the purpose of making reservations"
    )
    
    def say_hello(customer_input):
        if any(greeting in customer_input.lower() for greeting in ["hello", "hey", "hi"]):
            return "Hello! Welcome to LeChateau. How can I help you today?"
        return None
    
    greeting_tool = Tool.from_function(
        func=say_hello,
        name="say_hello_tool",
        description="use this tool to greet the customer after the customer has greeted you"
    )
    
    return llm, [reservation_tool, greeting_tool]

def create_agent(llm, tools):
    """Create agent with custom prompt"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a restaurant reservation assistant. Be concise and direct."),
        ("system", "For reservations: Check available tables and locations for the specified time."),
        ("system", "For greetings for example Hello,Hey: Use the say_hello_tool to respond."),
        ("system", "For confirmations: Confirm reservation details briefly and end conversation."),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    
    agent = create_openai_tools_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        llm=llm,
        handle_parsing_errors=True,
        verbose=True,
        max_iterations=3
    )

# Main UI
st.markdown("<h1 style='text-align: center; margin-bottom: 0;'>👑 LeChateau</h1>", unsafe_allow_html=True)
st.markdown("<p class='subheader' style='text-align: center; font-size: 1.5em; margin-top: 0;'>Luxury Dining Reservations</p>", unsafe_allow_html=True)

# Initialize LLM and agent
llm, tools = initialize_llm_tools()
agent_executor = create_agent(llm, tools)

# Create columns
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    st.markdown("### 🎙️ Voice Your Request")
    audio_value = st.audio_input("Speak your request")
    
    if audio_value:
        st.markdown("<div style='background: linear-gradient(145deg, #2d2d2d, #1a1a1a); padding: 20px; border-radius: 10px; border: 1px solid #DAA520;'>", unsafe_allow_html=True)
        st.success("Audio recorded successfully!")
        st.audio(audio_value)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Process audio with Gemini
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(audio_value.getvalue())
                
                # Use Gemini for transcription
                audio_file = genai.upload_file(tmp_file.name)
                model = genai.GenerativeModel("gemini-1.5-flash")
                result = model.generate_content([audio_file, "transcribe the audio as it is"])
                customer_input = result.text
                
                st.info(f"You said: {customer_input}")
                
                # Generate response
                with st.spinner("Processing..."):
                    response = agent_executor.invoke({
                        "input": customer_input,
                        "chat_history": st.session_state.chat_history
                    })
                    
                    bot_response = response['output']
                    st.success(f"Bot: {bot_response}")
                    
                    # Generate and play audio response
                    audio_file = text_to_speech(bot_response)
                    if audio_file:
                        st.audio(audio_file)
                        os.unlink(audio_file)
                    
                    # Update chat history
                    st.session_state.chat_history.append({
                        "user": customer_input,
                        "bot": bot_response
                    })
                    
        except Exception as e:
            st.error(f"Error: {str(e)}")

with col2:
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    st.markdown("### 💬 Conversation History")
    for idx, chat in enumerate(st.session_state.chat_history):
        st.markdown(f"<div style='margin-bottom: 15px;'>", unsafe_allow_html=True)
        st.text_area("You:", chat["user"], height=100, disabled=True, key=f"user_{idx}")
        st.text_area("Bot:", chat["bot"], height=100, disabled=True, key=f"bot_{idx}")
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("""
<div style='text-align: center; padding: 20px; color: #DAA520; font-family: "Cormorant Garamond", serif;'>
    <p>✨ Experience Exceptional Service ✨</p>
</div>
""", unsafe_allow_html=True)
