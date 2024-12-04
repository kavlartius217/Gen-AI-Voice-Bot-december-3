import streamlit as st
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

# Page config
st.set_page_config(
    page_title="LeChateau Reservation Bot",
    page_icon="🍽️",
    layout="wide"
)
st.markdown("""
<style>
    /* Main app background - matching Amazon gradient */
    .stApp {
        background: linear-gradient(135deg, 
            #000000 0%, 
            #004D2C 100%);
    }
    
    /* Headers - bolder white */
    h1, h2, h3 {
        color: #FFFFFF !important;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        letter-spacing: 0px;
    }
    
    /* Subheader */
    .subheader {
        color: #FFFFFF !important;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        opacity: 0.9;
    }
    
    /* Text areas */
    .stTextArea {
        background: linear-gradient(145deg, 
            rgba(0, 51, 30, 0.95) 0%, 
            rgba(0, 77, 44, 0.95) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        color: #FFFFFF !important;
        font-weight: 500;
    }
    
    /* Success messages */
    .st-emotion-cache-16idsys {
        background: rgba(0, 77, 44, 0.9);
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: #FFFFFF;
        font-weight: 500;
    }
    
    /* Info messages */
    .st-emotion-cache-16r3i8g {
        background: rgba(0, 51, 30, 0.9);
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: #FFFFFF;
        font-weight: 500;
    }
    
    /* Buttons */
    .stButton button {
        background: #FFFFFF;
        color: #004D2C;
        border: none;
        padding: 12px 24px;
        border-radius: 6px;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        background: #F0F0F0;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Audio player */
    .stAudio {
        background: rgba(0, 51, 30, 0.95);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 15px;
    }
    
    /* Chat container */
    .chat-container {
        background: linear-gradient(145deg,
            rgba(0, 51, 30, 0.95) 0%,
            rgba(0, 77, 44, 0.95) 100%);
        border-radius: 8px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin: 15px 0;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 6px;
        background: #000000;
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.2);
        border-radius: 3px;
    }
    
    /* Company branding */
    .company-brand {
        text-align: center;
        padding: 15px;
        color: #FFFFFF;
        font-family: 'Inter', sans-serif;
        font-size: 0.9em;
        font-weight: 600;
        letter-spacing: 0.5px;
        position: fixed;
        bottom: 0;
        width: 100%;
        background: rgba(0, 51, 30, 0.95);
    }

    /* Regular text */
    .stMarkdown {
        color: #FFFFFF;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
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
    csv_path = "table_data (1).csv"  # Update with your path
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
        if any(greeting in customer_input.lower() for greeting in ["Hello", "Hey", "Hi"]):
            return "Hello! Welcome to LeChateau. How can I help you today?"
        return None
    
    greeting_tool = Tool.from_function(
        func=say_hello,
        name="say_hello_tool",
        description="use this tool to greet the customer after the customer has greeted you"
    )
    
    return llm, [reservation_tool, greeting_tool]

def create_agent(llm, tools):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a Human like restaurant reservation assistant. Be concise and direct and always follow the prompt."),
        ("system", "For greetings such as Hi, Hey and Hello: Use the say_hello_tool to respond."),
        ("system", "When the user specifies the number of people and the time to make a reservation: Check the reservation_data_tool for the tables availabe during the specified time as well as the locations of these tables and convey this information to the user."),
        ("system", "After the user has chosen a table from the provided options: Confirm reservation details briefly and end conversation."),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    
    agent = create_openai_tools_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        llm=llm,
        temperature=0.5,
        handle_parsing_errors=True,
        verbose=True,
        max_iterations=10  # Reduced for faster response
    )

# Main UI
st.title("🍽️ LeChateau Reservation Bot")
st.subheader("Voice-enabled Reservation System")

# Initialize LLM and agent
llm, tools = initialize_llm_tools()
agent_executor = create_agent(llm, tools)

# Create columns
col1, col2 = st.columns([2, 1])

with col1:
    # Audio recording
    audio_value = st.audio_input("Speak your request")
    
    if audio_value:
        st.success("Audio recorded successfully!")
        st.audio(audio_value)
        
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
    st.markdown("### 💬 Chat History")
    for idx, chat in enumerate(st.session_state.chat_history):
        st.text_area("You:", value=chat["user"], height=100, disabled=True, key=f"user_message_{idx}")
        st.text_area("Bot:", value=chat["bot"], height=100, disabled=True, key=f"bot_response_{idx}")
        st.markdown("---")



