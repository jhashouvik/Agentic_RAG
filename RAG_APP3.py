import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from crewai import Agent, Task, Crew  # Import CrewAI
from tavily import TavilyClient  # Import Tavily
from langchain.tools import tool

# Load environment variables
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")  # Tavily API key

# Initialize LLM with explicit LiteLLM provider
llm = ChatGroq(
    model="llama-3.3-70b-versatile",  # Explicitly specify the provider and model
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# Initialize Tavily for internet search
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# Define Tavily as a tool for CrewAI
@tool
def tavily_search(query: str) -> str:
    """Search the internet using Tavily to find relevant information."""
    try:
        response = tavily.search(query=query, max_results=5)
        results = "\n".join([result.get('content', '') for result in response.get('results', [])])
        return results
    except Exception as e:
        return f"Error searching the internet: {e}"

# Define CrewAI Agents
def create_crewai_agent():
    # Define the Researcher Agent with Tavily as a tool
    researcher = Agent(
        role="Researcher",
        goal="Find relevant information to answer the user's query.",
        backstory="You are a skilled researcher who can find information from both local documents and the internet.",
        tools=[tavily_search],  # Add Tavily as a tool
        llm=llm,
        verbose=True
    )

    # Define the Analyst Agent
    analyst = Agent(
        role="Analyst",
        goal="Analyze the retrieved information and provide a clear and concise answer.",
        backstory="You are an expert analyst who can synthesize information from multiple sources.",
        tools=[],  # No tools needed for the analyst
        llm=llm,
        verbose=True
    )

    return researcher, analyst

# Streamlit app title
st.title("📚 NaiveRAG: Intelligent Document Chatbot 🤖")

# Initialize session state for conversation history
if "conversation" not in st.session_state:
    st.session_state.conversation = []

# File upload, URL input, directory input, and YouTube URL input in the sidebar
pdf_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")
web_url = st.sidebar.text_input("Enter a URL to load data from:")
csv_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

pages = []

# Load documents and create vector store
if pdf_file or web_url or csv_file:
    pages = []

    # Load PDF file
    if pdf_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_file.getbuffer())
            tmp_file_path = tmp_file.name

        try:
            loader = PyPDFLoader(tmp_file_path)
            pages.extend(loader.load())
        finally:
            os.unlink(tmp_file_path)

    # Load web URL
    if web_url:
        try:
            loader = WebBaseLoader(web_url)
            pages.extend(loader.load())
        except Exception as e:
            st.error(f"Error loading URL: {e}")

    # Load CSV file
    if csv_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
            tmp_file.write(csv_file.getbuffer())
            tmp_file_path = tmp_file.name

        try:
            loader = CSVLoader(tmp_file_path)
            pages.extend(loader.load())
        finally:
            os.unlink(tmp_file_path)

    # Create a vector store
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(pages, embeddings)

    # Store the retriever in session state
    st.session_state.retriever = vector_store.as_retriever()

# Chat input
query = st.chat_input("🤔 Ask a question about the document:")

# Display conversation history
for message in st.session_state.conversation:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Process user input
if query:
    # Add user query to conversation history
    st.session_state.conversation.append({"role": "user", "content": query})

    # Display user message
    with st.chat_message("user"):
        st.write(query)

    # Retrieve relevant documents
    if "retriever" in st.session_state:
        relevant_docs = st.session_state.retriever.invoke(query)  # Use `invoke` instead of `get_relevant_documents`
        context = "\n".join([doc.page_content for doc in relevant_docs])
    else:
        context = "No documents loaded."

    # Define CrewAI Agents and Tasks
    researcher, analyst = create_crewai_agent()

    # Define the Research Task
    research_task = Task(
        description=f"Find relevant information for the query: {query}\n\nContext:\n{context}",
        agent=researcher,
        expected_output="A detailed summary of relevant information with 3-5 points with in 200 words from both local documents and the internet."
    )

    # Define the Analysis Task
    analysis_task = Task(
        description="Analyze the retrieved information and provide a clear and concise answer in 3-5 points.",
        agent=analyst,
        expected_output="A detailed summary with well-structured and accurate answer in 3-5 points to the user's query with in 200 words."
    )

    # Create the Crew
    crew = Crew(
        agents=[researcher, analyst],
        tasks=[research_task, analysis_task],
        verbose=True
    )

    # Execute the Crew
    result = crew.kickoff()

    # Clean up the output
    def clean_output(result, word_limit=200):
        # Extract the raw output from the CrewOutput object
        if hasattr(result, "raw"):
            text = result.raw
        else:
            text = str(result)

        # Split the text into points
        points = [point.strip() for point in text.split("\n") if point.strip()]
        # Truncate to the specified word limit
        truncated_points = []
        word_count = 0
        for point in points:
            words = point.split()
            if word_count + len(words) <= word_limit:
                truncated_points.append(point)
                word_count += len(words)
            else:
                break
        return "\n".join(truncated_points)

    cleaned_result = clean_output(result)

    # Add bot response to conversation history
    st.session_state.conversation.append({"role": "assistant", "content": cleaned_result})

    # Display bot response
    with st.chat_message("assistant"):
        st.write(cleaned_result)

    # Handle contextual greetings
    if query.lower() in ["hi", "hello", "hey"]:
        st.session_state.conversation.append({
            "role": "assistant",
            "content": "Hello! I'm here to help you with any questions or concerns you may have about the uploaded documents. What would you like to know?"
        })
        with st.chat_message("assistant"):
            st.write("Hello! I'm here to help you with any questions or concerns you may have about the uploaded documents. What would you like to know?")
