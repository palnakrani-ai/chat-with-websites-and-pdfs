import os 
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader,PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

CONNECTION_STRING = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

def get_vectorstore_from_url(url):
    #get the text in documents form
    loader = WebBaseLoader(url)
    docs = loader.load()
    
    #split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter()
    docs_chunks = text_splitter.split_documents(docs)
    
    #embedding model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    
    COLLECTION_NAME = "WebBasedVectors"
    
    vector_store = PGVector(
        connection=CONNECTION_STRING, 
        collection_name=COLLECTION_NAME,
        embeddings=embeddings
    )
    
    # Add documents to the vector store
    vector_store.add_documents(docs_chunks)

    return vector_store

def get_vectorstore_from_pdf(pdf_file):
    # Save uploaded file temporarily
    with open("temp.pdf", "wb") as f:
        f.write(pdf_file.getbuffer())
    
    # Load PDF
    loader = PyPDFLoader("temp.pdf")
    docs = loader.load()
    
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter()
    docs_chunks = text_splitter.split_documents(docs)

    #embedding model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    COLLECTION_NAME = "PdfBasedVectors"
    
    vector_store = PGVector(
        connection=CONNECTION_STRING, 
        collection_name=COLLECTION_NAME,
        embeddings=embeddings
    )
    
    # Add documents to the vector store
    vector_store.add_documents(docs_chunks)

    return vector_store

def get_context_retriever_chain(vector_store):
    # Initialize HuggingFace LLM
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        huggingfacehub_api_token=os.getenv('HUGGINGFACE_API_KEY')
    )
    retriever = vector_store.as_retriever()
    
    prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            (
                "user",
                "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation",
            ),
        ]
    )
    
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

    return retriever_chain

def get_conversational_rag_chain(retriever_chain):

    # Initialize HuggingFace LLM
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        huggingfacehub_api_token=os.getenv('HUGGINGFACE_API_KEY')
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer the user's questions based on the below context:\n\n{context}",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
        ]
    )

    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)

    return create_retrieval_chain(retriever_chain, stuff_documents_chain)


def get_response(user_input):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)

    response = conversation_rag_chain.invoke(
        {"chat_history": st.session_state.chat_history, "input": user_input}
    )

    return response["answer"]

def init_chat_session():
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am a bot. How can I help you?"),
    ]

# app config
st.set_page_config(page_title="Chat with websites and PDFs", page_icon="🤖")
st.title("Chat with websites and PDFs")

# sidebar
with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Website URL")
    pdf_file = st.file_uploader("Upload PDF", type=['pdf'])

    if st.button("Clear Chat History"):
        init_chat_session()
        st.session_state.pop('vector_store', None)
        st.session_state.pop('current_source', None)
        st.success("Chat history cleared!")

# Main logic for handling input sources
if website_url and pdf_file:
    st.warning("Please provide either a website URL or a PDF file, not both.")
elif website_url:
    # Handle website URL
    if "vector_store" not in st.session_state or st.session_state.get('current_source') != website_url:
        with st.spinner("Loading website content..."):
            try:
                st.session_state.vector_store = get_vectorstore_from_url(website_url)
                st.session_state.current_source = website_url
                init_chat_session()
                st.success("Website content loaded successfully!")
            except Exception as e:
                st.error(f"Error loading website: {str(e)}")
elif pdf_file:
    # Handle PDF file
    if "vector_store" not in st.session_state or st.session_state.get('current_source') != pdf_file.name:
        with st.spinner("Processing PDF..."):
            vector_store = get_vectorstore_from_pdf(pdf_file)
            if vector_store:
                st.session_state.vector_store = vector_store
                st.session_state.current_source = pdf_file.name
                init_chat_session()
                st.success("PDF processed successfully!")
else:
    st.info("Please enter a website URL or upload a PDF file to start chatting.")

# Initialize chat history if not exists
if "chat_history" not in st.session_state:
    init_chat_session()

# Chat interface
if "vector_store" in st.session_state:
    # Display source information
    st.sidebar.success(f"Currently chatting with: {st.session_state.current_source}")
    
    # User input
    user_query = st.chat_input("Type your message here...")
    if user_query is not None and user_query != "":
        with st.spinner("Generating response..."):
            response = get_response(user_query)
            st.session_state.chat_history.append(HumanMessage(content=user_query))
            st.session_state.chat_history.append(AIMessage(content=response))

    # Display conversation
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)