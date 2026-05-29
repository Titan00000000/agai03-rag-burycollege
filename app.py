# Importing my libraries and running them into the programs memory
import os
import pandas as pd
import time
import streamlit as st

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
 

# Setting up the API Key so Streamlit can read it. The API key is already stored in load_dotenv() so no need for further calls to this.
load_dotenv()


# Main page set up. This is what the user will see when run as a webpage
st.set_page_config(page_title="Bury College AI Assistant", page_icon="🎓", layout="centered")
st.title("🎓 Bury College AI Assistant")
st.write("Welcome! Ask me anything about Bury College courses, campus facilities, or student services.")

# This will build the sidebar with the statistics requested from the assignment
with st.sidebar:
    st.markdown("# 🎓") # Or a local logo if you have one
    st.header("Project Information")
    st.markdown(
        """
        This intelligent assistant answers questions about **Bury College** using a 
        two-stage Hybrid Retrieval system.
        
        * **Target Site:** [Bury College Official](https://burycollege.ac.uk)
        * **Framework:** LangChain & Google Gemini
        * **Vector Store:** FAISS (CPU variant)
        """
    )
    
    st.divider()
    
    # Statistics Section
    st.subheader("📊 Scraping & Dataset Stats")
    st.metric(label="Pages Scraped", value="17")          
    st.metric(label="Synthetic Q/As", value="204")       
    st.caption("Data last indexed: May 2026")
    
    st.divider()

    # Clear Chat Button Layout & Logic
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Option to see sample questions to ask and to upload data if needed.  However, while you can upload the questions, they don't update the system
st.write("---")
expander = st.expander("💡 View Sample Questions & Project Options")
with expander:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            **Try asking things like:**
            * *What computing courses do you offer?*
            * *Where is the campus located?*
            * *How can I contact student services?*
            """
        )
    with col2:
        # File uploader option for user extension compliance
        uploaded_file = st.file_uploader("Upload new questions (.txt/.json)", type=["txt", "json"])
        if uploaded_file is not None:
            st.success("Questions uploaded successfully! (Mock integration)")
st.write("---")



# This is run once to store the data into memory once and enables chat history. It will speed up query response times
@st.cache_resource
def initialize_rag_system():
    """Loads pre-computed FAISS vector stores and initializes the LLM chain once."""
    # Initialise embeddings and active stable LLM model
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.2)
    
    # Load main web page document index from my SSD
    persist_directory = "data/processed/faiss_db"
    doc_vector_store = FAISS.load_local(
        persist_directory, 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    
    # Load curated Q/A vector store from my SSD
    qa_persist_directory = "data/processed/faiss_qa_db"
    qa_vector_store = FAISS.load_local(
        qa_persist_directory, 
        embeddings, 
        allow_dangerous_deserialization=True
    )
  
    # Ensures that the chatbot has a context to work with so it knows how to respond to my expected audience
    prompt_template = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an intelligent, helpful, and official AI assistant for Bury College.\n"
        "Your task is to answer the user's question accurately using ONLY the provided context below.\n\n"
        "=== CONTEXT ===\n"
        "{context}\n"
        "===============\n\n"
        "Guidelines:\n"
        "- Base your answer strictly on the provided context. If the answer cannot be found in the context, "
        "politely state that you do not have that information and suggest contacting the college directly.\n"
        "- Maintain a professional, welcoming tone.\n"
        "- Do not mention data sources, brackets, or database names in your conversational text."
    )),
    # This holds the rolling conversation log
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])


    # Build the final chain
    chain = prompt_template | llm | StrOutputParser()
    
    return qa_vector_store, doc_vector_store, chain

# Execute our cached startup function
qa_store, doc_store, llm_chain = initialize_rag_system()

# This allows the system to check two places. 
# In stage 1 it tries my qa_dataset.csv file first. If the threshold is too high, then it moves to stage 2
def hybrid_retrieve(query, qa_threshold=0.5):
    """Two-stage retrieval routing logic."""
    # Stage 1: Check curated Q/A index
    qa_results = qa_store.similarity_search_with_score(query, k=1)
    
    if qa_results:
        qa_doc, distance_score = qa_results[0]
        if distance_score < qa_threshold:
            return {
                "source_type": "Curated Q/A Pair",
                "context": f"Question: {qa_doc.page_content}\nAnswer: {qa_doc.metadata['answer']}",
                "source": qa_doc.metadata['source']
            }
            
    # Stage 2: Fallback to document scrapes. This happens if the matchup between the question asked and 
    # data we have in my qa_dataset.csv file is week (greater than 0.5, then we fall back to the full scraped data
    doc_results = doc_store.similarity_search(query, k=3)
    combined_context = "\n\n".join([d.page_content for d in doc_results])
    sources = list(set([d.metadata['source'] for d in doc_results]))
    
    return {
        "source_type": "Full Web Page Scrapes",
        "context": combined_context,
        "source": ", ".join(sources)
    }

# This will create the chat interface and also allow for the history of the conversation to be stored (subject to the amount of tokens left)
# Streamlit keeps track of chat history in st.session_state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display all existing messages from history on refresh
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # If there are sources attached to an old assistant message, display them
        if "source_info" in message:
            st.caption(message["source_info"])


# Accept live user chat input
if user_query := st.chat_input("Type your question here..."):
    
    # Display user's question instantly
    with st.chat_message("user"):
        st.markdown(user_query)
    
    # Save user's question to the session state history
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # Generate response inside assistant chat bubble
    with st.chat_message("assistant"):
        # A. Pull context using hybrid routing
        retrieval_data = hybrid_retrieve(user_query)
        
        # B. Generate conversational text from Gemini with history
        with st.spinner("Consulting college data..."):
            # Pull out all historical messages except the user's live question 
            # to feed as past context
            chat_history = st.session_state.messages[:-1] 
            
            response_text = llm_chain.invoke({
                "context": retrieval_data["context"],
                "history": chat_history, # FEED HISTORY TO GEMINI HERE!
                "question": user_query
            })
        
        # C. Format source citation string for assignment compliance
        source_citation = f"ℹ️ **Source Type:** {retrieval_data['source_type']} | **Verified URL(s):** {retrieval_data['source']}"
        
        # D. Render the elements on the web page
        st.markdown(response_text)
        st.caption(source_citation)
        
    # E. Save assistant response and citations to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": response_text,
        "source_info": source_citation
    })
    