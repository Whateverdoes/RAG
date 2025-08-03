import streamlit as st
import tempfile
import os
from pathlib import Path
import logging
from typing import List

# Import your original RAG pipeline classes
# Make sure your original code is saved as 'rag_pipeline.py'
try:
    from rag_pipeline import RAGPipeline, RAGConfig, DocumentProcessor
except ImportError:
    st.error("Could not import rag_pipeline.py - make sure the file exists in the same directory")
    st.stop()

# Configure page
st.set_page_config(
    page_title="RAG Document Q&A System",
    page_icon="📚",
    layout="wide"
)


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def save_uploaded_file(uploaded_file):
    """Save uploaded file to temporary location and return path"""
    try:
        # Create temp file with same extension
        suffix = Path(uploaded_file.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error saving file: {str(e)}")
        return None

def process_uploaded_files(uploaded_files, rag_pipeline):
    """Process uploaded files and add to vector store"""
    temp_files = []
    
    try:
        # Save uploaded files to temp locations
        for uploaded_file in uploaded_files:
            temp_path = save_uploaded_file(uploaded_file)
            if temp_path:
                temp_files.append(temp_path)
        
        if not temp_files:
            st.error("No files were successfully uploaded")
            return False
        
        # Process documents using your original pipeline
        all_documents = []
        
        with st.spinner("Processing documents..."):
            for temp_path in temp_files:
                try:
                    # Load document
                    documents = rag_pipeline.document_processor.load_document(temp_path)
                    # Process into chunks
                    chunks = rag_pipeline.document_processor.process_documents(documents)
                    all_documents.extend(chunks)
                    
                    # Show progress
                    filename = Path(temp_path).name
                    st.success(f"✅ Processed {filename}: {len(chunks)} chunks created")
                    
                except Exception as e:
                    st.error(f"Error processing {Path(temp_path).name}: {str(e)}")
                    continue
        
        if all_documents:
            # Create vector store
            with st.spinner("Creating vector store..."):
                rag_pipeline.vector_store.create_vector_store(all_documents)
            
            st.success(f"🎉 Successfully processed {len(all_documents)} document chunks!")
            return True
        else:
            st.error("No documents were successfully processed")
            return False
            
    except Exception as e:
        st.error(f"Error during processing: {str(e)}")
        return False
        
    finally:
        # Clean up temp files
        for temp_path in temp_files:
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            except Exception as e:
                logger.warning(f"Could not delete temp file {temp_path}: {e}")

def main():
    st.title("📚 RAG Document Q&A System")
    st.markdown("Upload your documents and ask questions about their content!")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # API Token input (now optional)
        hf_token = st.text_input(
            "Hugging Face API Token (Optional)",
            type="password",
            help="Enter your Hugging Face API token (optional - app can work with local models)"
        )
        
        # LLM Type selection
        llm_type = st.selectbox(
            "LLM Type",
            ["local", "ollama", "huggingface"],
            help="Choose your LLM backend"
        )
        
        # Advanced settings
        with st.expander("Advanced Settings"):
            chunk_size = st.slider("Chunk Size", 500, 2000, 1000)
            chunk_overlap = st.slider("Chunk Overlap", 50, 500, 200)
            temperature = st.slider("Temperature", 0.0, 1.0, 0.1)
            top_k = st.slider("Top K Results", 1, 10, 4)
    
    # Initialize session state
    if 'rag_pipeline' not in st.session_state:
        st.session_state.rag_pipeline = None
    if 'documents_loaded' not in st.session_state:
        st.session_state.documents_loaded = False
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Documents")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Choose files",
            accept_multiple_files=True,
            type=['pdf', 'txt', 'csv', 'docx', 'doc'],
            help="Drag and drop files here or click to browse"
        )
        
        # Process files button
        if uploaded_files:
            if st.button("Process Documents", type="primary"):
                try:
                    # Initialize RAG pipeline
                    config = RAGConfig(
                        hf_api_token=hf_token if hf_token else None,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        temperature=temperature,
                        top_k=top_k,
                        llm_type=llm_type
                    )
                    
                    with st.spinner("Initializing RAG pipeline..."):
                        st.session_state.rag_pipeline = RAGPipeline(config)
                    
                    # Process files
                    success = process_uploaded_files(uploaded_files, st.session_state.rag_pipeline)
                    st.session_state.documents_loaded = success
                    
                except ValueError as e:
                    if "API token" in str(e):
                        st.error("❌ Invalid Hugging Face API token. Please check your token and try again.")
                        st.info("💡 Get your token from: https://huggingface.co/settings/tokens")
                    else:
                        st.error(f"Configuration error: {str(e)}")
                except Exception as e:
                    st.error(f"Error initializing pipeline: {str(e)}")
        
        elif not uploaded_files:
            st.info("Please upload some documents to get started")
        
        # Show uploaded files
        if uploaded_files:
            st.subheader("Uploaded Files:")
            for file in uploaded_files:
                st.write(f"📄 {file.name} ({file.size} bytes)")
    
    with col2:
        st.header("❓ Ask Questions")
        
        if st.session_state.documents_loaded and st.session_state.rag_pipeline:
            # Query input
            query = st.text_area(
                "Enter your question:",
                placeholder="What is the main topic discussed in the documents?",
                height=100
            )
            
            # Ask question button
            if st.button("Ask Question", type="primary"):
                if query.strip():
                    with st.spinner("Searching for answer..."):
                        try:
                            response = st.session_state.rag_pipeline.run_query(query)
                            
                            # Display answer
                            st.subheader("Answer:")
                            st.write(response['result'])
                            
                            # Display source documents
                            if 'source_documents' in response and response['source_documents']:
                                with st.expander("📖 Source Documents"):
                                    for i, doc in enumerate(response['source_documents']):
                                        st.write(f"**Source {i+1}:**")
                                        st.write(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                                        if hasattr(doc, 'metadata') and doc.metadata:
                                            st.write(f"*Metadata: {doc.metadata}*")
                                        st.write("---")
                        
                        except Exception as e:
                            st.error(f"Error processing query: {str(e)}")
                else:
                    st.warning("Please enter a question")
        else:
            st.info("Please upload and process documents first to ask questions")
    
    # Instructions
    with st.expander("📋 Instructions"):
        st.markdown("""
        ### How to use this app:
        
        1. **Upload Documents:**
           - Drag and drop files or click to browse
           - Supported formats: PDF, TXT, CSV, DOCX, DOC
           - You can upload multiple files at once
        
        2. **Choose LLM Type:**
           - **Local**: Uses basic text processing (current mode)
           - **Ollama**: Better responses with local models
           - **HuggingFace**: Cloud-based models (requires API token)
        
        3. **Process Documents:**
           - Click "Process Documents" after uploading
           - Wait for processing to complete
        
        4. **Ask Questions:**
           - Type your question in the text area
           - Click "Ask Question" to get AI-powered answers
           - View source documents that were used for the answer
        
        ### 🚀 For Better Responses:
        
        **Option 1 - Install Transformers (Python):**
        ```bash
        pip install transformers torch
        ```
        Then restart the app and select "local" mode.
        
        **Option 2 - Install Ollama (Recommended):**
        1. Download from https://ollama.ai
        2. Install and run: `ollama pull llama2`
        3. Select "ollama" in the sidebar
        
        **Option 3 - Use HuggingFace API:**
        1. Get token from https://huggingface.co/settings/tokens
        2. Enter token in sidebar
        3. Select "huggingface" mode
        
        ### Tips:
        - Larger chunk sizes work better for longer documents
        - Higher temperature makes answers more creative
        - Top K controls how many document chunks are used for answering
        """)

    # Show current mode info
    if st.session_state.get('rag_pipeline'):
        with st.sidebar:
            st.info("✅ RAG Pipeline Active")
            if st.session_state.get('documents_loaded'):
                st.success("📚 Documents Loaded")
    
    # Add a note about basic mode
    if st.session_state.get('documents_loaded') and st.session_state.get('rag_pipeline'):
        st.info("💡 **Currently in basic mode** - For better answers, install transformers (`pip install transformers torch`) or Ollama and restart the app.")

if __name__ == "__main__":
    main()