import os
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path
import streamlit as st
from dataclasses import dataclass
import sys

# LangChain imports - Updated for latest version
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, CSVLoader, 
    UnstructuredWordDocumentLoader
)
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Try multiple LLM options
try:
    from langchain_community.llms import Ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RAGConfig:
    """Configuration class for RAG application"""
    hf_api_token: Optional[str] = None
    chunk_size: int = 1000
    chunk_overlap: int = 200
    temperature: float = 0.1
    max_tokens: int = 1000
    top_k: int = 4
    llm_type: str = "local"  # "huggingface", "ollama", or "local"
    model_name: str = "llama2"  # for ollama
    embedding_model: str = "all-MiniLM-L6-v2"

class SimpleLLM:
    """Simple local LLM wrapper using transformers"""
    
    def __init__(self, model_name="microsoft/DialoGPT-small", max_tokens=1000, temperature=0.1):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers package not available")
        
        self.generator = pipeline(
            "text-generation",
            model=model_name,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True if temperature > 0 else False,
            pad_token_id=50256
        )
        
    def __call__(self, prompt: str) -> str:
        try:
            # Generate response
            result = self.generator(prompt, max_new_tokens=200, num_return_sequences=1)
            return result[0]['generated_text'][len(prompt):].strip()
        except Exception as e:
            logger.error(f"Error in local LLM generation: {e}")
            return "I apologize, but I'm having trouble generating a response right now."

class DocumentProcessor:
    """Handles document loading and processing"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
    def load_document(self, file_path: str) -> List[Document]:
        """Load document based on file extension"""
        file_path = Path(file_path)
        
        # Debug information
        logger.info(f"Processing file: {file_path}")
        logger.info(f"File exists: {file_path.exists()}")
        logger.info(f"File suffix: '{file_path.suffix}'")
        logger.info(f"File suffix lower: '{file_path.suffix.lower()}'")
        
        try:
            # Check if file exists
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            file_extension = file_path.suffix.lower().strip()
            
            if file_extension == '.pdf':
                loader = PyPDFLoader(str(file_path))
            elif file_extension == '.txt':
                loader = TextLoader(str(file_path), encoding='utf-8')
            elif file_extension == '.csv':
                loader = CSVLoader(str(file_path))
            elif file_extension in ['.docx', '.doc']:
                loader = UnstructuredWordDocumentLoader(str(file_path))
            else:
                raise ValueError(f"Unsupported file type: '{file_extension}' for file: {file_path}")
            
            documents = loader.load()
            logger.info(f"Loaded {len(documents)} documents from {file_path}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {str(e)}")
            raise
    
    def process_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        try:
            chunks = self.text_splitter.split_documents(documents)
            logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
            return chunks
        except Exception as e:
            logger.error(f"Error processing documents: {str(e)}")
            raise

class VectorStore:
    """Manages vector storage and retrieval"""
    
    def __init__(self, embeddings):
        """Initialize VectorStore with embeddings"""
        self.embeddings = embeddings
        self.vector_store = None
        
    def create_vector_store(self, documents: List[Document]) -> None:
        """Create FAISS vector store from documents"""
        try:
            if not documents:
                raise ValueError("No documents provided for vector store creation")
            
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
            logger.info(f"Created vector store with {len(documents)} documents")
            
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            raise
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to existing vector store"""
        try:
            if self.vector_store is None:
                self.create_vector_store(documents)
            else:
                self.vector_store.add_documents(documents)
                logger.info(f"Added {len(documents)} documents to vector store")
                
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            raise
    
    def save_vector_store(self, path: str) -> None:
        """Save vector store to disk"""
        try:
            if self.vector_store is None:
                raise ValueError("No vector store to save")
            
            self.vector_store.save_local(path)
            logger.info(f"Vector store saved to {path}")
            
        except Exception as e:
            logger.error(f"Error saving vector store: {str(e)}")
            raise
    
    def load_vector_store(self, path: str) -> None:
        """Load vector store from disk"""
        try:
            self.vector_store = FAISS.load_local(path, self.embeddings, allow_dangerous_deserialization=True)
            logger.info(f"Vector store loaded from {path}")
            
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            raise
    
    def get_retriever(self, top_k: int = 4):
        """Get retriever for similarity search"""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")
        return self.vector_store.as_retriever(search_kwargs={"k": top_k})

class RAGPipeline:
    """Main RAG pipeline class"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name=config.embedding_model)
        
        # Initialize LLM based on config
        self.llm = self._initialize_llm(config)
            
        self.document_processor = DocumentProcessor(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )
        
        # Fixed: Pass embeddings to VectorStore constructor
        self.vector_store = VectorStore(self.embeddings)
        
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""Use the following pieces of context to answer the human's question. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Always provide specific details and cite relevant information from the context when possible.

Context:
{context}

Question:
{question}

Answer:"""
        )

    def _initialize_llm(self, config: RAGConfig):
        """Initialize LLM based on configuration"""
        if config.llm_type == "ollama" and OLLAMA_AVAILABLE:
            try:
                return Ollama(model=config.model_name, temperature=config.temperature)
            except Exception as e:
                logger.warning(f"Ollama initialization failed: {e}, falling back to local model")
        
        if config.llm_type == "local" and TRANSFORMERS_AVAILABLE:
            try:
                return SimpleLLM(
                    model_name="microsoft/DialoGPT-small",
                    max_tokens=config.max_tokens,
                    temperature=config.temperature
                )
            except Exception as e:
                logger.warning(f"Local model initialization failed: {e}")
        
        # Fallback to an enhanced response model
        class EnhancedBasicLLM:
            def __call__(self, prompt: str) -> str:
                # Extract context and question from prompt
                try:
                    lines = prompt.split('\n')
                    context_start = False
                    question_start = False
                    context_lines = []
                    question_lines = []
                    
                    for line in lines:
                        if line.startswith('Context:'):
                            context_start = True
                            continue
                        elif line.startswith('Question:'):
                            context_start = False
                            question_start = True
                            continue
                        elif line.startswith('Answer:'):
                            break
                        
                        if context_start:
                            context_lines.append(line.strip())
                        elif question_start:
                            question_lines.append(line.strip())
                    
                    context = ' '.join([line for line in context_lines if line])
                    question = ' '.join([line for line in question_lines if line])
                    
                    # Provide a more intelligent basic response
                    if context and question:
                        # Find relevant sentences in context that might answer the question
                        question_words = question.lower().split()
                        context_sentences = context.split('.')
                        
                        relevant_sentences = []
                        for sentence in context_sentences:
                            sentence_lower = sentence.lower()
                            if any(word in sentence_lower for word in question_words):
                                relevant_sentences.append(sentence.strip())
                        
                        if relevant_sentences:
                            return f"Based on the document content, here are the most relevant details:\n\n" + \
                                   "\n".join([f"• {sentence}." for sentence in relevant_sentences[:3]])
                        else:
                            return f"I found information related to your question in the documents. " + \
                                   f"The key points from the retrieved content include: {context[:300]}..."
                    
                    return "I can retrieve relevant information from your documents, but I'm currently running in basic mode. Please check the document chunks below for detailed information."
                except Exception:
                    return "I can retrieve relevant information from your documents, but I'm currently running in basic mode. Please check the document chunks below for detailed information."
        
        logger.warning("Using enhanced basic LLM - install transformers or ollama for better responses")
        return EnhancedBasicLLM()

    def get_qa_chain(self):
        """Create a simple QA chain"""
        retriever = self.vector_store.get_retriever(top_k=self.config.top_k)
        
        class SimpleQAChain:
            def __init__(self, llm, retriever, prompt_template):
                self.llm = llm
                self.retriever = retriever
                self.prompt_template = prompt_template
            
            def invoke(self, inputs: Dict[str, str]) -> Dict[str, Any]:
                query = inputs["query"]
                
                # Retrieve relevant documents
                docs = self.retriever.get_relevant_documents(query)
                
                # Format context
                context = "\n\n".join([doc.page_content for doc in docs])
                
                # Create prompt
                prompt = self.prompt_template.format(context=context, question=query)
                
                # Generate answer
                try:
                    answer = self.llm(prompt)
                except Exception as e:
                    logger.error(f"LLM generation error: {e}")
                    answer = f"I found relevant information in the documents, but had trouble generating a response. Please check the source documents below."
                
                return {
                    "result": answer,
                    "source_documents": docs
                }
        
        return SimpleQAChain(self.llm, retriever, self.prompt_template)

    def run_query(self, query: str) -> Dict[str, Any]:
        """Answer user query using the QA chain"""
        try:
            qa_chain = self.get_qa_chain()
            response = qa_chain.invoke({"query": query})
            return response
        except Exception as e:
            logger.error(f"Error running query: {str(e)}")
            raise