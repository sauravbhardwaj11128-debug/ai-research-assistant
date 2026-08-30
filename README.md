# 📚 AI Research Assistant

An AI-powered Research Assistant that allows users to upload PDF documents and ask questions based on their content. The application uses LangChain, Gemini, ChromaDB, and Retrieval-Augmented Generation (RAG) to retrieve relevant information and generate contextual answers.

## 🚀 Features

- Upload PDF documents
- Extract and preprocess document content
- Split documents into smaller chunks
- Generate Gemini embeddings
- Store embeddings using ChromaDB
- Retrieve relevant document sections using similarity search
- Generate answers using Gemini LLM
- Display retrieved document sources
- Interactive Streamlit interface
- Maintain previous questions and answers during the session

## 🛠️ Tech Stack

- Python
- LangChain
- Google Gemini
- ChromaDB
- PyPDF
- Streamlit

## 🔄 Project Workflow

PDF Upload
↓
PDF Document Loading
↓
Text Splitting
↓
Gemini Embeddings
↓
ChromaDB Vector Database
↓
Similarity Search / Retriever
↓
Relevant Context
↓
Gemini LLM
↓
Final Answer

## 📂 Project Structure

ai-research-assistant/

├── app.py  
├── requirements.txt  
├── README.md  
└── notebooks/  
&nbsp;&nbsp;&nbsp;&nbsp;└── AI_Research_Assistant.ipynb

## 📈 Development Progress

### Week 1
- Project repository initialized
- LangChain environment configured
- PDF document loading implemented
- Document preprocessing and text chunking completed

### Week 2
- Gemini embeddings integrated
- ChromaDB vector database created
- Document embeddings stored
- Similarity search implemented

### Week 3
- Document retriever implemented
- Retrieved relevant document chunks
- Tested similarity search with different queries

### Week 4
- Gemini LLM integrated
- RAG prompt created
- Initial RAG pipeline implemented
- Tested question answering using local PDF documents

### Current Development
- Streamlit application developed
- PDF upload functionality added
- RAG pipeline connected with the user interface
- Source retrieval and chat history implemented

## ▶️ How to Run

### 1. Clone the repository

git clone <YOUR_GITHUB_REPOSITORY_URL>

### 2. Install dependencies

pip install -r requirements.txt

### 3. Run the Streamlit application

streamlit run app.py

### 4. Enter Gemini API Key

Enter your Gemini API key in the application sidebar.

### 5. Upload a PDF

Upload a PDF document and click **Process Document**.

### 6. Ask Questions

Enter questions related to the uploaded document and click **Ask Question**.

## 🔮 Future Improvements

- Single AI Agent integration
- Better conversation memory
- Improved document retrieval
- Multiple document support
- Advanced response evaluation
- Authentication and user management
- Cloud deployment and performance optimization

## 👨‍💻 Project Status

The project is currently under active development. The core document processing, vector search, RAG pipeline, and Streamlit interface have been implemented. Further improvements and agent-based capabilities will be added in upcoming development stages.

