# 🤖 MediBot - AI-Powered Medical Document Q&A System

A Retrieval-Augmented Generation (RAG) system that creates an intelligent chatbot for answering questions based on medical documents. This project demonstrates a complete implementation of document processing, vector storage, and conversational AI.

## ✨ Features

- **📚 Document Processing**: Automatically processes PDF documents and creates searchable chunks
- **🔍 Intelligent Retrieval**: Uses FAISS vector database for efficient similarity search
- **💬 Conversational Interface**: Clean chat-based UI with conversation history
- **🎯 Context-Aware Responses**: Provides answers based only on loaded documents, preventing hallucination
- **⚡ Fast Performance**: Optimized with Groq's high-speed LLM inference
- **🛡️ Error Handling**: Robust error handling throughout the application

## 🏗️ Architecture

The project consists of three main components:

1. **Document Processing Pipeline** (`create_memory_for_llm.py`)
   - Loads and processes PDF documents
   - Splits documents into searchable chunks
   - Creates vector embeddings and stores in FAISS

2. **Core Q&A Engine** (`connect_memory_for_llm.py`)
   - Connects vector store with Large Language Model
   - Implements RetrievalQA chain for accurate responses
   - Uses custom prompt templates for context-aware answers

3. **Web Interface** (`medibot.py`)
   - Streamlit-based chat interface
   - Maintains conversation history
   - User-friendly error handling

## 🛠️ Tech Stack

- **LLM**: Groq's Llama 3.1 8B Instant model
- **Embeddings**: HuggingFace sentence-transformers/all-MiniLM-L6-v2
- **Vector Database**: FAISS for similarity search
- **Web Framework**: Streamlit
- **Document Processing**: PyPDFLoader
- **Text Splitting**: RecursiveCharacterTextSplitter

## 📋 Prerequisites

- Python 3.8+
- Groq API key
- Required Python packages (see Installation section)

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/medibot.git
   cd medibot
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   Create a `.env` file in the root directory:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

4. **Add your documents**
   Place your PDF files in the `data/` directory.

## 📖 Usage

### 1. Process Documents
First, run the document processing script to create the vector database:
```bash
python create_memory_for_llm.py
```

### 2. Test the System (Optional)
Test the Q&A system via command line:
```bash
python connect_memory_with_llm.py
```

### 3. Launch Web Interface
Start the Streamlit web application:
```bash
streamlit run medibot.py
```

The web interface will be available at `http://localhost:8501`

## 📁 Project Structure

```
medibot/
├── create_memory_for_llm.py    # Document processing pipeline
├── connect_memory_for_llm.py  # Core Q&A engine
├── medibot.py                  # Streamlit web interface
├── data/                       # Document storage
│   └── oracle.pdf             # Example medical document
├── vectorstore/               # FAISS vector database
│   └── db_faiss/
│       ├── index.faiss
│       └── index.pkl
├── requirements.txt            # Python dependencies
├── .env                       # Environment variables
└── README.md                  # This file
```

## 🔧 Configuration

### Customizing the System

1. **Change Embedding Model**: Modify the model name in the embedding initialization:
   ```python
   embedding_model = HuggingFaceEmbeddings(model_name="your_model_name")
   ```

2. **Adjust Chunk Size**: Modify chunk parameters in `create_memory_for_llm.py`:
   ```python
   text_splitter = RecursiveCharacterTextSplitter(
       chunk_size=500,  # Adjust chunk size
       chunk_overlap=50  # Adjust overlap
   )
   ```

3. **Customize Prompt Template**: Modify the prompt template in the scripts to change response style.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LangChain](https://langchain.com/) for the RAG framework
- [Groq](https://groq.com/) for high-speed LLM inference
- [HuggingFace](https://huggingface.co/) for embedding models
- [Streamlit](https://streamlit.io/) for the web interface
- [FAISS](https://github.com/facebookresearch/faiss) for vector similarity search

## 📞 Support

If you have any questions or need help with the project, please open an issue on GitHub.

## 🔄 Version History

- **v1.0.0** - Initial release with basic RAG functionality
  - Document processing pipeline
  - FAISS vector storage
  - Streamlit web interface
  - Groq LLM integration

---

**Made with ❤️ for the AI community** 