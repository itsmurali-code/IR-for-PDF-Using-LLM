# Information Retrieval from Documents by LLM

This Streamlit application enables users to upload PDF documents and leverage the power of Large Language Models (LLMs) and semantic search for context-aware question answering. The system utilizes Groq for language model completions and Google Generative AI for embeddings via LangChain.

## Features

- **Secure API Key Input:** Enter your GROQ and Google API keys directly in the sidebar.
- **PDF Upload:** Upload one or multiple PDF files for analysis.
- **Document Embedding:** Converts documents into vector embeddings using Google Generative AI.
- **Vector Store:** Embeds are stored with FAISS for efficient and accurate retrieval.
- **Natural Language Q&A:** Ask questions about your uploaded documents and receive precise, context-driven answers.
- **Similarity Search Insights:** View which document chunks were most relevant for each answer.
- **Response Time Display:** Instantly see how long each query takes.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
   ```

2. **Create & activate a Python environment (optional but recommended):**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # on Windows: .venv\Scripts\activate
   ```

3. **Install the dependencies:**
   ```bash
   pip install streamlit langchain langchain-groq langchain-google-genai \
               langchain-community faiss-cpu
   ```
   Adjust package names as needed.

## API Keys Needed

- **GROQ API Key:** Sign up at [GROQ](https://groq.com/) to obtain your API key.
- **Google API Key:** Obtain from [Google Generative AI](https://ai.google.dev/) or via Google Cloud Console.

## Usage

1. **Run the Streamlit app:**
   ```bash
   streamlit run your_script_name.py
   ```
   Replace `your_script_name.py` with the actual script filename.

2. **In the web interface:**
   - Enter your API keys in the sidebar.
   - Upload PDF(s) using the uploader.
   - Click **Documents Embedding** to prepare the search database.
   - Type your question about the documents.
   - View the answer, documents used, and response time.

## How It Works

- **Embedding and Chunking:** Uploaded PDFs are split into text chunks, embedded into vectors, and indexed in FAISS.
- **Retrieval:** When you ask a question, the system finds relevant vector chunks and feeds their context to LLM (Llama3 on Groq) to generate precise answers using only your documents.
- **Transparency:** You can inspect which document portions most influenced the LLM’s answer.

## Troubleshooting

- Both API keys are required for full functionality.
- Upload only readable, uncorrupted PDFs.
- If answers or embeddings are missing, double-check your API keys.

## License

[MIT License](LICENSE) (or specify your license here)

## Credits

Built with
[Streamlit] https://streamlit.io/, 
[LangChain] https://www.langchain.com/,
[FAISS] https://faiss.ai/,
[Google Generative AI] https://ai.google.dev/,
[Groq] https://groq.com/.

## YT Links
https://www.youtube.com/watch?v=wFdFLWc-W4k
https://www.youtube.com/watch?v=AjQPRomyd-k
