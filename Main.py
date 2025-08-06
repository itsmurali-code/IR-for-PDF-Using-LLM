import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import tempfile
import time

# Title of the app
st.title("Information Retrival from Documents By LLM ")

# Input fields for API keys in sidebar
st.sidebar.title("API Keys")
groq_api_key = st.sidebar.text_input("Enter your GROQ API Key", type="password")
google_api_key = st.sidebar.text_input("Enter your Google API Key", type="password")

if groq_api_key and google_api_key:
    # Set the Google API key in the environment for use by the GoogleGenerativeAIEmbeddings class
    os.environ["GOOGLE_API_KEY"] = google_api_key

    # Initialize the ChatGroq LLM with the provided API key
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

    # Define the prompt template
    prompt = ChatPromptTemplate.from_template(
        """
          Answer the questions based on the provided context only.
        Please provide the most accurate response based on the question
        <context>
        {context}
        <context>
        Questions: {input}
        """
    )

    # Function to create vector embeddings from documents
    def vector_embedding(uploaded_files):
        with tempfile.TemporaryDirectory() as temp_dir:
            all_docs = []
            for uploaded_file in uploaded_files:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Load the PDF file
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                all_docs.extend(docs)

            # Process the uploaded PDF files
            st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # Chunk Creation
            st.session_state.final_documents = st.session_state.text_splitter.split_documents(all_docs)  # Splitting
            st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)  # Vector embeddings

    # File uploader for user to upload PDF files
    uploaded_files = st.file_uploader("Upload PDF Files", type=["pdf"], accept_multiple_files=True)

    # Input field for user to enter their question
    prompt1 = st.text_input("Enter Your Question From Documents")

    # Button to trigger document embedding
    if st.button("Documents Embedding") and uploaded_files:
        vector_embedding(uploaded_files)
        st.write("Vector Store DB Is Ready")

    # Process the user's question if provided
    if prompt1 and "vectors" in st.session_state:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start = time.process_time()
        response = retrieval_chain.invoke({'input': prompt1})
        st.write("Response time: {:.2f} seconds".format(time.process_time() - start))
        st.write(response['answer'])

        # Display document similarity search results in an expander
        with st.expander("Document Similarity Search"):
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("--------------------------------")
else:
    st.write("Please enter both GROQ and Google API keys.")


