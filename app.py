import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import google.generativeai as genai
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()

# Configure Google Generative AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize a session state variable to store questions
if "previous_questions" not in st.session_state:
    st.session_state.previous_questions = []

def get_pdf_text(pdf_docs):
    """Extract text from uploaded PDF files."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    """Split text into manageable chunks."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
    chunks = splitter.split_text(text)
    return chunks

def get_vector_store(chunks):
    """Generate and save a FAISS vector store from text chunks."""
    embeddings = HuggingFaceEmbeddings(model_name="bert-base-uncased")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")  # Save the FAISS index locally
    return vector_store

def conversion_chain():
    """Set up the question-answering chain using Google Gemini model."""
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the provided context, respond with "Answer is not available in the context". Do not guess.\n\n
    Context:\n{context}\n
    Question:\n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, prompt=prompt, chain_type="stuff")
    return chain

def user_input(user_question):
    """Process user question and return the answer using the FAISS vector store."""
    embeddings = HuggingFaceEmbeddings(model_name="bert-base-uncased")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    if not docs:
        return "No answer found. No relevant documents were retrieved."

    chain = conversion_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    answer = response.get('output_text', 'No answer found') if isinstance(response, dict) else response

    st.write("Answer:", answer)

    with st.expander("Retrieved Documents and Debug Info"):
        st.write("Retrieved Documents:")
        for i, doc in enumerate(docs):
            st.write(f"Document {i+1}: {doc.page_content}")
            st.write("--------------------------------")

    # Add the question to session state
    st.session_state.previous_questions.append(user_question)
    
    return answer

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="Chat with PDF", page_icon="🤖")
    st.header("Chat with PDF using Gemini 🤖")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu")

        # Display previously asked questions at the top
        if st.session_state.previous_questions:
            st.write("Previously Asked Questions:")
            for q in reversed(st.session_state.previous_questions):
                st.write("- ", q)

        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)
        
        # Move the submit button to the bottom of the sidebar
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Processing complete!")

if __name__ == "__main__":
    main()
