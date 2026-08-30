import os
import tempfile

import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="📚",
    layout="wide"
)


# --------------------------------------------------
# CUSTOM TITLE
# --------------------------------------------------

st.title("📚 AI Research Assistant")
st.write(
    "Upload a PDF document and ask questions based on its content."
)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.header("⚙️ Settings")

    api_key = st.text_input(
        "Gemini API Key",
        type="password",
        help="Enter your Google Gemini API key."
    )

    st.markdown("---")

    st.info(
        "This application uses LangChain, "
        "Gemini, ChromaDB and RAG."
    )


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "document_processed" not in st.session_state:
    st.session_state.document_processed = False


# --------------------------------------------------
# PDF UPLOAD
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "📄 Upload your PDF",
    type=["pdf"]
)


# --------------------------------------------------
# PROCESS DOCUMENT
# --------------------------------------------------

if uploaded_file is not None:

    if st.button("🔄 Process Document", use_container_width=True):

        if not api_key:
            st.error("Please enter your Gemini API key first.")

        else:

            try:

                with st.spinner(
                    "Processing document... Please wait."
                ):

                    # Save uploaded PDF temporarily
                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=".pdf"
                    ) as temp_file:

                        temp_file.write(
                            uploaded_file.getvalue()
                        )

                        pdf_path = temp_file.name


                    # --------------------------------------------------
                    # LOAD PDF
                    # --------------------------------------------------

                    loader = PyPDFLoader(pdf_path)

                    documents = loader.load()


                    # --------------------------------------------------
                    # SPLIT DOCUMENT
                    # --------------------------------------------------

                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=500,
                        chunk_overlap=100
                    )

                    chunks = splitter.split_documents(
                        documents
                    )


                    # --------------------------------------------------
                    # GEMINI EMBEDDINGS
                    # --------------------------------------------------

                    embeddings = GoogleGenerativeAIEmbeddings(
                        model="models/gemini-embedding-001",
                        google_api_key=api_key
                    )


                    # --------------------------------------------------
                    # CHROMA VECTOR DATABASE
                    # --------------------------------------------------

                    vector_db = Chroma.from_documents(
                        documents=chunks,
                        embedding=embeddings
                    )


                    # --------------------------------------------------
                    # RETRIEVER
                    # --------------------------------------------------

                    retriever = vector_db.as_retriever(
                        search_type="similarity",
                        search_kwargs={"k": 3}
                    )


                    # --------------------------------------------------
                    # GEMINI LLM
                    # --------------------------------------------------

                    llm = ChatGoogleGenerativeAI(
                        model="gemini-2.5-flash",
                        temperature=0.2,
                        google_api_key=api_key
                    )


                    # --------------------------------------------------
                    # RAG PROMPT
                    # --------------------------------------------------

                    prompt = ChatPromptTemplate.from_template(
                        """
                        You are an AI Research Assistant.

                        Answer the user's question using only
                        the provided context.

                        If the answer is not available in the
                        context, say:

                        "I could not find the answer in the
                        provided document."

                        Context:
                        {context}

                        Question:
                        {input}

                        Answer:
                        """
                    )


                    # --------------------------------------------------
                    # RAG CHAIN
                    # --------------------------------------------------

                    rag_chain = (
                        {
                            "context": retriever,
                            "input": RunnablePassthrough()
                        }
                        | prompt
                        | llm
                    )


                    # Save objects
                    st.session_state.rag_chain = rag_chain
                    st.session_state.retriever = retriever
                    st.session_state.document_processed = True
                    st.session_state.chat_history = []


                    # Remove temporary file
                    os.remove(pdf_path)


                st.success(
                    f"Document processed successfully! "
                    f"{len(chunks)} chunks created."
                )


            except Exception as e:

                st.error(
                    f"An error occurred while processing "
                    f"the document: {e}"
                )


# --------------------------------------------------
# QUESTION ANSWERING SECTION
# --------------------------------------------------

if st.session_state.document_processed:

    st.markdown("---")

    st.subheader("💬 Ask Questions")

    question = st.text_input(
        "Enter your question:",
        placeholder="e.g. What is Cloud Computing?"
    )


    if st.button(
        "🔍 Ask Question",
        use_container_width=True
    ):

        if not question.strip():

            st.warning("Please enter a question.")

        else:

            try:

                with st.spinner(
                    "Searching document and generating answer..."
                ):

                    # Generate answer
                    response = st.session_state.rag_chain.invoke(
                        question
                    )

                    answer = response.content

                    # Retrieve source documents
                    source_docs = (
                        st.session_state.retriever.invoke(
                            question
                        )
                    )


                    # Store chat
                    st.session_state.chat_history.append(
                        {
                            "question": question,
                            "answer": answer
                        }
                    )


                # --------------------------------------------------
                # ANSWER
                # --------------------------------------------------

                st.subheader("🤖 Answer")

                st.write(answer)


                # --------------------------------------------------
                # SOURCES
                # --------------------------------------------------

                with st.expander(
                    "📚 View Retrieved Sources"
                ):

                    for i, doc in enumerate(
                        source_docs,
                        start=1
                    ):

                        st.markdown(
                            f"**Source {i}**"
                        )

                        page_number = (
                            doc.metadata.get(
                                "page",
                                "Unknown"
                            )
                        )

                        st.caption(
                            f"Page: {page_number + 1}"
                            if isinstance(
                                page_number,
                                int
                            )
                            else f"Page: {page_number}"
                        )

                        st.write(
                            doc.page_content[:500]
                        )

                        st.markdown("---")


            except Exception as e:

                st.error(
                    f"Unable to generate answer: {e}"
                )


# --------------------------------------------------
# CHAT HISTORY
# --------------------------------------------------

if st.session_state.chat_history:

    st.markdown("---")

    st.subheader("🕘 Previous Questions")

    for item in reversed(
        st.session_state.chat_history
    ):

        with st.expander(
            item["question"]
        ):

            st.write(item["answer"])


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("---")

st.caption(
    "AI Research Assistant • "
    "LangChain + Gemini + ChromaDB + RAG"
)
