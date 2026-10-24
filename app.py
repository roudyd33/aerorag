import os
from pathlib import Path

import streamlit as st
from pptx import Presentation
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

st.set_page_config(page_title="AeroRAG", page_icon="✈️")

st.title("✈️ AeroRAG")
st.subheader("Aviation Knowledge Assistant")
st.write("Ask questions based on your aviation presentation.")

api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("Gemini API key is not configured.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = api_key

@st.cache_resource
def build_vector_store():
    docs_dir = Path("documents")
    pptx_files = list(docs_dir.glob("*.pptx"))

    if not pptx_files:
        raise FileNotFoundError(
            "No PowerPoint file found in the documents folder."
        )

    presentation = Presentation(str(pptx_files[0]))
    docs = []

    for slide_number, slide in enumerate(presentation.slides, start=1):
        text = "\n".join(
            shape.text for shape in slide.shapes
            if shape.has_text_frame and shape.text.strip()
        )
        if text.strip():
            docs.append(Document(
                page_content=text,
                metadata={
                    "source": pptx_files[0].name,
                    "slide": slide_number
                }
            ))

    if not docs:
        raise ValueError("No readable text was found in the presentation.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key
    )

    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="aviation"
    )

question = st.text_input(
    "Your aviation question",
    placeholder="What does a V-n diagram show?"
)

if st.button("Ask AeroRAG"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        try:
            with st.spinner("Searching the presentation and generating an answer..."):
                vector_store = build_vector_store()
                results = vector_store.similarity_search(question, k=4)

                if not results:
                    st.warning("I couldn't find relevant information.")
                else:
                    context = "\n\n".join(
                        f"Slide {doc.metadata.get('slide', '?')}:\n"
                        f"{doc.page_content}"
                        for doc in results
                    )

                    llm = ChatGoogleGenerativeAI(
                        model="gemini-3.8-flash",
                        temperature=0,
                        google_api_key=api_key
                    )

                    prompt = f"""
Answer using only the presentation excerpts below.
If the excerpts do not contain the answer, say you cannot find it.
Mention slide numbers when possible.

Presentation excerpts:
{context}

Question: {question}
"""
                   response = llm.invoke(prompt)

answer_text = response.content

if isinstance(answer_text, list):
    answer_text = "\n".join(
        item.get("text", "")
        for item in answer_text
        if isinstance(item, dict) and item.get("type") == "text"
    )

st.markdown("### Answer")
st.markdown(answer_text)

with st.expander("Retrieved presentation excerpts"):
    st.write(context)
            st.caption(str(exc))
