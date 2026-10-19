import os
from pathlib import Path

import streamlit as st
from pptx import Presentation
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)

st.set_page_config(
   # ===== AVIATION BACKGROUND THEME =====

st.markdown("""
<style>
/* Full website: dark aviation gradient */
.stApp {
    background:
        radial-gradient(
            ellipse at 15% 10%,
            rgba(0, 119, 255, 0.22),
            transparent 42%
        ),
        radial-gradient(
            ellipse at 85% 85%,
            rgba(0, 76, 180, 0.18),
            transparent 45%
        ),
        linear-gradient(
            135deg,
            #050B16 0%,
            #0A192B 50%,
            #071426 100%
        );

    background-attachment: fixed;
}

/* Aviation-inspired sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #09172A 0%,
        #102844 100%
    );

    border-right: 1px solid rgba(73, 153, 255, 0.25);
}

/* Transparent top bar */
.stApp > header {
    background: transparent;
}

/* Blue accent headings */
.stApp h1,
.stApp h2,
.stApp h3 {
    color: #8CCBFF;
}

/* Subtle blue links */
.stApp a {
    color: #68B6FF;
}

/* Chat panels */
.stChatMessage {
    background: rgba(13, 32, 55, 0.72);
    border: 1px solid rgba(77, 145, 218, 0.20);
    border-radius: 14px;
}
</style>
""", unsafe_allow_html=True)

# ===== END AVIATION BACKGROUND ===== page_title="AeroRAG",
    page_icon="✈️",
    layout="wide",
)

st.title("✈️ AeroRAG")
st.subheader("AI-powered aviation knowledge assistant")
st.write(
    "Ask questions about your aviation presentation. "
    "AeroRAG retrieves relevant information and generates an answer."
)

# Read the API key from Streamlit Secrets
try:
    api_key = st.secrets.get("GOOGLE_API_KEY")
except Exception:
    api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error(
        "Google API key is missing. Add GOOGLE_API_KEY "
        "in your Streamlit app Secrets."
    )
    st.stop()

# Find PowerPoint files
documents_folder = Path("documents")
pptx_files = list(documents_folder.glob("*.pptx"))

if not pptx_files:
    st.error(
        "No PowerPoint file was found in the documents folder. "
        "Upload your aviation .pptx file to the documents folder on GitHub."
    )
    st.stop()


@st.cache_resource
def build_vector_store(file_path, key):
    # Extract text from the PowerPoint slides
    presentation = Presentation(file_path)
    slide_texts = []

    for slide_number, slide in enumerate(presentation.slides, start=1):
        slide_content = []

        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text.strip():
                slide_content.append(shape.text.strip())

        if slide_content:
            slide_texts.append(
                f"Slide {slide_number}\n" + "\n".join(slide_content)
            )

    if not slide_texts:
        raise ValueError(
            "No readable text was found in the PowerPoint presentation."
        )

    # Split slide text into searchable chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
    )

    chunks = splitter.create_documents(slide_texts)

    # Create embeddings and the searchable vector database
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=key,
    )

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    return vector_store


# Build the knowledge base
try:
    with st.spinner("Preparing the aviation knowledge base..."):
        vector_store = build_vector_store(
            str(pptx_files[0]),
            api_key,
        )
except Exception as error:
    st.error("Could not prepare the knowledge base.")
    st.caption(f"Technical details: {error}")
    st.stop()

# User question
question = st.text_input(
    "Your aviation question",
    placeholder="Example: What does a V-n diagram show?",
)

if st.button("Ask AeroRAG", type="primary"):
    if not question.strip():
        st.warning("Please enter a question first.")
    else:
        try:
            with st.spinner("Searching the presentation and generating an answer..."):
                # Retrieve relevant presentation excerpts
                retriever = vector_store.as_retriever(
                    search_kwargs={"k": 4}
                )
                retrieved_docs = retriever.invoke(question)

                if not retrieved_docs:
                    st.warning(
                        "No relevant information was found in the presentation."
                    )
                    st.stop()

                context = "\n\n".join(
                    doc.page_content for doc in retrieved_docs
                )

                # Generate a clear answer grounded in the retrieved context
                model = ChatGoogleGenerativeAI(
                    model="gemini-3.8-flash",
                    google_api_key=api_key,
                    temperature=0.2,
                )

                prompt = f"""
You are AeroRAG, an aviation knowledge assistant.

Answer the user's question using only the presentation excerpts below.
Write a clear, direct answer in plain text or simple Markdown.
Do not return JSON, metadata, signatures, or technical formatting.
If the excerpts do not contain the answer, say that the presentation
does not provide enough information.

PRESENTATION EXCERPTS:
{context}

USER QUESTION:
{question}

Give the answer only.
"""

                response = model.invoke(prompt)

                # Convert Gemini's response into clean readable text
                content = response.content

                if isinstance(content, str):
                    answer = content
                elif isinstance(content, list):
                    text_parts = []

                    for item in content:
                        if isinstance(item, str):
                            text_parts.append(item)
                        elif isinstance(item, dict):
                            if item.get("type") == "text" and item.get("text"):
                                text_parts.append(item["text"])

                    answer = "\n".join(text_parts)
                else:
                    answer = str(content)

                st.markdown("### Answer")
                st.markdown(answer.strip() or "No text answer was returned.")

                with st.expander("View retrieved presentation excerpts"):
                    for index, doc in enumerate(retrieved_docs, start=1):
                        st.markdown(f"**Excerpt {index}**")
                        st.write(doc.page_content)

        except Exception as error:
            st.error("Sorry, AeroRAG could not answer that question.")
            st.caption(f"Technical details: {error}")
