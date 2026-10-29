import os
import time
import logging
import tempfile
from pathlib import Path

import streamlit as st
from pptx import Presentation

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    force=True,
)

logger = logging.getLogger("aerorag")

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AeroRAG | Aviation Intelligence",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# AVIATION THEME
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(
                circle at 15% 10%,
                rgba(0, 110, 255, 0.18),
                transparent 35%
            ),
            linear-gradient(
                135deg,
                #06101f 0%,
                #0a192d 50%,
                #071222 100%
            );
        color: #edf5ff;
    }

    [data-testid="stHeader"] {
        background: rgba(4, 12, 25, 0.85);
    }

    .block-container {
        max-width: 1100px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        padding: 2rem;
        border: 1px solid rgba(75, 155, 255, 0.35);
        border-radius: 20px;
        background: linear-gradient(
            135deg,
            rgba(15, 43, 76, 0.95),
            rgba(5, 19, 38, 0.95)
        );
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.25);
        margin-bottom: 1.5rem;
    }

    .hero h1 {
        color: #ffffff;
        font-size: 2.5rem;
        margin-bottom: 0.4rem;
    }

    .hero p {
        color: #a9c9ef;
        font-size: 1rem;
    }

    .status {
        display: inline-block;
        padding: 0.35rem 0.75rem;
        border-radius: 30px;
        border: 1px solid rgba(56, 189, 248, 0.45);
        background: rgba(14, 116, 180, 0.18);
        color: #7dd3fc;
        font-size: 0.8rem;
        margin-bottom: 0.8rem;
    }

    div[data-testid="stTextInput"] input {
        background: #0c1b30;
        color: #ffffff;
        border: 1px solid #315780;
        border-radius: 10px;
    }

    div[data-testid="stButton"] button {
        background: linear-gradient(90deg, #0877ee, #16a5ff);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        min-height: 2.8rem;
    }

    div[data-testid="stButton"] button:hover {
        border: 1px solid #8ac5ff;
        color: white;
    }

    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li {
        color: #e1edff;
        line-height: 1.7;
    }

    hr {
        border-color: rgba(100, 160, 230, 0.22);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="status">● AVIATION AI SYSTEM</div>
        <h1>✈️ AeroRAG</h1>
        <p>
            Aviation knowledge retrieval powered by
            Retrieval-Augmented Generation. Ask questions about
            your aviation presentations and receive answers
            grounded in retrieved content.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

EMBEDDING_MODEL = os.getenv(
    "GEMINI_EMBEDDING_MODEL",
    "models/gemini-embedding-001",
)


def get_api_key():
    """Read the Google API key from environment or Streamlit Secrets."""

    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        try:
            api_key = st.secrets.get("GOOGLE_API_KEY")
        except Exception:
            api_key = None

    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is missing. Add it under "
            "Streamlit Cloud > App settings > Secrets."
        )

    return api_key


def find_presentation():
    """Find the aviation PowerPoint committed to the repository."""

    project_dir = Path(__file__).resolve().parent

    candidates = [
        project_dir / "documents" /
        "AAFC-POF-23-AerodynamicLimitationsv3.1.pptx",
        project_dir / "AAFC-POF-23-AerodynamicLimitationsv3.1.pptx",
    ]

    # Also accept another PPTX placed directly in documents/.
    documents_dir = project_dir / "documents"

    if documents_dir.exists():
        candidates.extend(
            sorted(documents_dir.glob("*.pptx"))
        )

    # Also accept a PPTX in the project root.
    candidates.extend(sorted(project_dir.glob("*.pptx")))

    for candidate in candidates:
        if candidate.is_file():
            logger.info("Using presentation: %s", candidate.name)
            return candidate

    raise FileNotFoundError(
        "No PowerPoint presentation was found. Upload your "
        "aviation .pptx file to the GitHub repository's "
        "documents/ folder and commit the file."
    )


def load_presentation_documents(presentation_path):
    """Extract slide text and retain slide numbers."""

    presentation = Presentation(str(presentation_path))
    documents = []

    for slide_number, slide in enumerate(
        presentation.slides,
        start=1,
    ):
        slide_text = []

        for shape in slide.shapes:
            if getattr(shape, "has_text_frame", False):
                text = shape.text.strip()

                if text:
                    slide_text.append(text)

            if getattr(shape, "has_table", False):
                for row in shape.table.rows:
                    cells = [
                        cell.text.strip()
                        for cell in row.cells
                        if cell.text.strip()
                    ]

                    if cells:
                        slide_text.append(" | ".join(cells))

        combined_text = "\n".join(slide_text).strip()

        if combined_text:
            documents.append(
                Document(
                    page_content=combined_text,
                    metadata={
                        "slide": slide_number,
                        "source": presentation_path.name,
                    },
                )
            )

    if not documents:
        raise RuntimeError(
            "The presentation was found, but no readable slide "
            "text was extracted."
        )

    logger.info(
        "Extracted text from %d slides",
        len(documents),
    )

    return documents


# ============================================================
# BUILD CLOUD RAG DATABASE
# ============================================================

@st.cache_resource(show_spinner=False)
def load_rag_components():
    """
    Build a cloud-side Chroma database from the PPTX.

    This deliberately creates a new database using Google
    embeddings instead of reusing a potentially incompatible
    local Ollama-embedding database.
    """

    api_key = get_api_key()
    presentation_path = find_presentation()

    slide_documents = load_presentation_documents(
        presentation_path
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )

    chunks = splitter.split_documents(slide_documents)

    if not chunks:
        raise RuntimeError(
            "No text chunks could be created from the presentation."
        )

    logger.info("Created %d presentation chunks", len(chunks))

    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=api_key,
    )

    # Use a new temporary database for this running app instance.
    # It is recreated from the PPTX when the app starts afresh.
    database_path = tempfile.mkdtemp(prefix="aerorag_chroma_")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="aviation",
        persist_directory=database_path,
    )

    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=api_key,
        temperature=0.2,
    )

    logger.info(
        "AeroRAG database and language model initialized successfully"
    )

    return vectorstore, llm


# ============================================================
# QUESTION INPUT
# ============================================================

st.markdown("### Aviation Knowledge Assistant")

question = st.text_input(
    "Ask an aviation question",
    placeholder="Example: What does the maneuver envelope represent?",
    key="aviation_question",
)

ask_button = st.button(
    "Ask AeroRAG",
    use_container_width=True,
)

# ============================================================
# RETRIEVAL AND ANSWER GENERATION
# ============================================================

if ask_button:
    if not question.strip():
        st.warning("Please enter a question first.")

    else:
        try:
            logger.info("Received a RAG question")

            with st.spinner(
                "Preparing aviation knowledge database..."
            ):
                vectorstore, llm = load_rag_components()

            with st.spinner("Searching aviation knowledge..."):
                retrieval_start = time.time()

                retrieved_docs = vectorstore.similarity_search(
                    question,
                    k=4,
                )

                logger.info(
                    "Retrieved %d documents in %.2f seconds",
                    len(retrieved_docs),
                    time.time() - retrieval_start,
                )

            if not retrieved_docs:
                st.warning(
                    "No relevant presentation excerpts were found."
                )
                st.stop()

            context_parts = []

            for index, doc in enumerate(
                retrieved_docs,
                start=1,
            ):
                metadata = doc.metadata or {}
                slide = metadata.get("slide")

                label = (
                    f"Slide {slide}"
                    if slide is not None
                    else f"Excerpt {index}"
                )

                context_parts.append(
                    f"{label}:\n{doc.page_content}"
                )

            context = "\n\n".join(context_parts)

            prompt = f"""
You are AeroRAG, an aviation knowledge assistant.

Answer the user's question using only the presentation excerpts
provided below.

Instructions:
- Give a clear, direct, professional answer.
- Mention slide numbers when possible.
- Do not invent aviation facts not supported by the excerpts.
- If the excerpts do not contain enough information, say so.
- Keep the answer easy to understand.

Presentation excerpts:
{context}

Question:
{question}

Answer:
"""

            with st.spinner("Generating your answer..."):
                generation_start = time.time()

                try:
                    response = llm.invoke(prompt)

                except Exception:
                    logger.exception(
                        "Error while generating the RAG answer"
                    )
                    st.error(
                        "The AI model could not generate an answer. "
                        "Check the model name, API key, and Cloud logs."
                    )
                    st.stop()

                logger.info(
                    "Answer generated in %.2f seconds",
                    time.time() - generation_start,
                )

            answer_text = response.content

            if isinstance(answer_text, list):
                answer_text = "\n".join(
                    item.get("text", "")
                    if isinstance(item, dict)
                    else str(item)
                    for item in answer_text
                )

            if not isinstance(answer_text, str):
                answer_text = str(answer_text)

            st.markdown("### Answer")
            st.markdown(answer_text)

            with st.expander("Retrieved presentation excerpts"):
                for index, doc in enumerate(
                    retrieved_docs,
                    start=1,
                ):
                    metadata = doc.metadata or {}
                    slide = metadata.get("slide")

                    heading = (
                        f"Slide {slide}"
                        if slide is not None
                        else f"Excerpt {index}"
                    )

                    st.markdown(f"**{heading}**")
                    st.write(doc.page_content)

            logger.info("Question completed successfully")

        except Exception as exc:
            logger.exception(
                "Unexpected error in the AeroRAG workflow"
            )
            st.error("AeroRAG could not complete your request.")
            st.caption(str(exc))

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AeroRAG | Aviation Intelligence | "
    "Retrieval-Augmented Generation"
)
