import os
import time
import logging
from pathlib import Path

import streamlit as st
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate

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
        <h1>✈ AeroRAG</h1>
        <p>
            Aviation knowledge retrieval powered by Retrieval-Augmented
            Generation. Ask questions about your aviation presentations
            and receive answers grounded in retrieved content.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# CONFIGURATION
# ============================================================

DB_PATH = os.getenv("CHROMA_DB_PATH", "chroma_db_local")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "aviation")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
EMBEDDING_MODEL = os.getenv(
    "GEMINI_EMBEDDING_MODEL",
    "models/gemini-embedding-001",
)

# ============================================================
# LOAD RAG COMPONENTS
# ============================================================

@st.cache_resource
def load_rag_components():
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        try:
            api_key = st.secrets.get("GOOGLE_API_KEY")
        except Exception:
            api_key = None

    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is not configured. Add it to "
            "Streamlit Cloud secrets."
        )

    db_directory = Path(DB_PATH)

    if not db_directory.exists():
        raise RuntimeError(
            f"Chroma database directory not found: {DB_PATH}"
        )

    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=api_key,
    )

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=DB_PATH,
        embedding_function=embeddings,
    )

    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=api_key,
        temperature=0.2,
    )

    logger.info("AeroRAG components initialized successfully")

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

ask_button = st.button("Ask AeroRAG", use_container_width=True)

# ============================================================
# RETRIEVAL AND ANSWER GENERATION
# ============================================================

if ask_button:
    if not question.strip():
        st.warning("Please enter a question first.")

    else:
        try:
            logger.info("Received a RAG question")

            with st.spinner("Searching aviation knowledge..."):
                vectorstore, llm = load_rag_components()

                retrieval_start = time.time()

                retrieved_docs = vectorstore.similarity_search(
                    question,
                    k=4,
                )

                retrieval_elapsed = time.time() - retrieval_start

                logger.info(
                    "Retrieved %d documents in %.2f seconds",
                    len(retrieved_docs),
                    retrieval_elapsed,
                )

                if not retrieved_docs:
                    st.warning(
                        "No relevant presentation excerpts were found."
                    )
                    st.stop()

                context_parts = []

                for index, doc in enumerate(retrieved_docs, start=1):
                    metadata = doc.metadata or {}
                    slide = metadata.get("slide")

                    if slide is None:
                        slide = metadata.get("slide_number")

                    slide_label = (
                        f"Slide {slide}"
                        if slide is not None
                        else f"Excerpt {index}"
                    )

                    context_parts.append(
                        f"{slide_label}:\n{doc.page_content}"
                    )

                context = "\n\n".join(context_parts)

                prompt = ChatPromptTemplate.from_template(
                    """
You are AeroRAG, an aviation knowledge assistant.

Answer the user's question using only the presentation excerpts
provided below.

Instructions:
- Give a clear, direct, professional answer.
- Mention slide numbers when possible.
- Do not invent aviation facts that are not supported by the excerpts.
- If the excerpts do not contain enough information, say so.
- Keep the answer easy to understand.

Presentation excerpts:
{context}

Question:
{question}
"""
                )

                formatted_prompt = prompt.format_messages(
                    context=context,
                    question=question,
                )

                generation_start = time.time()

                try:
                    response = llm.invoke(formatted_prompt)

                    generation_elapsed = (
                        time.time() - generation_start
                    )

                    logger.info(
                        "RAG answer generated successfully in %.2f seconds",
                        generation_elapsed,
                    )

                except Exception:
                    logger.exception(
                        "Error while generating RAG answer"
                    )
                    st.error(
                        "The app encountered an error while generating "
                        "the answer. Check the application logs."
                    )
                    st.stop()

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

                        if slide is None:
                            slide = metadata.get("slide_number")

                        heading = (
                            f"Slide {slide}"
                            if slide is not None
                            else f"Excerpt {index}"
                        )

                        st.markdown(f"**{heading}**")
                        st.write(doc.page_content)

                logger.info(
                    "Question completed successfully"
                )

        except Exception:
            logger.exception(
                "Unexpected error in the AeroRAG request workflow"
            )
            st.error(
                "AeroRAG encountered an unexpected error. "
                "Check the Streamlit Cloud logs for details."
            )

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AeroRAG | Aviation Intelligence | "
    "Retrieval-Augmented Generation"
)
