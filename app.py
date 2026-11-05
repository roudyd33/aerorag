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
# 1. LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    force=True,
)

logger = logging.getLogger("aerorag")

# ============================================================
# 2. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AeroRAG | Aviation Intelligence",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# 3. AVIATION THEME
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
# 4. HEADER
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
# 5. CONFIGURATION
# ============================================================

MODEL_NAME = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.8-flash",
)

EMBEDDING_MODEL = os.getenv(
    "GEMINI_EMBEDDING_MODEL",
    "models/gemini-embedding-001",
)

RETRIEVAL_COUNT = 4
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# ============================================================
# 6. ERROR CLASSIFICATION
# ============================================================

def is_temporary_error(exc):
    """
    Identify errors that may recover after waiting.

    Note: Some 429 quota errors are not temporary.
    Those should be resolved through quota or billing settings.
    """

    message = str(exc).upper()

    permanent_quota_messages = [
        "QUOTA EXCEEDED",
        "PER DAY",
        "DAILY LIMIT",
        "BILLING NOT ENABLED",
    ]

    if any(item in message for item in permanent_quota_messages):
        return False

    temporary_indicators = [
        "503",
        "UNAVAILABLE",
        "429",
        "RESOURCE_EXHAUSTED",
        "500",
        "INTERNAL",
        "502",
        "504",
        "DEADLINE_EXCEEDED",
        "TIMEOUT",
        "CONNECTION RESET",
        "SERVICE UNAVAILABLE",
    ]

    return any(
        item in message
        for item in temporary_indicators
    )


def retry_operation(
    operation,
    operation_name,
    max_attempts=5,
):
    """
    Retry an operation after temporary API failures.

    Wait schedule:
    3, 6, 12, 24 seconds.
    """

    for attempt in range(max_attempts):

        try:
            logger.info(
                "%s: attempt %s/%s",
                operation_name,
                attempt + 1,
                max_attempts,
            )

            result = operation()

            logger.info(
                "%s completed successfully",
                operation_name,
            )

            return result

        except Exception as exc:

            if (
                not is_temporary_error(exc)
                or attempt == max_attempts - 1
            ):
                logger.exception(
                    "%s failed",
                    operation_name,
                )
                raise

            wait_seconds = 3 * (2 ** attempt)

            logger.warning(
                "%s temporarily failed. "
                "Retrying in %s seconds. "
                "Attempt %s/%s. Error: %s",
                operation_name,
                wait_seconds,
                attempt + 1,
                max_attempts,
                str(exc),
            )

            time.sleep(wait_seconds)

    raise RuntimeError(
        f"{operation_name} failed after all retry attempts."
    )


# ============================================================
# 7. API KEY
# ============================================================

def get_api_key():
    """Get Google API key from Streamlit Secrets or environment."""

    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        try:
            api_key = st.secrets.get("GOOGLE_API_KEY")
        except Exception:
            api_key = None

    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is missing. "
            "Add it under Streamlit Cloud > App settings > Secrets."
        )

    return api_key


# ============================================================
# 8. FIND POWERPOINT
# ============================================================

def find_presentation():
    """Find the aviation PowerPoint."""

    project_dir = Path(__file__).resolve().parent

    documents_dir = project_dir / "documents"

    candidates = [
        documents_dir /
        "AAFC-POF-23-AerodynamicLimitationsv3.1.pptx",

        project_dir /
        "AAFC-POF-23-AerodynamicLimitationsv3.1.pptx",
    ]

    if documents_dir.exists():
        candidates.extend(
            sorted(documents_dir.glob("*.pptx"))
        )

    candidates.extend(
        sorted(project_dir.glob("*.pptx"))
    )

    for candidate in candidates:

        if candidate.is_file():

            logger.info(
                "Presentation found: %s",
                candidate.name,
            )

            return candidate

    raise FileNotFoundError(
        "No PowerPoint presentation was found. "
        "Check that your .pptx file is committed to "
        "the repository's documents folder."
    )


# ============================================================
# 9. EXTRACT POWERPOINT CONTENT
# ============================================================

def load_presentation_documents(presentation_path):
    """Extract readable slide text and preserve slide numbers."""

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
                        slide_text.append(
                            " | ".join(cells)
                        )

        combined_text = "\n".join(
            slide_text
        ).strip()

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
            "The PowerPoint was found, but no readable "
            "slide text could be extracted."
        )

    logger.info(
        "Extracted text from %s slides",
        len(documents),
    )

    return documents


# ============================================================
# 10. BUILD RAG COMPONENTS
# ============================================================

@st.cache_resource(show_spinner=False)
def load_rag_components():
    """
    Build the Chroma database and initialize Gemini.

    The resource is cached during the running Streamlit process.
    """

    api_key = get_api_key()

    presentation_path = find_presentation()

    slide_documents = load_presentation_documents(
        presentation_path
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = splitter.split_documents(
        slide_documents
    )

    if not chunks:

        raise RuntimeError(
            "No text chunks could be created."
        )

    logger.info(
        "Created %s presentation chunks",
        len(chunks),
    )

    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=api_key,
    )

    def create_vectorstore():
        """
        Create a fresh temporary database for each build attempt.
        """

        database_path = tempfile.mkdtemp(
            prefix="aerorag_chroma_"
        )

        return Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name="aviation",
            persist_directory=database_path,
        )

    vectorstore = retry_operation(
        create_vectorstore,
        operation_name="Chroma database creation",
        max_attempts=4,
    )

    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=api_key,
        temperature=0.2,
    )

    logger.info(
        "AeroRAG components initialized successfully"
    )

    return vectorstore, llm


# ============================================================
# 11. RETRIEVE RELEVANT INFORMATION
# ============================================================

def retrieve_documents(vectorstore, question):
    """Search ChromaDB with retry handling."""

    return retry_operation(
        lambda: vectorstore.similarity_search(
            question,
            k=RETRIEVAL_COUNT,
        ),
        operation_name="Document retrieval",
        max_attempts=4,
    )


# ============================================================
# 12. GENERATE ANSWER
# ============================================================

def generate_answer_with_retry(llm, prompt):
    """Generate a Gemini answer with temporary-error retries."""

    return retry_operation(
        lambda: llm.invoke(prompt),
        operation_name="Gemini answer generation",
        max_attempts=5,
    )


# ============================================================
# 13. PREPARE CONTEXT
# ============================================================

def prepare_context(retrieved_docs):
    """Prepare retrieved excerpts and preserve slide references."""

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

    return "\n\n".join(context_parts)


# ============================================================
# 14. CLEAN MODEL RESPONSE
# ============================================================

def clean_answer(response):
    """Convert Gemini response content into readable text."""

    answer = response.content

    if isinstance(answer, list):

        text_parts = []

        for item in answer:

            if isinstance(item, dict):

                if item.get("type") == "text":
                    text_parts.append(
                        item.get("text", "")
                    )

            else:
                text_parts.append(str(item))

        answer = "\n".join(text_parts)

    if not isinstance(answer, str):
        answer = str(answer)

    answer = answer.strip()

    if not answer:
        raise RuntimeError(
            "Gemini returned an empty answer."
        )

    return answer


# ============================================================
# 15. QUESTION INPUT
# ============================================================

st.markdown(
    "### Aviation Knowledge Assistant"
)

question = st.text_input(
    "Ask an aviation question",
    placeholder=(
        "Example: What does the maneuver envelope represent?"
    ),
    key="aviation_question",
)

ask_button = st.button(
    "Ask AeroRAG",
    use_container_width=True,
)

# ============================================================
# 16. MAIN RAG WORKFLOW
# ============================================================

if ask_button:

    if not question.strip():

        st.warning(
            "Please enter an aviation question first."
        )

    else:

        logger.info(
            "New question received"
        )

        retrieved_docs = []

        # ----------------------------------------------------
        # STEP A: INITIALIZE DATABASE AND MODELS
        # ----------------------------------------------------

        try:

            with st.spinner(
                "Preparing aviation knowledge database..."
            ):

                vectorstore, llm = load_rag_components()

        except Exception as exc:

            logger.exception(
                "RAG initialization failed"
            )

            st.error(
                "AeroRAG could not initialize its database "
                "or AI services."
            )

            if is_temporary_error(exc):

                st.warning(
                    "A Google AI service may be temporarily "
                    "unavailable. Please try again later."
                )

            else:

                st.info(
                    "Check your API key, PowerPoint file, "
                    "dependencies, and Streamlit Cloud logs."
                )

            st.stop()

        # ----------------------------------------------------
        # STEP B: RETRIEVE DOCUMENTS
        # ----------------------------------------------------

        try:

            with st.spinner(
                "Searching aviation knowledge..."
            ):

                retrieval_start = time.time()

                retrieved_docs = retrieve_documents(
                    vectorstore,
                    question,
                )

                logger.info(
                    "Retrieved %s documents in %.2f seconds",
                    len(retrieved_docs),
                    time.time() - retrieval_start,
                )

        except Exception as exc:

            logger.exception(
                "Document retrieval failed"
            )

            st.error(
                "AeroRAG could not search the aviation database."
            )

            if is_temporary_error(exc):

                st.warning(
                    "The Google embedding service may be "
                    "temporarily unavailable. Please try again."
                )

            else:

                st.info(
                    "Check the embedding configuration and "
                    "database initialization in the logs."
                )

            st.stop()

        # ----------------------------------------------------
        # STEP C: CHECK RETRIEVAL RESULTS
        # ----------------------------------------------------

        if not retrieved_docs:

            st.warning(
                "No relevant presentation excerpts were found."
            )

            st.stop()

        context = prepare_context(
            retrieved_docs
        )

        # ----------------------------------------------------
        # STEP D: DISPLAY RETRIEVED INFORMATION
        # ----------------------------------------------------

        with st.expander(
            "Retrieved presentation excerpts",
            expanded=False,
        ):

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

                st.markdown(
                    f"**{heading}**"
                )

                st.write(
                    doc.page_content
                )

        # ----------------------------------------------------
        # STEP E: PREPARE PROMPT
        # ----------------------------------------------------

        prompt = f"""
You are AeroRAG, an aviation knowledge assistant.

Your task is to answer the user's question using
ONLY the presentation excerpts provided below.

RULES:

1. Give a clear, direct, professional answer.
2. Use only information supported by the excerpts.
3. Do not invent aviation facts.
4. Mention slide numbers whenever possible.
5. If the excerpts do not contain enough information,
   clearly say that the information could not be found.
6. If the question is ambiguous, explain the ambiguity.
7. Keep the answer understandable and concise.

PRESENTATION EXCERPTS:

{context}

USER QUESTION:

{question}

ANSWER:
"""

        # ----------------------------------------------------
        # STEP F: GENERATE ANSWER
        # ----------------------------------------------------

        try:

            with st.spinner(
                "Generating your aviation answer..."
            ):

                generation_start = time.time()

                response = generate_answer_with_retry(
                    llm,
                    prompt,
                )

                answer_text = clean_answer(
                    response
                )

                logger.info(
                    "Answer generated in %.2f seconds",
                    time.time() - generation_start,
                )

            # ------------------------------------------------
            # STEP G: DISPLAY FINAL ANSWER
            # ------------------------------------------------

            st.markdown(
                "### Answer"
            )

            st.markdown(
                answer_text
            )

            st.success(
                "Answer generated successfully."
            )

            logger.info(
                "Question completed successfully"
            )

        except Exception as exc:

            logger.exception(
                "Answer generation failed"
            )

            st.warning(
                "The aviation information was successfully "
                "retrieved, but the AI model could not "
                "generate an answer."
            )

            if is_temporary_error(exc):

                st.info(
                    "Gemini is temporarily unavailable or "
                    "experiencing high demand. "
                    "Your retrieved aviation excerpts are "
                    "still available above. Please try again later."
                )

            else:

                st.error(
                    "An error occurred during answer generation."
                )

                st.caption(
                    "Check the Streamlit Cloud logs for details."
                )


# ============================================================
# 17. FOOTER
# ============================================================

st.divider()

st.caption(
    "AeroRAG | Aviation Intelligence | "
    "Retrieval-Augmented Generation"
)
