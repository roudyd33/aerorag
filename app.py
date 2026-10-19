import os
import streamlit as st

from pptx import Presentation
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)
from langchain_core.prompts import ChatPromptTemplate


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="AeroRAG",
    page_icon="✈️",
    layout="wide",
)


# ==========================================
# DARK AVIATION BACKGROUND
# ==========================================

st.markdown("""
<style>

/* Entire website background */
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
    color: #EAF2FF;
}

/* Transparent top toolbar */
.stApp > header {
    background: transparent;
}

/* Sidebar gradient */
section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #09172A 0%,
        #102844 100%
    );

    border-right: 1px solid rgba(73, 153, 255, 0.25);
}

/* Headings */
.stApp h1,
.stApp h2,
.stApp h3 {
    color: #8CCBFF;
}

/* General text */
.stApp p,
.stApp label {
    color: #D5E3F5;
}

/* Links */
.stApp a {
    color: #68B6FF;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(
        135deg,
        #0878E8,
        #1555B5
    );
    color: white;
    border: 1px solid rgba(113, 184, 255, 0.35);
    border-radius: 10px;
    font-weight: 600;
}

.stButton > button:hover {
    background: linear-gradient(
        135deg,
        #168DFF,
        #216DDA
    );
    color: white;
    border-color: #70BBFF;
}

/* Chat messages */
.stChatMessage {
    background: rgba(13, 32, 55, 0.72);
    border: 1px solid rgba(77, 145, 218, 0.20);
    border-radius: 14px;
}

/* Text input */
.stChatInput textarea,
.stTextInput input {
    background-color: rgba(8, 22, 40, 0.90);
    color: #F0F6FF;
    border: 1px solid #294765;
    border-radius: 10px;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background-color: rgba(13, 32, 55, 0.55);
    border: 1px dashed #31577E;
    border-radius: 12px;
}

/* Dividers */
hr {
    border-color: rgba(86, 151, 220, 0.20);
}

.stMainBlockContainer {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# AERORAG HEADER
# ==========================================

st.title("✈ AeroRAG")

st.subheader(
    "AI-powered aviation knowledge assistant"
)

st.write(
    "Ask questions about your aviation presentation. "
    "AeroRAG retrieves relevant information and generates "
    "an answer based on the available material."
)

st.divider()


# ==========================================
# API KEY
# ==========================================

try:
    api_key = st.secrets["GOOGLE_API_KEY"]
except Exception:
    api_key = os.getenv("GOOGLE_API_KEY")


# ==========================================
# EXISTING AVIATION DATABASE
# ==========================================

DB_PATH = "chroma_db"
COLLECTION_NAME = "aerorag"


@st.cache_resource
def load_database():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key,
    )

    database = Chroma(
        persist_directory=DB_PATH,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )

    return database


# ==========================================
# GEMINI MODEL
# ==========================================

@st.cache_resource
def load_model():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.1,
    )


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:
    st.title("✈ AeroRAG")

    st.caption("AVIATION INTELLIGENCE SYSTEM")

    st.divider()

    st.markdown("### About")

    st.write(
        "A Retrieval-Augmented Generation system "
        "designed to answer questions using aviation documents."
    )

    st.divider()

    st.markdown("### Architecture")

    st.write("• Document retrieval")
    st.write("• Vector database")
    st.write("• Gemini language model")
    st.write("• Source-grounded answers")


# ==========================================
# CHAT HISTORY
# ==========================================

if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ==========================================
# ASK AERORAG
# ==========================================

question = st.chat_input(
    "Ask an aviation question..."
)

if question:

    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        try:
            if not api_key:
                st.error(
                    "Google API key not found. "
                    "Configure GOOGLE_API_KEY in Streamlit Secrets."
                )
                st.stop()

            with st.spinner(
                "Searching aviation knowledge..."
            ):

                database = load_database()
                model = load_model()

                retrieved_docs = database.similarity_search(
                    question,
                    k=4,
                )

                if not retrieved_docs:
                    answer = (
                        "I couldn't find relevant information "
                        "in the available aviation documents."
                    )

                else:
                    context_parts = []

                    for index, doc in enumerate(
                        retrieved_docs,
                        start=1,
                    ):
                        source = doc.metadata.get(
                            "source",
                            "Unknown",
                        )

                        page = doc.metadata.get(
                            "page",
                            "Unknown",
                        )

                        context_parts.append(
                            f"[Source {index}: {source}, "
                            f"page/slide {page}]\n"
                            f"{doc.page_content}"
                        )

                    context = "\n\n".join(context_parts)

                    prompt = ChatPromptTemplate.from_messages([
                        (
                            "system",
                            """
You are AeroRAG, an aviation knowledge assistant.

Answer using the provided aviation context.

Rules:
- Ground factual answers in the supplied context.
- Do not invent information or citations.
- If the context is insufficient, say so.
- Explain concepts clearly and professionally.
- Do not replace official aviation manuals,
  instructor guidance, or operational procedures.
                            """,
                        ),
                        (
                            "human",
                            """
Aviation context:

{context}

Question:
{question}

Answer clearly and cite the relevant source numbers.
                            """,
                        ),
                    ])

                    response = (
                        prompt | model
                    ).invoke({
                        "context": context,
                        "question": question,
                    })

                    answer = response.content

                    if isinstance(answer, list):
                        answer = "\n".join(
                            str(item.get("text", item))
                            if isinstance(item, dict)
                            else str(item)
                            for item in answer
                        )

                    with st.expander("Retrieved sources"):
                        for index, doc in enumerate(
                            retrieved_docs,
                            start=1,
                        ):
                            source = doc.metadata.get(
                                "source",
                                "Unknown",
                            )

                            page = doc.metadata.get(
                                "page",
                                "Unknown",
                            )

                            st.markdown(
                                f"**Source {index}:** "
                                f"{source} — page/slide {page}"
                            )

                            st.write(doc.page_content[:1000])

            st.markdown(answer)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
            })

        except Exception as error:
            st.error(
                "AeroRAG encountered an error. "
                "Check the database, embeddings configuration, "
                "API key, and deployment logs."
            )

            st.code(str(error))


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "AeroRAG | Aviation Intelligence powered by RAG"
)
