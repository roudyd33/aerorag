import streamlit as st

st.set_page_config(
    page_title="AeroRAG",
    page_icon="✈️",
    layout="centered"
)

st.title("✈️ AeroRAG")
st.subheader("Aviation Knowledge Assistant")

st.write(
    "Ask questions about aviation using your "
    "aviation knowledge base."
)

st.info(
    "This is the web interface. In the next steps, "
    "we will connect it to your existing RAG system."
)

question = st.text_input(
    "Enter your aviation question:",
    placeholder="What does a V-n diagram show?"
)

if st.button("Ask AeroRAG"):
    if question.strip():
        st.write("Your question:", question)
        st.warning(
            "The RAG connection is not configured yet. "
            "We will connect your existing system next."
        )
    else:
        st.error("Please enter a question.")