import streamlit as st
from google import genai
from dotenv import load_dotenv
from pypdf import PdfReader
import chromadb
import os


# --------------------------------------------------
# LOAD ENVIRONMENT VARIABLES
# --------------------------------------------------

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    st.error("GEMINI_API_KEY is missing in the .env file.")
    st.stop()


# --------------------------------------------------
# GEMINI CLIENT
# --------------------------------------------------

client = genai.Client(api_key=API_KEY)


# --------------------------------------------------
# CHROMADB
# --------------------------------------------------

chroma_client = chromadb.Client()

collection = chroma_client.get_or_create_collection(
    name="gemini_rag_collection"
)


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Gemini RAG",
    page_icon="📚",
    layout="wide"
)


# --------------------------------------------------
# PROFESSIONAL CSS
# --------------------------------------------------

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

/* Main Background */
.stApp {
    background: linear-gradient(
        135deg,
        #e8f7ff 0%,
        #fdf0f6 50%,
        #eaf8ff 100%
    );
}

/* Main Container */
.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Main Title */
.main-title {
    text-align: center;
    color: #087ea4;
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 5px;
}

/* Subtitle */
.subtitle {
    text-align: center;
    color: #555555;
    font-size: 17px;
    margin-bottom: 30px;
}

/* Section Titles */
h1, h2, h3 {
    color: #087ea4 !important;
}

/* Normal Text */
p, label, .stMarkdown {
    color: #222222;
}

/* File Uploader */
[data-testid="stFileUploader"] {
    background: rgba(255, 255, 255, 0.85);
    border: 2px dashed #70c9e8;
    border-radius: 18px;
    padding: 20px;
    box-shadow: 0 5px 20px rgba(8, 126, 164, 0.10);
}

/* File uploader text */
[data-testid="stFileUploader"] section {
    background: transparent;
}

/* Text Input */
.stTextInput > div > div > input {
    background-color: #ffffff !important;
    color: #222222 !important;
    border: 2px solid #b9dff0 !important;
    border-radius: 12px !important;
    padding: 13px 15px !important;
    font-size: 15px !important;
}

.stTextInput > div > div > input:focus {
    border-color: #55b9df !important;
    box-shadow: 0 0 0 2px rgba(85, 185, 223, 0.15) !important;
}

/* Buttons */
.stButton > button {
    width: 100%;
    border-radius: 12px;
    border: none;
    padding: 12px 20px;
    background: linear-gradient(
        90deg,
        #55b9df,
        #f3a6c7
    );
    color: #ffffff;
    font-size: 15px;
    font-weight: 700;
    transition: all 0.3s ease;
    box-shadow: 0 5px 15px rgba(8, 126, 164, 0.18);
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(8, 126, 164, 0.25);
    background: linear-gradient(
        90deg,
        #3daed7,
        #ed8fb6
    );
}

/* Success Message */
[data-testid="stAlert"] {
    border-radius: 12px;
}

/* Info Cards */
.info-card {
    background: rgba(255, 255, 255, 0.90);
    border-left: 5px solid #55b9df;
    border-radius: 15px;
    padding: 18px 22px;
    margin: 15px 0;
    box-shadow: 0 5px 18px rgba(8, 126, 164, 0.10);
}

/* Answer Card */
.answer-card {
    background: linear-gradient(
        135deg,
        #ffffff,
        #fff3f8
    );
    border: 1px solid #f2bfd5;
    border-left: 6px solid #f3a6c7;
    border-radius: 16px;
    padding: 22px;
    margin-top: 15px;
    box-shadow: 0 6px 20px rgba(220, 130, 165, 0.12);
    color: #222222;
    font-size: 16px;
    line-height: 1.7;
}

/* Context Card */
.context-card {
    background: #ffffff;
    border: 1px solid #bde3f2;
    border-radius: 14px;
    padding: 18px;
    margin: 10px 0;
    box-shadow: 0 4px 15px rgba(8, 126, 164, 0.08);
    color: #222222;
}

/* Divider */
hr {
    border: none;
    height: 2px;
    background: linear-gradient(
        90deg,
        #55b9df,
        #f3a6c7,
        #55b9df
    );
    margin: 30px 0;
}

/* Expander */
[data-testid="stExpander"] {
    background: rgba(255, 255, 255, 0.9);
    border: 1px solid #bde3f2;
    border-radius: 14px;
}

/* Metric-like information */
.stat-box {
    background: linear-gradient(
        135deg,
        #e1f6ff,
        #fff0f6
    );
    border-radius: 15px;
    padding: 18px;
    text-align: center;
    border: 1px solid #c7e8f5;
    box-shadow: 0 4px 15px rgba(8, 126, 164, 0.08);
}

.stat-number {
    font-size: 25px;
    font-weight: 800;
    color: #087ea4;
}

.stat-label {
    color: #555555;
    font-size: 13px;
}

/* Footer */
.footer {
    text-align: center;
    margin-top: 40px;
    padding: 20px;
    color: #666666;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    '<div class="main-title">📚 Chat with your PDF</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-powered PDF Question Answering using Gemini API + ChromaDB</div>',
    unsafe_allow_html=True
)


# --------------------------------------------------
# PDF TEXT EXTRACTION
# --------------------------------------------------

def extract_text_from_pdf(pdf_file):

    reader = PdfReader(pdf_file)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# --------------------------------------------------
# TEXT CHUNKING
# --------------------------------------------------

def create_chunks(text, chunk_size=1000):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk)

        start = end

    return chunks


# --------------------------------------------------
# GEMINI EMBEDDING
# --------------------------------------------------

def create_embedding(text):

    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )

    return response.embeddings[0].values


# --------------------------------------------------
# STORE CHUNKS IN CHROMADB
# --------------------------------------------------

def store_chunks(chunks):

    try:
        chroma_client.delete_collection(
            "gemini_rag_collection"
        )
    except Exception:
        pass

    global collection

    collection = chroma_client.get_or_create_collection(
        name="gemini_rag_collection"
    )

    embeddings = []

    for chunk in chunks:

        embedding = create_embedding(chunk)

        embeddings.append(embedding)

    ids = []

    for i in range(len(chunks)):
        ids.append(f"chunk_{i}")

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings
    )

    return len(chunks)


# --------------------------------------------------
# ASK GEMINI
# --------------------------------------------------

def generate_answer(question, context):

    prompt = f"""
You are a helpful AI assistant.

Answer the user's question using ONLY the information
provided in the context below.

If the answer is not available in the context,
say:

"I could not find the answer in the uploaded document."

Do not make up information.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt
    )

    return response.text


# --------------------------------------------------
# UPLOAD SECTION
# --------------------------------------------------

st.markdown(
    '<div class="info-card"><b>📄 Upload Your Document</b><br>'
    'Upload a PDF and process it to create an AI-searchable knowledge base.</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"]
)


# --------------------------------------------------
# PROCESS PDF
# --------------------------------------------------

if uploaded_file is not None:

    st.success(
        f"📄 Uploaded: {uploaded_file.name}"
    )

    if st.button("🔄 Process PDF"):

        with st.spinner("📖 Reading PDF..."):

            text = extract_text_from_pdf(
                uploaded_file
            )

        if not text.strip():

            st.error(
                "❌ Could not extract text from this PDF."
            )

        else:

            st.info(
                f"📊 Extracted {len(text):,} characters."
            )

            with st.spinner("✂️ Creating text chunks..."):

                chunks = create_chunks(text)

            st.info(
                f"🧩 Created {len(chunks)} chunks."
            )

            with st.spinner(
                "🧠 Creating Gemini embeddings and storing in ChromaDB..."
            ):

                number_of_chunks = store_chunks(
                    chunks
                )

            st.success(
                f"✅ PDF processed successfully! "
                f"{number_of_chunks} chunks stored in ChromaDB."
            )

            st.session_state.pdf_processed = True


# --------------------------------------------------
# QUESTION SECTION
# --------------------------------------------------

if st.session_state.get(
    "pdf_processed",
    False
):

    st.divider()

    st.markdown(
        '<div class="info-card"><b>💬 Ask Questions About Your PDF</b><br>'
        'Ask anything related to the information contained in your uploaded document.</div>',
        unsafe_allow_html=True
    )

    question = st.text_input(
        "Enter your question:",
        placeholder="Example: What is the main topic of this document?"
    )

    if st.button("🔍 Get Answer"):

        if not question.strip():

            st.warning(
                "⚠️ Please enter a question."
            )

        else:

            with st.spinner(
                "🔎 Searching the document..."
            ):

                question_embedding = create_embedding(
                    question
                )

                results = collection.query(
                    query_embeddings=[
                        question_embedding
                    ],
                    n_results=3
                )

            documents = results["documents"][0]

            context = "\n\n".join(
                documents
            )

            # ------------------------------------------
            # RETRIEVED CONTEXT
            # ------------------------------------------

            st.subheader(
                "📖 Retrieved Context"
            )

            with st.expander(
                "🔎 View retrieved information"
            ):

                for i, doc in enumerate(
                    documents
                ):

                    st.markdown(
                        f"""
                        <div class="context-card">
                        <b>📌 Chunk {i + 1}</b>
                        <br><br>
                        {doc}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            # ------------------------------------------
            # GEMINI ANSWER
            # ------------------------------------------

            with st.spinner(
                "🤖 Gemini is generating the answer..."
            ):

                answer = generate_answer(
                    question,
                    context
                )

            st.subheader(
                "🤖 Gemini Answer"
            )

            st.markdown(
                f"""
                <div class="answer-card">
                {answer}
                </div>
                """,
                unsafe_allow_html=True
            )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown(
    """
    <div class="footer">
        📚 Gemini RAG • Powered by Gemini API & ChromaDB
        <br>
        Sky Blue × Light Pink Professional Interface
    </div>
    """,
    unsafe_allow_html=True
)