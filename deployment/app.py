import os
import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_astradb import AstraDBVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

# --- Load Environment Variables ---
load_dotenv()

# --- Configuration ---
PDF_PATH = "FlykiteAirlinesHRP.pdf"
ASTRA_COLLECTION_NAME = "flykite_hr_policies"

# Astra DB credentials
ASTRA_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
ASTRA_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_KEYSPACE = os.getenv("ASTRA_DB_KEYSPACE", "default_keyspace")

# LLM Configuration
model_name_or_path = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
model_basename = "mistral-7b-instruct-v0.2.Q6_K.gguf"

SYSTEM_PROMPT = """You are an expert HR assistant for Flykite Airlines. \
Answer employee questions clearly and accurately using only the HR policy \
document context provided below.

Guidelines:
- Be concise but detailed
- Use bullet points for lists or multi-part answers
- Always cite the relevant policy area (e.g. "Per the Leave Policy...")
- If the answer is not in the context, say: \
"I couldn't find that in the Flykite HR handbook. Please contact HR directly."

Context from HR Handbook:
{context}"""

# --- Caching Functions ---
@st.cache_resource
def load_retriever():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # Initialize Astra DB Vector Store
    vector_store = AstraDBVectorStore(
        collection_name="flykite_hr_policies",
        embedding=embeddings,
        api_endpoint=ASTRA_API_ENDPOINT,
        token=ASTRA_APPLICATION_TOKEN,
        namespace=ASTRA_KEYSPACE,
    )

    # Auto-ingest if collection is empty
    try:
        dummy_check = vector_store.similarity_search("test_query_placeholder", k=1)
    except Exception:
        dummy_check = []

    if not dummy_check:
        with st.spinner("Astra DB collection is empty. Ingesting PDF..."):
            loader = PyPDFLoader(PDF_PATH)
            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
            chunks = splitter.split_documents(docs)
            vector_store.add_documents(chunks)
        st.success("✅ PDF ingested successfully into Astra DB!")

    return vector_store.as_retriever(search_kwargs={"k": 4})

@st.cache_resource
def load_llm():
    with st.spinner("Downloading/loading Llama.cpp model..."):
        model_path = hf_hub_download(
            repo_id=model_name_or_path,
            filename=model_basename,
            resume_download=True,
            cache_dir="./huggingface_cache"
        )
    return Llama(
        model_path=model_path,
        n_threads=4,
        n_batch=512,
        n_gpu_layers=0, 
        n_ctx=4096,  # Increased to accommodate context + system prompt
    )

# --- Streamlit UI ---
st.set_page_config(page_title="Flykite HR Bot", page_icon="✈️", layout="centered")
st.title("✈️ Flykite Airlines HR Policy Assistant")
st.markdown(
    "Instant answers from the official Flykite Airlines HR handbook.\n\n"
    "**Ask about:** Leave policies · Benefits · Code of conduct · Disciplinary procedures · Compliance"
)

# Credential check
if not all([ASTRA_API_ENDPOINT, ASTRA_APPLICATION_TOKEN]):
    st.error("❌ Missing Astra DB credentials. Please set `ASTRA_DB_API_ENDPOINT` and `ASTRA_DB_APPLICATION_TOKEN` in your `.env` file.")
    st.stop()

# 1. Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

# 2. Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 3. Example Questions
example_questions = [
    "How do I apply for annual leave, and how much notice is required?",
    "What is the process for requesting emergency bereavement leave?",
    "Can unused sick days be carried over to the next year?",
    "What happens to my leave balance if I resign during the year?",
    "Are flight crew entitled to additional rest days after long-haul flights?"
]

st.caption("💡 **Try asking:**")
cols = st.columns(len(example_questions))
for i, col in enumerate(cols):
    if col.button(example_questions[i], use_container_width=True, key=f"example_{i}"):
        st.session_state.pending_prompt = example_questions[i]

# 4. Chat input
prompt = st.chat_input("Ask a question about the HR handbook...")

if st.session_state.pending_prompt:
    prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None

# 5. Process the prompt
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            retriever = load_retriever()
            llm = load_llm()
            
            docs = retriever.invoke(prompt)
            context = "\n\n".join(doc.page_content for doc in docs)

            chat_messages = [{"role": "system", "content": SYSTEM_PROMPT.format(context=context)}]
            for msg in st.session_state.messages:
                chat_messages.append({"role": msg["role"], "content": msg["content"]})

            response = llm.create_chat_completion(messages=chat_messages)
            answer = response["choices"][0]["message"]["content"]

            pages = sorted(set(doc.metadata.get("page", 0) + 1 for doc in docs))
            if pages:
                answer += f"\n\n*📄 Referenced pages: {', '.join(str(p) for p in pages)}*"
            
            st.markdown(answer)
            
    st.session_state.messages.append({"role": "assistant", "content": answer})
