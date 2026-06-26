import os
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

# --- Configuration ---
PDF_PATH = "FlykiteAirlinesHRP.pdf"
CHROMA_DB_PATH = "hr_flykite_db"
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
    if os.path.exists(CHROMA_DB_PATH):
        vs = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
    else:
        loader = PyPDFLoader(PDF_PATH)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
        chunks = splitter.split_documents(docs)
        vs = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_DB_PATH)
    return vs.as_retriever(search_kwargs={"k": 4})

@st.cache_resource
def load_llm():
    model_path = hf_hub_download(
        repo_id=model_name_or_path,
        filename=model_basename,
        resume_download=True,
        cache_dir="./huggingface_cache"
    )
    return Llama(
        model_path=model_path,
        n_threads=2,
        n_batch=256,
        n_gpu_layers=0, 
        n_ctx=2300,
    )

# --- Streamlit UI ---
st.set_page_config(page_title="Flykite HR Bot", page_icon="✈️")
st.title("✈️ Flykite Airlines HR Policy Assistant")
st.markdown(
    "Instant answers from the official Flykite Airlines HR handbook.\n\n"
    "**Ask about:** Leave policies · Benefits · Code of conduct · Disciplinary procedures · Compliance"
)

# 1. Initialize session state for chat history and pending prompts
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

# 2. Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 3. Define Example Questions
example_questions = [
    "How do I apply for annual leave, and how much notice is required?",
    "What is the process for requesting emergency bereavement leave?",
    "Can unused sick days be carried over to the next year?",
    "What happens to my leave balance if I resign during the year?",
    "Are flight crew entitled to additional rest days after long-haul flights?"
]

# 4. Create clickable buttons for the examples
st.caption("💡 **Try asking:**")
cols = st.columns(len(example_questions))
for i, col in enumerate(cols):
    # use_container_width=True makes them stretch evenly
    if col.button(example_questions[i], use_container_width=True, key=f"example_{i}"):
        st.session_state.pending_prompt = example_questions[i]

# 5. Chat input
prompt = st.chat_input("Ask a question about the HR handbook...")

# If an example button was clicked, use that as the prompt
if st.session_state.pending_prompt:
    prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None  # Clear it so it doesn't repeat

# 6. Process the prompt (either from text input or example button)
if prompt:
    # Add user message to history and display
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Load models (cached)
            retriever = load_retriever()
            llm = load_llm()
            
            docs = retriever.invoke(prompt)
            context = "\n\n".join(doc.page_content for doc in docs)

            # Format messages for llama_cpp native API
            chat_messages = [{"role": "system", "content": SYSTEM_PROMPT.format(context=context)}]
            for msg in st.session_state.messages:
                chat_messages.append({"role": msg["role"], "content": msg["content"]})

            # Call llama_cpp directly
            response = llm.create_chat_completion(messages=chat_messages)
            answer = response["choices"][0]["message"]["content"]

            # Add page references
            pages = sorted(set(doc.metadata.get("page", 0) + 1 for doc in docs))
            if pages:
                answer += f"\n\n*📄 Referenced pages: {', '.join(str(p) for p in pages)}*"
            
            st.markdown(answer)
            
    # Add assistant message to history
    st.session_state.messages.append({"role": "assistant", "content": answer})
