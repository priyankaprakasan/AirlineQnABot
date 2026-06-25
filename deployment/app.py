import os
import gradio as gr
# from langchain_groq import ChatGroq # Removed
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# New imports for LlamaCpp
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

PDF_PATH = "data/FlykiteAirlinesHRP.pdf"
CHROMA_DB_PATH = "hr_flykite_db"
# GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "") # No longer needed for LlamaCpp

# Hugging Face Token for model download (if needed, e.g., for gated models)
# Assumed to be set in the environment where the app is run
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Llama model parameters
model_name_or_path = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
model_basename = "mistral-7b-instruct-v0.2.Q6_K.gguf"

# Download the model
model_path = hf_hub_download(
    repo_id=model_name_or_path,
    filename=model_basename,
    resume_download=True,
    cache_dir="./huggingface_cache" # Store model weights in a local cache directory
)


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


def build_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    if os.path.exists(CHROMA_DB_PATH):
        return Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)

    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    vs = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_DB_PATH)
    return vs


print("Loading HR handbook and building index...")
vectorstore = build_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# Initialize Llama 
llm = Llama(
    model_path=model_path,
    n_threads=2,  # CPU cores
    n_batch=256,  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
    n_gpu_layers=32,  # Change this value based on your model and your GPU VRAM pool.
    n_ctx=2300,  # Context window
)
print("Ready.")


def respond(message: str, history: list) -> str:
    docs = retriever.invoke(message)
    context = "\n\n".join(doc.page_content for doc in docs)

    messages = [SystemMessage(content=SYSTEM_PROMPT.format(context=context))]
    # history is list of [user, assistant] pairs from gr.ChatInterface
    for human, assistant in history:
        messages.append(HumanMessage(content=human))
        if assistant:
            messages.append(AIMessage(content=assistant))
    messages.append(HumanMessage(content=message))

    answer = llm.invoke(messages).content

    pages = sorted(set(doc.metadata.get("page", 0) + 1 for doc in docs))
    if pages:
        answer += f"\n\n*📄 Referenced pages: {', '.join(str(p) for p in pages)}*"

    return answer


demo = gr.ChatInterface(
    fn=respond,
    title="✈️ Flykite Airlines HR Policy Assistant",
    description=(
        "Instant answers from the official Flykite Airlines HR handbook.\n\n"
        "**Ask about:** Leave policies · Benefits · Code of conduct · Disciplinary procedures · Compliance"
    ),
    examples=[
        1. "How do I apply for annual leave, and how much notice is required?"
        2. "What is the process for requesting emergency bereavement leave?"
        3. "Can unused sick days be carried over to the next year?"
        4. "What happens to my leave balance if I resign during the year?"
        5. "Are flight crew entitled to additional rest days after long-haul flights?"
    ],
)

if __name__ == "__main__":
    demo.launch()
