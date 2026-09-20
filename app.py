import os
import shutil
import tempfile
import streamlit as st
from git import Repo

from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_groq import ChatGroq

# --- Page Layout & Header ---
st.set_page_config(
    page_title="RepoLens | AI Repository Engineer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Session State Storage ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "agent_executor" not in st.session_state:
    st.session_state.agent_executor = None
if "indexed_repo" not in st.session_state:
    st.session_state.indexed_repo = None
if "repo_stats" not in st.session_state:
    st.session_state.repo_stats = {}

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    env_key = os.environ.get("GROQ_API_KEY", "")
    if not env_key and "GROQ_API_KEY" in st.secrets:
        env_key = st.secrets["GROQ_API_KEY"]
        
    groq_api_key = st.text_input(
        "Groq API Key",
        value=env_key,
        type="password",
        help="Paste your key from console.groq.com"
    )
    if groq_api_key:
        os.environ["GROQ_API_KEY"] = groq_api_key

    selected_model = st.selectbox(
        "Groq Model",
        ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"],
        index=0
    )

    st.divider()
    st.subheader("📦 Repository Ingestion")
    repo_url = st.text_input(
        "GitHub Repository URL",
        value="https://github.com/DevamDadhia/Explainable-Intrusion-Detection-System"
    )
    index_btn = st.button("🚀 Ingest & Index Repo", type="primary", use_container_width=True)

    if st.session_state.indexed_repo:
        st.divider()
        st.markdown(f"**Indexed Repo:** `{st.session_state.indexed_repo.split('/')[-1]}`")
        st.caption(f"📁 Files: {st.session_state.repo_stats.get('files', 0)} | 🧩 Chunks: {st.session_state.repo_stats.get('chunks', 0)}")
        if st.button("🧹 Clear Conversation", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

# --- Agent & Tools Builder ---
def build_repo_agent(repo_dir: str, vectorstore, api_key: str, model_name: str):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    @tool
    def list_repository_files(path: str = ".") -> str:
        """Lists all files in the repository directory tree. Use this first to explore codebase structure."""
        target = os.path.join(repo_dir, path.strip("/\\"))
        if not os.path.exists(target):
            return f"Error: Path '{path}' not found."
        collected = []
        for root, dirs, files in os.walk(target):
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', 'venv', '__pycache__', 'dist']]
            rel_root = os.path.relpath(root, repo_dir)
            for f in files:
                if not f.startswith('.'):
                    collected.append(os.path.normpath(os.path.join(rel_root, f)))
            if len(collected) > 150:
                collected.append("... [remaining files truncated]")
                break
        return "\n".join(collected) if collected else "Directory is empty."

    @tool
    def search_codebase(query: str) -> str:
        """Semantically searches indexed code snippets, docstrings, and functions."""
        docs = retriever.invoke(query)
        if not docs:
            return "No matching code snippets found."
        formatted = []
        for i, d in enumerate(docs, 1):
            src = d.metadata.get("source", "unknown")
            formatted.append(f"--- Result {i} [Source: {src}] ---\n{d.page_content}")
        return "\n\n".join(formatted)

    @tool
    def read_source_file(file_path: str, start_line: int = 1, line_count: int = 120) -> str:
        """Reads exact lines of a source file with line numbers to audit implementations."""
        clean = file_path.strip("/\\")
        full_path = os.path.join(repo_dir, clean)
        if not os.path.exists(full_path):
            return f"Error: File '{clean}' does not exist in repository."
        try:
            with open(full_path, "r", encoding="utf-8", errors="ignore") as fp:
                lines = fp.readlines()
            start_idx = max(0, start_line - 1)
            slice_lines = lines[start_idx : start_idx + line_count]
            numbered = [f"{start_idx + i + 1}: {line}" for i, line in enumerate(slice_lines)]
            return f"File: {clean} (Lines {start_idx + 1}-{min(len(lines), start_idx + line_count)} of {len(lines)}):\n" + "".join(numbered)
        except Exception as e:
            return f"Error reading file: {str(e)}"

    tools = [list_repository_files, search_codebase, read_source_file]
    llm = ChatGroq(model=model_name, groq_api_key=api_key, temperature=0.0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are RepoLens, an expert Principal AI Software Engineer.\n"
            "Guidelines:\n"
            "1. Always check directory structure using `list_repository_files` or locate symbols with `search_codebase`.\n"
            "2. Inspect specific code details and check logic using `read_source_file`.\n"
            "3. Cite exact file names and line numbers in your answers.\n"
            "4. Provide structured, technically accurate, and direct responses."
        )),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

# --- Ingestion Trigger ---
if index_btn:
    if not os.environ.get("GROQ_API_KEY"):
        st.error("⚠️ Please provide your Groq API key in the sidebar.")
    elif not repo_url.strip():
        st.error("⚠️ Please specify a valid GitHub repository URL.")
    else:
        with st.spinner("Cloning repository, parsing AST, and indexing vector store..."):
            try:
                work_dir = tempfile.mkdtemp(prefix="repolens_workspace_")
                Repo.clone_from(repo_url.strip(), work_dir, depth=1)

                loader = GenericLoader.from_filesystem(
                    work_dir,
                    glob="**/*",
                    suffixes=[".py", ".md", ".txt"],
                    exclude=["*.git*", "*venv*", "*__pycache__*", "*node_modules*"],
                    parser=LanguageParser(language=Language.PYTHON, parser_threshold=500)
                )
                raw_docs = loader.load()

                splitter = RecursiveCharacterTextSplitter.from_language(
                    language=Language.PYTHON,
                    chunk_size=1000,
                    chunk_overlap=150
                )
                chunks = splitter.split_documents(raw_docs)

                for doc in chunks:
                    doc.metadata["source"] = os.path.relpath(doc.metadata.get("source", ""), work_dir)

                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                vectorstore = Chroma.from_documents(chunks, embeddings)

                st.session_state.agent_executor = build_repo_agent(
                    work_dir, vectorstore, os.environ["GROQ_API_KEY"], selected_model
                )
                st.session_state.indexed_repo = repo_url.strip()
                st.session_state.repo_stats = {"files": len(raw_docs), "chunks": len(chunks)}
                st.session_state.chat_history = []
                st.success(f"✅ Successfully indexed {len(raw_docs)} files into {len(chunks)} semantic chunks!")
            except Exception as e:
                st.error(f"Error during repository ingestion: {str(e)}")

# --- Main Interface ---
st.title("🔍 RepoLens: Autonomous Repository AI Engineer")
st.caption("Agentic code inspection, vector retrieval, and architectural auditing powered by LangChain and Groq.")

# Suggested Prompts
if st.session_state.indexed_repo and not st.session_state.chat_history:
    st.markdown("##### 💡 Quick Start Inquiries:")
    c1, c2, c3 = st.columns(3)
    if c1.button("🏗️ Explain Architecture", use_container_width=True):
        st.session_state.pending_prompt = "Explain the architecture of this repository, key directories, and core responsibilities."
        st.rerun()
    if c2.button("🔍 Locate Core Models", use_container_width=True):
        st.session_state.pending_prompt = "Where are the primary models or algorithms defined? Cite exact files and line numbers."
        st.rerun()
    if c3.button("🛡️ Audit Codebase", use_container_width=True):
        st.session_state.pending_prompt = "Audit the main implementation files for potential edge cases, bugs, or unhandled exceptions."
        st.rerun()

# Render Message Log
for msg in st.session_state.chat_history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# Chat Input Handler
prompt_input = st.chat_input("Ask a question about the repository...")
active_prompt = prompt_input or st.session_state.pop("pending_prompt", None)

if active_prompt:
    if not st.session_state.agent_executor:
        st.warning("⚠️ Please index a repository from the sidebar first.")
    else:
        st.chat_message("user").markdown(active_prompt)
        st.session_state.chat_history.append(HumanMessage(content=active_prompt))

        with st.chat_message("assistant"):
            with st.spinner("RepoLens is traversing files and analyzing code..."):
                try:
                    result = st.session_state.agent_executor.invoke({
                        "input": active_prompt,
                        "chat_history": st.session_state.chat_history[:-1]
                    })
                    response_text = result["output"]
                    st.markdown(response_text)
                    st.session_state.chat_history.append(AIMessage(content=response_text))
                except Exception as err:
                    st.error(f"Agent execution error: {str(err)}")
