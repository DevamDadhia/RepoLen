import sys
import os

# Prevent Windows console encoding crashes
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import shutil
import tempfile
import unicodedata
from datetime import datetime
from textwrap import dedent
import streamlit as st

from git import Repo
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, HumanMessage
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent

def clean_str(val: str) -> str:
    if not val:
        return ""
    normalized = unicodedata.normalize("NFKD", str(val))
    return normalized.replace("\u00a0", " ").encode("ascii", "ignore").decode("ascii").strip()

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="RepoLens | AI Repository Engineer",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CSS THEME
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}
.stApp {
    background-color: #0B0F17 !important;
    color: #F3F4F6 !important;
}
.block-container {
    padding-top: 1.8rem !important;
    padding-bottom: 3.5rem !important;
    max-width: 1240px !important;
}
section[data-testid="stSidebar"] {
    background-color: #080C14 !important;
    border-right: 1px solid #161F30 !important;
}
section[data-testid="stSidebar"] > div {
    padding-top: 1.8rem !important;
    padding-left: 1.2rem !important;
    padding-right: 1.2rem !important;
}
.nav-item {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 14px;
    border-radius: 8px;
    color: #94A3B8;
    font-size: 14px;
    font-weight: 500;
    margin-bottom: 4px;
    text-decoration: none;
}
.nav-item:hover {
    background: #111A2E;
    color: #F8FAFC;
}
.nav-item.active {
    background: #152238;
    color: #60A5FA;
    border: 1px solid #1E3A5F;
}
.hero-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 22px;
}
.hero-title {
    font-size: 46px;
    font-weight: 800;
    letter-spacing: -1.5px;
    color: #FFFFFF;
    line-height: 1.1;
    margin: 0;
}
.hero-title span {
    color: #3B82F6;
}
.hero-subtitle {
    font-size: 15px;
    color: #94A3B8;
    margin-top: 8px;
}
.value-card {
    background: #0E1524;
    border: 1px solid #1A263D;
    border-radius: 12px;
    padding: 18px 16px;
    min-height: 110px;
    display: flex;
    gap: 14px;
    align-items: flex-start;
}
.val-icon-box {
    width: 38px;
    height: 38px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
    flex-shrink: 0;
}
.val-title {
    font-size: 14px;
    font-weight: 600;
    color: #F1F5F9;
    margin-bottom: 3px;
}
.val-desc {
    font-size: 12px;
    line-height: 1.4;
}
.repo-summary-card {
    background: #0D1424;
    border: 1px solid #1E2B45;
    border-radius: 14px;
    padding: 20px 24px;
    margin-top: 16px;
    margin-bottom: 20px;
}
.repo-header-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.repo-avatar {
    width: 42px;
    height: 42px;
    background: #1D2A4A;
    border: 1px solid #2B3D66;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 20px;
    margin-right: 14px;
}
.repo-title-text {
    font-size: 18px;
    font-weight: 700;
    color: #F8FAFC;
}
.badge-indexed {
    background: #064E3B;
    color: #34D399;
    border: 1px solid #059669;
    font-size: 11px;
    font-weight: 600;
    padding: 2px 10px;
    border-radius: 9999px;
    display: inline-flex;
    align-items: center;
    margin-left: 10px;
}
.substat-card {
    background: #0A0F1D;
    border: 1px solid #172338;
    border-radius: 10px;
    padding: 14px 16px;
    display: flex;
    align-items: center;
    gap: 14px;
}
.substat-icon {
    font-size: 20px;
    color: #60A5FA;
}
.substat-value {
    font-size: 18px;
    font-weight: 700;
    color: #FFFFFF;
    line-height: 1.1;
}
.substat-label {
    font-size: 11px;
    color: #64748B;
    margin-top: 2px;
}
.stButton > button {
    background-color: #10192A !important;
    color: #CBD5E1 !important;
    border: 1px solid #1E2D47 !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
}
.stButton > button:hover {
    border-color: #3B82F6 !important;
    color: #FFFFFF !important;
    background-color: #14223B !important;
}
div[data-testid="stTextInput"] input {
    background-color: #0A0F1D !important;
    border: 1px solid #1B283F !important;
    color: #F8FAFC !important;
    border-radius: 8px !important;
}
div[data-testid="stButton"] button[kind="primary"] {
    background-color: #2563EB !important;
    border-color: #2563EB !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
}
.suggestion-pill button {
    background: #0B101E !important;
    border: 1px solid #1C273C !important;
    border-radius: 9999px !important;
    font-size: 12px !important;
    color: #94A3B8 !important;
    padding: 4px 14px !important;
}
.suggestion-pill button:hover {
    color: #F8FAFC !important;
    border-color: #3B82F6 !important;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# STATE MANAGEMENT
# ============================================================

defaults = {
    "chat_history": [],
    "agent_executor": None,
    "vectorstore": None,
    "work_dir": None,
    "indexed_repo": "https://github.com/DevamDadhia/Explainable-Intrusion-Detection-System",
    "repo_stats": {
        "files": 42,
        "chunks": 186,
        "language": "Python",
        "submodules": 3,
        "analyzed_at": "Sept 20, 2026, 19:23"
    },
    "pending_prompt": None,
    "temp_dir": None
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ============================================================
# AGENT BUILDER
# ============================================================

def build_repo_agent(repo_dir: str, vectorstore: Chroma, api_key: str, model_name: str):
    clean_api_key = clean_str(api_key)
    clean_model = clean_str(model_name) or "openai/gpt-oss-120b"

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    @tool
    def list_repository_files(path: str = ".") -> str:
        """Lists repository files to inspect project structure."""
        target = os.path.join(repo_dir, path.strip("/\\"))
        if not os.path.exists(target):
            return f"Error: Path '{path}' not found."

        collected = []
        for root, dirs, files in os.walk(target):
            dirs[:] = [
                d for d in dirs
                if not d.startswith(".")
                and d not in ["node_modules", "venv", ".venv", "__pycache__", "dist", "build", ".git"]
            ]
            rel_root = os.path.relpath(root, repo_dir)
            for f in files:
                if not f.startswith("."):
                    collected.append(os.path.normpath(os.path.join(rel_root, f)))

            if len(collected) > 150:
                collected.append("... [remaining files truncated]")
                break

        res = "\n".join(collected) if collected else "Directory is empty."
        return res.encode("ascii", "ignore").decode("ascii")

    @tool
    def search_codebase(query: str) -> str:
        """Semantically searches indexed repository chunks for symbols or functions."""
        safe_query = query.encode("ascii", "ignore").decode("ascii")
        docs = retriever.invoke(safe_query)
        if not docs:
            return "No matching code snippets found."

        formatted = []
        for i, doc in enumerate(docs, 1):
            src = doc.metadata.get("source", "unknown")
            content = doc.page_content.encode("ascii", "ignore").decode("ascii")
            formatted.append(f"--- Result {i} [Source: {src}] ---\n{content}")
        return "\n\n".join(formatted)

    @tool
    def read_source_file(file_path: str, start_line: int = 1, line_count: int = 120) -> str:
        """Reads exact lines of code from a specific file."""
        clean = file_path.strip("/\\")
        full_path = os.path.join(repo_dir, clean)

        if not os.path.exists(full_path):
            return f"Error: File '{clean}' does not exist."

        try:
            with open(full_path, "r", encoding="utf-8", errors="ignore") as fp:
                lines = fp.readlines()

            start_idx = max(0, start_line - 1)
            slice_lines = lines[start_idx: start_idx + line_count]
            numbered = [f"{start_idx + i + 1}: {line}" for i, line in enumerate(slice_lines)]
            end_line = min(len(lines), start_idx + line_count)
            res = f"File: {clean} (Lines {start_idx + 1}-{end_line} of {len(lines)}):\n" + "".join(numbered)
            return res.encode("ascii", "ignore").decode("ascii")
        except Exception as e:
            return f"Error reading file: {str(e)}"

    tools = [list_repository_files, search_codebase, read_source_file]

    llm = ChatGroq(
        model=clean_model,
        groq_api_key=clean_api_key,
        temperature=0.1
    )

    system_prompt = (
        "You are RepoLens, an expert Principal AI Software Engineer. "
        "Analyze repositories carefully using your tools. Inspect file layouts, "
        "search relevant snippets, read exact lines, and answer questions directly with specific file citations."
    )

    return create_react_agent(model=llm, tools=tools, prompt=system_prompt)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 4px;">
        <div style="background: #1D4ED8; color: #FFFFFF; border-radius: 8px; width: 34px; height: 34px; display: flex; align-items: center; justify-content: center; font-weight: 800; font-size: 16px;">
            &lt;/&gt;
        </div>
        <div style="font-size: 22px; font-weight: 800; color: #FFFFFF; letter-spacing: -0.5px;">
            Repo<span style="color: #3B82F6;">Lens</span>
        </div>
    </div>
    <div style="color: #64748B; font-size: 12px; margin-bottom: 24px;">
        Understand any GitHub repo with AI
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <a class="nav-item active" href="#"><span style="font-size:16px;">🏠</span> Home</a>
    <a class="nav-item" href="#"><span style="font-size:16px;">💬</span> Chat</a>
    <a class="nav-item" href="#"><span style="font-size:16px;">📄</span> Repository Info</a>
    <a class="nav-item" href="#"><span style="font-size:16px;">📊</span> Analysis</a>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    with st.expander("⚙️ Settings", expanded=True):
        env_key = clean_str(os.environ.get("GROQ_API_KEY", ""))
        try:
            if not env_key and "GROQ_API_KEY" in st.secrets:
                env_key = clean_str(st.secrets["GROQ_API_KEY"])
        except Exception:
            pass

        groq_api_key = st.text_input(
            "Groq API Key",
            value=env_key,
            type="password",
            placeholder="gsk_..."
        )
        if groq_api_key:
            os.environ["GROQ_API_KEY"] = clean_str(groq_api_key)

        selected_model = st.selectbox(
            "Model",
            [
                "openai/gpt-oss-120b",
                "openai/gpt-oss-20b",
                "qwen/qwen3.8-27b"
            ],
            index=0
        )

    with st.expander("ℹ️ About"):
        st.caption("RepoLens indexes GitHub codebases into vector embeddings and uses tool-calling agents to inspect and audit them.")

# ============================================================
# HERO HEADER & REPO INPUT
# ============================================================

st.markdown("""
<div class="hero-header">
    <div>
        <div class="hero-title">Repo<span>Lens</span></div>
        <div class="hero-subtitle">Your AI engineer for understanding, debugging and exploring GitHub repositories.</div>
    </div>
</div>
""", unsafe_allow_html=True)

repo_col, btn_col = st.columns([0.80, 0.20])

with repo_col:
    repo_url_input = st.text_input(
        "Repo URL",
        value=st.session_state.indexed_repo or "https://github.com/DevamDadhia/Explainable-Intrusion-Detection-System",
        placeholder="https://github.com/username/repository",
        label_visibility="collapsed"
    )

with btn_col:
    index_clicked = st.button("Analyze Repository  →", type="primary", use_container_width=True)

st.markdown("<p style='color:#64748B; font-size:12px; margin-top:-6px; margin-bottom: 22px;'>Enter a public GitHub repository URL to get started</p>", unsafe_allow_html=True)

# ============================================================
# INGESTION WORKFLOW
# ============================================================

if index_clicked:
    active_key = clean_str(os.environ.get("GROQ_API_KEY", ""))
    target_repo = clean_str(repo_url_input)

    if not active_key:
        st.error("Please provide your Groq API key in the sidebar.")
    elif not target_repo:
        st.error("Please enter a valid GitHub repository URL.")
    else:
        with st.status("Ingesting and Analyzing repository...", expanded=True) as status:
            try:
                if st.session_state.temp_dir and os.path.exists(st.session_state.temp_dir):
                    shutil.rmtree(st.session_state.temp_dir, ignore_errors=True)

                work_dir = tempfile.mkdtemp(prefix="repolens_workspace_")
                st.session_state.temp_dir = work_dir
                st.session_state.work_dir = work_dir

                Repo.clone_from(target_repo, work_dir, depth=1)

                loader = GenericLoader.from_filesystem(
                    work_dir,
                    glob="**/*",
                    suffixes=[".py", ".md", ".txt"],
                    exclude=["*git*", "*venv*", "*__pycache__*", "*node_modules*"],
                    parser=LanguageParser(language=Language.PYTHON, parser_threshold=500)
                )
                raw_docs = loader.load()

                for d in raw_docs:
                    d.page_content = d.page_content.encode("ascii", "ignore").decode("ascii")

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
                st.session_state.vectorstore = vectorstore

                st.session_state.agent_executor = build_repo_agent(
                    work_dir,
                    vectorstore,
                    active_key,
                    clean_str(selected_model)
                )

                st.session_state.indexed_repo = target_repo
                st.session_state.repo_stats = {
                    "files": len(raw_docs),
                    "chunks": len(chunks),
                    "language": "Python",
                    "submodules": 3,
                    "analyzed_at": datetime.now().strftime("%b %d, %Y, %H:%M")
                }
                st.session_state.chat_history = []
                status.update(label="Repository indexed successfully!", state="complete")
                st.rerun()

            except Exception as e:
                status.update(label="Analysis failed", state="error")
                st.error(f"Failed to clone/analyze repository: {e}")

# ============================================================
# 4 VALUE CARDS
# ============================================================

v1, v2, v3, v4 = st.columns(4)

with v1:
    st.markdown(dedent("""
    <div class="value-card">
        <div class="val-icon-box" style="background:#2E1065; color:#A855F7;">📄</div>
        <div>
            <div class="val-title">Understand</div>
            <div class="val-desc" style="color:#94A3B8;">Get clear explanations of complex codebases</div>
        </div>
    </div>
    """), unsafe_allow_html=True)

with v2:
    st.markdown(dedent("""
    <div class="value-card">
        <div class="val-icon-box" style="background:#064E3B; color:#34D399;">🪲</div>
        <div>
            <div class="val-title">Find Issues</div>
            <div class="val-desc" style="color:#94A3B8;">Detect potential bugs and security risks</div>
        </div>
    </div>
    """), unsafe_allow_html=True)

with v3:
    st.markdown(dedent("""
    <div class="value-card">
        <div class="val-icon-box" style="background:#451A03; color:#F59E0B;">⚗️</div>
        <div>
            <div class="val-title">Analyze</div>
            <div class="val-desc" style="color:#94A3B8;">Explore architecture and code structure</div>
        </div>
    </div>
    """), unsafe_allow_html=True)

with v4:
    st.markdown(dedent("""
    <div class="value-card">
        <div class="val-icon-box" style="background:#4C0519; color:#FB7185;">⚡</div>
        <div>
            <div class="val-title">Optimize</div>
            <div class="val-desc" style="color:#94A3B8;">Get performance insights and improvement suggestions</div>
        </div>
    </div>
    """), unsafe_allow_html=True)

# ============================================================
# METRICS CONTAINER
# ============================================================

stats = st.session_state.repo_stats
repo_full = st.session_state.indexed_repo or "Explainable-Intrusion-Detection-System"
repo_name = repo_full.rstrip("/").split("/")[-1]
user_slug = "/".join(repo_full.rstrip("/").split("/")[-2:]) if "github.com" in repo_full else repo_name

st.markdown(dedent(f"""
<div class="repo-summary-card">
    <div class="repo-header-row">
        <div style="display: flex; align-items: center;">
            <div class="repo-avatar">📦</div>
            <div>
                <div style="display:flex; align-items:center;">
                    <span class="repo-title-text">{repo_name}</span>
                    <span class="badge-indexed">● Indexed</span>
                </div>
                <div style="font-size:13px; color:#3B82F6; margin-top:2px;">
                    {user_slug} &nbsp;↗
                </div>
            </div>
        </div>
        <div style="text-align: right; color:#64748B; font-size:12px;">
            Last analyzed<br>
            <span style="color:#94A3B8;">{stats.get("analyzed_at", "Sept 20, 2026, 19:23")}</span>
        </div>
    </div>
</div>
"""), unsafe_allow_html=True)

s1, s2, s3, s4 = st.columns(4)

with s1:
    st.markdown(dedent(f"""
    <div class="substat-card">
        <div class="substat-icon">📄</div>
        <div>
            <div class="substat-value">{stats.get("files", 42)}</div>
            <div class="substat-label">Files</div>
        </div>
    </div>
    """), unsafe_allow_html=True)

with s2:
    st.markdown(dedent(f"""
    <div class="substat-card">
        <div class="substat-icon">🧩</div>
        <div>
            <div class="substat-value">{stats.get("chunks", 186)}</div>
            <div class="substat-label">Code Chunks</div>
        </div>
    </div>
    """), unsafe_allow_html=True)

with s3:
    st.markdown(dedent(f"""
    <div class="substat-card">
        <div class="substat-icon">🗄️</div>
        <div>
            <div class="substat-value" style="font-size:16px;">{stats.get("language", "Python")}</div>
            <div class="substat-label">Primary Language</div>
        </div>
    </div>
    """), unsafe_allow_html=True)

with s4:
    st.markdown(dedent(f"""
    <div class="substat-card">
        <div class="substat-icon">🌿</div>
        <div>
            <div class="substat-value">{stats.get("submodules", 3)}</div>
            <div class="substat-label">Submodules</div>
        </div>
    </div>
    """), unsafe_allow_html=True)

# ============================================================
# QUICK ANALYSIS BUTTONS
# ============================================================

st.markdown("<p style='font-size: 14px; font-weight:600; color:#CBD5E1; margin-top:24px; margin-bottom:12px;'>Quick Analysis</p>", unsafe_allow_html=True)

qa1, qa2, qa3, qa4, qa5 = st.columns(5)

with qa1:
    if st.button("📄 Explain this repository", use_container_width=True):
        st.session_state.pending_prompt = "Provide a comprehensive architectural and purpose overview of this repository."
        st.rerun()

with qa2:
    if st.button("🏛️ Analyze architecture", use_container_width=True):
        st.session_state.pending_prompt = "Explain the architecture of this repository. Map the data flow and identify key modules."
        st.rerun()

with qa3:
    if st.button("🪲 Find potential bugs", use_container_width=True):
        st.session_state.pending_prompt = "Perform a code review to find potential bugs, edge cases, and runtime exceptions. Cite exact files and lines."
        st.rerun()

with qa4:
    if st.button("🛡️ Security audit", use_container_width=True):
        st.session_state.pending_prompt = "Perform a security audit looking for hardcoded secrets, unsafe dependency calls, or vulnerabilities."
        st.rerun()

with qa5:
    if st.button("⚡ Performance review", use_container_width=True):
        st.session_state.pending_prompt = "Analyze this codebase for performance bottlenecks and identify algorithmic improvements."
        st.rerun()

# ============================================================
# CHAT LOG & INTERACTION
# ============================================================

st.markdown("<p style='font-size: 14px; font-weight:600; color:#CBD5E1; margin-top:26px; margin-bottom:6px;'>Chat with RepoLens</p>", unsafe_allow_html=True)

for msg in st.session_state.chat_history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    avatar = "🧑‍💻" if role == "user" else "🤖"
    with st.chat_message(role, avatar=avatar):
        st.markdown(msg.content)

# Suggestion Pill Buttons
sg1, sg2, sg3, sg4, sg5 = st.columns(5)
with sg1:
    st.markdown('<div class="suggestion-pill">', unsafe_allow_html=True)
    if st.button("Summarize the project", key="s_sum"):
        st.session_state.pending_prompt = "Give a concise summary of this repository project."
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

with sg2:
    st.markdown('<div class="suggestion-pill">', unsafe_allow_html=True)
    if st.button("How does the model work?", key="s_mod"):
        st.session_state.pending_prompt = "Explain how the model or primary logic in this project functions."
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

with sg3:
    st.markdown('<div class="suggestion-pill">', unsafe_allow_html=True)
    if st.button("Explain the data flow", key="s_flow"):
        st.session_state.pending_prompt = "Explain the step-by-step data flow across components in this project."
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

with sg4:
    st.markdown('<div class="suggestion-pill">', unsafe_allow_html=True)
    if st.button("What are the key components?", key="s_comp"):
        st.session_state.pending_prompt = "List and describe the key components and files in this repository."
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

with sg5:
    st.markdown('<div class="suggestion-pill">', unsafe_allow_html=True)
    if st.button("List main dependencies", key="s_deps"):
        st.session_state.pending_prompt = "Identify and explain the main third-party dependencies used in this repository."
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

prompt_input = st.chat_input("Ask anything about this repository...")
active_prompt = prompt_input or st.session_state.pop("pending_prompt", None)

if active_prompt:
    clean_prompt = clean_str(active_prompt)

    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(clean_prompt)
    st.session_state.chat_history.append(HumanMessage(content=clean_prompt))

    with st.chat_message("assistant", avatar="🤖"):
        active_key = clean_str(os.environ.get("GROQ_API_KEY", ""))
        target_model = clean_str(selected_model) or "openai/gpt-oss-120b"

        if st.session_state.vectorstore and st.session_state.work_dir:
            st.session_state.agent_executor = build_repo_agent(
                st.session_state.work_dir,
                st.session_state.vectorstore,
                active_key,
                target_model
            )

        if not st.session_state.agent_executor:
            mock_reply = "Please click Analyze Repository above first to build the codebase index."
            st.markdown(mock_reply)
            st.session_state.chat_history.append(AIMessage(content=mock_reply))
        else:
            with st.spinner("RepoLens is inspecting the codebase..."):
                try:
                    result = st.session_state.agent_executor.invoke({
                        "messages": st.session_state.chat_history
                    })
                    response_text = result["messages"][-1].content
                    st.markdown(response_text)
                    st.session_state.chat_history.append(AIMessage(content=response_text))
                except Exception as err:
                    st.error(f"Execution failed: {err}")
