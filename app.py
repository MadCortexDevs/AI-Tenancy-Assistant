# import streamlit as st
# from langchain_ollama import ChatOllama
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
# from PyPDF2 import PdfReader

# # Custom CSS styling
# st.markdown("""
# <style>
#     .main { background-color: #1a1a1a; color: #ffffff; }
#     .sidebar .sidebar-content { background-color: #2d2d2d; }
#     .stTextInput textarea { color: #ffffff !important; }
#     .stSelectbox div[data-baseweb="select"] { color: white !important; background-color: #3d3d3d !important; }
#     .stSelectbox svg { fill: white !important; }
#     .stSelectbox option, div[role="listbox"] div { background-color: #2d2d2d !important; color: white !important; }
#     .analysis-section { border-left: 4px solid #4CAF50; padding-left: 1rem; margin: 1rem 0; }
# </style>
# """, unsafe_allow_html=True)

# st.title("⚖️ AI Tenancy Legal Assistant")
# st.caption("🔍 Expert Analysis of Indian Rent Agreements | 100% Legal Compliance")

# # Sidebar configuration
# with st.sidebar:
#     st.header("⚙️ Legal Configurations")
#     selected_model = st.selectbox(
#         "Choose Legal Analysis Model",
#         ["deepseek-r1:1.5b", "deepseek-r1:7b"],
#         index=0
#     )
#     st.divider()
#     st.markdown("### Legal Capabilities")
#     st.markdown("""
#     - 📑 Rent Agreement Analysis
#     - ⚖️ Legal Rights Explanation
#     - ✍️ Clause Redrafting
#     - ❓ Critical Question Generation
#     """)
#     st.divider()
#     st.markdown("Powered by [Ollama](https://ollama.ai/) | [LangChain](https://python.langchain.com/)")

# # Initialize chat engine
# llm_engine = ChatOllama(
#     model=selected_model,
#     base_url="http://localhost:11434",
#     temperature=0.1  # Lower temperature for legal precision
# )

# # System prompt configuration
# legal_system_prompt = SystemMessagePromptTemplate.from_template(
#     """You are an expert AI Tenancy Lawyer specializing in Indian rent laws. Perform exhaustive analysis of rent agreements with:
# 1. Line-by-line legal compliance check using Model Tenancy Act 2021, state Rent Control Acts, and relevant case laws
# 2. Identification of ambiguous/unfair clauses with section-wise citations
# 3. Tenant protection-focused redrafting suggestions
# 4. Generate 5 critical questions per contested clause
# 5. Strict adherence to latest legal amendments (2024) and SC/HC judgments
# Never answer non-tenancy queries. Respond in structured markdown with legal references."""
# )

# # Session state management
# if "legal_messages" not in st.session_state:
#     st.session_state.legal_messages = [{"role": "ai", "content": "Welcome! Upload your rent agreement for comprehensive legal analysis. 🧾"}]

# if "analysis_stage" not in st.session_state:
#     st.session_state.analysis_stage = 0

# if "file_processed" not in st.session_state:
#     st.session_state.file_processed = False

# # Chat container
# chat_container = st.container()

# # Display messages
# with chat_container:
#     for msg in st.session_state.legal_messages:
#         with st.chat_message(msg["role"]):
#             st.markdown(msg["content"], unsafe_allow_html=True)

# # Analysis options
# analysis_options = {
#     1: "Analyze Agreement for Risks",
#     2: "Know Your Legal Rights",
#     3: "Redraft Problematic Clauses",
#     4: "Generate Clarification Questions"
# }

# # File upload and processing
# uploaded_file = st.file_uploader("Upload Rent Agreement (PDF/Text)", type=["pdf", "txt"])
# if uploaded_file and not st.session_state.file_processed:
#     try:
#         # Process text files
#         if uploaded_file.type == "text/plain":
#             text = uploaded_file.read().decode()
        
#         # Process PDF files
#         elif uploaded_file.type == "application/pdf":
#             pdf_reader = PdfReader(uploaded_file)
#             text = ""
#             for page in pdf_reader.pages:
#                 text += page.extract_text()
        
#         st.session_state.agreement_text = text
#         st.session_state.file_processed = True
#         st.session_state.legal_messages.append({"role": "user", "content": f"Uploaded file: {uploaded_file.name}"})
#         st.session_state.legal_messages.append({"role": "ai", "content": "File successfully uploaded! Choose an analysis option below."})
#         st.rerun()
    
#     except Exception as e:
#         st.error(f"Error processing file: {str(e)}")

# # Analysis handler
# def perform_legal_analysis(option):
#     analysis_prompts = {
#         1: f"""Analyze this rent agreement for tenant risks with:
#         1. Clause-by-clause breakdown
#         2. Legal non-compliance issues
#         3. Risk severity assessment
#         Agreement Text: {st.session_state.agreement_text}""",
        
#         2: f"""Explain tenant rights for this agreement with:
#         1. Relevant sections from Model Tenancy Act
#         2. State-specific rent control provisions
#         3. Recent judicial precedents
#         Agreement Text: {st.session_state.agreement_text}""",
        
#         3: f"""Redraft problematic clauses with:
#         1. Legal justification for changes
#         2. Balanced tenant-landlord obligations
#         3. Enforceable language
#         Agreement Text: {st.session_state.agreement_text}""",
        
#         4: f"""Generate clarification questions with:
#         1. 5 questions per contested clause
#         2. Focus on ambiguous terms
#         3. Financial/legal implications
#         Agreement Text: {st.session_state.agreement_text}"""
#     }
    
#     with st.spinner("🔍 Performing deep legal analysis..."):
#         legal_prompt = ChatPromptTemplate.from_messages([
#             legal_system_prompt,
#             HumanMessagePromptTemplate.from_template("{input}")
#         ])
#         legal_chain = legal_prompt | llm_engine | StrOutputParser()
#         return legal_chain.invoke({"input": analysis_prompts[option]})

# # Show analysis options after file upload
# if st.session_state.file_processed and st.session_state.analysis_stage == 0:
#     st.write("## Legal Analysis Options")
#     cols = st.columns(2)
#     for idx, (key, val) in enumerate(analysis_options.items()):
#         with cols[idx%2]:
#             if st.button(val, key=f"opt{key}"):
#                 st.session_state.analysis_stage = key
#                 analysis_result = perform_legal_analysis(key)
#                 st.session_state.legal_messages.append({
#                     "role": "ai",
#                     "content": f"## {analysis_options[key]}\n{analysis_result}"
#                 })
#                 st.session_state.analysis_stage = 0  # Reset for new analysis
#                 st.rerun()

# # User query handling
# if prompt := st.chat_input("Ask specific questions about your agreement..."):
#     if not st.session_state.file_processed:
#         st.warning("Please upload a rent agreement first!")
#         st.stop()
    
#     st.session_state.legal_messages.append({"role": "user", "content": prompt})
    
#     with st.spinner("🔍 Analyzing legal query..."):
#         legal_prompt = ChatPromptTemplate.from_messages([
#             legal_system_prompt,
#             HumanMessagePromptTemplate.from_template("""User question: {question}
#             Agreement Context: {context}""")
#         ])
        
#         legal_chain = legal_prompt | llm_engine | StrOutputParser()
#         response = legal_chain.invoke({
#             "question": prompt,
#             "context": st.session_state.agreement_text
#         })
#         print(response)
#     st.session_state.legal_messages.append({"role": "ai", "content": response})
#     st.rerun()

import streamlit as st
import fitz  # PyMuPDF
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# --- 1. Page Configuration and Styling ---
st.set_page_config(
    page_title="AI Tenancy Legal Assistant",
    page_icon="⚖️",
    layout="wide"
)

# Polished CSS
st.markdown("""
<style>
    .main {
        background-color: #1a1a1a;
        color: #ffffff;
    }
    .st-emotion-cache-1y4p8pa {
        padding: 2rem 2rem 10rem;
    }
    .stButton > button {
        border-radius: 8px;
        border: 1px solid #4CAF50;
        color: #4CAF50;
        background-color: transparent;
        transition: all 0.2s ease-in-out;
        width: 100%;
    }
    .stButton > button:hover {
        border-color: #ffffff;
        color: #ffffff;
        background-color: #4CAF50;
    }
    .stButton > button:focus {
        box-shadow: 0 0 0 2px #2d2d2d, 0 0 0 4px #4CAF50;
        outline: none;
    }
    #bottom_anchor {
        display: block;
        height: 1px;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. Sidebar Configuration ---
with st.sidebar:
    st.header("⚖️ AI Legal Assistant")
    st.markdown("Expert analysis for **Indian tenancy agreements**.")
    st.divider()
    st.markdown("### Core Capabilities")
    st.markdown("""
    - 📑 **Risk Analysis**
    - ⚖️ **Rights Explanation**
    - ✍️ **Clause Redrafting**
    - ❓ **Question Generation**
    """)
    st.divider()
    st.info("ℹ️ **Disclaimer**: This is an AI tool, not a substitute for advice from a qualified human lawyer.")
    st.divider()
    st.markdown("Powered by **Google Gemini** & **LangChain**")

# --- 3. LLM Initialization ---
try:
    llm_engine = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.1,
        google_api_key=st.secrets["GEMINI_API_KEY"]
    )
except KeyError:
    st.error("🚨 GEMINI_API_KEY not found. Please add it to your Streamlit secrets.")
    st.stop()

# System Prompt (strictly tenancy law)
legal_system_prompt = SystemMessagePromptTemplate.from_template(
    """
    You are an expert AI Tenancy Lawyer specializing in **Indian tenancy & rent laws**. 
    You must always analyze user questions strictly in relation to the uploaded rent agreement. 
    If no agreement is uploaded, politely refuse and ask the user to upload one.

    Perform exhaustive, clause-by-clause analysis of rent agreements to identify ambiguities, 
    unfair clauses, and legal non-compliance. Provide redrafting, risk analysis, 
    and cite Indian laws (Model Tenancy Act 2021, Transfer of Property Act 1882, 
    Contract Act 1872, Stamp Act 1899, state Rent Acts, Arbitration Act 1996, etc.).

    Always keep answers structured, precise, and tenant-protection oriented.
    """
)

# --- 4. Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agreement_text" not in st.session_state:
    st.session_state.agreement_text = None
if "current_file_name" not in st.session_state:
    st.session_state.current_file_name = None
if "scroll_to_bottom" not in st.session_state:
    st.session_state.scroll_to_bottom = False

# --- 5. Main UI ---
st.title("Indian Rent Agreement Analyzer")

# File Upload
upload_container = st.container(border=True)
with upload_container:
    st.markdown("#### 📂 Start Here: Upload Your Agreement")
    uploaded_file = st.file_uploader(
        "Upload your tenancy agreement (PDF or TXT). The analysis will appear below.",
        type=["pdf", "txt"],
        label_visibility="collapsed"
    )

# Extract text on new file
if uploaded_file and uploaded_file.name != st.session_state.current_file_name:
    st.session_state.current_file_name = uploaded_file.name
    st.session_state.agreement_text = None
    st.session_state.messages = []

    with st.spinner(f"Analyzing `{uploaded_file.name}`..."):
        try:
            text = ""
            if uploaded_file.type == "text/plain":
                text = uploaded_file.read().decode("utf-8")
            elif uploaded_file.type == "application/pdf":
                with fitz.open(stream=uploaded_file.getvalue(), filetype="pdf") as doc:
                    text = "".join(page.get_text() for page in doc)

            if not text.strip():
                st.error("❌ Could not extract text. The file may be scanned or empty.")
            else:
                st.session_state.agreement_text = text
                st.session_state.messages.append(
                    {"role": "ai", "content": f"✅ Agreement `{uploaded_file.name}` uploaded. Ask your tenancy-law questions below."}
                )
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    st.rerun()

# --- Chat History ---
st.markdown("#### 💬 Chat & Analysis Results")
with st.container():
    if not st.session_state.messages:
        st.info("Upload a rent agreement to begin analysis.")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"], unsafe_allow_html=True)
    st.markdown('<div id="bottom_anchor"></div>', unsafe_allow_html=True)

# --- Utility Functions ---
def perform_legal_analysis(analysis_type, agreement_text):
    prompts = {
        "Analyze Risks": "Analyze this rent agreement clause-by-clause for risks. Focus on Issue, Legal Violation, and Tenant Risk.",
        "Know Your Rights": "Explain all tenant rights based on this agreement, citing Indian laws and case law.",
        "Redraft Agreement": "Identify problematic clauses, redraft them favorably for tenant, and provide justification.",
        "Generate Questions": "For each unfair clause, generate 3–5 critical questions a tenant must ask the landlord."
    }
    human_prompt = HumanMessagePromptTemplate.from_template(
        f"{prompts[analysis_type]}\n\n--- AGREEMENT TEXT ---\n\n{{context}}"
    )
    chat_prompt = ChatPromptTemplate.from_messages([legal_system_prompt, human_prompt])
    chain = chat_prompt | llm_engine | StrOutputParser()
    with st.spinner(f"Performing '{analysis_type}' analysis..."):
        return chain.invoke({"context": agreement_text})

def handle_chat_query(prompt, agreement_text):
    human_prompt = HumanMessagePromptTemplate.from_template(
        "User question: {question}\n\n--- RENT AGREEMENT CONTEXT ---\n\n{context}"
    )
    chat_prompt = ChatPromptTemplate.from_messages([legal_system_prompt, human_prompt])
    chain = chat_prompt | llm_engine | StrOutputParser()
    with st.spinner("Analyzing under Indian tenancy law..."):
        return chain.invoke({"question": prompt, "context": agreement_text})

# --- Quick Actions ---
if st.session_state.agreement_text:
    st.divider()
    st.markdown("##### **Quick Actions**")
    cols = st.columns(4)
    options = {
        "Analyze Risks": "🕵️‍♂️",
        "Know Your Rights": "⚖️",
        "Redraft Agreement": "✍️",
        "Generate Questions": "❓"
    }
    for i, (option, icon) in enumerate(options.items()):
        if cols[i].button(f"{icon} {option}", key=f"opt_{i}"):
            st.session_state.messages.append({"role": "user", "content": f"Request: **{option}**"})
            response = perform_legal_analysis(option, st.session_state.agreement_text)
            st.session_state.messages.append({"role": "ai", "content": response})
            st.session_state.scroll_to_bottom = True
            st.rerun()

# --- Chat Input ---
if prompt := st.chat_input("Ask a tenancy-law question about your agreement..."):
    if not st.session_state.agreement_text:
        st.warning("⚠️ Please upload a rent agreement PDF/TXT first.")
        time.sleep(2)
        st.rerun()
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        response = handle_chat_query(prompt, st.session_state.agreement_text)
        st.session_state.messages.append({"role": "ai", "content": response})
        st.session_state.scroll_to_bottom = True
        st.rerun()

# --- Auto-scroll ---
if st.session_state.scroll_to_bottom:
    st.markdown(
        """
        <script>
            const anchor = document.getElementById('bottom_anchor');
            if (anchor) {
                anchor.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        </script>
        """,
        unsafe_allow_html=True,
    )
    st.session_state.scroll_to_bottom = False
