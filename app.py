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

# Polished CSS for a better UI
st.markdown("""
<style>
    /* General Styling */
    .main {
        background-color: #1a1a1a;
        color: #ffffff;
    }
    .st-emotion-cache-1y4p8pa { /* Main container padding */
        padding: 2rem 2rem 10rem; /* Added bottom padding to avoid overlap with action bar */
    }

    /* Action buttons styling */
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
    
    /* Anchor for scrolling */
    #bottom_anchor {
        display: block;
        height: 1px;
    }
</style>
""", unsafe_allow_html=True)


# --- 2. Sidebar Configuration ---
with st.sidebar:
    st.header("⚖️ AI Legal Assistant")
    st.markdown("Expert analysis for **Indian tenancy & rent** matters only.")
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


# --- 3. LLM and The "Perfect Prompt" Initialization ---
try:
    llm_engine = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.1,  # Lowered for higher legal precision
        google_api_key=st.secrets["GEMINI_API_KEY"]
    )
except KeyError:
    st.error("🚨 GEMINI_API_KEY not found. Please add it to your Streamlit secrets.")
    st.stop()

# System prompt — **strict tenancy scope**
legal_system_prompt = SystemMessagePromptTemplate.from_template(
    """
    *Role & Objective:*
    You are an expert AI Tenancy Lawyer specializing in **Indian tenancy & rent laws**. You must keep every response strictly within this domain. If a user asks outside this scope, politely steer them back by explaining that you only answer tenancy/rent questions under Indian law and suggest how to reframe their query.

    Perform exhaustive, line-by-line analysis of uploaded rent agreements to identify ambiguities, unfair clauses, and legal non-compliance. Prioritize tenant protection while ensuring recommendations are legally enforceable under Indian law. Leave no clause, term, or legal implication unexamined.

    ---
    *Core Instructions for Analysis:*
    1.  *Jurisdiction & Legal Compliance Check:*
        - Confirm property jurisdiction (state/city) and apply *all relevant laws*:
          - Primary Laws: Model Tenancy Act 2021, Transfer of Property Act 1882, Indian Contract Act 1872, state-specific Rent Control Acts.
          - Secondary Laws: Indian Stamp Act 1899, CPC 1908, Arbitration & Conciliation Act 1996, Registration Act 1908, Evidence Act 1872.
        - Cross-reference with latest amendments and landmark judgments from High Courts/Supreme Court when making assertions.

    2.  *Clause-by-Clause Deep Dive:*
        - For *every clause*: Highlight ambiguities, flag unfair terms, and identify gaps.
        - Use *structured markdown output* for clarity.

    3.  *Legally Sound Recourse & Redrafting:*
        - For each unfair clause, provide critical questions and redraft with balance, citing legal justification.

    4.  *Validation & Precision:*
        - Replace vague terms like “reasonable” with quantifiable metrics wherever feasible.
        - Anticipate disputes and analyze alignment with state-specific thresholds.

    ---
    *Mandatory Output Structure (when doing full agreement analysis):*
    ## Rent Agreement Analysis Report

    ### 1. Jurisdiction & Applicable Laws
    - **Property Location**: [State/City, if determinable]
    - **Primary Laws Applied**: [List with relevant sections]

    ### 2. Clause-wise Breakdown
    #### Clause [X]: [Clause Title]
    - **Issue**: [Unfair term/ambiguity]
    - **Legal Violation**: [Act/Section/Case Law citation]
    - **Tenant Risk**: [Financial/legal/privacy harm]
    - **Recourse**:
      - **Redrafted Clause**: [Revised, balanced text]
      - **Questions for Landlord**:
        1. [Question 1]
        2. [Question 2]
        3. [Add more]

    ### 3. Additional Considerations
    - **Missed Clauses**: [Highlight omissions, e.g., "No dispute resolution clause."]
    - **Strategic Advice**: [e.g., "Insist on adding a mediation clause."]

    ---
    *Model Behavior Rules:*
    - *Assumption-Free Analysis*: Reject vague clauses like "standard terms apply."
    - *Zero Vagueness*: Replace "reasonable" with quantifiable metrics.
    - *Bias Mitigation*: Balance landlord-tenant rights in all suggestions.
    - *Strict Scope*: Only Indian tenancy/rent law. If outside scope, respond: 
      "I’m focused on Indian tenancy & rent law. If you’d like, please reframe your question within this domain (e.g., how it affects tenants/landlords, rent agreements, deposits, notice, eviction, maintenance, registration, stamp duty, dispute resolution)."
    """
)

# --- 4. Session State Management ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agreement_text" not in st.session_state:
    st.session_state.agreement_text = None
if "current_file_name" not in st.session_state:
    st.session_state.current_file_name = None
if "scroll_to_bottom" not in st.session_state:
    st.session_state.scroll_to_bottom = False

# --- 5. Main Application UI & Logic ---
st.title("Indian Rent Agreement Analyzer")

# Step 1: File Uploader
upload_container = st.container(border=True)
with upload_container:
    st.markdown("#### 📂 Start Here: Upload Your Agreement")
    uploaded_file = st.file_uploader(
        "Upload your tenancy agreement (PDF or TXT). The analysis will appear below.",
        type=["pdf", "txt"],
        label_visibility="collapsed"
    )

# Process the file only if it's a new file
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
                st.error("Could not extract text from the file. It might be a scanned image or empty.")
            else:
                st.session_state.agreement_text = text
                st.session_state.messages.append({"role": "ai", "content": f"✅ **Success!** Agreement `{uploaded_file.name}` is ready. Choose an analysis option or ask a tenancy-law question below."})
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    st.rerun()

# Step 2: Display Chat History
st.markdown("#### 💬 Chat & Analysis Results")
chat_container = st.container()
with chat_container:
    if not st.session_state.messages:
         st.info("The analysis results and conversation will appear here.")
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"], unsafe_allow_html=True)
    
    st.markdown('<div id="bottom_anchor"></div>', unsafe_allow_html=True)

# --- UTILITY FUNCTIONS ---

def perform_legal_analysis(analysis_type, agreement_text):
    """Pre-baked analysis actions (always tenancy-law scoped)."""
    prompts = {
        "Analyze Risks": "Analyze this rent agreement clause-by-clause for potential risks. For each clause, clearly identify the Issue, the Legal Violation, and the Tenant Risk. Do not redraft or ask questions, focus only on the risk analysis.",
        "Know Your Rights": "Based on the provided agreement, explain all relevant legal rights and obligations of the tenant (and where necessary, the landlord). For every point of contention, cite the specific Indian Law, Act, Section, or relevant case law that provides justification.",
        "Redraft Agreement": "Identify the most problematic clauses in this agreement. Provide a legally sound, tenant-favorable redrafted version for each. Include the original clause for comparison and a concise legal justification for the changes.",
        "Generate Questions": "For every contested, unfair, or ambiguous clause in this agreement, generate 3–5 critical questions that the tenant must ask the landlord for clarification before signing. Group the questions by clause."
    }
    human_prompt = HumanMessagePromptTemplate.from_template(
        f"{prompts[analysis_type]}\n\n--- AGREEMENT TEXT ---\n\n{{context}}"
    )
    chat_prompt = ChatPromptTemplate.from_messages([legal_system_prompt, human_prompt])
    chain = chat_prompt | llm_engine | StrOutputParser()
    with st.spinner(f"Performing '{analysis_type}' analysis..."):
        response = chain.invoke({"context": agreement_text})
        return response

def handle_chat_query(prompt, agreement_text=None):
    """
    Always answer as an expert in Indian tenancy & rent law.
    - If agreement_text is present → use it as context.
    - If absent → answer in 'law-only' mode, still within tenancy scope.
    - If user asks outside scope → steer them back (handled in system prompt).
    """
    if agreement_text:
        # Agreement-aware mode
        human_prompt = HumanMessagePromptTemplate.from_template(
            "User question: {question}\n\n--- RENT AGREEMENT CONTEXT ---\n\n{context}"
        )
        variables = {"question": prompt, "context": agreement_text}
    else:
        # Law-only mode (no uploaded agreement)
        human_prompt = HumanMessagePromptTemplate.from_template(
            "User question: {question}\n\n(Answer strictly using Indian tenancy & rent laws. Provide section/case citations where helpful.)"
        )
        variables = {"question": prompt}

    chat_prompt = ChatPromptTemplate.from_messages([legal_system_prompt, human_prompt])
    chain = chat_prompt | llm_engine | StrOutputParser()
    with st.spinner("Analyzing under Indian tenancy law..."):
        response = chain.invoke(variables)
        return response

# --- Step 3: Persistent Bottom Action Bar ---
if st.session_state.agreement_text:
    st.divider()
    st.markdown("##### **Quick Actions**")
    cols = st.columns(4)
    analysis_options = {
        "Analyze Risks": "️🕵️‍♂️",
        "Know Your Rights": "⚖️",
        "Redraft Agreement": "✍️",
        "Generate Questions": "❓"
    }
    
    for i, (option, icon) in enumerate(analysis_options.items()):
        if cols[i].button(f"{icon} {option}", key=f"opt_{i}"):
            st.session_state.messages.append({"role": "user", "content": f"Request: **{option}**"})
            response = perform_legal_analysis(option, st.session_state.agreement_text)
            st.session_state.messages.append({"role": "ai", "content": response})
            st.session_state.scroll_to_bottom = True
            st.rerun()

# --- Step 4: Chat Input (always tenancy-law focused) ---
placeholder = "Ask a tenancy-law question..." if st.session_state.agreement_text else "Ask a tenancy-law question (you can upload an agreement for deeper analysis)..."
if prompt := st.chat_input(placeholder):
    st.session_state.messages.append({"role": "user", "content": prompt})
    response = handle_chat_query(prompt, st.session_state.agreement_text)
    st.session_state.messages.append({"role": "ai", "content": response})
    st.session_state.scroll_to_bottom = True
    st.rerun()

# --- Step 5: Auto-scroll Javascript Injection ---
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
