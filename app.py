import streamlit as st
from transformers import pipeline
import PyPDF2

# --- Removed custom CSS injection to use Streamlit's default theme ---

# --- Dummy Credentials ---
# Financial Institution: username: fi_user / password: password
# Supervisor: username: sup_user / password: password
users = {
    "fi_user": {"password": "password", "role": "financial"},
    "sup_user": {"password": "password", "role": "supervisor"}
}

# --- Global Shared Data Using st.cache_resource ---
@st.cache_resource
def get_shared_data():
    # This dictionary is shared across all sessions.
    return {
        "policies": {},    # {username: {doc_name: doc_content}}
        "standards": {},   # {doc_name: doc_content}
        "evaluations": []  # list of dicts: {"username": ..., "document": ..., "result": ...}
    }

shared_data = get_shared_data()

# --- Utility Function to Extract Text from PDF ---
def extract_text_from_pdf(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted
    return text

# --- Simple Login System ---
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username in users and users[username]["password"] == password:
            st.session_state["logged_in"] = True
            st.session_state["username"] = username
            st.session_state["role"] = users[username]["role"]
            st.success("Logged in successfully!")
            st.rerun()  # Refresh to load the appropriate dashboard
        else:
            st.error("Invalid username or password")
else:
    # --- Logout Button ---
    if st.sidebar.button("Logout"):
        st.session_state.clear()
        st.rerun()

    user_role = st.session_state["role"]
    username = st.session_state["username"]

    # ============================
    # Financial Institution (FI)
    # ============================
    if user_role == "financial":
        st.title("Financial Institution Dashboard")
        st.write("Welcome! As a Financial Institution, you can upload your AMLCFT policy documents and evaluate compliance.")
        
        nav_option = st.sidebar.radio("Navigation", ["Upload Documents", "Evaluate Compliance"])
        
        # --- Upload Documents Page ---
        if nav_option == "Upload Documents":
            st.header("Upload AMLCFT Policy Documents")
            uploaded_files = st.file_uploader("Upload Policy Documents (txt, pdf)", type=["txt", "pdf"], accept_multiple_files=True)
            if uploaded_files:
                # Initialize FI user's policy docs if not already present.
                if username not in shared_data["policies"]:
                    shared_data["policies"][username] = {}
                for uploaded_file in uploaded_files:
                    if uploaded_file.name.lower().endswith(".pdf"):
                        try:
                            content = extract_text_from_pdf(uploaded_file)
                        except Exception as e:
                            st.error(f"Error reading {uploaded_file.name}: {e}")
                            continue
                    else:
                        try:
                            content = uploaded_file.read().decode('utf-8')
                        except Exception as e:
                            st.error(f"Error reading {uploaded_file.name}: {e}")
                            continue
                    shared_data["policies"][username][uploaded_file.name] = content
                    st.success(f"Uploaded {uploaded_file.name}")
                st.rerun()  # Refresh after upload

            # --- List and Delete Uploaded Documents ---
            if username in shared_data["policies"] and shared_data["policies"][username]:
                st.subheader("Your Uploaded Policy Documents:")
                for doc_name in list(shared_data["policies"][username].keys()):
                    col1, col2 = st.columns([3, 1])
                    col1.write(doc_name)
                    if col2.button("Delete", key=f"delete_{doc_name}"):
                        del shared_data["policies"][username][doc_name]
                        st.success(f"Deleted {doc_name}")
                        st.rerun()

        # --- Evaluate Compliance Page ---
        # --- Evaluate Compliance Page ---
        elif nav_option == "Evaluate Compliance":
            st.header("Compliance Evaluation")
            if username not in shared_data["policies"] or not shared_data["policies"][username]:
                st.warning("No policy documents uploaded. Please upload your documents first.")
                st.stop()
            if not shared_data["standards"]:
                st.warning("No benchmark standards available. Please contact your supervisor.")
                st.stop()
            
            # Let the FI user select one of their policy documents.
            policy_choice = st.selectbox("Select a Policy Document", list(shared_data["policies"][username].keys()))
            policy_text = shared_data["policies"][username][policy_choice]
            st.subheader("Policy Document Content")
            st.text_area("Policy Document", policy_text, height=200, key="policy_doc", disabled=True)
            
            if st.button("Evaluate Compliance"):
                with st.spinner("Evaluating compliance using vector retrieval..."):
                    # --- Build the vector index from the uploaded benchmark standards ---
                    from sentence_transformers import SentenceTransformer
                    import faiss
                    import numpy as np
                    
                    # Load the embedding model.
                    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                    # Get the list of benchmark standard texts.
                    benchmark_texts = list(shared_data["standards"].values())
                    
                    # Compute embeddings for benchmark standards.
                    benchmark_embeddings = embedding_model.encode(benchmark_texts, convert_to_numpy=True)
                    dimension = benchmark_embeddings.shape[1]
                    
                    # Create a FAISS index.
                    index = faiss.IndexFlatL2(dimension)
                    index.add(benchmark_embeddings)
                    
                    # Compute embedding for the policy document (the query).
                    query_embedding = embedding_model.encode([policy_text], convert_to_numpy=True)
                    k = min(3, len(benchmark_texts))  # retrieve top 3 contexts (or fewer if less available)
                    distances, indices = index.search(query_embedding, k)
                    
                    # Retrieve the most relevant benchmark context.
                    retrieved_context = "\n".join([benchmark_texts[i] for i in indices[0]])
                    
                    # --- Build the prompt for the LLM ---
                    compliance_generator = pipeline("text2text-generation", model="google/flan-t5-base")
                    prompt = (
                        f"""
                        You are an Anti-money laundering and counter-terrorism financing compliance (AMLCFT) expert. The following are some of the standards for AMLCFT compliance:\n
                        {retrieved_context}\n\n
                        The following policy document is an AMLCFT policy document adopted by a financial institution:\n"
                        "{policy_text}"\n\n
                        Evaluate the compliance of the policy document against the AMLCFT standards. Please return two things in your answer which include the compliance level and suggestions how to improve the AMLCFT compliance, if any.
                        The compliance level must be one of: Compliant, Largely Compliant, Partially Compliant, or Non-Compliant.
                        """
                    )
                    
                    result = compliance_generator(prompt, max_length=300, do_sample=False)
                    evaluation_output = result[0]['generated_text'].strip()
                    
                    # Store the evaluation result for supervisor access.
                    shared_data["evaluations"].append({
                        "username": username,
                        "document": policy_choice,
                        "result": evaluation_output
                    })
                    st.success("Evaluation Completed!")
                    st.text_area("Compliance Evaluation Result", evaluation_output, height=200, key="eval_result", disabled=True)

                    # Do not rerun here so that the result remains visible.

    # ============================
    # Supervisor
    # ============================
    elif user_role == "supervisor":
        st.title("Supervisor Dashboard")
        st.write("Welcome! As a Supervisor, you can upload benchmark AMLCFT standards, view Financial Institution documents, and review compliance evaluations.")
        
        nav_option = st.sidebar.radio("Navigation", ["Upload Benchmark Standards", "View FI Documents", "View Evaluation Results"])
        
        # --- Upload Benchmark Standards ---
        if nav_option == "Upload Benchmark Standards":
            st.header("Upload Benchmark AMLCFT Standards")
            uploaded_files = st.file_uploader("Upload Benchmark Standards (txt, pdf)", type=["txt", "pdf"], accept_multiple_files=True)
            if uploaded_files:
                for uploaded_file in uploaded_files:
                    if uploaded_file.name.lower().endswith(".pdf"):
                        try:
                            content = extract_text_from_pdf(uploaded_file)
                        except Exception as e:
                            st.error(f"Error reading {uploaded_file.name}: {e}")
                            continue
                    else:
                        try:
                            content = uploaded_file.read().decode('utf-8')
                        except Exception as e:
                            st.error(f"Error reading {uploaded_file.name}: {e}")
                            continue
                    shared_data["standards"][uploaded_file.name] = content
                    st.success(f"Uploaded {uploaded_file.name}")
                st.rerun()
            
            if shared_data["standards"]:
                st.subheader("Your Uploaded Benchmark Standards:")
                for doc_name in list(shared_data["standards"].keys()):
                    col1, col2 = st.columns([3, 1])
                    col1.write(doc_name)
                    if col2.button("Delete", key=f"delete_std_{doc_name}"):
                        del shared_data["standards"][doc_name]
                        st.success(f"Deleted {doc_name}")
                        st.rerun()
        
        # --- View FI Documents ---
        elif nav_option == "View FI Documents":
            st.header("View Financial Institution Documents")
            if not shared_data["policies"]:
                st.info("No Financial Institution documents have been uploaded yet.")
            else:
                for fi_user, docs in shared_data["policies"].items():
                    st.subheader(f"Documents from {fi_user}")
                    for doc_name, doc_content in docs.items():
                        st.text_area(f"{doc_name}", doc_content, height=150, key=f"fi_doc_{fi_user}_{doc_name}", disabled=True)
        
        # --- View Evaluation Results ---
        elif nav_option == "View Evaluation Results":
            st.header("View Compliance Evaluation Results")
            if not shared_data["evaluations"]:
                st.info("No compliance evaluations have been performed yet.")
            else:
                for idx, eval_item in enumerate(shared_data["evaluations"]):
                    st.subheader(f"User: {eval_item['username']} - Document: {eval_item['document']}")
                    st.text_area("Evaluation Result", eval_item["result"], height=100, key=f"eval_result_{idx}", disabled=True)
