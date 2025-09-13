import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os
import plotly.express as px
import plotly.figure_factory as ff
from streamlit_ace import st_ace

# Import our plagiarism detection modules
from plagiarism_checker.token_based import TokenBasedChecker
from plagiarism_checker.ast_based import ASTBasedChecker
from plagiarism_checker.codebert_based import CodeBERTChecker

# Set page configuration
st.set_page_config(
    page_title="Code Plagiarism Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {font-size: 3rem; color: #1f77b4; text-align: center}
    .sub-header {font-size: 1.5rem; color: #ff7f0e; border-bottom: 2px solid #ff7f0e}
    .highlight {background-color: #f0f2f6; padding: 15px; border-radius: 5px; margin: 10px 0}
    .result-box {padding: 20px; border-radius: 10px; margin: 10px 0}
    .similar {background-color: #e6f4ea}
    .suspicious {background-color: #fce8e6}
    .normal {background-color: #e8f0fe}
</style>
""", unsafe_allow_html=True)

# App header
st.markdown('<p class="main-header">🔍 Code Plagiarism Detector</p>', unsafe_allow_html=True)
st.markdown("### An AI-powered tool to detect plagiarism in source code")

# Sidebar
with st.sidebar:
    st.image("https://streamlit.io/images/brand/streamlit-mark-color.png", width=100)
    st.title("Settings")
    
    detection_method = st.selectbox(
        "Detection Method",
        ["Token-based", "AST-based", "CodeBERT-based"],
        help="Choose the plagiarism detection method"
    )
    
    threshold = st.slider(
        "Similarity Threshold", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.7,
        help="Threshold to flag code as potentially plagiarized"
    )
    
    language = st.selectbox(
        "Programming Language",
        ["Python", "Java", "C++", "JavaScript"],
        help="Select the programming language of the submitted code"
    )
    
    st.info("""
    This tool helps educators detect potential plagiarism in student assignments.
    Upload multiple files or paste code directly to compare.
    """)

# Main content
tab1, tab2, tab3 = st.tabs(["Upload Files", "Paste Code", "Results & Analysis"])

with tab1:
    st.markdown('<p class="sub-header">Upload Code Files</p>', unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "Choose source code files", 
        accept_multiple_files=True,
        type=['py', 'java', 'cpp', 'js', 'c', 'h']
    )
    
    if uploaded_files:
        file_data = []
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                file_data.append({
                    'name': uploaded_file.name,
                    'path': tmp_file.name,
                    'content': uploaded_file.getvalue().decode('utf-8')
                })
        
        st.session_state['file_data'] = file_data
        st.success(f"Uploaded {len(uploaded_files)} files successfully!")
        
        # Show file preview
        with st.expander("Preview Uploaded Files"):
            selected_file = st.selectbox("Select file to preview", [f['name'] for f in file_data])
            file_content = next((f['content'] for f in file_data if f['name'] == selected_file), "")
            st.code(file_content, language=language.lower())

with tab2:
    st.markdown('<p class="sub-header">Paste Code Directly</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("First Code Snippet")
        code1 = st_ace(
            language=language.lower(),
            theme="github",
            key="ace-1",
            height=300
        )
    
    with col2:
        st.subheader("Second Code Snippet")
        code2 = st_ace(
            language=language.lower(),
            theme="github",
            key="ace-2",
            height=300
        )
    
    if code1 and code2:
        st.session_state['pasted_code'] = [code1, code2]
        st.success("Code snippets ready for analysis!")

with tab3:
    st.markdown('<p class="sub-header">Analysis Results</p>', unsafe_allow_html=True)
    
    if ('file_data' in st.session_state or 'pasted_code' in st.session_state) and st.button("Run Plagiarism Check"):
        with st.spinner("Analyzing code for plagiarism..."):
            # Initialize the appropriate checker based on selection
            if detection_method == "Token-based":
                checker = TokenBasedChecker()
            elif detection_method == "AST-based":
                checker = ASTBasedChecker()
            else:
                checker = CodeBERTChecker()
            
            # Process files or pasted code
            if 'file_data' in st.session_state:
                file_data = st.session_state['file_data']
                results = []
                similarity_matrix = np.zeros((len(file_data), len(file_data)))
                
                for i, file1 in enumerate(file_data):
                    row_results = []
                    for j, file2 in enumerate(file_data):
                        if i == j:
                            similarity = 1.0
                        else:
                            similarity = checker.compare(file1['content'], file2['content'])
                        similarity_matrix[i][j] = similarity
                        row_results.append({
                            'file1': file1['name'],
                            'file2': file2['name'],
                            'similarity': similarity,
                            'status': "Potential Plagiarism" if similarity > threshold else "OK"
                        })
                    results.extend(row_results)
                
                # Convert results to DataFrame
                df = pd.DataFrame(results)
                
            else:  # Pasted code
                code1, code2 = st.session_state['pasted_code']
                similarity = checker.compare(code1, code2)
                
                df = pd.DataFrame({
                    'file1': ['Snippet 1'],
                    'file2': ['Snippet 2'],
                    'similarity': [similarity],
                    'status': ["Potential Plagiarism" if similarity > threshold else "OK"]
                })
                
                similarity_matrix = np.array([[1.0, similarity], [similarity, 1.0]])
            
            # Display results
            st.subheader("Similarity Matrix")
            
            if 'file_data' in st.session_state:
                file_names = [f['name'] for f in st.session_state['file_data']]
            else:
                file_names = ['Snippet 1', 'Snippet 2']
                
            fig = px.imshow(
                similarity_matrix,
                labels=dict(x="Files", y="Files", color="Similarity"),
                x=file_names,
                y=file_names,
                color_continuous_scale="RdYlGn_r",
                aspect="auto"
            )
            fig.update_layout(width=800, height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed results table
            st.subheader("Detailed Comparison Results")
            st.dataframe(df.style.applymap(
                lambda x: 'background-color: #fce8e6' if x == "Potential Plagiarism" else 'background-color: #e6f4ea', 
                subset=['status']
            ))
            
            # Highlight potential plagiarism cases
            suspicious_cases = df[df['similarity'] > threshold]
            if not suspicious_cases.empty:
                st.warning(f"Found {len(suspicious_cases)} potential plagiarism cases!")
                
                for _, case in suspicious_cases.iterrows():
                    if case['file1'] != case['file2']:  # Don't show self-comparisons
                        with st.expander(f"Potential plagiarism: {case['file1']} vs {case['file2']}"):
                            st.write(f"Similarity score: {case['similarity']:.3f}")
                            
                            # Show code comparison if only two files
                            if len(file_names) == 2 or 'pasted_code' in st.session_state:
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write(f"**{case['file1']}**")
                                    if 'file_data' in st.session_state:
                                        content1 = next(f['content'] for f in st.session_state['file_data'] 
                                                       if f['name'] == case['file1'])
                                    else:
                                        content1 = st.session_state['pasted_code'][0]
                                    st.code(content1, language=language.lower())
                                
                                with col2:
                                    st.write(f"**{case['file2']}**")
                                    if 'file_data' in st.session_state:
                                        content2 = next(f['content'] for f in st.session_state['file_data'] 
                                                       if f['name'] == case['file2'])
                                    else:
                                        content2 = st.session_state['pasted_code'][1]
                                    st.code(content2, language=language.lower())
            else:
                st.success("No potential plagiarism detected!")
                
            # Download results
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name="plagiarism_results.csv",
                mime="text/csv"
            )
    else:
        st.info("Upload files or paste code to analyze for plagiarism.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>This plagiarism detection tool uses multiple methods to identify similar code:</p>
    <ul style='display: inline-block; text-align: left'>
        <li><strong>Token-based</strong>: Compares code tokens after normalization</li>
        <li><strong>AST-based</strong>: Compares abstract syntax trees for structural similarity</li>
        <li><strong>CodeBERT-based</strong>: Uses AI-powered code embeddings for semantic analysis</li>
    </ul>
</div>
""", unsafe_allow_html=True)