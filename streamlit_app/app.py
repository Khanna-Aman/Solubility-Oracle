"""
Streamlit Dashboard for Solubility Oracle
Quick prototyping and visualization interface
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional
import torch
import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from models.ensemble import SolubilityEnsemble
from utils.molecular import validate_smiles, smiles_to_graph, get_molecular_descriptors

# Page configuration
st.set_page_config(
    page_title="Solubility Oracle",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">💧 Solubility Oracle</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Aqueous Solubility Prediction using AttentiveFP</p>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("⚙️ Configuration")
st.sidebar.markdown("---")

# Model loading (placeholder - in production, load from checkpoint)
@st.cache_resource
def load_model():
    """Load model - placeholder for actual checkpoint"""
    # In production, this would load from checkpoint
    # For demo purposes, we'll create a simple model
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Note: This is a placeholder - actual implementation would load trained model
        st.sidebar.success("Model ready (demo mode)")
        return None  # Return None for demo
    except Exception as e:
        st.sidebar.error(f"Model loading error: {e}")
        return None

model = load_model()

# Main content
tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Batch Analysis", "ℹ️ About"])

with tab1:
    st.header("Single Molecule Prediction")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        smiles_input = st.text_input(
            "Enter SMILES String",
            value="CC(=O)OC1=CC=CC=C1C(=O)O",
            help="Enter a valid SMILES string representing the molecular structure"
        )
        
        example_smiles = [
            "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
            "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
            "CC1=CC=C(C=C1)C(C)CC(=O)O",  # Naproxen
            "C1=CC=C(C=C1)C(=O)O",  # Benzoic acid
        ]
        
        st.markdown("**Example SMILES:**")
        for ex in example_smiles:
            if st.button(f"Use: {ex}", key=f"ex_{ex}"):
                smiles_input = ex
                st.rerun()
    
    with col2:
        st.markdown("### Quick Info")
        st.info("""
        **LogS Values:**
        - > 0: Highly Soluble
        - -2 to 0: Moderately Soluble
        - -4 to -2: Poorly Soluble
        - < -4: Very Poorly Soluble
        """)
    
    if st.button("🔮 Predict Solubility", type="primary", use_container_width=True):
        if not smiles_input:
            st.error("Please enter a SMILES string")
        elif not validate_smiles(smiles_input):
            st.error("Invalid SMILES string. Please check your input.")
        else:
            with st.spinner("Predicting solubility..."):
                # In production, use actual model
                # For demo, show placeholder
                st.warning("⚠️ Demo Mode: Model checkpoint not loaded. Install dependencies and train model for actual predictions.")
                
                # Show molecular info
                graph = smiles_to_graph(smiles_input)
                descriptors = get_molecular_descriptors(smiles_input)
                
                if graph is not None and descriptors is not None:
                    st.success("✅ Valid molecule processed!")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Atoms", graph.x.shape[0])
                    with col2:
                        st.metric("Bonds", graph.edge_index.shape[1] // 2)
                    with col3:
                        st.metric("Descriptors", len(descriptors))
                    
                    # Placeholder prediction
                    st.markdown("""
                    <div class="prediction-box">
                        <h2>Predicted LogS: N/A (Demo Mode)</h2>
                        <p>Train model to see actual predictions</p>
                    </div>
                    """, unsafe_allow_html=True)

with tab2:
    st.header("Batch Analysis")
    
    st.markdown("Upload a CSV file with SMILES strings for batch prediction.")
    
    uploaded_file = st.file_uploader(
        "Choose CSV file",
        type=['csv'],
        help="CSV should contain a 'SMILES' column"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            if 'SMILES' not in df.columns:
                st.error("CSV file must contain a 'SMILES' column")
            else:
                st.success(f"Loaded {len(df)} molecules")
                
                # Validate SMILES
                from utils.molecular import validate_smiles
                valid_mask = df['SMILES'].apply(validate_smiles)
                valid_df = df[valid_mask].copy()
                
                st.info(f"Valid SMILES: {valid_mask.sum()} / {len(df)}")
                
                if st.button("Process Batch", type="primary"):
                    st.warning("⚠️ Demo Mode: Batch processing requires trained model.")
                    
                    # Show sample data
                    st.dataframe(valid_df.head(10), use_container_width=True)
                    
                    # Visualization placeholder
                    if len(valid_df) > 0:
                        fig = px.histogram(
                            valid_df,
                            x='SMILES',
                            title="Molecule Distribution",
                            labels={'SMILES': 'Molecule Index'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error reading file: {e}")

with tab3:
    st.header("About Solubility Oracle")
    
    st.markdown("""
    ### 🎯 Overview
    
    **Solubility Oracle** is a comprehensive machine learning system for predicting 
    aqueous solubility (LogS) of chemical compounds using Graph Neural Networks 
    with AttentiveFP architecture.
    
    ### 🏗️ Architecture
    
    - **AttentiveFP**: State-of-the-art graph attention mechanism
    - **Hybrid Model**: Combines GNN features with molecular descriptors
    - **Ensemble**: Multiple model averaging for robustness
    
    ### 📊 Features
    
    - SMILES-based predictions
    - Real-time inference
    - Batch processing
    - Interactive visualizations
    
    ### 🔬 Technical Stack
    
    - **Deep Learning**: PyTorch, PyTorch Geometric
    - **Chemistry**: RDKit, Mordred
    - **Backend**: FastAPI
    - **Frontend**: React.js, TypeScript
    - **Dashboard**: Streamlit
    
    ### 📚 Usage
    
    1. Enter a SMILES string in the Prediction tab
    2. Click "Predict Solubility" to get LogS prediction
    3. Use Batch Analysis for multiple molecules
    
    ### ⚠️ Note
    
    This is a demo interface. For actual predictions, ensure:
    - Model checkpoint is available in `checkpoints/best_model.pt`
    - All dependencies are installed
    - Model has been trained on solubility data
    """)
    
    st.markdown("---")
    st.markdown("**Version**: 1.0.0 | **Status**: Active Development")
