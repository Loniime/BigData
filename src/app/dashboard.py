"""
GitHub Stars Predictor - All-in-One Dashboard
Everything on homepage: Metrics + Training + Predictions
"""

import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime
import subprocess
import json
import time
import os

from src.core.azure_storage import AzureStorage
from src.core.settings import Settings
from src.core.data_build import snapshots_to_dataframe

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="GitHub Stars MLOps Platform",
    page_icon="⭐",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =============================================================================
# CUSTOM CSS
# =============================================================================

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .main { padding: 0; }

    .hero {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        margin-bottom: 2rem;
    }
    .hero h1 { font-size: 2.5rem; margin-bottom: 0.5rem; }
    .hero p { font-size: 1rem; opacity: 0.9; }

    .card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
    }
    .card h3 { color: #1e3c72; margin-bottom: 1rem; }

    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
    }
    .metric-value { font-size: 2rem; font-weight: bold; margin: 0.5rem 0; }
    .metric-label { font-size: 0.9rem; opacity: 0.9; }

    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: bold;
        width: 100%;
    }
    
    .status-success {
        background: #10b981;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SESSION STATE
# =============================================================================

if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False
if "df_cache" not in st.session_state:
    st.session_state.df_cache = None
if "stats_cache" not in st.session_state:
    st.session_state.stats_cache = None
if "train_running" not in st.session_state:
    st.session_state.train_running = False
if "pred_generated" not in st.session_state:
    st.session_state.pred_generated = False
if "pred_df" not in st.session_state:
    st.session_state.pred_df = None

# =============================================================================
# HELPERS
# =============================================================================

@st.cache_resource
def get_storage():
    return AzureStorage()

def load_data_once():
    if st.session_state.data_loaded:
        return st.session_state.df_cache, st.session_state.stats_cache
    
    try:
        storage = get_storage()
        blob_names = storage.list_blobs(Settings.SNAPSHOT_PREFIX)
        snapshots = []
        for name in blob_names:
            snap = storage.download_json(name)
            if snap and "repos" in snap:
                snapshots.append(snap)
        
        if not snapshots:
            return None, None

        df = snapshots_to_dataframe(snapshots)
        df_sorted = df.sort_values(["repo_id", "date"])
        first = df_sorted.groupby("repo_id").first().reset_index()
        last = df_sorted.groupby("repo_id").last().reset_index()

        stats = first[["repo_id", "full_name", "language", "stars"]].rename(columns={"stars": "initial_stars"})
        stats = stats.merge(last[["repo_id", "stars"]].rename(columns={"stars": "stars_t"}), on="repo_id")
        n_days = df_sorted.groupby("repo_id")["date"].nunique().reset_index().rename(columns={"date": "n_days"})
        stats = stats.merge(n_days, on="repo_id")
        stats["stars_growth"] = stats["stars_t"] - stats["initial_stars"]
        stats["daily_growth"] = stats["stars_growth"] / stats["n_days"].clip(lower=1)

        st.session_state.df_cache = df
        st.session_state.stats_cache = stats
        st.session_state.data_loaded = True

        return df, stats
    except Exception as e:
        st.error(f"Error: {e}")
        return None, None

# =============================================================================
# HEADER
# =============================================================================

st.markdown("""
<div class="hero">
    <h1>⭐ GitHub Stars MLOps Platform</h1>
    <p>All-in-One: Metrics • Training • Predictions</p>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# LOAD DATA
# =============================================================================

if not st.session_state.data_loaded:
    with st.spinner("⏳ Loading data..."):
        load_data_once()

df, stats = load_data_once()

if df is None:
    st.error("No data available")
    st.stop()

# =============================================================================
# SECTION 1: MÉTRIQUES (3 métriques au lieu de 4)
# =============================================================================

st.markdown("## 📊 Quick Stats")

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(f"""
<div class="metric-card">
    <div class="metric-label">Repositories Tracked</div>
    <div class="metric-value">{stats['repo_id'].nunique():,}</div>
</div>
""", unsafe_allow_html=True)

with c2:
    st.markdown(f"""
<div class="metric-card">
    <div class="metric-label">Total Stars</div>
    <div class="metric-value">{int(stats['stars_t'].sum()):,}</div>
</div>
""", unsafe_allow_html=True)

with c3:
    days = int((df["date"].max() - df["date"].min()).days + 1)
    st.markdown(f"""
<div class="metric-card">
    <div class="metric-label">Days of Data</div>
    <div class="metric-value">{days}</div>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# SECTION 2: TRAINING
# =============================================================================

st.markdown("## 🎓 Model Training")

with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("**Configure Hyperparameters:**")
        
        ca, cb = st.columns(2)
        with ca:
            n_estimators = st.number_input("N Estimators", 50, 2000, 200, 50, key="train_n_est")
            max_depth = st.number_input("Max Depth", 2, 200, 20, 2, key="train_depth")
        with cb:
            window_days = st.number_input("Window Days", 1, 60, 7, 1, key="train_window")
            horizon_days = st.number_input("Horizon Days", 1, 30, 7, 1, key="train_horizon")
    
    with col2:
        st.markdown("**Current Model:**")
        storage = get_storage()
        meta = storage.download_json(Settings.META_BLOB)
        if meta:
            st.markdown('<div class="status-success">Ready</div>', unsafe_allow_html=True)
            st.metric("R²", f"{meta['metrics']['r2']:.3f}")
            st.metric("MAE", f"{meta['metrics']['mae']:.2f}")
        else:
            st.warning("No model trained yet")
    
    st.markdown("---")
    
    if st.button("🚀 START TRAINING", type="primary", disabled=st.session_state.train_running):
        st.session_state.train_running = True
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        log_box = st.empty()
        
        try:
            status_text.text("Preparing...")
            progress_bar.progress(10)
            
            env = os.environ.copy()
            env["RF_N_ESTIMATORS"] = str(n_estimators)
            env["RF_MAX_DEPTH"] = str(max_depth)
            env["ML_WINDOW_DAYS"] = str(window_days)
            env["ML_HORIZON_DAYS"] = str(horizon_days)
            
            status_text.text("Training... (logs below)")
            progress_bar.progress(20)
            
            proc = subprocess.Popen(
                ["python", "-m", "src.jobs.step4_train_model"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
                cwd="/app",
            )
            
            logs = []
            for line in proc.stdout:
                logs.append(line)
                logs = logs[-200:]
                log_box.code("".join(logs), language="text")
                progress_bar.progress(min(90, 20 + len(logs) // 4))
            
            proc.wait(timeout=5)
            
            model_exists = storage.download_bytes(Settings.MODEL_BLOB) is not None
            new_meta = storage.download_json(Settings.META_BLOB)
            
            if model_exists and new_meta:
                progress_bar.progress(100)
                status_text.text("✅ Completed!")
                st.success("✅ Training successful!")
                
                st.markdown("### 🎉 New Metrics")
                m1, m2, m3 = st.columns(3)
                m1.metric("R²", f"{new_meta['metrics']['r2']:.3f}")
                m2.metric("MAE", f"{new_meta['metrics']['mae']:.2f}")
                m3.metric("MSE", f"{new_meta['metrics']['mse']:.1f}")
                
                st.info("✅ Model saved to Azure Blob Storage")
                st.info("✅ Metrics logged to MLflow at http://localhost:5000")
            else:
                st.error("❌ Training failed")
        
        except Exception as e:
            st.error(f"❌ Error: {e}")
        
        st.session_state.train_running = False
    
    st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# SECTION 3: PRÉDICTIONS
# =============================================================================

st.markdown("## 🔮 Predictions")

with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    model_meta = storage.download_json(Settings.META_BLOB)
    
    if not model_meta:
        st.warning("⚠️ No model available. Train a model first!")
    else:
        col_a, col_b = st.columns([2, 1])
        
        with col_a:
            st.markdown(f"""
**Model Info:**
- R²: {model_meta['metrics']['r2']:.3f}
- MAE: {model_meta['metrics']['mae']:.2f}
- Last trained: {model_meta.get('created_at', 'N/A')[:16]}
            """)
        
        with col_b:
            st.markdown('<div class="status-success">Model Ready</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        if st.button("🔮 GENERATE PREDICTIONS", type="primary"):
            with st.spinner("Generating predictions..."):
                try:
                    prog = st.progress(0)
                    stat = st.empty()
                    
                    stat.text("Loading data...")
                    prog.progress(20)
                    
                    blob_names = storage.list_blobs(Settings.SNAPSHOT_PREFIX)
                    snapshots = []
                    for name in blob_names:
                        s = storage.download_json(name)
                        if s and "repos" in s:
                            snapshots.append(s)
                    
                    df_all = snapshots_to_dataframe(snapshots)
                    df_all["date"] = pd.to_datetime(df_all["date"])
                    last_date = df_all["date"].max()
                    
                    stat.text("Loading model...")
                    prog.progress(40)
                    
                    model_bytes = storage.download_bytes(Settings.MODEL_BLOB)
                    from joblib import load
                    with open("/tmp/model.joblib", "wb") as f:
                        f.write(model_bytes)
                    model = load("/tmp/model.joblib")
                    
                    stat.text("Building features...")
                    prog.progress(60)
                    
                    from src.core.data_build import build_features_for_prediction
                    X_pred, meta_pred, _ = build_features_for_prediction(
                        df_all,
                        window_days=int(getattr(Settings, "ML_WINDOW_DAYS", 7)),
                        as_of_date=last_date,
                    )
                    
                    stat.text("Predicting...")
                    prog.progress(80)
                    
                    y_pred = model.predict(X_pred)
                    
                    pred_df = meta_pred.copy()
                    if "stars" in pred_df.columns:
                        pred_df = pred_df.rename(columns={"stars": "stars_t"})
                    pred_df["pred_delta_stars_h"] = y_pred
                    pred_df["pred_stars_t_plus_h"] = pred_df["stars_t"] + pred_df["pred_delta_stars_h"]
                    
                    st.session_state.pred_df = pred_df
                    st.session_state.pred_generated = True
                    
                    prog.progress(100)
                    stat.text("✅ Done!")
                    st.success(f"✅ {len(pred_df)} predictions generated!")
                    
                except Exception as e:
                    st.error(f"❌ Error: {e}")
        
        # Afficher résultats
        if st.session_state.pred_generated and st.session_state.pred_df is not None:
            st.markdown("---")
            st.markdown("### 📊 Top Predictions")
            
            pred_df = st.session_state.pred_df
            
            top_n = st.slider("Show top N", 5, 50, 20, key="pred_slider")
            only_pos = st.checkbox("Only positive", value=True, key="pred_pos")
            
            show = pred_df.copy()
            if only_pos:
                show = show[show["pred_delta_stars_h"] > 0]
            
            show = show.sort_values("pred_delta_stars_h", ascending=False).head(top_n)
            
            st.dataframe(
                show[["full_name", "stars_t", "pred_delta_stars_h", "pred_stars_t_plus_h"]],
                use_container_width=True,
                hide_index=True,
            )
            
            # Export
            csv = pred_df.to_csv(index=False)
            st.download_button("📥 Download CSV", csv, "predictions.csv", "text/csv")
    
    st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# SECTION 4: TOP REPOS
# =============================================================================

st.markdown("## 🏆 Top Growing Repositories")

top_repos = stats.nlargest(10, "daily_growth")[
    ["full_name", "language", "initial_stars", "stars_t", "stars_growth", "daily_growth"]
]

st.dataframe(
    top_repos.style.background_gradient(subset=["daily_growth"], cmap="Blues"),
    use_container_width=True,
    hide_index=True,
)

# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; padding: 1rem;'>
    <p>GitHub Stars MLOps Platform • Azure + MLflow</p>
</div>
""", unsafe_allow_html=True)