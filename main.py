import streamlit as st
import os
import mlflow
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from src.core.azure_storage import AzureStorage
from src.core.settings import Settings
from src.core.data_build import snapshots_to_dataframe

# =============================================================================
# HELPERS
# =============================================================================

def ms_to_dt(ms: int):
    if ms is None:
        return None
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).astimezone()

@st.cache_resource
def get_storage():
    return AzureStorage()

def format_number(num):
    if isinstance(num, float):
        return f"{num:,.2f}"
    return f"{num:,}"

# =============================================================================
# FONCTION TRAINING RÉGRESSION
# =============================================================================

def run_training(progress_callback=None):
    """Exécute le training régression avec les paramètres par défaut"""
    import sys
    from io import StringIO
    
    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()
    
    try:
        from src.jobs.step4_train_model import main as train_main
        
        if progress_callback:
            progress_callback(30, "Loading data...")
        
        if progress_callback:
            progress_callback(60, "Training model...")
        
        train_main()
        
        if progress_callback:
            progress_callback(100, "Completed!")
        
        return True, "Training completed successfully"
    
    except Exception as e:
        return False, str(e)
    
    finally:
        sys.stdout = old_stdout

# =============================================================================
# FONCTION TRAINING CLASSIFICATION
# =============================================================================

def run_classifier_training(progress_callback=None):
    """Exécute le training du classifier"""
    import sys
    from io import StringIO
    
    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()
    
    try:
        from src.jobs.step4b_train_classifier import main as train_clf_main
        
        if progress_callback:
            progress_callback(30, "Loading data...")
        
        if progress_callback:
            progress_callback(60, "Training classifier...")
        
        train_clf_main()
        
        if progress_callback:
            progress_callback(100, "Completed!")
        
        return True, "Classifier training completed successfully"
    
    except Exception as e:
        return False, str(e)
    
    finally:
        sys.stdout = old_stdout

# =============================================================================
# FONCTION PREDICTION
# =============================================================================

def run_prediction(progress_callback=None):
    """Exécute les prédictions (régression + classification)"""
    import sys
    from io import StringIO
    
    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()
    
    try:
        from src.jobs.step5_predict_monitor import main as predict_main
        
        if progress_callback:
            progress_callback(30, "Loading models...")
        
        if progress_callback:
            progress_callback(60, "Generating predictions...")
        
        predict_main()
        
        if progress_callback:
            progress_callback(100, "Completed!")
        
        return True, "Predictions generated successfully"
    
    except Exception as e:
        return False, str(e)
    
    finally:
        sys.stdout = old_stdout

# =============================================================================
# SESSION STATE
# =============================================================================

if "train_running" not in st.session_state:
    st.session_state.train_running = False
if "clf_train_running" not in st.session_state:
    st.session_state.clf_train_running = False
if "pred_generated" not in st.session_state:
    st.session_state.pred_generated = False
if "pred_df" not in st.session_state:
    st.session_state.pred_df = None
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False
if "df_cache" not in st.session_state:
    st.session_state.df_cache = None
if "stats_cache" not in st.session_state:
    st.session_state.stats_cache = None
if "current_page" not in st.session_state:
    st.session_state.current_page = "Home"

# =============================================================================
# LOAD DATA
# =============================================================================

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
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="GitHub Stars MLOps Platform",
    layout="wide"
)

st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: scale(1.05);
    }
    .metric-value { font-size: 2rem; font-weight: bold; }
    .metric-label { font-size: 0.9rem; opacity: 0.9; }
    .main-title {
        text-align: center;
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .intro-text {
        text-align: center;
        font-size: 1.1rem;
        color: #666;
        max-width: 800px;
        margin: 0 auto 2rem auto;
        line-height: 1.6;
    }
    .section-title {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1e3c72;
        margin-top: 3rem;
        margin-bottom: 1rem;
        text-align: center;
    }
    .section-desc {
        font-size: 1rem;
        color: #666;
        margin-bottom: 2rem;
        line-height: 1.6;
        text-align: center;
        max-width: 700px;
        margin-left: auto;
        margin-right: auto;
    }
    .action-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        padding: 2.5rem;
        margin-bottom: 3rem;
        box-shadow: 0 8px 16px rgba(102, 126, 234, 0.3);
    }
    .action-title {
        font-size: 1.8rem;
        font-weight: bold;
        margin-bottom: 1rem;
        text-align: center;
    }
    .action-desc {
        font-size: 1rem;
        margin-bottom: 2rem;
        opacity: 0.95;
        text-align: center;
        line-height: 1.6;
    }
    .stButton > button {
        background: white !important;
        color: #667eea !important;
        border: none !important;
        font-weight: bold !important;
        padding: 0.75rem 2rem !important;
        border-radius: 25px !important;
        transition: all 0.3s !important;
    }
    .stButton > button:hover {
        transform: scale(1.05) !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2) !important;
    }
    .explain-box {
        background: #f0f4ff;
        border-left: 4px solid #667eea;
        padding: 1.5rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
    }
    .explain-title {
        font-size: 1.1rem;
        font-weight: bold;
        color: #1e3c72;
        margin-bottom: 0.5rem;
    }
    .explain-text {
        font-size: 0.95rem;
        color: #555;
        line-height: 1.6;
    }
    .column-explain {
        background: white;
        border: 1px solid #e5e7eb;
        padding: 1rem;
        border-radius: 6px;
        margin-bottom: 0.5rem;
    }
    .column-name {
        font-weight: bold;
        color: #667eea;
        font-size: 0.9rem;
    }
    .column-desc {
        color: #666;
        font-size: 0.85rem;
        margin-top: 0.3rem;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# MAIN
# =============================================================================

def main():
    
    if st.session_state.current_page == "Home":
        # Titre centré
        st.markdown('<h1 class="main-title">GitHub Stars MLOps Platform</h1>', unsafe_allow_html=True)
        
        # Introduction
        st.markdown("""
<div class="intro-text">
This platform provides an end-to-end machine learning pipeline for predicting GitHub repository growth. 
Using historical star data, the system trains both regression and classification models to forecast 
repository performance and detect potential viral growth patterns.
</div>
""", unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Load data
        if not st.session_state.data_loaded:
            with st.spinner("Loading data..."):
                load_data_once()
        
        df, stats = load_data_once()
        
        if df is None:
            st.error("No data available")
            st.stop()
        
        # Quick Stats
        st.markdown('<div class="section-title">Current Dataset</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-desc">The platform continuously tracks GitHub repositories and collects daily snapshots of their metrics.</div>', unsafe_allow_html=True)
        
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
        
        # Top Growing Repositories
        st.markdown('<div class="section-title">Top Growing Repositories</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-desc">These repositories have shown the highest growth in stars over the tracking period.</div>', unsafe_allow_html=True)
        
        top_repos = stats.nlargest(10, "daily_growth")[
            ["full_name", "language", "initial_stars", "stars_t", "stars_growth"]
        ]
        st.dataframe(
            top_repos.style.background_gradient(subset=["stars_growth"], cmap="Blues"),
            use_container_width=True,
            hide_index=True,
        )
        
        # Time Series Chart
        st.markdown('<div class="section-title">Repository Stars Over Time</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-desc">Evolution of star counts for the top 5 fastest-growing repositories.</div>', unsafe_allow_html=True)
        
        import plotly.express as px
        top_5_repos = stats.nlargest(5, "daily_growth")["full_name"].tolist()
        df_top = df[df["full_name"].isin(top_5_repos)]
        fig = px.area(
            df_top, 
            x="date", 
            y="stars", 
            color="full_name",
            title="",
            labels={"date": "Date", "stars": "Stars", "full_name": "Repository"}
        )
        fig.update_layout(
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # =========================================================================
        # SECTION: MODEL PERFORMANCE MONITORING
        # =========================================================================
        
        st.markdown('<div class="section-title">Model Performance Monitoring</div>', unsafe_allow_html=True)
        st.markdown("""
<div class="section-desc">
Track how well the model performs over time. The backtest evaluates the model on historical data 
to measure prediction accuracy and detect when retraining is needed.
</div>
""", unsafe_allow_html=True)
        
        storage = get_storage()
        backtest_data = storage.download_json("monitoring/metrics_history_backtest.json")
        
        if backtest_data and "history" in backtest_data and len(backtest_data["history"]) > 0:
            history = pd.DataFrame(backtest_data["history"])
            history["t"] = pd.to_datetime(history["t"])
            history["r2"] = history["metrics"].apply(lambda x: x.get("r2"))
            history["mae"] = history["metrics"].apply(lambda x: x.get("mae"))
            history["rmse"] = history["metrics"].apply(lambda x: x.get("rmse"))
            
            # Identifier les points de retraining
            retrains = history[history["retrained_today"] == True].copy()
            
            # Graphique R² avec points de retraining
            import plotly.graph_objects as go
            
            fig_r2 = go.Figure()
            
            # Ligne R²
            fig_r2.add_trace(go.Scatter(
                x=history["t"],
                y=history["r2"],
                mode='lines+markers',
                name='R² Score',
                line=dict(color='#667eea', width=2),
                marker=dict(size=6)
            ))
            
            # Points de retraining
            if len(retrains) > 0:
                fig_r2.add_trace(go.Scatter(
                    x=retrains["t"],
                    y=retrains["r2"],
                    mode='markers',
                    name='Model Retrained',
                    marker=dict(
                        size=12,
                        color='#10b981',
                        symbol='star',
                        line=dict(color='white', width=2)
                    )
                ))
            
            # Ligne de seuil
            r2_threshold = backtest_data.get("r2_threshold", 0.10)
            fig_r2.add_hline(
                y=r2_threshold, 
                line_dash="dash", 
                line_color="red",
                annotation_text=f"Retrain Threshold ({r2_threshold})"
            )
            
            fig_r2.update_layout(
                title="R² Score Evolution (Regression)",
                xaxis_title="Date",
                yaxis_title="R² Score",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig_r2, use_container_width=True)
            
            # Graphique MAE
            fig_mae = go.Figure()
            
            fig_mae.add_trace(go.Scatter(
                x=history["t"],
                y=history["mae"],
                mode='lines+markers',
                name='MAE (Mean Absolute Error)',
                line=dict(color='#764ba2', width=2),
                marker=dict(size=6),
                fill='tozeroy',
                fillcolor='rgba(118, 75, 162, 0.1)'
            ))
            
            # Points de retraining
            if len(retrains) > 0:
                fig_mae.add_trace(go.Scatter(
                    x=retrains["t"],
                    y=retrains["mae"],
                    mode='markers',
                    name='Model Retrained',
                    marker=dict(
                        size=12,
                        color='#10b981',
                        symbol='star',
                        line=dict(color='white', width=2)
                    )
                ))
            
            fig_mae.update_layout(
                title="Prediction Error Over Time (MAE)",
                xaxis_title="Date",
                yaxis_title="MAE (stars)",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig_mae, use_container_width=True)
            
            # Graphique RMSE
            fig_rmse = go.Figure()
            
            fig_rmse.add_trace(go.Scatter(
                x=history["t"],
                y=history["rmse"],
                mode='lines+markers',
                name='RMSE (Root Mean Squared Error)',
                line=dict(color='#f59e0b', width=2),
                marker=dict(size=6),
                fill='tozeroy',
                fillcolor='rgba(245, 158, 11, 0.1)'
            ))
            
            # Points de retraining
            if len(retrains) > 0:
                fig_rmse.add_trace(go.Scatter(
                    x=retrains["t"],
                    y=retrains["rmse"],
                    mode='markers',
                    name='Model Retrained',
                    marker=dict(
                        size=12,
                        color='#10b981',
                        symbol='star',
                        line=dict(color='white', width=2)
                    )
                ))
            
            fig_rmse.update_layout(
                title="Prediction Error Over Time (RMSE)",
                xaxis_title="Date",
                yaxis_title="RMSE (stars)",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig_rmse, use_container_width=True)
        else:
            st.info("No backtest data available yet. Run the backtest job to generate monitoring data.")
        # Ajouter APRÈS le monitoring de régression (ligne ~400 environ)

        # =========================================================================
        # SECTION: CLASSIFICATION MODEL PERFORMANCE MONITORING
        # =========================================================================
        
        st.markdown('<div class="section-title">Classification Model Performance Monitoring</div>', unsafe_allow_html=True)
        st.markdown("""
<div class="section-desc">
Track the classification model's ability to correctly identify growth regimes over time. 
Monitor F1 score, recall for explosion class, and precision to ensure early detection of viral repositories.
</div>
""", unsafe_allow_html=True)
        
        clf_backtest_data = storage.download_json("monitoring/classifier_metrics_history_backtest.json")
        
        if clf_backtest_data and "history" in clf_backtest_data and len(clf_backtest_data["history"]) > 0:
            clf_history = pd.DataFrame(clf_backtest_data["history"])
            clf_history["t"] = pd.to_datetime(clf_history["t"])
            clf_history["accuracy"] = clf_history["metrics"].apply(lambda x: x.get("accuracy"))
            clf_history["f1_macro"] = clf_history["metrics"].apply(lambda x: x.get("f1_macro"))
            clf_history["recall_explosion"] = clf_history["metrics"].apply(lambda x: x.get("recall_explosion"))
            clf_history["precision_explosion"] = clf_history["metrics"].apply(lambda x: x.get("precision_explosion"))
            
            # Identifier les points de retraining
            clf_retrains = clf_history[clf_history["retrained_today"] == True].copy()
            
            import plotly.graph_objects as go
            
            # Graphique F1 Macro
            fig_f1 = go.Figure()
            
            fig_f1.add_trace(go.Scatter(
                x=clf_history["t"],
                y=clf_history["f1_macro"],
                mode='lines+markers',
                name='F1 Macro Score',
                line=dict(color='#667eea', width=2),
                marker=dict(size=6)
            ))
            
            # Points de retraining
            if len(clf_retrains) > 0:
                fig_f1.add_trace(go.Scatter(
                    x=clf_retrains["t"],
                    y=clf_retrains["f1_macro"],
                    mode='markers',
                    name='Classifier Retrained',
                    marker=dict(
                        size=12,
                        color='#10b981',
                        symbol='star',
                        line=dict(color='white', width=2)
                    )
                ))
            
            # Ligne de seuil F1
            f1_threshold = clf_backtest_data.get("f1_threshold", 0.40)
            fig_f1.add_hline(
                y=f1_threshold, 
                line_dash="dash", 
                line_color="red",
                annotation_text=f"Retrain Threshold ({f1_threshold})"
            )
            
            fig_f1.update_layout(
                title="F1 Macro Score Evolution (Classification)",
                xaxis_title="Date",
                yaxis_title="F1 Macro",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig_f1, use_container_width=True)
            
            # Graphique Recall Explosion
            fig_recall = go.Figure()
            
            # Filtrer les valeurs non-null pour recall
            clf_history_recall = clf_history[clf_history["recall_explosion"].notna()].copy()
            
            fig_recall.add_trace(go.Scatter(
                x=clf_history_recall["t"],
                y=clf_history_recall["recall_explosion"],
                mode='lines+markers',
                name='Recall Explosion Class',
                line=dict(color='#dc2626', width=2),
                marker=dict(size=6),
                fill='tozeroy',
                fillcolor='rgba(220, 38, 38, 0.1)'
            ))
            
            # Points de retraining
            if len(clf_retrains) > 0:
                clf_retrains_recall = clf_retrains[clf_retrains["recall_explosion"].notna()].copy()
                if len(clf_retrains_recall) > 0:
                    fig_recall.add_trace(go.Scatter(
                        x=clf_retrains_recall["t"],
                        y=clf_retrains_recall["recall_explosion"],
                        mode='markers',
                        name='Classifier Retrained',
                        marker=dict(
                            size=12,
                            color='#10b981',
                            symbol='star',
                            line=dict(color='white', width=2)
                        )
                    ))
            
            # Ligne de seuil Recall
            recall_threshold = clf_backtest_data.get("recall_threshold", 0.30)
            fig_recall.add_hline(
                y=recall_threshold, 
                line_dash="dash", 
                line_color="red",
                annotation_text=f"Retrain Threshold ({recall_threshold})"
            )
            
            fig_recall.update_layout(
                title="Recall for Explosion Class Over Time",
                xaxis_title="Date",
                yaxis_title="Recall (Explosion)",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig_recall, use_container_width=True)
            
            # Graphique Precision Explosion
            fig_precision = go.Figure()
            
            # Filtrer les valeurs non-null pour precision
            clf_history_precision = clf_history[clf_history["precision_explosion"].notna()].copy()
            
            fig_precision.add_trace(go.Scatter(
                x=clf_history_precision["t"],
                y=clf_history_precision["precision_explosion"],
                mode='lines+markers',
                name='Precision Explosion Class',
                line=dict(color='#f59e0b', width=2),
                marker=dict(size=6),
                fill='tozeroy',
                fillcolor='rgba(245, 158, 11, 0.1)'
            ))
            
            # Points de retraining
            if len(clf_retrains) > 0:
                clf_retrains_precision = clf_retrains[clf_retrains["precision_explosion"].notna()].copy()
                if len(clf_retrains_precision) > 0:
                    fig_precision.add_trace(go.Scatter(
                        x=clf_retrains_precision["t"],
                        y=clf_retrains_precision["precision_explosion"],
                        mode='markers',
                        name='Classifier Retrained',
                        marker=dict(
                            size=12,
                            color='#10b981',
                            symbol='star',
                            line=dict(color='white', width=2)
                        )
                    ))
            
            fig_precision.update_layout(
                title="Precision for Explosion Class Over Time",
                xaxis_title="Date",
                yaxis_title="Precision (Explosion)",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig_precision, use_container_width=True)
            
            # Graphique Accuracy
            fig_accuracy = go.Figure()
            
            fig_accuracy.add_trace(go.Scatter(
                x=clf_history["t"],
                y=clf_history["accuracy"],
                mode='lines+markers',
                name='Overall Accuracy',
                line=dict(color='#8b5cf6', width=2),
                marker=dict(size=6),
                fill='tozeroy',
                fillcolor='rgba(139, 92, 246, 0.1)'
            ))
            
            # Points de retraining
            if len(clf_retrains) > 0:
                fig_accuracy.add_trace(go.Scatter(
                    x=clf_retrains["t"],
                    y=clf_retrains["accuracy"],
                    mode='markers',
                    name='Classifier Retrained',
                    marker=dict(
                        size=12,
                        color='#10b981',
                        symbol='star',
                        line=dict(color='white', width=2)
                    )
                ))
            
            fig_accuracy.update_layout(
                title="Overall Accuracy Over Time",
                xaxis_title="Date",
                yaxis_title="Accuracy",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig_accuracy, use_container_width=True)
        else:
            st.info("No classifier backtest data available yet. Run the classifier backtest job (step6b_backtest_classifier.py) to generate monitoring data.")
        # =========================================================================
        # SECTION: REGRESSION TRAINING
        # =========================================================================
        
        st.markdown('<div class="section-title">Train Regression Model</div>', unsafe_allow_html=True)
        
        st.markdown("""
<div class="action-box">
    <div class="action-title">Regression Model Training</div>
    <div class="action-desc">
    The regression model predicts the exact number of stars a repository will gain over the next 7 days. 
    It uses temporal features and historical patterns to forecast precise numeric growth values.
    </div>
</div>
""", unsafe_allow_html=True)
        
        # Current model status
        meta = storage.download_json(Settings.META_BLOB)
        
        col_status1, col_status2, col_status3 = st.columns(3)
        
        if meta:
            with col_status1:
                st.metric("Current Model R²", f"{meta['metrics']['r2']:.3f}")
            with col_status2:
                st.metric("Current Model MAE", format_number(meta['metrics']['mae']))
            with col_status3:
                st.caption(f"Last trained: {meta.get('created_at', 'N/A')[:16]}")
        else:
            with col_status1:
                st.info("No regression model trained yet")
            with col_status2:
                st.write("")
            with col_status3:
                st.write("")
        
        st.markdown("")
        
        # Bouton training centré
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            if st.button("START REGRESSION TRAINING", type="primary", use_container_width=True, disabled=st.session_state.train_running, key="train_reg"):
                st.session_state.train_running = True
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(pct, msg):
                    progress_bar.progress(pct)
                    status_text.text(msg)
                
                try:
                    success, message = run_training(progress_callback=update_progress)
                    
                    if success:
                        st.success("Regression training completed!")
                        
                        new_meta = storage.download_json(Settings.META_BLOB)
                        if new_meta:
                            st.markdown("**New Model Performance:**")
                            m1, m2, m3 = st.columns(3)
                            m1.metric("R²", f"{new_meta['metrics']['r2']:.3f}")
                            m2.metric("MAE", format_number(new_meta['metrics']['mae']))
                            m3.metric("RMSE", format_number(new_meta['metrics'].get('rmse', 0)))
                            
                            st.info("Model saved to Azure and logged to MLflow")
                    else:
                        st.error(f"Training failed: {message}")
                
                except Exception as e:
                    st.error(f"Error: {e}")
                
                st.session_state.train_running = False
        
        # =========================================================================
        # SECTION: CLASSIFICATION TRAINING
        # =========================================================================
        
        st.markdown('<div class="section-title">Train Classification Model</div>', unsafe_allow_html=True)
        
        st.markdown("""
<div class="action-box">
    <div class="action-title">Classification Model Training</div>
    <div class="action-desc">
    The classification model predicts the growth regime of a repository: Stable, Moderate Growth (Faible), 
    or Explosive Growth (Explosion). This helps detect potential viral repositories early by categorizing 
    their future trajectory. The classes are defined using relative gain thresholds to ensure fairness 
    across repositories of different sizes.
    </div>
</div>
""", unsafe_allow_html=True)
        
        # Current classifier status
        clf_meta = storage.download_json(getattr(Settings, "CLASSIFIER_META_BLOB", "models/classifier_meta.json"))
        
        col_clf1, col_clf2, col_clf3 = st.columns(3)
        
        if clf_meta:
            with col_clf1:
                st.metric("F1 Macro", f"{clf_meta['metrics'].get('f1_macro', 0):.3f}")
            with col_clf2:
                st.metric("Recall Explosion", f"{clf_meta['metrics'].get('recall_explosion', 0):.3f}")
            with col_clf3:
                st.caption(f"Last trained: {clf_meta.get('created_at', 'N/A')[:16]}")
        else:
            with col_clf1:
                st.info("No classifier trained yet")
            with col_clf2:
                st.write("")
            with col_clf3:
                st.write("")
        
        st.markdown("")
        
        # Bouton classifier training
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            if st.button("START CLASSIFICATION TRAINING", type="primary", use_container_width=True, disabled=st.session_state.clf_train_running, key="train_clf"):
                st.session_state.clf_train_running = True
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(pct, msg):
                    progress_bar.progress(pct)
                    status_text.text(msg)
                
                try:
                    success, message = run_classifier_training(progress_callback=update_progress)
                    
                    if success:
                        st.success("Classification training completed!")
                        
                        new_clf_meta = storage.download_json(getattr(Settings, "CLASSIFIER_META_BLOB", "models/classifier_meta.json"))
                        if new_clf_meta:
                            st.markdown("**New Classifier Performance:**")
                            m1, m2, m3, m4 = st.columns(4)
                            m1.metric("Accuracy", f"{new_clf_meta['metrics'].get('accuracy', 0):.3f}")
                            m2.metric("F1 Macro", f"{new_clf_meta['metrics'].get('f1_macro', 0):.3f}")
                            m3.metric("Recall Explosion", f"{new_clf_meta['metrics'].get('recall_explosion', 0):.3f}")
                            m4.metric("Precision Explosion", f"{new_clf_meta['metrics'].get('precision_explosion', 0):.3f}")
                            
                            st.info("Classifier saved to Azure and logged to MLflow")
                    else:
                        st.error(f"Training failed: {message}")
                
                except Exception as e:
                    st.error(f"Error: {e}")
                
                st.session_state.clf_train_running = False
        
        # =========================================================================
        # SECTION: PREDICTION (Régression + Classification)
        # =========================================================================
        
        st.markdown('<div class="section-title">Generate Predictions</div>', unsafe_allow_html=True)
        
        st.markdown("""
<div class="action-box">
    <div class="action-title">Prediction Generation</div>
    <div class="action-desc">
    Generate both regression predictions (exact star gain) and classification predictions (growth regime) 
    for all tracked repositories. The system uses both models to provide a complete view of future repository performance.
    </div>
</div>
""", unsafe_allow_html=True)
        
        model_meta = storage.download_json(Settings.META_BLOB)
        
        if not model_meta:
            st.warning("Please train the regression model first before generating predictions.")
        else:
            col_pred1, col_pred2, col_pred3 = st.columns(3)
            
            with col_pred1:
                st.metric("Regression Model", "Ready")
            with col_pred2:
                clf_status = "Ready" if clf_meta else "Not Trained"
                st.metric("Classification Model", clf_status)
            with col_pred3:
                st.metric("Prediction Horizon", f"{model_meta.get('horizon_days', 7)} days")
            
            st.markdown("")
            
            # Bouton prediction centré
            col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
            with col_btn2:
                if st.button("GENERATE PREDICTIONS", type="primary", use_container_width=True, key="pred_main"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    def update_progress(pct, msg):
                        progress_bar.progress(pct)
                        status_text.text(msg)
                    
                    try:
                        success, message = run_prediction(progress_callback=update_progress)
                        
                        if success:
                            st.success("Predictions generated successfully!")
                            
                            blob_names = storage.list_blobs(Settings.PRED_PREFIX)
                            if blob_names:
                                latest = sorted(blob_names)[-1]
                                pred_data = storage.download_json(latest)
                                
                                if pred_data:
                                    pred_df = pd.DataFrame(pred_data.get("predictions", []))
                                    st.session_state.pred_df = pred_df
                                    st.session_state.pred_date = pred_data.get("as_of_date")
                                    st.session_state.pred_generated = True
                        else:
                            st.error(f"Prediction failed: {message}")
                    
                    except Exception as e:
                        st.error(f"Error: {e}")
        
        # =========================================================================
        # SECTION: CLASSIFICATION RESULTS
        # =========================================================================
        
        if st.session_state.pred_generated and st.session_state.pred_df is not None:
            pred_df = st.session_state.pred_df
            
            # Check if classification predictions exist
            has_classification = "pred_class" in pred_df.columns and pred_df["pred_class"].notna().any()
            
            if has_classification:
                st.markdown("---")
                st.markdown('<div class="section-title">Classification Results</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="section-desc">Growth regime predictions for {st.session_state.pred_date}</div>', unsafe_allow_html=True)
                
                # Mapping des classes
                class_names = {0: "Stable", 1: "Faible", 2: "Explosion"}
                pred_df["class_name"] = pred_df["pred_class"].map(class_names)
                
                # Distribution des classes
                st.markdown("### Class Distribution")
                class_counts = pred_df["class_name"].value_counts()
                
                import plotly.graph_objects as go
                
                fig_dist = go.Figure(data=[
                    go.Bar(
                        x=class_counts.index,
                        y=class_counts.values,
                        marker=dict(
                            color=['#4338ca', '#92400e', '#991b1b'],
                        ),
                        text=class_counts.values,
                        textposition='auto',
                    )
                ])
                
                fig_dist.update_layout(
                    title="Predicted Growth Regimes",
                    xaxis_title="Class",
                    yaxis_title="Number of Repositories",
                    showlegend=False
                )
                
                st.plotly_chart(fig_dist, use_container_width=True)
            
            # =========================================================================
            # SECTION: REGRESSION RESULTS (TABLE)
            # =========================================================================
            
            st.markdown("---")
            st.markdown('<div class="section-title">Prediction Details</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="section-desc">Detailed predictions generated on {st.session_state.pred_date}</div>', unsafe_allow_html=True)
            
            # EXPLICATION DU TABLEAU
            st.markdown("""
<div class="explain-box">
    <div class="explain-title">Understanding the Predictions Table</div>
    <div class="explain-text">
    This table shows forecasted growth for GitHub repositories over the next 7 days. Each row represents one repository with its current state and predicted future.
    </div>
</div>
""", unsafe_allow_html=True)
            
            col_ex1, col_ex2, col_ex3, col_ex4 = st.columns(4)
            
            with col_ex1:
                st.markdown("""
<div class="column-explain">
    <div class="column-name">full_name</div>
    <div class="column-desc">Repository identifier (owner/repo)</div>
</div>
""", unsafe_allow_html=True)
            
            with col_ex2:
                st.markdown("""
<div class="column-explain">
    <div class="column-name">stars_t</div>
    <div class="column-desc">Current star count (today)</div>
</div>
""", unsafe_allow_html=True)
            
            with col_ex3:
                st.markdown("""
<div class="column-explain">
    <div class="column-name">pred_delta_stars_h</div>
    <div class="column-desc">Predicted gain over next 7 days</div>
</div>
""", unsafe_allow_html=True)
            
            with col_ex4:
                st.markdown("""
<div class="column-explain">
    <div class="column-name">pred_stars_t_plus_h</div>
    <div class="column-desc">Predicted total stars in 7 days</div>
</div>
""", unsafe_allow_html=True)
            
            st.markdown("---")
            
            col_a, col_b = st.columns(2)
            with col_a:
                top_n = st.slider("Show top N repositories", 5, 100, 20)
            with col_b:
                only_pos = st.checkbox("Only positive predictions", True)
            
            show = pred_df.copy()
            if only_pos and "pred_delta_stars_h" in show.columns:
                show = show[show["pred_delta_stars_h"] > 0]
            
            show = show.sort_values("pred_delta_stars_h", ascending=False).head(top_n)
            
            # Colonnes à afficher
            display_cols = ["full_name", "stars_t", "pred_delta_stars_h", "pred_stars_t_plus_h"]
            if has_classification and "class_name" in show.columns:
                display_cols.append("class_name")
            
            st.dataframe(
                show[display_cols],
                use_container_width=True,
                hide_index=True,
            )
            
            csv = pred_df.to_csv(index=False)
            st.download_button("Download Full Predictions (CSV)", csv, "predictions.csv", "text/csv")
    
    # =============================================================================
    # PAGES SECONDAIRES
    # =============================================================================
    
    elif st.session_state.current_page == "MLflow":
        if st.button("← Back to Home"):
            st.session_state.current_page = "Home"
            st.rerun()
        
        st.title("MLflow Runs Explorer")
        
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        exp_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "github-stars")
        
        mlflow.set_tracking_uri(tracking_uri)
        client = mlflow.tracking.MlflowClient()
        
        exp = client.get_experiment_by_name(exp_name)
        
        if exp is None:
            st.error(f"Experiment '{exp_name}' not found.")
            return
        
        st.success(f"Experiment: {exp.name} (id={exp.experiment_id})")
        
        col1, col2 = st.columns(2)
        with col1:
            max_runs = st.number_input("Max runs", 1, 500, 50, 10)
        with col2:
            status_filter = st.selectbox("Status", ["ALL", "FINISHED", "FAILED", "RUNNING"])
        
        filter_str = "" if status_filter == "ALL" else f"attributes.status = '{status_filter}'"
        
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string=filter_str,
            max_results=int(max_runs),
            order_by=["attributes.start_time DESC"],
        )
        
        st.write(f"**{len(runs)} runs found**")
        
        if runs:
            rows = []
            for r in runs:
                info, data = r.info, r.data
                row = {
                    "run_id": info.run_id[:8],
                    "status": info.status,
                    "start_time": ms_to_dt(info.start_time),
                }
                
                for k in ["r2", "mae", "mse", "rmse", "f1_macro", "recall_explosion"]:
                    if k in data.metrics:
                        row[k] = data.metrics.get(k)
                
                for k in ["window_days", "horizon_days", "n_estimators"]:
                    if k in data.params:
                        row[k] = data.params.get(k)
                
                if "mlflow.runName" in data.tags:
                    row["run_name"] = data.tags.get("mlflow.runName")
                
                rows.append(row)
            
            df_runs = pd.DataFrame(rows)
            st.dataframe(df_runs, use_container_width=True, hide_index=True)
    
    elif st.session_state.current_page == "Training":
        if st.button("← Back to Home"):
            st.session_state.current_page = "Home"
            st.rerun()
        
        st.title("Advanced Training Configuration")
        st.caption("For advanced users: customize model hyperparameters")
        
        st.info("This page is for advanced configuration. For quick training, use the buttons on the home page.")
    
    elif st.session_state.current_page == "Predictions":
        if st.button("← Back to Home"):
            st.session_state.current_page = "Home"
            st.rerun()
        
        st.title("Prediction History")
        st.caption("View all past predictions")
        
        storage = get_storage()
        pred_blobs = storage.list_blobs(Settings.PRED_PREFIX)
        
        if not pred_blobs:
            st.info("No predictions generated yet. Go to the home page to generate predictions.")
        else:
            st.write(f"**{len(pred_blobs)} prediction files found**")
            
            for blob in sorted(pred_blobs, reverse=True)[:10]:
                date_str = blob.split("_")[-1].replace(".json", "")
                st.write(f"- {date_str}")

if __name__ == "__main__":
    main()