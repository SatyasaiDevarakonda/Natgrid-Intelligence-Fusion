"""
NATGRID Intelligence Fusion System
Modern, clean Streamlit application with horizontal tabs
Enhanced with dataset labels, descriptions, and improved visualizations
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add project root to path
PROJECT_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_DIR))

# Page configuration
st.set_page_config(
    page_title="AI-Driven Intelligence Fusion for Automated Threat Detection and Analysis",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Import utilities
from utils import get_llm, get_ner_extractor, get_search_engine, get_anomaly_detector
from config import get_config

# Enhanced Custom CSS
st.markdown("""
<style>
    /* Hide sidebar by default */
    [data-testid="stSidebar"] {
        display: none;
    }
    
    /* Main header styling */
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        color: #1E293B;
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
        margin-bottom: 0;
    }
    
    .main-subtitle {
        text-align: center;
        color: #64748B;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* LLM Status Banner */
    .llm-status {
        padding: 0.8rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
        font-weight: 600;
        font-size: 0.95rem;
    }
    .llm-connected {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        color: white;
        box-shadow: 0 4px 6px rgba(16, 185, 129, 0.2);
    }
    .llm-disconnected {
        background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%);
        color: white;
        box-shadow: 0 4px 6px rgba(239, 68, 68, 0.2);
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #F8FAFC 0%, #F1F5F9 100%);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border: 1px solid #E2E8F0;
        text-align: center;
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
    }
    
    /* Threat badges - Updated for RED/ORANGE/PINK */
    .threat-badge {
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 700;
        display: inline-block;
    }
    .threat-red { 
        background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%);
        color: white;
        box-shadow: 0 2px 4px rgba(239, 68, 68, 0.3);
    }
    .threat-orange { 
        background: linear-gradient(135deg, #F59E0B 0%, #D97706 100%);
        color: white;
        box-shadow: 0 2px 4px rgba(245, 158, 11, 0.3);
    }
    .threat-pink { 
        background: linear-gradient(135deg, #EC4899 0%, #DB2777 100%);
        color: white;
        box-shadow: 0 2px 4px rgba(236, 72, 153, 0.3);
    }
    
    /* Risk badges for reports */
    .risk-badge {
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 700;
        display: inline-block;
    }
    .risk-high { 
        background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%);
        color: white;
        box-shadow: 0 2px 4px rgba(239, 68, 68, 0.3);
    }
    .risk-medium { 
        background: linear-gradient(135deg, #F59E0B 0%, #D97706 100%);
        color: white;
        box-shadow: 0 2px 4px rgba(245, 158, 11, 0.3);
    }
    .risk-low { 
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        color: white;
        box-shadow: 0 2px 4px rgba(16, 185, 129, 0.3);
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        transition: all 0.3s;
        box-shadow: 0 4px 6px rgba(99, 102, 241, 0.25);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(99, 102, 241, 0.4);
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #EEF2FF 0%, #E0E7FF 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #6366F1;
        margin: 1rem 0;
    }
    
    /* Dataset info box */
    .dataset-box {
        background: linear-gradient(135deg, #F0FDF4 0%, #DCFCE7 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #10B981;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1E293B;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 3px solid #6366F1;
        padding-bottom: 0.5rem;
    }
    
    /* Graph labels */
    .graph-label {
        font-size: 0.85rem;
        color: #64748B;
        text-align: center;
        margin-top: -10px;
        font-style: italic;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #F8FAFC;
        padding: 0.5rem;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: white;
        border-radius: 8px;
        padding: 0 24px;
        font-weight: 600;
        font-size: 1rem;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%);
        color: white;
    }
    
    /* Formula box */
    .formula-box {
        background: linear-gradient(135deg, #FEF3C7 0%, #FDE68A 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #F59E0B;
        margin: 0.5rem 0;
        font-family: monospace;
    }
</style>
""", unsafe_allow_html=True)


# Cache functions
@st.cache_data
def load_data():
    """Load all datasets"""
    config = get_config()
    try:
        reports_df = pd.read_csv(config.data_dir / 'intelligence_reports.csv')
        events_df = pd.read_csv(config.data_dir / 'event_logs.csv')
        entities_df = pd.read_csv(config.data_dir / 'entity_master.csv')
        return reports_df, events_df, entities_df, None
    except FileNotFoundError as e:
        return None, None, None, f"Data files not found: {e}"


@st.cache_resource
def init_llm():
    """Initialize LLM and return status"""
    try:
        llm = get_llm()
        provider = llm.provider
        return llm, True, f"Connected to {provider.upper()}"
    except Exception as e:
        return None, False, f"LLM Error: {str(e)}"


@st.cache_resource
def init_search_engine(_documents, _doc_ids):
    """Initialize search engine"""
    config = get_config()
    engine = get_search_engine()
    index_path = config.models_dir / "search_index.pkl"
    
    if index_path.exists():
        engine.load_index(str(index_path))
    else:
        engine.index_documents(_documents, _doc_ids)
    
    return engine


@st.cache_resource
def init_anomaly_detector():
    """Initialize anomaly detector"""
    config = get_config()
    detector = get_anomaly_detector(contamination=config.anomaly_contamination)
    model_path = config.models_dir / "anomaly_detector.pkl"
    
    if model_path.exists():
        detector.load_model(str(model_path))
    
    return detector


@st.cache_resource
def init_ner():
    """Initialize NER model"""
    return get_ner_extractor()


def render_llm_status(llm, connected, status_msg):
    """Render LLM connection status banner"""
    if connected:
        st.markdown(
            f'<div class="llm-status llm-connected">🤖 AI Powered | {status_msg}</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="llm-status llm-disconnected">⚠️ AI Offline | {status_msg}</div>',
            unsafe_allow_html=True
        )


def get_threat_badge_class(threat_level):
    """Get CSS class for threat level badge"""
    level_map = {
        'RED': 'threat-red',
        'ORANGE': 'threat-orange', 
        'PINK': 'threat-pink',
        'HIGH': 'threat-red',
        'MEDIUM': 'threat-orange',
        'LOW': 'threat-pink'
    }
    return level_map.get(threat_level.upper(), 'threat-pink')


def main():
    # Initialize LLM first
    llm, llm_connected, llm_status = init_llm()
    
    # Header
    st.markdown('<h1 class="main-header">🛡️ AI-Driven Intelligence Fusion for Automated Threat Detection and Analysis</h1>', unsafe_allow_html=True)
    st.markdown('<p class="main-subtitle">National Intelligence Grid • Advanced Analytics Platform</p>', unsafe_allow_html=True)
    
    # Load data
    reports_df, events_df, entities_df, error = load_data()
    
    if error:
        st.error(error)
        st.info("Run data generation scripts: `cd data && python generate_reports.py && python generate_events.py && python generate_entities.py`")
        return
    
    # Horizontal tabs with dataset names
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏠 Dashboard (All Datasets)",
        "🔍 Intelligence Search (intelligence_reports.csv)", 
        "👤 Entity Analysis (entity_master.csv)",
        "⚠️ Anomaly Detection (event_logs.csv)",
        "📊 Intelligence Fusion (intelligence_reports.csv)"
    ])
    
    with tab1:
        render_dashboard(reports_df, events_df, entities_df, llm, llm_connected)
    
    with tab2:
        render_intelligence_search(reports_df, llm, llm_connected)
    
    with tab3:
        render_entity_analysis(entities_df, reports_df, llm, llm_connected)
    
    with tab4:
        render_anomaly_detection(events_df, llm, llm_connected)
    
    with tab5:
        render_intelligence_fusion(reports_df, llm, llm_connected)


def render_dashboard(reports_df, events_df, entities_df, llm, llm_connected):
    """Dashboard with key metrics, visualizations, dataset descriptions, and snapshots"""
    render_llm_status(llm, llm_connected, "AI analytics enabled" if llm_connected else "Check API configuration")
    
    # Key metrics - Only 3 numbers (removed Total Reports)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        high_priority = len(reports_df[reports_df['priority'] == 'HIGH'])
        st.metric("🔴 High Priority Reports", high_priority, delta=f"{high_priority/len(reports_df)*100:.1f}%")
    
    with col2:
        anomalies = len(events_df[events_df['is_anomaly'] == 1])
        st.metric("⚠️ Detected Anomalies", anomalies, delta=f"{anomalies/len(events_df)*100:.1f}%")
    
    with col3:
        # Handle both old (risk_level) and new (threat_level) column names
        threat_col = 'threat_level' if 'threat_level' in entities_df.columns else 'risk_level'
        high_threat = len(entities_df[entities_df[threat_col].isin(['RED', 'HIGH'])])
        st.metric("🎯 High Threat Entities", high_threat)
    
    st.markdown("---")
    
    # Dataset Descriptions Section
    st.markdown('<div class="section-header">📁 Dataset Overview</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="dataset-box">
            <strong>📄 intelligence_reports.csv</strong><br>
            Intelligence reports from multiple sources including OSINT, Field Reports, Cyber Intel, and Informants. 
            Contains threat assessments across categories like Terrorism, Smuggling, Cyber Attacks, and Civil Unrest.
            <br><br>
            <strong>Records:</strong> """ + str(len(reports_df)) + """ | 
            <strong>Features:</strong> """ + str(len(reports_df.columns)) + """
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="dataset-box">
            <strong>📊 event_logs.csv</strong><br>
            System event logs capturing user activities including logins, database access, file operations, and API calls. 
            Contains labeled anomalies for training detection models.
            <br><br>
            <strong>Records:</strong> """ + str(len(events_df)) + """ | 
            <strong>Features:</strong> """ + str(len(events_df.columns)) + """
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="dataset-box">
            <strong>👤 entity_master.csv</strong><br>
            Master database of persons and organizations with threat level assessments (RED/ORANGE/PINK). 
            Includes known affiliations, threat scores, and surveillance notes.
            <br><br>
            <strong>Records:</strong> """ + str(len(entities_df)) + """ | 
            <strong>Features:</strong> """ + str(len(entities_df.columns)) + """
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Dataset Snapshots
    st.markdown('<div class="section-header">📋 Dataset Snapshots (First 5 Rows)</div>', unsafe_allow_html=True)
    
    st.markdown("**📄 intelligence_reports.csv**")
    st.dataframe(reports_df.head(5), use_container_width=True, hide_index=True)
    
    st.markdown("**📊 event_logs.csv**")
    st.dataframe(events_df.head(5), use_container_width=True, hide_index=True)
    
    st.markdown("**👤 entity_master.csv**")
    st.dataframe(entities_df.head(5), use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Enhanced visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="section-header">📊 Reports by Priority</div>', unsafe_allow_html=True)
        st.markdown('<p class="graph-label">Data Source: intelligence_reports.csv</p>', unsafe_allow_html=True)
        priority_counts = reports_df['priority'].value_counts()
        
        fig = go.Figure(data=[go.Pie(
            labels=priority_counts.index,
            values=priority_counts.values,
            hole=0.5,
            marker=dict(
                colors=['#EF4444', '#F59E0B', '#10B981'],
                line=dict(color='white', width=3)
            ),
            textinfo='label+percent',
            textfont=dict(size=14, color='white', family='Arial Black'),
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
        )])
        
        fig.update_layout(
            height=350,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
            margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown('<div class="section-header">📈 Top Threat Categories</div>', unsafe_allow_html=True)
        st.markdown('<p class="graph-label">Data Source: intelligence_reports.csv</p>', unsafe_allow_html=True)
        category_counts = reports_df['category'].value_counts().head(8)
        
        fig = go.Figure(data=[go.Bar(
            x=category_counts.values,
            y=category_counts.index,
            orientation='h',
            marker=dict(
                color=category_counts.values,
                colorscale='Viridis',
                line=dict(color='white', width=1)
            ),
            text=category_counts.values,
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Reports: %{x}<extra></extra>'
        )])
        
        fig.update_layout(
            height=350,
            showlegend=False,
            yaxis_title="",
            xaxis_title="Number of Reports",
            margin=dict(l=20, r=20, t=20, b=40),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=True, gridcolor='#E2E8F0')
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Timeline chart
    st.markdown('<div class="section-header">📅 Intelligence Reports Timeline</div>', unsafe_allow_html=True)
    st.markdown('<p class="graph-label">Data Source: intelligence_reports.csv</p>', unsafe_allow_html=True)
    
    reports_df['date'] = pd.to_datetime(reports_df['date'])
    timeline_data = reports_df.groupby([pd.Grouper(key='date', freq='D'), 'priority']).size().reset_index(name='count')
    
    fig = px.line(
        timeline_data,
        x='date',
        y='count',
        color='priority',
        color_discrete_map={'HIGH': '#EF4444', 'MEDIUM': '#F59E0B', 'LOW': '#10B981'},
        title='Daily Intelligence Report Trends'
    )
    
    fig.update_layout(
        height=300,
        xaxis_title="Date",
        yaxis_title="Number of Reports",
        legend_title="Priority",
        hovermode='x unified',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)


def render_intelligence_search(reports_df, llm, llm_connected):
    """Semantic search with AI summarization"""
    render_llm_status(llm, llm_connected, "AI summarization available" if llm_connected else "Search available (AI offline)")
    
    st.markdown('<div class="section-header">🔍 Semantic Intelligence Search</div>', unsafe_allow_html=True)
    st.markdown('<p class="graph-label">Data Source: intelligence_reports.csv</p>', unsafe_allow_html=True)
    
    with st.spinner("Initializing search engine..."):
        search_engine = init_search_engine(
            tuple(reports_df['report_text'].tolist()),
            tuple(reports_df['report_id'].tolist())
        )
    
    st.markdown('<div class="info-box">💡 Search using natural language. The system uses AI to find semantically similar reports, not just keyword matching.</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input("🔎 Enter your search query", placeholder="e.g., weapons smuggling, cyber attack, border security, terrorism financing")
    with col2:
        top_k = st.selectbox("Results", [5, 10, 15, 20], index=0)
    
    if query:
        with st.spinner("🔍 Searching intelligence database..."):
            results = search_engine.search(query, top_k=top_k)
        
        st.success(f"✅ Found {len(results)} relevant reports from intelligence_reports.csv")
        
        # AI Summarization
        if llm_connected and len(results) > 0:
            if st.button("🤖 Generate AI Summary of Top Results"):
                with st.spinner("🤖 AI generating comprehensive summary..."):
                    combined = [r['text'] for r in results[:3]]
                    summary = llm.summarize_intelligence_reports(combined)
                    st.info("**🤖 AI-Generated Summary:**")
                    st.markdown(summary)
        
        st.markdown("---")
        
        # Results
        for i, result in enumerate(results, 1):
            report_data = reports_df[reports_df['report_id'] == result['document_id']].iloc[0]
            
            priority_class = f"risk-{report_data['priority'].lower()}"
            
            with st.expander(
                f"#{i} • {result['document_id']} • {report_data['category']} • "
                f"Match: {result['similarity']:.1%}",
                expanded=(i <= 3)
            ):
                st.markdown(f"**Priority:** <span class='risk-badge {priority_class}'>{report_data['priority']}</span> | **Date:** {report_data['date']} | **Source:** {report_data['source']}", unsafe_allow_html=True)
                st.write(result['text'])


def render_entity_analysis(entities_df, reports_df, llm, llm_connected):
    """Entity threat analysis with AI threat assessment - Updated for threat_level with RED/ORANGE/PINK"""
    render_llm_status(llm, llm_connected, "AI threat assessments available" if llm_connected else "Entity data available (AI offline)")
    
    st.markdown('<div class="section-header">👤 Entity Intelligence Analysis</div>', unsafe_allow_html=True)
    st.markdown('<p class="graph-label">Data Source: entity_master.csv</p>', unsafe_allow_html=True)
    
    # Determine column name (support both old and new)
    threat_col = 'threat_level' if 'threat_level' in entities_df.columns else 'risk_level'
    
    # Get unique threat levels
    unique_levels = entities_df[threat_col].unique().tolist()
    
    # Filters
    col1, col2 = st.columns(2)
    with col1:
        entity_type = st.selectbox("Entity Type", ["All", "PERSON", "ORGANIZATION"])
    with col2:
        # Dynamic filter based on available levels
        default_filter = [l for l in ['RED', 'ORANGE', 'HIGH', 'MEDIUM'] if l in unique_levels]
        threat_filter = st.multiselect("Threat Level", unique_levels, default=default_filter[:2] if default_filter else unique_levels[:2])
    
    # Filter entities
    filtered = entities_df.copy()
    if entity_type != "All":
        filtered = filtered[filtered['entity_type'] == entity_type]
    filtered = filtered[filtered[threat_col].isin(threat_filter)]
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("📊 Total Entities", len(filtered))
    col2.metric("👤 Persons", len(filtered[filtered['entity_type'] == 'PERSON']))
    col3.metric("🏢 Organizations", len(filtered[filtered['entity_type'] == 'ORGANIZATION']))
    
    # Threat Level Distribution Chart
    st.markdown("---")
    st.markdown('<div class="section-header">📊 Threat Level Distribution</div>', unsafe_allow_html=True)
    st.markdown('<p class="graph-label">Data Source: entity_master.csv</p>', unsafe_allow_html=True)
    
    threat_counts = entities_df[threat_col].value_counts()
    
    # Color map for threat levels
    color_map = {
        'RED': '#EF4444', 'HIGH': '#EF4444',
        'ORANGE': '#F59E0B', 'MEDIUM': '#F59E0B',
        'PINK': '#EC4899', 'LOW': '#10B981'
    }
    colors = [color_map.get(level, '#6366F1') for level in threat_counts.index]
    
    fig = go.Figure(data=[go.Bar(
        x=threat_counts.index,
        y=threat_counts.values,
        marker=dict(color=colors, line=dict(color='white', width=2)),
        text=threat_counts.values,
        textposition='outside'
    )])
    
    fig.update_layout(
        height=300,
        xaxis_title="Threat Level",
        yaxis_title="Number of Entities",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Entity list
    for _, entity in filtered.head(20).iterrows():
        threat_level = entity[threat_col]
        badge_class = get_threat_badge_class(threat_level)
        
        st.markdown(
            f"### {entity['entity_name']} <span class='threat-badge {badge_class}'>{threat_level}</span>",
            unsafe_allow_html=True
        )
        st.write(f"**Type:** {entity['entity_type']} • **ID:** {entity['entity_id']}")
        st.write(f"**Known Affiliations:** {entity['known_affiliations']}")
        
        # Show threat score if available
        if 'threat_score' in entity:
            st.write(f"**Threat Score:** {entity['threat_score']}/100")
        
        if llm_connected:
            if st.button("🤖 AI Threat Assessment", key=entity['entity_id']):
                with st.spinner("🤖 Generating threat assessment..."):
                    related = reports_df[
                        reports_df['report_text'].str.contains(
                            entity['entity_name'].split()[0], 
                            case=False, 
                            na=False
                        )
                    ]['report_text'].head(3).tolist()
                    
                    assessment = llm.generate_threat_assessment(
                        entity['entity_name'],
                        related if related else []
                    )
                
                st.markdown("---")
                st.info("**🤖 AI Threat Assessment:**")
                st.markdown(assessment)
        
        st.markdown("---")


def render_anomaly_detection(events_df, llm, llm_connected):
    """Anomaly detection with detailed explanation, risk score formula, and AI insights"""
    render_llm_status(llm, llm_connected, "AI explanations available" if llm_connected else "Detection available (AI offline)")
    
    st.markdown('<div class="section-header">⚠️ Anomaly Detection System</div>', unsafe_allow_html=True)
    st.markdown('<p class="graph-label">Data Source: event_logs.csv</p>', unsafe_allow_html=True)
    
    # Explanation box
    st.markdown("""
    <div class="info-box">
        <h3 style="margin-top: 0;">🎯 What is Anomaly Detection?</h3>
        <p><strong>Anomaly detection</strong> identifies unusual patterns or behaviors in system event logs that deviate from normal activity. This is critical for:</p>
        <ul>
            <li><strong>🔒 Security Threats:</strong> Detect unauthorized access attempts, data breaches, or insider threats</li>
            <li><strong>🚨 Early Warning:</strong> Identify suspicious activities before they escalate into major incidents</li>
            <li><strong>📊 Behavioral Analysis:</strong> Find unusual login times, locations, or access patterns</li>
            <li><strong>🎯 Risk Prioritization:</strong> Focus investigation efforts on the most suspicious activities</li>
        </ul>
        <p><strong>How it works:</strong> The system uses <em>Isolation Forest</em> machine learning algorithm to analyze event patterns including time, location, access level, IP addresses, and user behavior. Events that significantly deviate from normal patterns receive higher risk scores (0-100).</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.spinner("🔍 Analyzing event logs for anomalies..."):
        detector = init_anomaly_detector()
        
        if not detector.is_fitted:
            results_df = detector.fit_predict(events_df)
        else:
            results_df = detector.predict(events_df)
    
    # Metrics with better visualization
    normal_count = len(results_df[results_df['predicted_anomaly'] == 0])
    anomaly_count = len(results_df[results_df['predicted_anomaly'] == 1])
    detection_rate = anomaly_count/len(results_df)*100
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "✅ Normal Events", 
            f"{normal_count:,}",
            help="Events matching expected behavior patterns"
        )
    
    with col2:
        st.metric(
            "⚠️ Anomalous Events", 
            f"{anomaly_count:,}",
            delta=f"{detection_rate:.1f}%",
            delta_color="inverse",
            help="Events flagged as suspicious or unusual"
        )
    
    with col3:
        st.metric(
            "🎯 Detection Rate", 
            f"{detection_rate:.1f}%",
            help="Percentage of events identified as anomalous"
        )
    
    # Risk Score Distribution Formula
    st.markdown("---")
    st.markdown('<div class="section-header">📐 Risk Score Calculation Formula</div>', unsafe_allow_html=True)
    st.markdown('<p class="graph-label">Data Source: event_logs.csv (Processed by Isolation Forest)</p>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="formula-box">
        <strong>📊 Risk Score Distribution Formula:</strong><br><br>
        <code>anomaly_score = model.score_samples(X_scaled)</code><br>
        <em># More negative score = more anomalous</em><br><br>
        <code>risk_score = ((1 - (score - score_min) / (score_max - score_min)) × 100</code><br><br>
        <strong>Where:</strong><br>
        • <code>score</code> = Raw anomaly score from Isolation Forest (negative values)<br>
        • <code>score_min</code> = Minimum anomaly score in dataset<br>
        • <code>score_max</code> = Maximum anomaly score in dataset<br><br>
        <strong>Interpretation:</strong><br>
        • <span style="color: #EF4444;">Risk Score > 90:</span> CRITICAL - Immediate investigation required<br>
        • <span style="color: #F59E0B;">Risk Score 70-90:</span> HIGH - Priority review needed<br>
        • <span style="color: #FBBF24;">Risk Score 50-70:</span> MEDIUM - Enhanced monitoring<br>
        • <span style="color: #10B981;">Risk Score < 50:</span> LOW - Normal activity
    </div>
    """, unsafe_allow_html=True)
    
    # Anomaly distribution chart
    st.markdown("---")
    st.markdown('<div class="section-header">📊 Risk Score Distribution</div>', unsafe_allow_html=True)
    st.markdown('<p class="graph-label">Data Source: event_logs.csv (Anomalies Only)</p>', unsafe_allow_html=True)
    
    anomalies_only = results_df[results_df['predicted_anomaly'] == 1]
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=anomalies_only['risk_score'],
        nbinsx=20,
        marker=dict(
            color=anomalies_only['risk_score'],
            colorscale='Reds',
            line=dict(color='white', width=1)
        ),
        hovertemplate='Risk Score: %{x:.0f}<br>Count: %{y}<extra></extra>'
    ))
    
    # Add vertical lines for thresholds
    fig.add_vline(x=90, line_dash="dash", line_color="#EF4444", annotation_text="CRITICAL (>90)")
    fig.add_vline(x=70, line_dash="dash", line_color="#F59E0B", annotation_text="HIGH (>70)")
    
    fig.update_layout(
        height=300,
        xaxis_title="Risk Score (0-100)",
        yaxis_title="Number of Events",
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Top anomalies
    st.markdown("---")
    st.markdown('<div class="section-header">🚨 Highest Risk Anomalies</div>', unsafe_allow_html=True)
    st.markdown('<p class="graph-label">Data Source: event_logs.csv (Top 10 Anomalies)</p>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        💡 <strong>Understanding Anomalies:</strong> These are unusual system activities that don't match normal patterns. 
        Higher risk scores indicate more suspicious behavior that may require investigation.
    </div>
    """, unsafe_allow_html=True)
    
    anomalies = results_df[results_df['predicted_anomaly'] == 1].nsmallest(10, 'anomaly_score')
    
    for i, (_, row) in enumerate(anomalies.iterrows(), 1):
        risk_score = row['risk_score']
        
        # Determine risk level and color
        if risk_score > 90:
            risk_level = "CRITICAL"
            risk_emoji = "🔴"
            risk_color = "#EF4444"
        elif risk_score > 70:
            risk_level = "HIGH"
            risk_emoji = "🟠"
            risk_color = "#F59E0B"
        else:
            risk_level = "MEDIUM"
            risk_emoji = "🟡"
            risk_color = "#F59E0B"
        
        # Create a clean card display
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #F8FAFC 0%, #FFFFFF 100%); 
                    padding: 1.5rem; 
                    border-radius: 12px; 
                    border-left: 5px solid {risk_color};
                    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                    margin-bottom: 1rem;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                <h3 style="margin: 0; color: #1E293B;">
                    {risk_emoji} Anomaly #{i} - {risk_level} RISK
                </h3>
                <span style="background: {risk_color}; 
                             color: white; 
                             padding: 0.5rem 1rem; 
                             border-radius: 20px;
                             font-weight: bold;">
                    Risk: {risk_score:.1f}/100
                </span>
            </div>
            <p style="color: #64748B; margin: 0.5rem 0;"><strong>Event ID:</strong> {row['event_id']} • <strong>Time:</strong> {row['timestamp']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Simple readable information
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🔍 What Happened:**")
            st.write(f"• **User:** {row['user_id']}")
            st.write(f"• **Action:** {row['event_type'].replace('_', ' ').title()}")
            st.write(f"• **Location:** {row['location']}")
            st.write(f"• **Duration:** {row['duration_mins']} minutes")
        
        with col2:
            st.markdown("**⚠️ Why It's Suspicious:**")
            
            # Generate simple explanation based on the data
            reasons = []
            if row['duration_mins'] > 120:
                reasons.append(f"Unusually long duration ({row['duration_mins']} mins)")
            if row['access_level'] in ['Level-3', 'Level-4', 'Level-5']:
                reasons.append(f"High security access level ({row['access_level']})")
            if row['location'] in ['Unknown', 'Proxy Server', 'Beijing', 'Karachi', 'Dubai', 'Singapore', 'London']:
                reasons.append(f"Unusual location ({row['location']})")
            if row['status'] == 'failed':
                reasons.append("Failed operation attempt")
            if risk_score > 90:
                reasons.append("Highly abnormal behavior pattern detected")
            elif risk_score > 70:
                reasons.append("Significant deviation from normal activity")
            
            # Check for anomaly_type if available
            if 'anomaly_type' in row and pd.notna(row['anomaly_type']):
                reasons.insert(0, f"Type: {row['anomaly_type']}")
            
            if not reasons:
                reasons.append("Pattern doesn't match typical user behavior")
            
            for reason in reasons:
                st.write(f"• {reason}")
        
        # AI Explanation button
        if llm_connected:
            if st.button(f"🤖 Get Detailed AI Analysis", key=f"exp_{row['event_id']}"):
                with st.spinner("🤖 AI analyzing anomaly pattern..."):
                    anomaly_data = {
                        'user_id': row['user_id'],
                        'event_type': row['event_type'],
                        'timestamp': row['timestamp'],
                        'location': row['location'],
                        'risk_score': risk_score,
                        'details': f"Access: {row['access_level']}, IP: {row['ip_address']}, Duration: {row['duration_mins']}mins, Device: {row['device_id']}"
                    }
                    explanation = llm.explain_anomaly(anomaly_data)
                    st.info("**🤖 AI Analysis:**")
                    st.markdown(explanation)
        
        st.markdown("---")


def render_intelligence_fusion(reports_df, llm, llm_connected):
    """Multi-source intelligence fusion with NER and AI"""
    render_llm_status(llm, llm_connected, "AI fusion available" if llm_connected else "NER available (AI offline)")
    
    st.markdown('<div class="section-header">📊 Multi-Source Intelligence Fusion</div>', unsafe_allow_html=True)
    st.markdown('<p class="graph-label">Data Source: intelligence_reports.csv</p>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        💡 <strong>Intelligence Fusion</strong> combines multiple intelligence reports to identify patterns, extract key entities, and generate comprehensive threat assessments.
    </div>
    """, unsafe_allow_html=True)
    
    with st.spinner("Loading NER model..."):
        ner = init_ner()
    
    # Priority filter that updates dropdown
    col1, col2 = st.columns(2)
    with col1:
        priority_filter = st.multiselect(
            "🎯 Filter by Priority", 
            ["HIGH", "MEDIUM", "LOW"], 
            default=["HIGH"],
            help="Select priority levels to filter available reports"
        )
    with col2:
        max_reports = st.slider("📊 Maximum Reports to Fuse", 3, 10, 5)
    
    # Filter reports based on priority
    filtered = reports_df[reports_df['priority'].isin(priority_filter)] if priority_filter else reports_df
    
    # Create report options with ID and name (category)
    report_options = [
        f"{row['report_id']} - {row['category']}" 
        for _, row in filtered.head(50).iterrows()
    ]
    
    # Extract report IDs from selections
    def extract_report_id(option):
        return option.split(' - ')[0]
    
    st.markdown("**📄 Select Reports to Fuse:**")
    selected_options = st.multiselect(
        "Select reports by ID and category",
        report_options,
        default=None,
        max_selections=max_reports,
        label_visibility="collapsed"
    )
    
    selected = [extract_report_id(opt) for opt in selected_options]
    
    if selected and st.button("🚀 Generate Fusion Intelligence Report", type="primary"):
        selected_texts = filtered[filtered['report_id'].isin(selected)]['report_text'].tolist()
        selected_details = filtered[filtered['report_id'].isin(selected)]
        
        # Show selected reports summary
        st.markdown("---")
        st.markdown("**📋 Fusing Intelligence from:**")
        for _, report in selected_details.iterrows():
            priority_class = f"risk-{report['priority'].lower()}"
            st.markdown(
                f"• {report['report_id']} - {report['category']} "
                f"<span class='risk-badge {priority_class}'>{report['priority']}</span>",
                unsafe_allow_html=True
            )
        
        st.markdown("---")
        
        with st.spinner("🔍 Extracting entities and analyzing intelligence..."):
            # Extract entities
            all_entities = {'persons': [], 'organizations': [], 'locations': []}
            
            for text in selected_texts:
                entities = ner.extract_entities(text)
                all_entities['persons'].extend(entities['persons'])
                all_entities['organizations'].extend(entities['organizations'])
                all_entities['locations'].extend(entities['locations'])
            
            # Deduplicate and count
            entity_counts = {
                'persons': {},
                'organizations': {},
                'locations': {}
            }
            
            for entity_type in ['persons', 'organizations', 'locations']:
                for entity in all_entities[entity_type]:
                    entity_lower = entity.lower()
                    entity_counts[entity_type][entity_lower] = entity_counts[entity_type].get(entity_lower, 0) + 1
            
            # AI Fusion Summary
            if llm_connected:
                st.markdown('<div class="section-header">🤖 AI-Generated Fusion Report</div>', unsafe_allow_html=True)
                summary = llm.summarize_intelligence_reports(selected_texts)
                st.success("**Comprehensive Intelligence Analysis:**")
                st.markdown(summary)
            else:
                st.warning("⚠️ AI offline - showing entity extraction only")
        
        # Show extracted entities with counts
        st.markdown("---")
        st.markdown('<div class="section-header">🎯 Extracted Intelligence Entities</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**👤 Key Persons**")
            sorted_persons = sorted(entity_counts['persons'].items(), key=lambda x: x[1], reverse=True)
            for person, count in sorted_persons[:10]:
                mentions = f"({count} mention{'s' if count > 1 else ''})"
                st.write(f"• {person.title()} {mentions}")
            if not sorted_persons:
                st.write("_No persons identified_")
        
        with col2:
            st.markdown("**🏢 Organizations**")
            sorted_orgs = sorted(entity_counts['organizations'].items(), key=lambda x: x[1], reverse=True)
            for org, count in sorted_orgs[:10]:
                mentions = f"({count} mention{'s' if count > 1 else ''})"
                st.write(f"• {org.title()} {mentions}")
            if not sorted_orgs:
                st.write("_No organizations identified_")
        
        with col3:
            st.markdown("**📍 Locations**")
            sorted_locs = sorted(entity_counts['locations'].items(), key=lambda x: x[1], reverse=True)
            for loc, count in sorted_locs[:10]:
                mentions = f"({count} mention{'s' if count > 1 else ''})"
                st.write(f"• {loc.title()} {mentions}")
            if not sorted_locs:
                st.write("_No locations identified_")


if __name__ == "__main__":
    main()
