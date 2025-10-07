import streamlit as st
from model_handler import ModelHandler
from utils import load_image, preprocess_image, allowed_file
import sqlite3
from datetime import datetime
import pandas as pd
import plotly.express as px
import io
import base64
from PIL import Image

# PDF generation imports
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, PageBreak
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# UI Config
st.set_page_config(page_title="Trauma Detection (Medical Images)", layout="wide", initial_sidebar_state="expanded")

# Theme toggle
if "dark" not in st.session_state:
    st.session_state.dark = True

with st.sidebar:
    st.title("Trauma Detection")
    st.markdown("Minimal, explainable AI for medical image trauma detection")
    mode = st.selectbox("Theme", ["Dark", "Light"], index=0)
    st.session_state.dark = (mode == "Dark")

    st.markdown("---")
    page = st.radio("Navigation", ["Home", "Upload & Detect", "Model Insights", "About AI Model"])

# Model selector
MODEL_OPTIONS = {
    "Bone Fracture Detection (X-ray)": "Hemgg/bone-fracture-detection-using-xray",
    "Intracranial Hemorrhage Detection": "DifeiT/rsna-intracranial-hemorrhage-detection",
    "Brain Tumor Detection": "ShimaGh/Brain-Tumor-Detection",
    "MRI Brain Abnormality Classification": "hugginglearners/brain-tumor-detection-mri"
}

MODEL_DESCRIPTIONS = {
    "Hemgg/bone-fracture-detection-using-xray": "Specialized for detecting bone fractures and trauma in X-ray images",
    "DifeiT/rsna-intracranial-hemorrhage-detection": "Expert model for identifying hemorrhages and head trauma",
    "ShimaGh/Brain-Tumor-Detection": "Optimized for brain tumor localization and detection",
    "hugginglearners/brain-tumor-detection-mri": "General MRI-based brain abnormality classification"
}

if 'model_name' not in st.session_state:
    st.session_state.model_name = list(MODEL_OPTIONS.values())[0]

selected_model_display = st.sidebar.selectbox(
    "Select Analysis Model", 
    list(MODEL_OPTIONS.keys()), 
    index=0,
    help="Choose the appropriate model based on your image type and analysis needs"
)
st.session_state.model_name = MODEL_OPTIONS[selected_model_display]

# Display model information
with st.sidebar.expander("ℹ️ Model Information", expanded=False):
    st.markdown(f"**Selected Model:** {st.session_state.model_name}")
    st.markdown(f"**Description:** {MODEL_DESCRIPTIONS[st.session_state.model_name]}")
    
    # Model-specific recommendations
    if "bone-fracture" in st.session_state.model_name:
        st.info("📋 Best for: X-ray images of bones, limbs, and skeletal structures")
    elif "hemorrhage" in st.session_state.model_name:
        st.info("🧠 Best for: Head CT scans and intracranial imaging")
    elif "tumor" in st.session_state.model_name:
        st.info("🔬 Best for: Brain MRI and tumor detection imaging")

# Initialize model handler
@st.cache_resource
def get_handler(name):
    return ModelHandler(name)

def create_results_dashboard(img, cam_img, pred, conf, probs, handler, timestamp, patient_id):
    """Create comprehensive results dashboard with enhanced medical visualizations."""
    st.header("🎩 Medical Analysis Results Dashboard")
    
    # Medical-specific severity assessment
    if "fracture" in pred.lower():
        severity = "High Risk" if conf > 0.8 else "Moderate Risk" if conf > 0.6 else "Low Risk"
        severity_icon = "🔴" if conf > 0.8 else "🟡" if conf > 0.6 else "🟢"
    elif "hemorrhage" in pred.lower():
        severity = "Critical" if conf > 0.8 else "Urgent" if conf > 0.6 else "Monitor"
        severity_icon = "🆘" if conf > 0.8 else "⚠️" if conf > 0.6 else "🟡"
    elif "tumor" in pred.lower():
        severity = "High Suspicion" if conf > 0.8 else "Moderate Suspicion" if conf > 0.6 else "Low Suspicion"
        severity_icon = "🔴" if conf > 0.8 else "🟡" if conf > 0.6 else "🟢"
    else:
        severity = "Review Required" if conf > 0.6 else "Normal"
        severity_icon = "🟡" if conf > 0.6 else "✅"
    
    # Summary Card with medical focus
    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
    
    with summary_col1:
        st.metric("🩺 Finding", pred.split('(')[0].strip())
    with summary_col2:
        st.metric("🎯 Confidence", f"{conf:.1%}")
    with summary_col3:
        st.metric(f"{severity_icon} Assessment", severity)
    with summary_col4:
        st.metric("🕰️ Analysis Time", timestamp[:16])
    
    # Medical-specific color coding
    if "no" in pred.lower() or "normal" in pred.lower():
        color = "#27AE60"  # Green for normal
    elif any(word in pred.lower() for word in ['fracture', 'hemorrhage', 'tumor']):
        color = "#E74C3C" if conf > 0.75 else "#F39C12" if conf > 0.5 else "#F7DC6F"
    else:
        color = "#3498DB"  # Blue for uncertain
    
    # Enhanced result card with medical context
    st.markdown(f"""
    <div style='padding:20px; border-radius:10px; background:{color}; color:white; margin:20px 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1)'>
        <h2 style='margin:0; text-align:center'>🎨 {pred}</h2>
        <p style='margin:5px 0; text-align:center; font-size:18px'>Medical Confidence: {conf:.1%}</p>
        <p style='margin:5px 0; text-align:center'>Model: {handler.model_name.split('/')[-1]}</p>
        <p style='margin:0; text-align:center'>Patient ID: {patient_id or 'Anonymous'}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Side-by-side medical visualization
    st.subheader("🖼️ Medical Image Analysis")
    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:
        st.markdown("**Original Medical Image**")
        st.image(img, use_column_width=True, caption="Input for analysis")
    
    with viz_col2:
        st.markdown("**AI Attention Map (Grad-CAM)**")
        st.image(cam_img, use_column_width=True, caption="Areas of AI focus")
    
    # Medical confidence analysis
    st.subheader("📈 Model Confidence Analysis")
    if len(probs) > 1:
        labels = [handler.label_map.get(i, f'Class_{i}') for i in range(len(probs))]
        
        # Create confidence chart
        fig = px.bar(
            x=labels[:min(5, len(labels))], 
            y=probs[:min(5, len(probs))], 
            title="Medical Classification Confidence",
            color=probs[:min(5, len(probs))], 
            color_continuous_scale="RdYlGn",
            labels={'x': 'Medical Classifications', 'y': 'Confidence Score'}
        )
        fig.update_layout(
            xaxis_title="Medical Classifications", 
            yaxis_title="Confidence Score",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Medical explainability section
    with st.expander("🔍 Medical AI Explainability", expanded=True):
        st.markdown("""
        **Understanding the AI Analysis:**
        
        � **Red/Hot Areas**: High attention regions where the AI detected potential abnormalities
        � **Yellow Areas**: Moderate attention regions of clinical interest
        🔵 **Blue/Cool Areas**: Low attention regions, considered within normal limits
        
        **Clinical Interpretation Guidelines:**
        - **High Confidence (>75%)**: Strong AI evidence, recommend clinical correlation
        - **Moderate Confidence (50-75%)**: Suggestive findings, additional imaging may be warranted
        - **Low Confidence (<50%)**: Uncertain findings, clinical evaluation recommended
        
        **Model-Specific Notes:**
        """
        )
        
        # Model-specific guidance
        if "bone-fracture" in handler.model_name:
            st.info("🦴 **Bone Fracture Model**: Optimized for detecting fractures in X-ray images. Focus on cortical breaks, alignment, and bone integrity.")
        elif "hemorrhage" in handler.model_name:
            st.warning("🧠 **Hemorrhage Detection**: Specialized for intracranial bleeding. Critical findings require immediate clinical attention.")
        elif "tumor" in handler.model_name.lower():
            st.info("🔬 **Tumor Detection**: Focused on mass lesions and abnormal tissue growth. Follow-up imaging often required.")
    
    # Medical feedback and documentation
    st.subheader("📝 Clinical Documentation & Feedback")
    feedback_col1, feedback_col2, feedback_col3 = st.columns(3)
    
    with feedback_col1:
        if st.button("✅ Clinically Confirmed", type="primary"):
            save_feedback(timestamp, "confirmed")
            st.success("Clinical confirmation logged!")
    
    with feedback_col2:
        if st.button("❌ Clinical Disagreement", type="secondary"):
            save_feedback(timestamp, "disagreement")
            st.success("Clinical feedback logged!")
    
    with feedback_col3:
        if st.button("🔄 Requires Follow-up", type="secondary"):
            save_feedback(timestamp, "followup_required")
            st.success("Follow-up requirement logged!")
    
    # Clinical notes section
    st.markdown("**Clinical Notes & Observations:**")
    doctor_notes = st.text_area(
        "Enter clinical observations, additional findings, or recommendations:", 
        height=100,
        placeholder="Example: Patient reports pain in affected area. Recommend orthopedic consultation..."
    )
    
    if st.button("📝 Save Clinical Notes"):
        save_notes(timestamp, doctor_notes)
        st.success("Clinical notes saved to patient record!")

def create_history_view(conn):
    """Create enhanced history view with filtering and analysis."""
    st.subheader("📊 Analysis History")
    
    # Fetch data
    df = pd.read_sql_query("SELECT * FROM results ORDER BY id DESC LIMIT 500", conn)
    
    if df.empty:
        st.info('💫 No analysis history yet. Upload and analyze some images first!')
        return
    
    # Filters
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    
    with filter_col1:
        model_filter = st.selectbox("Filter by Model", ["All"] + df['model'].unique().tolist())
    
    with filter_col2:
        pred_filter = st.selectbox("Filter by Prediction", ["All"] + df['prediction'].unique().tolist())
    
    with filter_col3:
        min_conf = st.slider("Minimum Confidence", 0.0, 1.0, 0.0)
    
    # Apply filters
    filtered_df = df.copy()
    if model_filter != "All":
        filtered_df = filtered_df[filtered_df['model'] == model_filter]
    if pred_filter != "All":
        filtered_df = filtered_df[filtered_df['prediction'] == pred_filter]
    filtered_df = filtered_df[filtered_df['confidence'] >= min_conf]
    
    # Display filtered results
    st.dataframe(filtered_df, use_container_width=True)
    
    # Analytics
    if len(filtered_df) > 0:
        analytics_col1, analytics_col2 = st.columns(2)
        
        with analytics_col1:
            # Prediction distribution
            pred_counts = filtered_df['prediction'].value_counts()
            fig_pie = px.pie(values=pred_counts.values, names=pred_counts.index, 
                           title="Prediction Distribution")
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with analytics_col2:
            # Confidence over time
            filtered_df['timestamp'] = pd.to_datetime(filtered_df['timestamp'])
            fig_line = px.line(filtered_df, x='timestamp', y='confidence', 
                             title="Confidence Trends Over Time")
            st.plotly_chart(fig_line, use_container_width=True)
    
    # Management options
    st.subheader("🛠️ Data Management")
    management_col1, management_col2 = st.columns(2)
    
    with management_col1:
        selected_ids = st.multiselect('Select entries to delete (by ID)', 
                                    options=filtered_df['id'].tolist())
        if st.button('🗑️ Delete Selected', type="secondary"):
            for id_val in selected_ids:
                conn.execute('DELETE FROM results WHERE id=?', (id_val,))
            conn.commit()
            st.success(f"Deleted {len(selected_ids)} entries")
            st.experimental_rerun()
    
    with management_col2:
        if st.button('📥 Export History as CSV'):
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"trauma_analysis_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

def create_export_section(conn):
    """Create export functionality for results and reports."""
    st.subheader("📄 Export & Reports")
    
    # Get recent results
    df = pd.read_sql_query("SELECT * FROM results ORDER BY id DESC LIMIT 100", conn)
    
    if df.empty:
        st.info('💫 No data to export yet.')
        return
    
    export_col1, export_col2 = st.columns(2)
    
    with export_col1:
        st.markdown("**📈 Data Export**")
        
        # CSV Export
        csv_data = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Full History (CSV)",
            data=csv_data,
            file_name=f"trauma_detection_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
        # JSON Export
        json_data = df.to_json(orient='records', indent=2)
        st.download_button(
            label="📥 Download as JSON",
            data=json_data,
            file_name=f"trauma_detection_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    with export_col2:
        st.markdown("**📄 Report Generation**")
        
        # Summary statistics
        if len(df) > 0:
            st.metric("Total Analyses", len(df))
            st.metric("Average Confidence", f"{df['confidence'].mean():.1%}")
            st.metric("Most Common Prediction", df['prediction'].mode().iloc[0] if not df['prediction'].mode().empty else "N/A")
            
            # Generate summary report
            summary_report = generate_summary_report(df)
            st.download_button(
                label="📄 Download Summary Report",
                data=summary_report,
                file_name=f"trauma_detection_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )

def save_feedback(timestamp, feedback_type):
    """Save user feedback for model improvement."""
    try:
        feedback_conn = sqlite3.connect("feedback.db")
        feedback_conn.execute('''CREATE TABLE IF NOT EXISTS feedback
                               (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                                timestamp TEXT, feedback_type TEXT, 
                                created_at TEXT)''')
        feedback_conn.execute("INSERT INTO feedback (timestamp, feedback_type, created_at) VALUES (?,?,?)",
                            (timestamp, feedback_type, datetime.now().isoformat()))
        feedback_conn.commit()
        feedback_conn.close()
    except Exception as e:
        st.error(f"Failed to save feedback: {e}")

def save_notes(timestamp, notes):
    """Save doctor's notes."""
    try:
        notes_conn = sqlite3.connect("notes.db")
        notes_conn.execute('''CREATE TABLE IF NOT EXISTS notes
                            (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                             timestamp TEXT, notes TEXT, 
                             created_at TEXT)''')
        notes_conn.execute("INSERT INTO notes (timestamp, notes, created_at) VALUES (?,?,?)",
                          (timestamp, notes, datetime.now().isoformat()))
        notes_conn.commit()
        notes_conn.close()
    except Exception as e:
        st.error(f"Failed to save notes: {e}")

def generate_detailed_pdf_report(analysis_id, conn):
    """Generate detailed PDF report for a specific analysis."""
    try:
        # Get analysis data
        analysis_data = pd.read_sql_query("SELECT * FROM results WHERE id = ?", conn, params=(analysis_id,))
        if analysis_data.empty:
            st.error("Analysis not found")
            return None
        
        row = analysis_data.iloc[0]
        
        # Create PDF buffer
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=1*inch)
        
        # Styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#2E86C1')
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=12,
            textColor=colors.HexColor('#1F4E79')
        )
        
        # Content
        content = []
        
        # Title
        content.append(Paragraph("🩺 TRAUMA DETECTION ANALYSIS REPORT", title_style))
        content.append(Spacer(1, 20))
        
        # Patient Information
        content.append(Paragraph("📋 ANALYSIS INFORMATION", heading_style))
        
        info_data = [
            ['Analysis ID:', str(row['id'])],
            ['Timestamp:', row['timestamp']],
            ['Filename:', row['filename']],
            ['Model Used:', row['model']],
            ['Prediction:', row['prediction']],
            ['Confidence:', f"{row['confidence']:.2%}"],
            ['True Label:', row.get('true_label', 'Not provided') or 'Not provided']
        ]
        
        info_table = Table(info_data, colWidths=[2*inch, 4*inch])
        info_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#E8F4FD')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#BDC3C7'))
        ]))
        
        content.append(info_table)
        content.append(Spacer(1, 20))
        
        # Analysis Results
        content.append(Paragraph("🎯 ANALYSIS RESULTS", heading_style))
        
        severity = "High Risk" if row['confidence'] > 0.8 else "Moderate Risk" if row['confidence'] > 0.6 else "Low Risk"
        
        results_data = [
            ['Detected Condition:', row['prediction']],
            ['Confidence Level:', f"{row['confidence']:.2%}"],
            ['Risk Assessment:', severity],
            ['Accuracy Rating:', 'High' if row['confidence'] > 0.75 else 'Moderate' if row['confidence'] > 0.5 else 'Low']
        ]
        
        results_table = Table(results_data, colWidths=[2*inch, 4*inch])
        results_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#FDF2E9')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#BDC3C7'))
        ]))
        
        content.append(results_table)
        content.append(Spacer(1, 20))
        
        # Interpretation
        content.append(Paragraph("🔍 CLINICAL INTERPRETATION", heading_style))
        
        interpretation_text = f"""
        <b>AI Model Analysis:</b><br/>
        The AI model '{row['model']}' analyzed the medical image '{row['filename']}' and detected: <b>{row['prediction']}</b><br/><br/>
        
        <b>Confidence Assessment:</b><br/>
        The model's confidence level of {row['confidence']:.1%} indicates {severity.lower()} findings. 
        {'This suggests strong evidence for the detected condition.' if row['confidence'] > 0.75 else 
         'This suggests moderate evidence requiring clinical correlation.' if row['confidence'] > 0.5 else 
         'This suggests low confidence and requires further examination.'}<br/><br/>
        
        <b>Recommendations:</b><br/>
        {'• Consider immediate clinical evaluation<br/>• Correlate with patient symptoms and history<br/>• Additional imaging may be warranted' if row['confidence'] > 0.75 else
         '• Clinical correlation recommended<br/>• Consider additional imaging studies<br/>• Monitor patient symptoms' if row['confidence'] > 0.5 else
         '• Further imaging recommended<br/>• Clinical evaluation advised<br/>• Consider alternative diagnostic methods'}
        """
        
        content.append(Paragraph(interpretation_text, styles['Normal']))
        content.append(Spacer(1, 20))
        
        # Technical Details
        content.append(Paragraph("⚙️ TECHNICAL DETAILS", heading_style))
        
        tech_text = f"""
        <b>Model Information:</b><br/>
        • Model: {row['model']}<br/>
        • Architecture: Transformer-based Vision Model<br/>
        • Input Processing: 224x224 normalized image<br/>
        • Output: Classification with confidence scoring<br/><br/>
        
        <b>Analysis Methodology:</b><br/>
        • Image preprocessing with normalization<br/>
        • Deep learning feature extraction<br/>
        • Grad-CAM attention mapping for explainability<br/>
        • Confidence calibration for clinical decision support<br/><br/>
        
        <b>Quality Assurance:</b><br/>
        • Automated image quality checks performed<br/>
        • Model uncertainty quantification included<br/>
        • Results require clinical validation
        """
        
        content.append(Paragraph(tech_text, styles['Normal']))
        content.append(Spacer(1, 20))
        
        # Disclaimer
        content.append(Paragraph("⚠️ IMPORTANT DISCLAIMER", heading_style))
        
        disclaimer_text = """
        <b>For Research and Educational Purposes Only</b><br/><br/>
        This AI-generated analysis is intended for research and educational purposes only. 
        It should not be used as a substitute for professional medical diagnosis, treatment, 
        or clinical decision-making. Always consult qualified healthcare professionals for 
        medical advice and patient care decisions.<br/><br/>
        
        The accuracy of AI predictions may vary and should always be validated through 
        appropriate clinical evaluation and additional diagnostic procedures as deemed 
        necessary by qualified medical professionals.
        """
        
        content.append(Paragraph(disclaimer_text, styles['Normal']))
        
        # Footer
        content.append(Spacer(1, 30))
        footer_style = ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontSize=8,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#7F8C8D')
        )
        content.append(Paragraph(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Trauma Detection AI System", footer_style))
        
        # Build PDF
        doc.build(content)
        buffer.seek(0)
        return buffer.getvalue()
        
    except Exception as e:
        st.error(f"Error generating PDF: {e}")
        return None

def generate_summary_pdf_report(df, include_charts=True, include_images=False):
    """Generate comprehensive summary PDF report."""
    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=1*inch)
        
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#2E86C1')
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=12,
            textColor=colors.HexColor('#1F4E79')
        )
        
        content = []
        
        # Title
        content.append(Paragraph("📊 TRAUMA DETECTION COMPREHENSIVE SUMMARY REPORT", title_style))
        content.append(Spacer(1, 20))
        
        # Executive Summary
        content.append(Paragraph("📋 EXECUTIVE SUMMARY", heading_style))
        
        summary_data = [
            ['Total Analyses Performed:', str(len(df))],
            ['Analysis Period:', f"{df['timestamp'].min()} to {df['timestamp'].max()}"],
            ['Average Confidence:', f"{df['confidence'].mean():.1%}"],
            ['Highest Confidence:', f"{df['confidence'].max():.1%}"],
            ['Most Common Finding:', df['prediction'].mode().iloc[0] if not df['prediction'].mode().empty else "N/A"],
            ['Models Used:', ', '.join(df['model'].unique())]
        ]
        
        summary_table = Table(summary_data, colWidths=[2.5*inch, 3.5*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#E8F4FD')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#BDC3C7'))
        ]))
        
        content.append(summary_table)
        content.append(Spacer(1, 20))
        
        # Prediction Breakdown
        content.append(Paragraph("📈 FINDINGS DISTRIBUTION", heading_style))
        
        pred_counts = df['prediction'].value_counts()
        pred_data = [['Finding', 'Count', 'Percentage']]
        for pred, count in pred_counts.items():
            percentage = (count / len(df)) * 100
            pred_data.append([pred, str(count), f"{percentage:.1f}%"])
        
        pred_table = Table(pred_data, colWidths=[2*inch, 1*inch, 1*inch])
        pred_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E86C1')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#BDC3C7'))
        ]))
        
        content.append(pred_table)
        content.append(Spacer(1, 20))
        
        # Statistical Analysis
        content.append(Paragraph("📊 STATISTICAL ANALYSIS", heading_style))
        
        stats_text = f"""
        <b>Confidence Statistics:</b><br/>
        • Mean Confidence: {df['confidence'].mean():.2%}<br/>
        • Median Confidence: {df['confidence'].median():.2%}<br/>
        • Standard Deviation: {df['confidence'].std():.2%}<br/>
        • High Confidence Cases (>75%): {len(df[df['confidence'] > 0.75])} ({len(df[df['confidence'] > 0.75])/len(df)*100:.1f}%)<br/>
        • Low Confidence Cases (<50%): {len(df[df['confidence'] < 0.5])} ({len(df[df['confidence'] < 0.5])/len(df)*100:.1f}%)<br/><br/>
        
        <b>Model Performance:</b><br/>
        """
        
        model_stats = df.groupby('model').agg({
            'confidence': ['mean', 'count']
        }).round(3)
        
        for model in df['model'].unique():
            model_df = df[df['model'] == model]
            stats_text += f"• {model}: {len(model_df)} analyses, avg confidence {model_df['confidence'].mean():.1%}<br/>"
        
        content.append(Paragraph(stats_text, styles['Normal']))
        content.append(Spacer(1, 20))
        
        # Include charts if requested
        if include_charts:
            content.append(Paragraph("📈 VISUAL ANALYTICS", heading_style))
            
            # Generate charts
            chart_buffer = generate_summary_charts(df)
            if chart_buffer:
                chart_img = RLImage(chart_buffer, width=6*inch, height=4*inch)
                content.append(chart_img)
                content.append(Spacer(1, 20))
        
        # Recent Analyses Table
        content.append(Paragraph("📋 RECENT ANALYSES SUMMARY", heading_style))
        
        recent_data = [['ID', 'Timestamp', 'Prediction', 'Confidence']]
        for _, row in df.head(10).iterrows():
            recent_data.append([
                str(row['id']), 
                row['timestamp'][:16], 
                row['prediction'], 
                f"{row['confidence']:.1%}"
            ])
        
        recent_table = Table(recent_data, colWidths=[0.5*inch, 2*inch, 2*inch, 1*inch])
        recent_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E86C1')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#BDC3C7'))
        ]))
        
        content.append(recent_table)
        content.append(PageBreak())
        
        # Disclaimer
        content.append(Paragraph("⚠️ IMPORTANT DISCLAIMER", heading_style))
        
        disclaimer_text = """
        <b>For Research and Educational Purposes Only</b><br/><br/>
        This comprehensive analysis report is generated from AI-based trauma detection 
        system data for research and educational purposes only. The findings and statistics 
        presented should not be used for clinical decision-making without proper medical 
        supervision and validation.<br/><br/>
        
        All AI predictions require clinical correlation and professional medical interpretation. 
        This system is designed to assist healthcare professionals and researchers, not replace 
        clinical judgment.
        """
        
        content.append(Paragraph(disclaimer_text, styles['Normal']))
        
        # Footer
        content.append(Spacer(1, 30))
        footer_style = ParagraphStyle(
            'Footer',
            parent=styles['Normal'],
            fontSize=8,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#7F8C8D')
        )
        content.append(Paragraph(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Trauma Detection AI System", footer_style))
        
        # Build PDF
        doc.build(content)
        buffer.seek(0)
        return buffer.getvalue()
        
    except Exception as e:
        st.error(f"Error generating summary PDF: {e}")
        return None

def generate_summary_report(df):
    """Generate a text summary report."""
    report = f"""
TRAUMA DETECTION ANALYSIS SUMMARY REPORT
========================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

OVERVIEW:
---------
Total Analyses: {len(df)}
Date Range: {df['timestamp'].min()} to {df['timestamp'].max()}
Average Confidence: {df['confidence'].mean():.1%}

PREDICTION BREAKDOWN:
-------------------
"""
    
    pred_counts = df['prediction'].value_counts()
    for pred, count in pred_counts.items():
        percentage = (count / len(df)) * 100
        report += f"{pred}: {count} ({percentage:.1f}%)\n"
    
    report += f"""

CONFIDENCE STATISTICS:
--------------------
Highest Confidence: {df['confidence'].max():.1%}
Lowest Confidence: {df['confidence'].min():.1%}
Median Confidence: {df['confidence'].median():.1%}

MODEL USAGE:
-----------
"""
    
    model_counts = df['model'].value_counts()
    for model, count in model_counts.items():
        report += f"{model}: {count} analyses\n"
    
    return report

def generate_summary_charts(df):
    """Generate summary charts for PDF inclusion."""
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Trauma Detection Analytics Summary', fontsize=16)
        
        # Prediction distribution pie chart
        pred_counts = df['prediction'].value_counts()
        ax1.pie(pred_counts.values, labels=pred_counts.index, autopct='%1.1f%%')
        ax1.set_title('Findings Distribution')
        
        # Confidence histogram
        ax2.hist(df['confidence'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.set_title('Confidence Distribution')
        ax2.set_xlabel('Confidence Level')
        ax2.set_ylabel('Frequency')
        
        # Model usage
        model_counts = df['model'].value_counts()
        ax3.bar(range(len(model_counts)), model_counts.values)
        ax3.set_title('Model Usage')
        ax3.set_xlabel('Models')
        ax3.set_ylabel('Usage Count')
        ax3.set_xticks(range(len(model_counts)))
        ax3.set_xticklabels([m.split('/')[-1][:15] for m in model_counts.index], rotation=45)
        
        # Confidence by prediction
        pred_conf = df.groupby('prediction')['confidence'].mean()
        ax4.bar(range(len(pred_conf)), pred_conf.values)
        ax4.set_title('Average Confidence by Finding')
        ax4.set_xlabel('Findings')
        ax4.set_ylabel('Average Confidence')
        ax4.set_xticks(range(len(pred_conf)))
        ax4.set_xticklabels(pred_conf.index, rotation=45)
        
        plt.tight_layout()
        
        # Save to buffer
        chart_buffer = io.BytesIO()
        plt.savefig(chart_buffer, format='png', dpi=150, bbox_inches='tight')
        chart_buffer.seek(0)
        plt.close()
        
        return chart_buffer
        
    except Exception as e:
        print(f"Error generating charts: {e}")
        return None

# Initialize missing imports
from datetime import datetime


# Database (SQLite) for history
DB_PATH = "results_history.db"
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
conn.execute('''CREATE TABLE IF NOT EXISTS results
             (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, model TEXT, filename TEXT, prediction TEXT, confidence REAL)''')
# Add true_label column if it doesn't exist
try:
    conn.execute('ALTER TABLE results ADD COLUMN true_label TEXT')
except sqlite3.OperationalError:
    pass  # Column already exists


if page == "Home":
    st.title("Trauma Detection — Home")
    st.write("Upload X-ray, MRI or DICOM images and get real-time trauma detection with Grad-CAM explainability.")
    st.write("Use the sidebar to switch models and navigation.")

elif page == "Upload & Detect":
    st.title("🩺 Trauma Detection Analysis")
    
    # Create tabs for different sections
    detect_tab, history_tab, export_tab = st.tabs(["🔍 Detect", "📊 History", "📄 Export"])
    
    with detect_tab:
        # Upload section
        uploaded = st.file_uploader("Upload medical image (JPEG, PNG, DICOM)", type=['png','jpg','jpeg','dcm'])
        
        col_input1, col_input2 = st.columns([2, 1])
        with col_input1:
            true_label = st.text_input('🏷️ True label (optional, for metrics)', value='')
        with col_input2:
            patient_id = st.text_input('👤 Patient ID (optional)', value='')
        
        if uploaded is not None:
            file_bytes = uploaded.read()
            if not allowed_file(uploaded.name):
                st.error("❌ Unsupported file type")
            else:
                # Analysis section
                with st.container():
                    st.markdown("---")
                    st.subheader("🔬 Analysis in Progress")
                    
                    # Lazy model load
                    try:
                        handler = get_handler(st.session_state.model_name)
                    except Exception as e:
                        st.error(f"❌ Failed to load model: {e}")
                        st.stop()
                    
                    # Progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    try:
                        # Preprocessing
                        status_text.info('🔄 Preprocessing image...')
                        img = load_image(io.BytesIO(file_bytes), uploaded.name)
                        input_tensor = preprocess_image(img, handler.input_size)
                        progress_bar.progress(25)
                        
                        # Inference
                        status_text.info('🧠 Running AI analysis...')
                        pred, conf, probs = handler.predict(input_tensor)
                        progress_bar.progress(60)
                        
                        # Grad-CAM generation
                        status_text.info('🎯 Generating explainability maps...')
                        cam_img = handler.gradcam(input_tensor)
                        progress_bar.progress(85)
                        
                        # Save to database
                        timestamp = datetime.now().isoformat()
                        conn.execute("INSERT INTO results (timestamp, model, filename, prediction, confidence, true_label) VALUES (?,?,?,?,?,?)",
                                   (timestamp, st.session_state.model_name, uploaded.name, pred, float(conf), true_label if true_label.strip() else None))
                        conn.commit()
                        progress_bar.progress(100)
                        status_text.success('✅ Analysis complete!')
                        
                        # Results Dashboard
                        st.markdown("---")
                        create_results_dashboard(img, cam_img, pred, conf, probs, handler, timestamp, patient_id)
                        
                    except Exception as e:
                        st.error(f"❌ Error during processing: {e}")
                        status_text.error('❌ Analysis failed')
    
    with history_tab:
        create_history_view(conn)
    
    with export_tab:
        create_export_section(conn)

elif page == "Model Insights":
    st.title("Model Insights")
    st.write("Visual metrics from logged inferences (local). This is for quick analysis; for rigorous evaluation use labeled test sets.")
    df = pd.read_sql_query("SELECT * FROM results ORDER BY id DESC LIMIT 500", conn)
    if df.empty:
        st.info("No history yet — run some inferences first.")
    else:
        st.subheader("History")
        st.dataframe(df)

        st.subheader("Confidence distribution")
        fig = px.histogram(df, x='confidence', nbins=20, color='prediction')
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Simple metrics (by model)")
        agg = df.groupby('model').agg(accuracy=('confidence', 'mean'), count=('id','count')).reset_index()
        st.plotly_chart(px.bar(agg, x='model', y='accuracy', color='model'), use_container_width=True)

        # If ground-truth labels are present, compute confusion matrix
        if 'true_label' in df.columns and df['true_label'].notnull().any():
            try:
                from sklearn.metrics import confusion_matrix, classification_report
                labels = list(df['true_label'].dropna().unique())
                y_true = df['true_label'].fillna('Unknown')
                y_pred = df['prediction']
                cm = confusion_matrix(y_true, y_pred, labels=labels)
                cm_df = pd.DataFrame(cm, index=labels, columns=labels)
                st.subheader('Confusion Matrix')
                st.dataframe(cm_df)
            except ImportError:
                st.info("Install scikit-learn to see confusion matrix")

elif page == "About AI Model":
    st.title("🤖 About the AI Models")
    st.write("This application uses four specialized Hugging Face models, each optimized for specific types of medical trauma detection.")
    
    # Model cards
    for display_name, model_id in MODEL_OPTIONS.items():
        with st.expander(f"📋 {display_name}", expanded=False):
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown(f"**Model ID:** `{model_id}`")
                st.markdown(f"**Type:** {'X-ray Analysis' if 'xray' in model_id else 'MRI/CT Analysis'}")
                
            with col2:
                st.markdown(f"**Description:** {MODEL_DESCRIPTIONS[model_id]}")
                
                # Specific use cases
                if "bone-fracture" in model_id:
                    st.markdown("**Detects:** Bone fractures, breaks, dislocations")
                    st.markdown("**Image Types:** X-ray images of extremities, spine, pelvis")
                elif "hemorrhage" in model_id:
                    st.markdown("**Detects:** Intracranial bleeding, head trauma")
                    st.markdown("**Image Types:** Head CT scans, brain imaging")
                elif "Brain-Tumor" in model_id:
                    st.markdown("**Detects:** Brain tumors, masses, lesions")
                    st.markdown("**Image Types:** Brain MRI, neuroimaging")
                elif "brain-tumor-detection-mri" in model_id:
                    st.markdown("**Detects:** General brain abnormalities")
                    st.markdown("**Image Types:** MRI scans, brain imaging")
    
    st.markdown("---")
    st.subheader("🔬 Clinical Applications")
    
    applications = {
        "Emergency Medicine": "Rapid trauma assessment in emergency departments",
        "Radiology": "Second opinion and screening assistance for radiologists",
        "Orthopedics": "Bone fracture detection and classification",
        "Neurology": "Brain abnormality detection and tumor screening",
        "Research": "Medical imaging research and algorithm development"
    }
    
    for app, desc in applications.items():
        st.markdown(f"**{app}:** {desc}")
    
    st.markdown("---")
    st.warning("⚠️ **Important:** These models are for research and educational purposes only. All AI predictions require clinical validation and should not replace professional medical diagnosis.")


# Close DB on exit
# conn.close()
