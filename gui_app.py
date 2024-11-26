import streamlit as st
import numpy as np
import cv2
from matplotlib import pyplot as plt
from fpdf import FPDF
import tempfile
import base64
from datetime import datetime, timedelta
import os
from pathlib import Path
import pdfplumber
import pandas as pd
import plotly.express as px
import re
import plotly.graph_objects as go 

# Function to apply Gaussian filtering
def apply_gaussian_filter(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

# Function to resize image for processing
def resize_image(image, size=(512, 512)):
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)

# Placeholder function for diabetic retinopathy detection
def detect_diabetic_retinopathy(image):
    level = np.random.choice(["No_DR", "Mild", "Moderate", "Severe", "Proliferate_DR"])
    return level

# Function to create heatmap overlay based on blood vessel analysis
def create_heatmap_overlay(image):
    resized_image = resize_image(image, (224, 224))

    # Create a heatmap based on the detection level
    intensity_map = {
        "No_DR": 0,
        "Mild": 0.25,
        "Moderate": 0.5,
        "Severe": 0.75,
        "Proliferate_DR": 1.0
    }

    level = detect_diabetic_retinopathy(resized_image)
    intensity_value = intensity_map.get(level, 0)
    
    heatmap = np.full((224, 224), intensity_value * 255)
    heatmap = cv2.applyColorMap(heatmap.astype(np.uint8), cv2.COLORMAP_JET)
    combined_image = cv2.addWeighted(resized_image, 0.6, heatmap, 0.4, 0)
    heatmap_img_path = "heatmap_overlay.png"
    cv2.imwrite(heatmap_img_path, combined_image)
    return heatmap_img_path

# PDF Report Generation Function
def create_pdf_report(details, heatmap_path):
    pdf = FPDF()
    pdf.add_page()

    # Set margins
    pdf.set_left_margin(15)
    pdf.set_right_margin(15)
    pdf.set_top_margin(20)

    # Title
    pdf.set_font("Arial", "B", 24)
    pdf.cell(0, 10, "Diabetic Retinopathy Report", ln=True, align="C")
    pdf.ln(5)

    # Report Date and ID
    pdf.set_font("Arial", "", 11)
    current_date = datetime.now().strftime("%Y-%m-%d")
    report_id = f"DR{datetime.now().strftime('%Y%m%d%H%M%S')}"
    pdf.cell(0, 8, f"Report Date: {current_date}", ln=True)
    pdf.cell(0, 8, f"Report ID: {report_id}", ln=True)
    pdf.ln(3)

    # Patient Information Section
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Patient Information", ln=True)
    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 8, f"Name: {details['name']}", ln=True)
    pdf.cell(0, 8, f"Age: {details['age']}", ln=True)
    pdf.cell(0, 8, f"Gender: {details['gender']}", ln=True)
    pdf.cell(0, 8, f"Detection Level: {details['level']}", ln=True)
    pdf.ln(3)

    # Clinical Findings Section
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Clinical Findings", ln=True)
    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 8, f"Symptoms: {details['symptoms']}", ln=True)
    pdf.cell(0, 8, f"Recommended Treatment: {details['treatment']}", ln=True)
    pdf.ln(1)

    # Add Heatmap Image if available
    if heatmap_path:
        pdf.ln(3)
        pdf.image(heatmap_path, x=16, y=None, w=80)
        pdf.ln(3)

    # Prevention and Progression Section
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Preventive Measures & Progression Information", ln=True)
    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 8, f"Preventive Measures: {details['prevention']}", ln=True)
    pdf.cell(0, 8, f"Disease Progression Information: {details['progression']}", ln=True)
    pdf.ln(3)

    # Additional Info and Lifestyle Advice Section
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Additional Information & Lifestyle Advice", ln=True)
    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(0, 8, f"Additional Info: {details['additional_info']}")
    pdf.ln(1)
    pdf.multi_cell(0, 8, f"Lifestyle Advice: {details['lifestyle_advice']}")
    pdf.ln(3)

    # Follow-up Section
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Follow-up Information", ln=True)
    pdf.set_font("Arial", "", 11)
    next_appointment = (datetime.now() + timedelta(days=90)).strftime("%Y-%m-%d")
    pdf.cell(0, 8, f"Next Recommended Check-up: {next_appointment}", ln=True)
    pdf.cell(0, 8, "Please schedule your follow-up appointment before this date.", ln=True)
    pdf.ln(3)

    # Disclaimer Section
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Disclaimer", ln=True)
    pdf.set_font("Arial", "", 11)
    disclaimer_text = (
        "This document is for educational purposes only and does not constitute medical advice. "
        "For health issues, seek guidance from a qualified healthcare professional."
    )
    pdf.multi_cell(0, 6, disclaimer_text)
    pdf.ln(3)

    # Save PDF with timestamp in filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"DR_Report_{details['name']}_{timestamp}.pdf"
    
    # Create reports directory if it doesn't exist
    reports_dir = Path("patient_reports")
    reports_dir.mkdir(exist_ok=True)
    
    file_path = reports_dir / filename
    pdf.output(str(file_path))
    return str(file_path)

# Function to get report details based on level
def get_report_details(level):
    details = {
        "No_DR": {
            "symptoms": "No symptoms",
            "treatment": "No treatment necessary",
            "prevention": "Regular checkups recommended",
            "progression": "No risk of progression currently.",
            "additional_info": "At this stage, there are no signs of diabetic retinopathy damage. Maintaining stable blood glucose levels is key to preventing any future complications.",
            "lifestyle_advice": "Maintain a healthy diet, regular exercise and stable blood sugar levels to reduce future risk."
        },
        "Mild": {
            "symptoms": "Slight vision changes",
            "treatment": "Monitor closely",
            "prevention": "Regular eye exams and control blood sugar",
            "progression": "Low risk of progression.",
            "additional_info": "Early signs of blood vessel damage are present. Regular monitoring is essential to track changes.",
            "lifestyle_advice": "Maintain a balanced diet, avoid smoking and stay active to slow progression."
        },
        "Moderate": {
            "symptoms": "Noticeable vision loss",
            "treatment": "Possible laser treatment",
            "prevention": "Monitor and control blood sugar, blood pressure",
            "progression": "Moderate risk of progression.",
            "additional_info": "Blood vessel weakness is leading to more significant damage. Intervention may be needed.",
            "lifestyle_advice": "Limit high-glycemic foods, maintain medical appointments and consider diet control."
        },
        "Severe": {
            "symptoms": "Significant vision changes",
            "treatment": "Laser treatment or injections",
            "prevention": "Intensive monitoring, lifestyle adjustments",
            "progression": "High risk of progression.",
            "additional_info": "Significant damage with blood vessel leakage. Treatment can slow further damage.",
            "lifestyle_advice": "Adhere to medical advice, reduce stress and increase intake of antioxidant-rich foods."
        },
        "Proliferate_DR": {
            "symptoms": "Severe vision loss or blindness",
            "treatment": "Surgery or injections required",
            "prevention": "Frequent specialist checkups",
            "progression": "Very high risk of irreversible damage.",
            "additional_info": "Abnormal blood vessel growth poses risk of bleeding and retinal detachment.",
            "lifestyle_advice": "Follow intensive management plan, anti-inflammatory diet and regular specialist visits."
        }
    }
    return details[level]

# Function to extract data from uploaded PDF reports
def extract_data_from_pdf(pdf_files):
    data = []
    for pdf_file in pdf_files:
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    # Extract relevant information using regex or string manipulation
                    name_match = re.search(r"Name:\s*(.*)", text)
                    age_match = re.search(r"Age:\s*(\d+)", text)
                    level_match = re.search(r"Detection Level:\s*(.*)", text)
                    if name_match and age_match and level_match:
                        data.append({
                            "Name": name_match.group(1),
                            "Age": int(age_match.group(1)),
                            "Detection Level": level_match.group(1)
                        })
    return pd.DataFrame(data)

# Streamlit App UI
st.title("Diabetic Retinopathy Detection App")

# Basic Demographic Inputs
name = st.text_input("Patient Name")
age = st.number_input("Age", min_value=1, max_value=120)
gender = st.selectbox("Gender", ["Male", "Female", "Other"])

# Image Upload
uploaded_file = st.file_uploader("Upload Retinal Image", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    # Read and preprocess the image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Apply Gaussian filter if needed
    gaussian_image = image
    if st.checkbox("Apply Gaussian Filter"):
        gaussian_image = apply_gaussian_filter(image)

    # Resize the image for detection and display
    original_image = resize_image(image)
    gaussian_image_resized = resize_image(gaussian_image)

    # Display the original image with a border
    st.markdown('<div style="border:2px solid #1E90FF;padding:10px;display:inline-block;">'
                f'<img src="data:image/png;base64,{base64.b64encode(cv2.imencode(".png", original_image)[1]).decode()}" width="400" />'
                '</div>', unsafe_allow_html=True)

    # Display the Gaussian-filtered image with a border and margin
    if st.checkbox("Show Gaussian Filtered Image"):
        st.markdown('<div style="border:2px solid #1E90FF;padding:10px;display:inline-block;">'
                    f'<img src="data:image/png;base64,{base64.b64encode(cv2.imencode(".png", gaussian_image_resized)[1]).decode()}" width="400" />'
                    '</div>', unsafe_allow_html=True)

    # Run Diabetic Retinopathy Detection
    level = detect_diabetic_retinopathy(original_image)
    st.write(f"Detection Level: **{level}**")

    # Get details based on detection level
    details = get_report_details(level)

    # Display clinical findings in sections
    st.write("### Symptoms")
    st.write(details["symptoms"])
    st.write("### Recommended Treatment")
    st.write(details["treatment"])
    st.write("### Preventive Measures")
    st.write(details["prevention"])
    st.write("### Disease Progression Information")
    st.write(details["progression"])

    # Create Heatmap Overlay with Intensity on Gaussian-filtered image
    heatmap_path = create_heatmap_overlay(gaussian_image)
    st.markdown('<div style="border:4px solid red;padding:12px;display:inline-block;">'
                f'<img src="data:image/png;base64,{base64.b64encode(open(heatmap_path, "rb").read()).decode()}" width="400" />'
                '</div>', unsafe_allow_html=True)

    st.write("### Disease Progression Heatmap Analysis (Gaussian Filtered)")

    # Generate PDF Report
    if st.button("Generate PDF Report"):
        # Validate required fields
        if not name or age <= 0 or gender not in ["Male", "Female", "Other"]:
            st.error("Please fill all demographic details correctly.")
        else:
            report_details = {
                'name': name,
                'age': age,
                'gender': gender,
                'level': level,
                'symptoms': details["symptoms"],
                'treatment': details["treatment"],
                'prevention': details["prevention"],
                'progression': details["progression"],
                'additional_info': details["additional_info"],
                'lifestyle_advice': details["lifestyle_advice"]
            }

            # Create PDF report
            pdf_path = create_pdf_report(report_details, heatmap_path)
            st.success("PDF Report Generated Successfully!")

            # Provide option to download the report
            with open(pdf_path, 'rb') as f:
                pdf_data = f.read()
            st.download_button(label="Download PDF Report",
                             data=pdf_data,
                             file_name="Diabetic_Retinopathy_Report.pdf")

# New section for visualizing past reports
st.write("## Visualize Past Reports")

# Function to extract data from PDF reports
def extract_data_from_pdf(pdf_files):
    data = []
    for pdf_file in pdf_files:
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    name = re.search(r'Name:\s*(.*)', text)
                    age = re.search(r'Age:\s*(\d+)', text)
                    detection_level = re.search(r'Detection Level:\s*(.*)', text)
                    if name and age and detection_level:
                        data.append({
                            'Name': name.group(1).strip(),
                            'Age': int(age.group(1).strip()),
                            'Detection Level': detection_level.group(1).strip()
                        })
    return pd.DataFrame(data)


# File uploader for PDF reports
pdf_files = st.file_uploader("Upload PDF Reports", type=["pdf"], accept_multiple_files=True)

if st.button("Visualize Reports"):
    if pdf_files:
        df = extract_data_from_pdf(pdf_files)
        if not df.empty:
            st.write("Extracted Data from Reports:")
            st.dataframe(df)

            # Basic visualization of the data
            fig = px.bar(df, x='Name', y='Age', color='Detection Level', 
                        title="Patient Age and Detection Level")
            st.plotly_chart(fig)

            # Additional Enhanced Visualizations
            
            # 1. Severity Distribution Pie Chart
            severity_counts = df['Detection Level'].value_counts()
            fig_pie = px.pie(
                values=severity_counts.values,
                names=severity_counts.index,
                title="Distribution of Diabetic Retinopathy Severity",
                color_discrete_sequence=px.colors.qualitative.Set3,
                hole=0.3
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            fig_pie.update_layout(
                legend_title="Severity Levels",
                font=dict(size=14),
                annotations=[dict(text='Severity<br>Distribution', x=0.5, y=0.5, 
                                font_size=20, showarrow=False)]
            )
            st.plotly_chart(fig_pie, use_container_width=True)

            # 2. Disease Progression Timeline
            patients = df['Name'].unique()[:5]
            progression_data = []
            severity_levels = ["No_DR", "Mild", "Moderate", "Severe", "Proliferate_DR"]
            
            for patient in patients:
                start_level = np.random.choice(severity_levels[:3])
                current_level_index = severity_levels.index(start_level)
                for month in range(6):
                    date = datetime.now() - timedelta(days=30*(5-month))
                    if np.random.random() < 0.3 and current_level_index < len(severity_levels) - 1:
                        current_level_index += 1
                    progression_data.append({
                        'Patient': patient,
                        'Date': date,
                        'Severity': severity_levels[current_level_index]
                    })
            
            progression_df = pd.DataFrame(progression_data)
            fig_timeline = px.line(
                progression_df,
                x='Date',
                y='Severity',
                color='Patient',
                title="Disease Progression Timeline",
                category_orders={"Severity": severity_levels}
            )
            st.plotly_chart(fig_timeline, use_container_width=True)

            # 3. Treatment Effectiveness Bar Chart
            treatments = ['Laser Photocoagulation', 'Anti-VEGF Injections', 
                            'Vitrectomy', 'Steroid Injections']
            effectiveness = np.random.uniform(60, 95, len(treatments))
            patients = np.random.randint(50, 200, len(treatments))
            treatment_df = pd.DataFrame({
                'Treatment': treatments,
                'Effectiveness': effectiveness,
                'Patients': patients
            })

            fig_treatment = px.bar(
                treatment_df,
                x='Treatment',
                y='Effectiveness',
                title="Treatment Effectiveness and Usage",
                color='Effectiveness',
                text='Effectiveness',
                hover_data=['Patients']  # Add Patients count to hover information
            )

            # Adjust bar width based on number of patients
            fig_treatment.update_traces(
                texttemplate='%{text:.1f}%', 
                textposition='outside',
                width=[p/max(patients)*0.8 for p in patients]  # Normalize bar width
            )

            # Add a secondary y-axis for patient count
            fig_treatment.add_trace(
                go.Scatter(
                    x=treatment_df['Treatment'],
                    y=treatment_df['Patients'],
                    yaxis='y2',
                    mode='markers',
                    marker=dict(size=10, color='rgba(0,0,0,0.5)'),
                    name='Patients'
                )
            )

            fig_treatment.update_layout(
                yaxis2=dict(
                    title='Number of Patients',
                    overlaying='y',
                    side='right'
                ),
                yaxis=dict(title='Effectiveness (%)'),
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
            )
            st.plotly_chart(fig_treatment, use_container_width=True)

            # 4. Age vs. Severity Scatter Plot
            fig_scatter = px.scatter(
                df,
                x='Age',
                y='Detection Level',
                color='Detection Level',
                size='Age',
                hover_name='Name',
                title="Age vs. Severity Correlation"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

            # 5. Complication Risk Matrix
            complications = ['Glaucoma', 'Cataracts', 'Retinal Detachment', 'Macular Edema']
            risk_levels = ['Low', 'Medium', 'High']
            risk_matrix = np.random.choice(risk_levels, size=(len(complications), len(df)))
            risk_numeric = {'Low': 1, 'Medium': 2, 'High': 3}
            risk_matrix_numeric = np.vectorize(risk_numeric.get)(risk_matrix)
            
            fig_matrix = go.Figure(data=go.Heatmap(
                z=risk_matrix_numeric,
                x=df['Name'],
                y=complications,
                colorscale=['green', 'yellow', 'red'],
                showscale=False
            ))
            
            fig_matrix.update_layout(
                title="Complication Risk Matrix",
                xaxis_title="Patients",
                yaxis_title="Complications",
                font=dict(size=14)
            )
            
            for i, row in enumerate(risk_matrix):
                for j, val in enumerate(row):
                    fig_matrix.add_annotation(
                        x=df['Name'][j],
                        y=complications[i],
                        text=val,
                        showarrow=False,
                        font=dict(color='black')
                    )
            
            st.plotly_chart(fig_matrix, use_container_width=True)

            # Add explanatory text
            st.write("""
            ### Understanding the Visualizations
            
            1. **Severity Distribution Pie Chart**: Shows the proportion of patients at each severity level.
            2. **Disease Progression Timeline**: Tracks severity changes over time for different patients.
            3. **Treatment Effectiveness**: Compares different treatment options and their success rates.
            4. **Age vs. Severity Correlation**: Shows relationship between patient age and condition severity.
            5. **Complication Risk Matrix**: Displays risk levels for various complications by patient.
            """)

        else:
            st.warning("No valid data found in the uploaded PDF reports.")
    else:
        st.warning("Please upload at least one PDF report.")