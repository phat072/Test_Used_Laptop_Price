import streamlit as st
import PyPDF2
import json
import re
from io import BytesIO
from CVParser import CVParser
from CVParserGPT import CVParserGPT
import config as config
from JobRecommender import JobRecommender
import pandas as pd
import os

# Function to read PDF and extract text
def extract_text_from_pdf(uploaded_file):
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None

# Initialize CVParserGPT
model_name = config.GPT_MODEL_NAME  # or another model
api_key = config.GPT_TOKEN
cv_format = '{"NAME": "string", "EXPERIENCE": "string", "SKILLS": "list"}'  # Example format

# Initialize CVParser
cv_parser_gpt = CVParserGPT(model_name, api_key, cv_format)
cv_parser = CVParser(cv_parser_gpt)

# Initialize Job Recommender
job_recommender = JobRecommender()

# Streamlit UI
st.title("CV Information Extractor & Job Recommender")
st.write("Upload a CV (PDF or Text format) and extract information, then get job recommendations.")

# File upload widget
uploaded_file = st.file_uploader("Upload CV", type=["pdf", "txt"])

# Load job data from JSON file
@st.cache_data
def load_job_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        job_data = json.load(file)
    return job_data

# Assuming the JSON file is located at "data/jobs.json"
job_data = load_job_data("data/jobs.json")

# Convert job data to DataFrame for better processing
job_df = pd.DataFrame(job_data)

# Streamlit section to display job data
if uploaded_file:
    # Extract text from PDF or text file
    if uploaded_file.type == "application/pdf":
        cv_raw_text = extract_text_from_pdf(uploaded_file)
    elif uploaded_file.type == "text/plain":
        cv_raw_text = uploaded_file.read().decode("utf-8")
    
    if cv_raw_text:
        # Show the raw CV text
        st.subheader("Raw CV Text")
        st.text(cv_raw_text)

        # Extract Information using the CVParser
        st.subheader("Extracted Information")
        extracted_info = cv_parser.extractInformation(cv_raw_text)
        st.json(extracted_info)  # Display extracted info as JSON
        
        # Attach the CV data to the Job Recommender
        job_recommender.attachJobs(job_df)
        job_recommender.attachCV(extracted_info)

        # Compute job recommendations based on CV
        recommendations = job_recommender.computeJobsSimilarity(sort=True, top_f=5)  # Get top 5 job recommendations
        
        # Display recommended jobs
        st.subheader("Job Recommendations")
        for job in recommendations:
            st.write(f"**Job Title:** {job['job_title']}")
            st.write(f"**Company:** {job['company_name']}")
            st.write(f"**Location:** {', '.join(job['location'])}")
            st.write(f"**Post Date:** {job['post_date']}")
            st.write(f"**Job URL:** [Link]({job['job_url']})")
            st.write("------")
