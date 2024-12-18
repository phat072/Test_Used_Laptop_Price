import streamlit as st
import pandas as pd
from io import BytesIO
from CVParser import CVParser
from JobRecommender import JobRecommender
import config as config
from CVParserGPT import CVParserGPT

# Load the cv_format configuration file
cv_format_path = 'data/cv_format.json'

# Load job and field data
jobinfo_path = 'data/jobs.json'
job_df = pd.read_json(jobinfo_path, encoding="utf-8")

encoded_fields_path = 'data/encoded_fields.json'
field_df = pd.read_json(encoded_fields_path, encoding="utf-8")

# Initialize CVParserGPT with model, API token, and cv_format configuration
cv_parser = CVParser(model=CVParserGPT(config.GPT_MODEL_NAME, config.GPT_TOKEN, cv_format=cv_format_path))

# Initialize JobRecommender
job_recommender = JobRecommender()
job_recommender.attachFields(field_df=field_df)
job_recommender.attachJobs(job_df=job_df)

# Streamlit UI
st.title('CV Parser and Job Recommendation')
st.write("Upload your CV in PDF format to get job recommendations based on your profile.")

# Upload CV
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
        # Parse the uploaded CV
    st.write("Parsing CV...")

    # Use CVParser to process the uploaded PDF
    cv_pdf_data = uploaded_file.read()
    try:
        # Step 1: Parse the PDF and extract raw text
        st.write("Extracting raw text from PDF...")
        raw_text = cv_parser.parseFromPDF(cv_pdf_data, extract_json=False)
        st.subheader("Extracted Raw Text")
        st.text_area("Raw Text", raw_text[:1000], height=300)  # Show the first 1000 characters for review

        # Step 2: Attempt JSON extraction
        st.write("Attempting to extract JSON from raw text...")
        cv_info = cv_parser.extractJSONFromText(raw_text)

        if not cv_info:
            st.error("No JSON-like structure was found in the CV. Please ensure the CV format matches the expected structure.")
            st.stop()

        # Step 3: Display extracted CV data
        st.subheader("Extracted CV Information")
        st.json(cv_info)

        # Standardize the extracted information
        standardized_cv_dict = cv_parser.standardizeCVDict(cv_info)

        # Step 4: Proceed with job recommendations
        job_recommender.attachCV(cv_dict=standardized_cv_dict)
        similarity_job_df = job_recommender.computeJobsSimilarity(sort=True, top_f=5)
        st.subheader("Top 5 Recommended Jobs")
        st.dataframe(similarity_job_df[['job_title', 'fields', 'similarity_score']].head(5))
    except Exception as e:
        st.error(f"Error parsing CV: {str(e)}")
