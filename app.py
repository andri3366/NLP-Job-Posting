import streamlit as st
import pickle
import joblib
import os
from src.predict import predict_posting
from src.predict_text import predict_posting_text
from src.llm_explain import explain_prediction
# with open('model/lr_model.pkl', 'rb') as f:
#     model = joblib.load(f)

# with open('model/lr_vectorizer.pkl', 'rb') as f:
#     vectorizer = joblib.load(f)

# with open('model/lr_cat_features.pkl', 'rb') as f:
#     cat_columns = joblib.load(f)

base_dir = os.path.dirname(__file__)
model_path = os.path.join(base_dir, 'model/best_model.pkl')
vectorizer_path = os.path.join(base_dir, 'model/vectorizer.pkl')
cat_features_path = os.path.join(base_dir, 'model/cat_features.pkl')
text_model_path = os.path.join(base_dir, 'model/text_best_model.pkl')
text_vectorizer_path = os.path.join(base_dir, 'model/text_vectorizer.pkl')

# model = joblib.load(model_path)
# vectorizer = joblib.load(vectorizer_path)
# cat_columns = joblib.load(cat_features_path)

# text_model = joblib.load(text_model_path)
# text_vectorizer = joblib.load(text_vectorizer_path)

@st.cache_resource
def load_full_resources():
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    cat_columns = joblib.load(cat_features_path)

    return model, vectorizer, cat_columns


@st.cache_resource
def load_text_resources():
    text_model = joblib.load(text_model_path)
    text_vectorizer = joblib.load(text_vectorizer_path)

    return text_model, text_vectorizer

if 'result' not in st.session_state:
    st.session_state.result = None
if 'text' not in st.session_state:
    st.session_state.text = ""
if 'mode' not in st.session_state:
    st.session_state.mode = "Full Model"
if 'explanation' not in st.session_state:
    st.session_state.explanation = None
    
st.title("Fake Job Postings Detection")
st.write("This app detects whether a job posting is fake or not based on its description and other features.")
st.write("Please enter the job posting details below:")

# toggle button
mode = st.radio("Select Mode", ("Full Model", "Text Only Model"))

if mode != st.session_state.mode:
    st.session_state.mode = mode
    st.session_state.result = None
    st.session_state.explanation = None
    
if mode == "Text Only Model":
    text = st.text_area("Job Description")
    
    text_model, text_vectorizer = load_text_resources()
    
    if st.button('Analyze Text', key="text_analyze_btn"):
        st.session_state.result = predict_posting_text(text)
        st.session_state.text = text
        st.session_state.explanation = None
        
    if st.session_state.result:
        result = st.session_state.result

        st.success(
            f"Prediction: {result['label']} "
        )
        
        st.write(f"Confidence: {result['confidence']:.2f}")
        
        st.divider()
        st.subheader("Model Explanation")
        
        prompt = st.text_input(
            "Ask a question about the prediction",
            max_chars=200,
            placeholder="e.g., Why was this job posting classified as fraudulent?"
        )
        
        if st.button('Get Explanation', key="text_explanation_btn"):
            try:
                
                with st.spinner("Generating explanation..."):
                    st.session_state.explanation = explain_prediction(
                        prediction=result['label'],
                        text=st.session_state.text,
                        prompt=prompt
                    )     
                
                st.info(st.session_state.explanation)
            except Exception as e:
                st.error(f"App error: {str(e)}")
                print("Full error details:", e )
else:
    model, vectorizer, cat_columns = load_full_resources()
    title = st.text_input("Job Title")
    company_profile = st.text_area("Company Profile")
    text = st.text_area("Job Description")
    requirements = st.text_area("Job Requirements")
    benefits = st.text_area("Job Benefits")
    
    telecommuting = st.selectbox("Work From Home ", ["No", "Yes"])
    has_company_logo = st.selectbox("Has Company Logo", ["No", "Yes"])
    has_questions = st.selectbox("Has Questions", ["No", "Yes"])

    employment_type = st.selectbox(
        "Employment Type", 
        ["Missing","Full-time", "Part-time", "Contract", "Temporary", "Other"]
    )

    required_education = st.selectbox(
        "Required Education",
        ["Missing", "Bachelor's Degree", "High School or equivalent", "Unspecified", "Master's Degree", "Associate Degree", "Certification", "Some College Coursework Completed", "Professional", "Vocational", "Some High School Coursework", "Doctorate", "Vocational - Degree", "Vocational - HS Degree"]
    )

    required_experience = st.selectbox(
        "Required Experience",
        ["Missing","Internship", "Entry level", "Mid-Senior level", "Associate", "Director", "Internship", "Executive"]
    )

    country = st.selectbox(
        "Country",
        ["Missing","AU", "CA", "DE", "GB", "IN", "NZ", "Other", "US"]
    )

    if st.button('Analyze Posting', key="full_analyze_btn"):
        
        # st.session_state.result = predict_posting(text)
        # st.session_state.text = text
        
        # result = st.session_state.result
        # text = st.session_state.text
        
        combined_text = (title + ' ' + company_profile + ' ' + text + ' ' + requirements + ' ' + benefits)
        result = predict_posting(
            combined_text,
            telecommuting,
            has_company_logo,
            has_questions,
            employment_type,
            required_experience,
            required_education,
            country
        )
        
        # text = combined_text
        
        st.session_state.result =result
        st.session_state.text = combined_text
        st.session_state.explanation = None
        
    if st.session_state.result:
            result = st.session_state.result
            st.success(
                f"Prediction: {result['label']} "
            )
        
            st.write(f"Confidence: {result['confidence']:.2f}")
        
            st.divider()
            st.subheader("Model Explanation")
        
            prompt = st.text_input(
                "Ask a question about the prediction",
                max_chars=200,
                placeholder="e.g., Why was this job posting classified as fraudulent?"
            )
        
            if st.button('Get Explanation', key="full_explanation_btn"):
                try:
                    with st.spinner("Generating explanation..."):
                        
                        st.session_state.explanation = explain_prediction(
                            prediction=result['label'],
                            text=st.session_state.text,
                            prompt=prompt
                        )
                    
                    st.info(st.session_state.explanation)
                except Exception as e:
                    st.error(f"App error: {str(e)}")
                    print("Full error details:", e )