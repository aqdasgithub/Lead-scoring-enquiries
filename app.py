
import streamlit as st
import pandas as pd

st.title("🎯 Lead Scoring Prediction App")

st.write("Upload a CSV file of new leads to predict conversion probability")

uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    new_leads = pd.read_csv(uploaded_file)
    st.write("Uploaded Data:")
    st.dataframe(new_leads.head())

    # Encode and scale (reuse preprocessing pipeline)
    for col in ['Email_Source','Contacted','Location','Profession','Course_Interest']:
        new_leads[col] = le.fit_transform(new_leads[col])
    new_scaled = scaler.transform(new_leads)

    # Predict scores
    scores = best_model.predict_proba(new_scaled)[:,1]
    new_leads['Lead_Score'] = scores
    new_leads = new_leads.sort_values(by='Lead_Score', ascending=False)

    st.write("### Predicted Lead Scores")
    st.dataframe(new_leads[['Lead_Score'] + [col for col in new_leads.columns if col != 'Lead_Score']])

    # Download option
    csv = new_leads.to_csv(index=False).encode('utf-8')
    st.download_button("Download Results", csv, "Lead_Scoring_Results.csv", "text/csv")

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# --- MOCK MODEL AND PREPROCESSING OBJECTS ---
# In a real application, you would load these objects using pickle or joblib:
# le = joblib.load('label_encoder.pkl')
# scaler = joblib.load('scaler.pkl')
# best_model = joblib.load('model.pkl')

# Mock Preprocessing Objects (trained on dummy data)
class MockModel:
    """Simulates a trained model with a predict_proba method."""
    def predict_proba(self, X):
        # Generates a random probability array (for demonstration)
        # Probabilities are biased towards higher values for visual effect
        n_samples = X.shape[0]
        # Using the last two features (numerical) for mock score generation
        base_scores = np.clip(np.mean(X[:, -2:], axis=1) * 0.1 + 0.5, 0.1, 0.9)
        scores = np.column_stack([1 - base_scores, base_scores])
        return scores

# Initialize mock model
best_model = MockModel()

# Initialize real scikit-learn preprocessing objects
# NOTE: We now use a dictionary to store one LabelEncoder per categorical feature
label_encoders = {}
scaler = StandardScaler()

# --- DEFINING FEATURES BASED ON UPLOADED DATASET STRUCTURE ---
# Categories derived from the CSV snippet (using common values)
CATEGORICAL_FEATURES = {
    'Email_Source': ['Email', 'Social Media', 'Website', 'Referral', 'Walk-in', 'Other'],
    'Contacted': ['Yes', 'No'],
    'Location': ['Pune', 'Delhi', 'Bangalore', 'Hyderabad', 'Kolkata', 'Nagpur', 'Chennai', 'Ahmedabad', 'Jaipur', 'Other'],
    'Profession': ['Student', 'Working Professional', 'Freelancer', 'Other'],
    'Course_Interest': ['Data Science - Beginner', 'Data Science - Advanced', 'Machine Learning', 'AI & Deep Learning', 'Other']
}

# Numerical features confirmed by the document/CSV: Engagement_Score (1-100) and Follow_Up_Count
NUMERICAL_COLS = ['Engagement_Score', 'Follow_Up_Count']

# Fit the encoders/scaler on dummy data to set up the transformations.
n_samples = 50
dummy_data = pd.DataFrame({
    'Email_Source': np.random.choice(CATEGORICAL_FEATURES['Email_Source'], n_samples),
    'Contacted': np.random.choice(CATEGORICAL_FEATURES['Contacted'], n_samples),
    'Location': np.random.choice(CATEGORICAL_FEATURES['Location'], n_samples),
    'Profession': np.random.choice(CATEGORICAL_FEATURES['Profession'], n_samples),
    'Course_Interest': np.random.choice(CATEGORICAL_FEATURES['Course_Interest'], n_samples),
    'Engagement_Score': np.random.randint(1, 101, n_samples), # Range 1 to 100
    'Follow_Up_Count': np.random.randint(0, 6, n_samples)      # Range 0 to 5
})

# Preprocessing setup (Fit LabelEncoder)
for col in CATEGORICAL_FEATURES.keys():
    # FIX: Create and fit a unique LabelEncoder for each column
    encoder = LabelEncoder()
    # Fit on all possible categories defined above
    encoder.fit(CATEGORICAL_FEATURES[col]) 
    label_encoders[col] = encoder # Store the fitted encoder instance
    # Transform dummy data for use in scaler fitting (though not strictly necessary)
    dummy_data[col] = encoder.transform(dummy_data[col])

# Fit StandardScaler
scaler.fit(dummy_data[NUMERICAL_COLS].values)

# --- APP LAYOUT AND LOGIC ---
st.set_page_config(page_title="Lead Scoring Prediction", layout="centered")
st.title("🎯 Lead Scoring Prediction App")
st.markdown("Use the form below for a single lead prediction, or upload a CSV for batch scoring.")


# 1. USER INPUT SECTION (SINGLE LEAD)
st.header("1. Predict Single Lead Score")

with st.form("single_lead_form"):
    st.subheader("Lead Information")

    # Categorical Features
    col1, col2 = st.columns(2)
    with col1:
        email_source = st.selectbox("Email Source", CATEGORICAL_FEATURES['Email_Source'])
        profession = st.selectbox("Profession", CATEGORICAL_FEATURES['Profession'])
        contacted = st.radio("Was Contacted?", CATEGORICAL_FEATURES['Contacted'])
    with col2:
        location = st.selectbox("Location", CATEGORICAL_FEATURES['Location'])
        course_interest = st.selectbox("Course Interest", CATEGORICAL_FEATURES['Course_Interest'])

    # Numerical Features (Updated to use Engagement_Score and Follow_Up_Count)
    st.subheader("Activity Metrics")
    col3, col4 = st.columns(2)
    with col3:
        # Engagement_Score (Range 1-100 as per document)
        engagement_score = st.number_input(
            "Engagement Score (1-100)", 
            min_value=1, 
            max_value=100, 
            value=50, 
            step=1
        )
    with col4:
        # Follow_Up_Count (Assuming integer, typically low count)
        follow_up_count = st.number_input(
            "Follow Up Count", 
            min_value=0, 
            max_value=10, 
            value=2, 
            step=1
        )

    submitted = st.form_submit_button("Get Lead Score")

    if submitted:
        try:
            # Create a DataFrame from the input with correct column names
            input_data = pd.DataFrame([{
                'Email_Source': email_source,
                'Contacted': contacted,
                'Location': location,
                'Profession': profession,
                'Course_Interest': course_interest,
                'Engagement_Score': engagement_score,
                'Follow_Up_Count': follow_up_count
            }])

            # 1. Encode Categorical Data
            processed_data = input_data.copy()
            for col in CATEGORICAL_FEATURES.keys():
                # FIX: Use the specific LabelEncoder from the dictionary for this column
                le_instance = label_encoders[col]
                input_value = processed_data[col].iloc[0]

                try:
                    # Transform the single input value
                    processed_data[col] = le_instance.transform([input_value])[0]
                except ValueError:
                    # Fallback to the 'Other' or last category if unseen
                    st.warning(f"Unseen category '{input_value}' found for {col}. Mapping to 'Other'.")
                    processed_data[col] = le_instance.transform([le_instance.classes_[-1]])[0]


            # 2. Scale Numerical Data (Uses NUMERICAL_COLS: Engagement_Score, Follow_Up_Count)
            numerical_input = processed_data[NUMERICAL_COLS].values
            scaled_numerical = scaler.transform(numerical_input)
            
            # Combine all features for prediction (Categorical first, then Numerical)
            encoded_categorical = processed_data[CATEGORICAL_FEATURES.keys()].values
            final_features = np.hstack([encoded_categorical, scaled_numerical])


            # 3. Predict Score
            score = best_model.predict_proba(final_features)[:, 1][0]
            
            # 4. Display Result
            st.success("✅ Prediction Complete!")
            st.markdown(f"**Conversion Probability (Lead Score):**")
            st.metric("", f"{score:.2f} ({(score * 100):.0f}%)")

            if score > 0.5:
                st.balloons()
                st.info("🔥 High Priority Lead! Assign immediately to sales for follow-up.")
            elif score > 0.3:
                st.info("⭐ Medium Priority Lead. Nurture with targeted content.")
            else:
                st.info("🐢 Low Priority Lead. Keep in the general pool.")

        except Exception as e:
            st.error(f"An error occurred during single prediction. Please ensure all inputs are valid. Error: {e}")


# --- 2. BATCH UPLOAD SECTION (EXISTING FUNCTIONALITY) ---
st.header("2. Batch Score Prediction (CSV Upload)")
st.write("Upload a CSV file of new leads to predict conversion probability")

uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the file
    new_leads = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview:")
    st.dataframe(new_leads.head())

    try:
        # Preprocessing on Batch Data
        processed_leads = new_leads.copy()
        
        # 1. Encode and Scale
        for col in CATEGORICAL_FEATURES.keys():
            # FIX: Use the specific LabelEncoder for this column
            le_instance = label_encoders[col]
            
            if col in processed_leads.columns:
                # Apply transformation to the entire column
                processed_leads[col] = processed_leads[col].apply(
                    lambda x: le_instance.transform([x])[0] if x in le_instance.classes_ else le_instance.transform([le_instance.classes_[-1]])[0]
                )
            else:
                st.warning(f"Feature column '{col}' missing from uploaded file. Assuming default category.")
                processed_leads[col] = le_instance.transform([le_instance.classes_[-1]])[0] * len(processed_leads)


        # Ensure numerical columns are present before scaling
        if all(col in processed_leads.columns for col in NUMERICAL_COLS):
            scaled_batch = scaler.transform(processed_leads[NUMERICAL_COLS].values)
        else:
            raise ValueError(f"Uploaded CSV is missing required numerical columns: {NUMERICAL_COLS}")
        
        # Reconstruct the feature matrix
        encoded_categorical_batch = processed_leads[CATEGORICAL_FEATURES.keys()].values
        final_batch_features = np.hstack([encoded_categorical_batch, scaled_batch])

        # 2. Predict scores
        scores = best_model.predict_proba(final_batch_features)[:, 1]
        
        # 3. Add score and sort
        new_leads['Lead_Score'] = scores
        new_leads = new_leads.sort_values(by='Lead_Score', ascending=False)

        st.write("### Predicted Lead Scores (Ranked)")
        score_df = new_leads[['Lead_Score'] + [col for col in new_leads.columns if col not in ['Lead_Score', 'Name', 'Lead_ID', 'Converted']]]
        st.dataframe(score_df)

        # Download option
        @st.cache_data
        def convert_df(df):
            # Cache the conversion to prevent computation on every rerun
            return df.to_csv(index=False).encode('utf-8')

        csv = convert_df(new_leads) # Download the full results with all original columns
        st.download_button(
            "Download Full Results CSV",
            csv,
            "Lead_Scoring_Results.csv",
            "text/csv",
            key='download-csv'
        )

    except Exception as e:
        st.error(f"Error processing batch file. Check if the CSV columns and data types match the expected training features. Details: {e}")

