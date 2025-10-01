
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
