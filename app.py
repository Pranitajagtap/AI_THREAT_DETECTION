import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

st.title("AI Threat Detection System (Lite)")

# 1️⃣ Load trained Random Forest model
model = joblib.load('phishing_model.pkl')

# 2️⃣ Lightweight feature extractor for a single URL
def extract_features(url):
    return pd.DataFrame({
        'NumDots': [url.count('.')],
        'UrlLength': [len(url)],
        'NumDash': [url.count('-')],
        'NoHttps': [1 if not url.startswith('https') else 0],
        'AtSymbol': [1 if '@' in url else 0],
        'NumQueryComponents': [url.count('?')],
        'NumNumericChars': [sum(c.isdigit() for c in url)]
    })

# 3️⃣ Input URL
url_input = st.text_input("Enter a URL to check:")

if st.button("Predict"):
    if not url_input:
        st.warning("Please enter a URL first!")
    else:
        features = extract_features(url_input)
        prediction = model.predict(features)[0]

        st.write("⚠️ Malicious" if prediction == 1 else "✅ Safe")

        # Optional SHAP explanation
        try:
            explainer = shap.Explainer(model, features)
            shap_values = explainer(features)

            st.subheader("Feature Importance (SHAP)")

            fig, ax = plt.subplots(figsize=(8, 5))
            shap.summary_plot(shap_values, features, plot_type="bar", show=False)
            st.pyplot(fig)
        except Exception as e:
            st.write("⚠️ SHAP explanation not available:", e)
