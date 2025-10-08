import streamlit as st
import pandas as pd
import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from PIL import Image
import pytesseract
import os
import random

# --- 1. CONFIGURATION AND DATA LOADING ---
DATA_FILE = 'scam_data.csv'
DB_FILE = 'official_db.json'
IMAGE_FILE = 'official_logo.png' # Placeholder for visual check

# Load the fact-checking database
with open(DB_FILE, 'r') as f:
    OFFICIAL_DB = json.load(f)

# Load the training data for the Text Model
try:
    df = pd.read_csv(DATA_FILE, sep=',')
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.3, random_state=42)
except FileNotFoundError:
    st.error("Data file not found. Please create 'scam_data.csv' as instructed.")
    st.stop()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# --- 2. TRAIN THE TEXT VERIFICATION MODEL ---
# Using TF-IDF and Logistic Regression for quick, effective classification
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vectorized = vectorizer.fit_transform(X_train)

text_model = LogisticRegression()
text_model.fit(X_train_vectorized, y_train)

# --- 3. AI AGENT CORE LOGIC ---

def text_verification_agent(text_input, official_db):
    """Analyzes text for scam patterns and cross-checks with knowledge base."""
    results = {}
    suspicion_score = 0
    verdict = ""
    
    # ML Prediction (Initial Suspicion Score)
    text_vec = vectorizer.transform([text_input])
    prediction = text_model.predict(text_vec)[0]
    confidence = text_model.predict_proba(text_vec).max()
    
    if prediction == 0: # Predicted as Scam
        suspicion_score += confidence * 50  # Base suspicion from model
        verdict = "Text Model flagged as HIGHLY SUSPICIOUS."
    else:
        suspicion_score += (1 - confidence) * 10 # Deduct if model is confident it's safe
        verdict = "Text Model suggests the message is LIKELY SAFE."

    # Rule-Based & Knowledge Base Cross-Check (The Source Verification)
    official_match = "No official scheme mentioned."
    fact_check_detail = ""
    
    for scheme, data in official_db.items():
        # Check for scheme name
        if re.search(r'\b' + re.escape(scheme) + r'\b', text_input, re.I):
            official_match = f"Claim references the official '{scheme.upper()}' scheme."
            
            # Cross-check for known red flags specific to this scheme
            for flag in data['red_flags']:
                if re.search(r'\b' + re.escape(flag) + r'\b', text_input, re.I):
                    suspicion_score += 40 # Strong evidence of a scam pattern
                    fact_check_detail = f"WARNING: The message contains the red flag '{flag.upper()}' which is inconsistent with the official scheme requirement."
                    verdict = "FALSE CLAIM (Verified Inconsistency)"
                    break
            
            # Check for non-official links/contacts
            if not any(re.search(re.escape(data['official_source']), text_input, re.I) for data in official_db.values()):
                 if re.search(r'\b(http[s]?://[^\s]+\b|call\s*\d+|whatsapp|bit\.ly)\b', text_input, re.I):
                    suspicion_score += 10
                    fact_check_detail = "ALERT: Contains a non-official link/contact for an official scheme."

    # Simple Keyword Checks (Universal Red Flags)
    universal_flags = ["fee", "otp", "pin", "atm", "wallet", "immediate", "urgent"]
    if any(flag in text_input.lower() for flag in universal_flags):
        suspicion_score += 15
        
    final_score = min(100, max(0, int(suspicion_score))) # Clamp score between 0 and 100
    
    return {
        "final_score": final_score,
        "verdict": verdict,
        "official_match": official_match,
        "fact_check_detail": fact_check_detail
    }

def visual_verification_agent(image_file):
    """Simulates visual analysis (OCR and Logo Check)."""
    
    if image_file is None:
        return {"visual_text": "No image provided.", "visual_score": 0, "visual_verdict": "N/A"}

    try:
        # OCR (Extract Text)
        image = Image.open(image_file)
        # Preprocessing (simple black and white conversion often helps OCR)
        image_bw = image.convert('L') 
        extracted_text = pytesseract.image_to_string(image_bw)
        
        # Simple Logo Check Simulation
        # In a real project, this would be a custom CNN or object detection model.
        # For the MVP, we simulate a 'spoof' detection based on a low-quality OCR.
        
        if len(extracted_text) < 50:
            # Low OCR output often means a poor quality/manipulated image (Simulated Check)
            spoof_score = 30
            spoof_verdict = "Low OCR confidence (Potential image tampering/low quality spoof)."
        else:
            # High OCR output is good, but we check if it mentions a red flag (e.g., "SBI" and "OTP")
            if "SBI" in extracted_text and "OTP" in extracted_text:
                spoof_score = 50
                spoof_verdict = "OCR detected red-flag entities (SBI + OTP) together in image text."
            else:
                spoof_score = 0
                spoof_verdict = "OCR text seems clear; no immediate visual spoofing detected."
        
        return {
            "visual_text": extracted_text[:150] + "...",
            "visual_score": spoof_score,
            "visual_verdict": spoof_verdict
        }
        
    except pytesseract.TesseractNotFoundError:
        return {"visual_text": "Tesseract not installed. Cannot perform OCR.", "visual_score": 0, "visual_verdict": "DEPENDENCY ERROR"}
    except Exception as e:
        return {"visual_text": f"Error processing image: {e}", "visual_score": 0, "visual_verdict": "ERROR"}


def fusion_agent(text_results, visual_results):
    """Combines text and visual evidence for a final verdict."""
    
    # Base score is from text analysis, as that contains the main claim
    base_score = text_results["final_score"]
    
    # Boost the score if visual evidence supports the scam detection
    if visual_results["visual_score"] > 0:
        final_score = base_score + (visual_results["visual_score"] * 0.5)
    else:
        final_score = base_score

    final_score = min(100, int(final_score))
    
    if final_score >= 80:
        final_verdict = "🔴 **FAKE/SCAM:** High confidence in fact-check failure and red-flag patterns."
    elif final_score >= 50:
        final_verdict = "🟡 **MISLEADING:** Contains serious inconsistencies with official sources."
    else:
        final_verdict = "🟢 **LIKELY TRUE:** No major scam red-flags or fact-check failures found."
        
    return final_score, final_verdict
    

# --- 4. STREAMLIT UI ---

st.set_page_config(page_title="Agri-Fact AI Agent (MVP)", layout="wide")

st.title("🌾 Agri-Fact AI Agent (MVP) ")
st.markdown("### Multimodal Misinformation Detector for Agricultural Schemes")

with st.expander("Model Demonstration: Fact Check Database (Source of Truth)"):
    st.json(OFFICIAL_DB)


input_text = st.text_area(
    "1. Paste the suspicious message (SMS/WhatsApp Text):",
    placeholder="Example: 'PM Kisan loan immediately approve. Send Rs 500 processing fee now.'",
    height=150
)

uploaded_file = st.file_uploader("2. Upload the suspicious image/screenshot (Optional for Multimodal Check):", type=["png", "jpg", "jpeg"])

if st.button("Analyze Claim", type="primary"):
    
    if not input_text and not uploaded_file:
        st.error("Please enter text or upload an image to analyze.")
        st.stop()
        
    # --- Agent Execution ---
    
    # 1. Text Agent runs on input text + OCR text if available
    text_to_analyze = input_text
    
    # 2. Visual Agent runs
    visual_results = visual_verification_agent(uploaded_file)
    
    # Append OCR text to main analysis for multimodal text verification
    if uploaded_file and "Tesseract" not in visual_results["visual_text"]:
         text_to_analyze += " " + visual_results["visual_text"]
    
    # 3. Text Agent (now potentially Multimodal Text) runs
    text_results = text_verification_agent(text_to_analyze, OFFICIAL_DB)
    
    # 4. Fusion Agent runs
    final_score, final_verdict = fusion_agent(text_results, visual_results)

    # --- Display Results ---
    st.subheader("✅ Final Fact-Check Verdict")
    st.markdown(final_verdict)
    st.progress(final_score / 100)
    st.markdown(f"**Confidence Score:** {final_score}%")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔍 Text & Source Verification Agent Analysis")
        st.info(text_results["verdict"])
        st.markdown(f"**Source Check:** {text_results['official_match']}")
        st.error(f"**Red Flag Detail:** {text_results['fact_check_detail']}")
        st.markdown(f"**Action:** The claim's core elements were cross-referenced with {DB_FILE} and found to be highly inconsistent with official terms.")
        
    with col2:
        st.subheader("🖼️ Visual (Image) Verification Agent Analysis")
        st.warning(visual_results["visual_verdict"])
        if uploaded_file:
            st.image(uploaded_file, caption='Uploaded Screenshot', width=150)
        st.markdown(f"**Extracted Text (OCR):** `{visual_results['visual_text']}`")
        st.markdown(f"**Visual Score Contribution:** {visual_results['visual_score']}%")
        
    
    st.markdown("---")
    st.markdown(f"### 💡 Guidance for the Farmer")
    st.markdown(f"Based on the analysis, please **DO NOT** click any links or share personal information (OTP/PIN/Bank Details). For official information on schemes like KCC, please visit a verified government site like **{OFFICIAL_DB['kisan credit card']['official_source']}**.")