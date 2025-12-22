import streamlit as st
import joblib
from utils import preprocess_text

# Page configuration
st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="🔮",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("🔮 Multi-Class Sentiment Analysis")
st.markdown("### Daraz E-Commerce Review Sentiment Classifier")
st.markdown("---")

# Sidebar information
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    **Project**: Multi-class Sentiment Analysis
    
    **Model**: SVM (Support Vector Machine)
    
    **Features**:
    - TF-IDF Vectorization
    - 5000 features
    - Linear Kernel
    
    **Classes**:
    - 😞 Negative (-1)
    - 😐 Neutral (0)
    - 😊 Positive (1)
    
    **Language**: English + Roman Urdu (Code-mixed)
    """)
    
    st.markdown("---")
    st.markdown("**How it works:**")
    st.markdown("""
    1. Enter a product review
    2. Text is preprocessed
    3. SVM model predicts sentiment
    4. Result is displayed
    """)

# Load model and vectorizer
@st.cache_resource
def load_model():
    """Load the trained SVM model and vectorizer"""
    try:
        model = joblib.load('svm_model.pkl')
        vectorizer = joblib.load('vectorizer.pkl')
        return model, vectorizer
    except FileNotFoundError:
        st.error("❌ Model files not found! Please run the main.ipynb notebook first to train and save the model.")
        st.stop()

# Load model
svm_model, vectorizer = load_model()

# Main content
st.markdown("### 📝 Enter Your Review")

# Create input box
review_input = st.text_area(
    "Paste your product review here:",
    placeholder="Example: This product is amazing! It works perfectly and arrived quickly.",
    height=150,
    label_visibility="collapsed"
)

# Create columns for buttons
col1, col2 = st.columns([3, 1])

with col2:
    predict_button = st.button("Classify", type="primary", use_container_width=True)

# Process and predict
if predict_button:
    if not review_input.strip():
        st.warning("⚠️ Please enter a review before clicking Classify")
    else:
        with st.spinner("🔄 Processing your review..."):
            # Preprocess the text
            processed_review = preprocess_text(review_input)
            
            # Vectorize
            vectorized_input = vectorizer.transform([processed_review])
            
            # Make prediction
            prediction = svm_model.predict(vectorized_input)[0]
            
            # Get confidence scores (decision function)
            confidence_scores = svm_model.decision_function(vectorized_input)[0]
            
            # Map prediction to sentiment
            sentiment_map = {-1: "Negative", 0: "Neutral", 1: "Positive"}
            emoji_map = {-1: "😞", 0: "😐", 1: "😊"}
            color_map = {-1: "negative", 0: "neutral", 1: "positive"}
            
            sentiment_label = sentiment_map[prediction]
            emoji = emoji_map[prediction]
            color_class = color_map[prediction]
            
            # Display results
            st.markdown("---")
            st.markdown("### 🎯 Prediction Results")
            
            # Sentiment display
            sentiment_html = f"**Sentiment:** {emoji} **{sentiment_label.upper()}** (Class: {prediction})"
            st.success(sentiment_html)
            
            # Processed review
            with st.expander("📖 View Processed Review"):
                st.write(processed_review)
            
            # Model information
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Model", "SVM")
            with col2:
                st.metric("Vectorizer", "TF-IDF")
            with col3:
                st.metric("Features", "5000")
            
            st.markdown("---")
            st.success("✅ Classification complete!")

# Footer
st.markdown("---")
st.markdown("Sentiment Analysis System | Multi-Class Classification")
