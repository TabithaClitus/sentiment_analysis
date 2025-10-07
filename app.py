import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- App Title and Description ---
st.title("Sentiment Analyzer App")
st.write(
    "This app uses a fine-tuned DistilBERT model to predict the sentiment of a sentence about airlines. "
    "Enter a sentence below and click 'Analyze'."
)

# --- Model Loading ---
# This decorator caches the model so it's only loaded once.
@st.cache_resource
def load_model():
    """Loads the fine-tuned model and tokenizer."""
    model_path = "./my_sentiment_model"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval() # Set model to evaluation mode
    return tokenizer, model, device

# Load the model and tokenizer
tokenizer, model, device = load_model()

# --- Prediction Function ---
def predict(text):
    """Makes a prediction using the loaded model."""
    sentiment_labels = ['Negative', 'Neutral', 'Positive']
    
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    
    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Process the output
    logits = outputs.logits
    probabilities = F.softmax(logits, dim=-1)
    predicted_class_idx = torch.argmax(probabilities, dim=-1).item()
    
    predicted_label = sentiment_labels[predicted_class_idx]
    confidence_scores = {label: prob.item() for label, prob in zip(sentiment_labels, probabilities[0])}
    
    return predicted_label, confidence_scores

# --- User Interface ---
user_input = st.text_area("Enter your sentence here:", "The flight was on time and the crew was friendly.")

if st.button("Analyze Sentiment"):
    if user_input:
        # Get the prediction
        predicted_label, scores = predict(user_input)
        
        # Display the result
        st.subheader("Prediction Result")
        st.write(f"The predicted sentiment is: **{predicted_label}**")
        
        st.subheader("Confidence Scores")
        st.write(scores)
        
        # Display a bar chart of the scores
        st.bar_chart(scores)
        
    else:
        st.warning("Please enter a sentence to analyze.")