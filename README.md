Sentiment Analyzer 

This project is a web application that analyzes the sentiment of text (Positive, Negative, or Neutral). It uses a DistilBERT model that has been fine-tuned on a dataset of tweets about U.S. airlines. The user interface is built with Gradio.

APP:https://huggingface.co/spaces/tabithaclitus/sentiment_analyzer

Demo

Here is a screenshot of the running application:

<img width="1907" height="785" alt="Screenshot 2025-10-07 202539" src="https://github.com/user-attachments/assets/024b1a40-5bdf-43f6-b68d-f809c3d7853c" />

Real-time Sentiment Prediction: Enter any text and get an instant sentiment classification.

Confidence Scores: See the model's confidence for each sentiment category (Positive, Negative, Neutral).

Interactive Interface: A simple and clean web interface built with Gradio.

Custom-Trained Model: The model is not a generic sentiment model; it has been specifically fine-tuned on the language and context of airline tweets.

Technology Stack

Backend & ML: Python

ML Framework: PyTorch

NLP Library: Hugging Face Transformers (for the DistilBERT model and Trainer API)

Data Handling: Hugging Face Datasets, Pandas

Web Framework: Streamlit

Project Structure

sentiment-app/

├── my_sentiment_model/   # The saved fine-tuned model and tokenizer

│   ├── config.json

│   ├── pytorch_model.bin

│   └── ...

├── app.py                # The main Streamlit application script

├── requirements.txt      # List of Python dependencies

└── README.md             # This file

Model Training Process

The core of this project is the fine-tuned model. The training was performed in a Google Colab notebook and followed these key steps:

Dataset: The model was trained on the TweetEval "sentiment" dataset from the Hugging Face Hub. This dataset contains tweets about U.S. airlines, each labeled as positive, negative, or neutral.

Base Model: We used the pre-trained distilbert-base-uncased model as a starting point. This model has a general understanding of the English language.

Fine-Tuning: Using the Hugging Face Trainer API, we fine-tuned the base model on the airline tweet dataset for 3 epochs. This process specialized the model for the task of sentiment analysis in this specific context.

Evaluation: The best-performing model (from epoch 2) achieved a validation accuracy of 73.75%. This model was saved and is the one used in the Gradio application.


