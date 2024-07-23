import streamlit as st
from textblob import TextBlob
from transformers import pipeline
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(page_title="Sentiment and Emotion Analyzer", page_icon=":smiley:", layout="wide")

# Page title
st.title("Sentiment and Emotion Analyzer")

# User input
user_input = st.text_area("Enter text or paragraph to analyze:", height=150)

if st.button("Analyze"):
    if user_input:
        # Sentiment analysis
        blob = TextBlob(user_input)
        senti = blob.sentiment.polarity

        if senti < 0:
            sentiment = 'negative'
            col = "#FF6666"
        elif senti > 0:
            sentiment = 'positive'
            col = "#b3d23e"
        else:
            sentiment = 'neutral'
            col = "#d3d3d3"

        # Emotion classification
        classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)
        model_outputs = classifier(user_input)

        high_score_1 = 0
        emotion_1 = ''
        for output in model_outputs[0]:
            if output['score'] > high_score_1:
                high_score_1 = output['score']
                emotion_1 = output['label']

        # Display sentiment and emotion
        st.subheader("Analysis Results")
        st.write(f"**Sentiment:** {sentiment.capitalize()}")
        st.write(f"**Emotion:** {emotion_1.capitalize()}")

        # Graphical representation of sentiment
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.barh(y=0, width=senti, height=1, color=col)

        ax.set_xlim(-1, 1)
        ax.set_ylim(-0.5, 0.5)
        ax.grid(True)
        ax.axvline(0, color='black', linewidth=0.5)
        ax.set_title('Graphical Representation of Sentiment')
        ax.set_xlabel('Sentiment Value')
        ax.get_yaxis().set_visible(False)
        ax.text(senti, 0, f'{senti:.2f}', color='black', va='center', ha='left' if senti < 0 else 'right')

        st.pyplot(fig)

        # Optional: Display the entire emotion distribution
        st.subheader("Emotion Distribution")
        emotion_scores = {output['label']: output['score'] for output in model_outputs[0]}
        emotion_labels = list(emotion_scores.keys())
        emotion_values = list(emotion_scores.values())

        plt.figure(figsize=(12, 6))
        sns.barplot(x=emotion_values, y=emotion_labels, palette='viridis')
        plt.xlabel('Score')
        plt.ylabel('Emotion')
        plt.title('Emotion Distribution')

        st.pyplot(plt)
    else:
        st.error("Please enter some text to analyze.")
