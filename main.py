import streamlit as st
from textblob import TextBlob
from transformers import pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import pyttsx3

st.set_page_config(page_title="EmoSent", page_icon=":smiley:", layout="centered")

st.title("Sentiment and Emotion Analyzer")

user_input = st.text_area("Enter text or paragraph to analyze:", height=150)

if st.button("Analyze"):
    if user_input:
        blob = TextBlob(user_input)
        senti = blob.sentiment.polarity

        if senti < 0:
            sentiment = 'Negative'
            col = "#FF6666"
        elif senti > 0:
            sentiment = 'Positive'
            col = "#b3d23e"
        else:
            sentiment = 'Neutral'
            col = "#d3d3d3"

        # Emotions
        classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)
        model_outputs = classifier(user_input)
        emotion_score = model_outputs[0][0]['score']
        emotion = model_outputs[0][0]['label']

        # Summary
        summarizer = pipeline("summarization", model="philschmid/bart-large-cnn-samsum")
        summary = summarizer(user_input, max_length=100, min_length=30, do_sample=False)[0]['summary_text']

        st.subheader("Analysis Results")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Sentiment:** {sentiment}")
        with col2:
            st.write(f"**Emotion:** {emotion.capitalize()}")

        col3, col4 = st.columns(2)
        with col3:
            st.subheader("Summary")
            st.write(f"{summary}")
        
        with col4:
            st.subheader("Graphs")
            # Graph 1 : Sentiment
            fig, ax = plt.subplots(figsize=(10, 0.5))
            ax.barh(y=0, width=senti, height=1, color=col)

            ax.set_xlim(-1, 1)
            ax.set_ylim(-0.5, 0.5)
            ax.grid(True)
            ax.axvline(0, color='black', linewidth=0.5)
            ax.set_xlabel(f'Sentiment = {sentiment}')
            ax.get_yaxis().set_visible(False)
            ax.text(senti, 0, f'{senti:.2f}', color='black', va='center', rotation=90 if senti > 0 else 270, ha='left' if senti < 0 else 'right')
            plt.title("Senitments Value Representation")
            st.pyplot(fig)

            # Graph 2 : Emotion
            emotion_scores = {output['label']: output['score'] for output in model_outputs[0]}
            graphEmotions = list(emotion_scores.keys())[:5]
            graphValue = list(emotion_scores.values())[:5]

            plt.figure(figsize=(12, 6))
            sns.barplot(x=graphValue, y=graphEmotions, palette='viridis')
            plt.xlabel('Accuracy')
            plt.ylabel('Top 5 Emotions')
            plt.title('Emotion Distribution')
            st.pyplot(plt)

        # Text to speech
        tts_engine = pyttsx3.init()
        tts_engine.say(f'The above paragraph is {sentiment}')
        tts_engine.say(f'The emotion in the above paragraph is {emotion}')
        tts_engine.runAndWait()

    else:
        st.error("Please enter some text to analyze.")
