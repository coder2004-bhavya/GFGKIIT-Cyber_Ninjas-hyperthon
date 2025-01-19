!pip install streamlit
!pip install altair
!pip install requests
!pip install bs4
!pip install joblib
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import requests
from bs4 import BeautifulSoup
import joblib
LOAD THE TRAINED MODEL
pipe_lr = joblib.load(open("text_emotion.pkl", "rb"))
emotions_emoji_dict = {
    "anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗", "joy": "😂",
    "neutral": "😐", "sad": "😔", "sadness": "😔", "shame": "😳", "surprise": "😮"
}

# Function to predict emotions
def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]
# Function to get prediction probabilities
def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results
# Function to scrape text from a URL
def scrape_text_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        text = ' '.join([p.get_text() for p in paragraphs])
        return text
    except Exception as e:
        st.error(f"Error scraping the URL: {e}")
        return ""
def main():
    st.title("Text Emotion Detection")
    st.subheader("Detect Emotions In Text or From Web Links")
    st.markdown('<h5 style="color:gray;">By IOT Lab</h5>', unsafe_allow_html=True)

    # Sidebar options for input type
    st.sidebar.title("Input Options")
    input_type = st.sidebar.radio("Select Input Type:", ["Manual Text", "Web Link Scraping"])

    # Input based on the selected option
    if input_type == "Manual Text":
        with st.form(key='manual_text_form'):
            raw_text = st.text_area("Type Here")
            submit_text = st.form_submit_button(label='Submit')

        if submit_text:
            col1, col2 = st.columns(2)

            prediction = predict_emotions(raw_text)
            probability = get_prediction_proba(raw_text)

            with col1:
                st.success("Original Text")
                st.write(raw_text)

                st.success("Prediction")
                emoji_icon = emotions_emoji_dict[prediction]
                st.write("{}:{}".format(prediction, emoji_icon))
                st.write("Confidence:{}".format(np.max(probability)))

            with col2:
                st.success("Prediction Probability")
                proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
                proba_df_clean = proba_df.T.reset_index()
                proba_df_clean.columns = ["emotions", "probability"]

                fig = alt.Chart(proba_df_clean).mark_bar().encode(
                    x='emotions', y='probability', color='emotions'
                )
                st.altair_chart(fig, use_container_width=True)

    elif input_type == "Web Link Scraping":
        with st.form(key='web_scraping_form'):
            url = st.text_input("Enter Website URL:")
            submit_url = st.form_submit_button(label='Scrape and Predict')

        if submit_url:
            scraped_text = scrape_text_from_url(url)

            if scraped_text:
                col1, col2 = st.columns(2)

                prediction = predict_emotions(scraped_text)
                probability = get_prediction_proba(scraped_text)

                with col1:
                    st.success("Scraped Text")
                    st.write(scraped_text[:1000])  # Display the first 1000 characters

                    st.success("Prediction")
                    emoji_icon = emotions_emoji_dict[prediction]
                    st.write("{}:{}".format(prediction, emoji_icon))
                    st.write("Confidence:{}".format(np.max(probability)))

                with col2:
                    st.success("Prediction Probability")
                    proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
                    proba_df_clean = proba_df.T.reset_index()
                    proba_df_clean.columns = ["emotions", "probability"]

                    fig = alt.Chart(proba_df_clean).mark_bar().encode(
                        x='emotions', y='probability', color='emotions'
                    )
                    st.altair_chart(fig, use_container_width=True)


if __name__ == '__main__':
    main()
