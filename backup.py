
from textblob import TextBlob
import pandas as pd
import streamlit as st
import cleantext
import nltk
from nltk.corpus import stopwords, wordnet
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Set up stopwords for cleaning text
stop_words = set(stopwords.words('english'))

# Function to get synonyms for a word
def get_synonyms(word):
    synonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.append(lemma.name())
    return synonyms

st.header('Sentiment Analysis')

# Define sentiment categories and corresponding thresholds
sentiment_categories = {
    'Very Negative': (-1, -0.5),
    'Negative': (-0.5, 0),
    'Neutral': (0, 0.5),
    'Positive': (0.5, 1)
}

# Text input for single text analysis
with st.expander('Analyze Text'):
    text = st.text_input('Text here: ')
    if text:
        # Display synonyms for each word in the input text
        st.write("Synonyms:")
        words = text.split()
        for word in words:
            synonyms = get_synonyms(word)
            st.write(f"{word}: {synonyms}")

        # Perform sentiment analysis
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        for category, (lower, upper) in sentiment_categories.items():
            if lower <= polarity < upper:
                st.write('Sentiment: ', category)
                break
        st.write('Polarity: ', round(polarity, 2))
        st.write('Subjectivity: ', round(blob.sentiment.subjectivity, 2))

    pre = st.text_input('Clean Text: ')
    if pre:
        cleaned_text = cleantext.clean(pre, clean_all=False, extra_spaces=True,
                                       stopwords=True, lowercase=True, numbers=True, punct=True)
        st.write(cleaned_text)

# File upload for CSV analysis
with st.expander('Analyze CSV'):
    upl = st.file_uploader('Upload file')

    def score(x):
        blob1 = TextBlob(x)
        return blob1.sentiment.polarity

    def analyze(x):
        for category, (lower, upper) in sentiment_categories.items():
            if lower <= x < upper:
                return category

    if upl:
        df = pd.read_csv(upl)  # Read CSV file directly

        # Check if the column 'Unnamed: 0' exists before attempting to delete it
        if 'Unnamed: 0' in df.columns:
            del df['Unnamed: 0']

        df['score'] = df['tweets'].apply(score)
        df['analysis'] = df['score'].apply(analyze)
        st.write(df.head(10))

        # Visualize sentiment distribution
        st.subheader('Sentiment Distribution')
        sns.countplot(x='analysis', data=df)
        st.pyplot()

        # Create and display word cloud
        st.subheader('Word Cloud')
        wordcloud = WordCloud(stopwords=stop_words, background_color='white').generate(' '.join(df['tweets']))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot()

        @st.cache
        def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(df)

        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='sentiment.csv',
            mime='text/csv',
        )
