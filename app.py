import streamlit as st
import praw
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline
from datetime import datetime
import os
from config import reddit_config
# Initialize Reddit instance
reddit = praw.Reddit(client_id=reddit_config['client_id'],
                     client_secret=reddit_config['client_secret'],
                     user_agent=reddit_config['user_agent'],
                     username=reddit_config['username'],
                     password=reddit_config['password'])

# Function to perform the search
def perform_search(subreddit_name, keyword, sort):
    if subreddit_name == "all":
        submissions = reddit.subreddit("all").search(keyword, sort=sort, limit=100)
    else:
        subreddit = reddit.subreddit(subreddit_name)
        submissions = subreddit.search(keyword, sort=sort, limit=100)
    return submissions

def draw_plot_for_keyword(keyword, subreddit_name="all"):
    sort_types = ['relevance', 'hot', 'top', 'new', 'comments']
    combined_submissions = {}

    for sort_type in sort_types:
        submissions = perform_search(subreddit_name, keyword, sort=sort_type)
        for submission in submissions:
            combined_submissions[submission.id] = submission

    sorted_submissions = sorted(combined_submissions.values(), key=lambda x: x.created_utc, reverse=True)

    df = pd.DataFrame([{
        'ID': submission.id,
        'Subreddit': submission.subreddit.display_name,
        'Title': submission.title,
        'Text': submission.selftext,
        'URL': submission.url,
        'Time': datetime.utcfromtimestamp(submission.created_utc).strftime('%d-%m-%Y %H:%M:%S')
    } for submission in sorted_submissions])

    df['Time'] = pd.to_datetime(df['Time'])
    df['created_year'] = df['Time'].dt.year

    sentiment_classifier = pipeline(model="finiteautomata/bertweet-base-sentiment-analysis")

    def get_sentiment(text):
        try:
            sentiment = sentiment_classifier(text)[0]['label']
        except:
            try:
                text_chunks = [text[i:i+512] for i in range(0, len(text), 512)]
                sentiment_results = [sentiment_classifier(chunk)[0]['label'] for chunk in text_chunks]
                sentiment = max(set(sentiment_results), key=sentiment_results.count)
            except:
                sentiment = "Not Classified"
        return sentiment

    df.loc[df['Text'].isna(), 'Text'] = df.loc[df['Text'].isna(), 'Title']
    df['sentiment'] = df['Text'].apply(lambda x: get_sentiment(x))

    sentiment_counts = df['sentiment'].value_counts()

    plt.figure(figsize=(8, 6))
    plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=140)
    plt.title(f'Sentiment Distribution for "{keyword}"')

    if not os.path.exists('static'):
        os.makedirs('static')
    plt.savefig('static/sentiment_pie_chart.png')
    plt.close()

    return 'static/sentiment_pie_chart.png'

# Streamlit app
st.title('Reddit Sentiment Analysis')
st.write("Analyze the sentiment of Reddit posts based on a given keyword.")

keyword = st.text_input('Enter a keyword:')
subreddit = st.text_input('Enter a subreddit (optional):', 'all')

if st.button('Search'):
    with st.spinner('Fetching and analyzing data...'):
        image_path = draw_plot_for_keyword(keyword, subreddit)
        st.image(image_path)
        st.success('Analysis complete!')

