import streamlit as st
import praw
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline
from datetime import datetime
from wordcloud import WordCloud
import os
# from config import reddit_config
# Initialize Reddit instance
#reddit = praw.Reddit(client_id=reddit_config['client_id'],
#                    client_secret=reddit_config['client_secret'],
#                    user_agent=reddit_config['user_agent'],
#                    username=reddit_config['username'],
#                    password=reddit_config['password'])'''
reddit = praw.Reddit(client_id='yMQwdMYVS1J3wVfeb_3fuw',
                     client_secret='3sFTR2aigu8d0D2rXMFkNxJpyM3KLQ',
                     user_agent='testscript by u/sentiment',
                     username='Gulugulugulu1607',
                     password='gulugulugulu')
#Function to perform the search
def perform_search(subreddit_name, keyword, sort):
    if subreddit_name == "all":
        submissions = reddit.subreddit("all").search(keyword,sort=sort,limit=None)
    else:
        subreddit = reddit.subreddit(subreddit_name)
        submissions = subreddit.search(keyword,sort=sort, limit = None)
    return submissions
def draw_plot_for_keyword(keyword,subreddit_name):
    sort_types = ['relevance','hot','top','new','comments']
    combined_submissions = {}
    for sort_type in sort_types:
        submissions = perform_search(subreddit_name,keyword,sort = sort_type)
        for submission in submissions:
            combined_submissions[submission.id] = submission
    sorted_submissions = sorted(combined_submissions.values(), key=lambda x:x.created_utc,reverse = True )
    df = pd.DataFrame([{
        'ID': submission.id,
        'Subreddit': submission.subreddit.display_name,
        'Title': submission.title,
        'Text': submission.selftext,
        'URL':submission.url,
        'Time': datetime.utcfromtimestamp(submission.created_utc).strftime('%d-%m-%Y %H:%M:%S')
    } for submission in sorted_submissions])
    df['Time'] = pd.to_datetime(df['Time'])
    df['created_year'] = df['Time'].dt.year
    sentiment_classifier = pipeline(model='finiteautomata/bertweet-base-sentiment-analysis')
    def get_sentiment(row):
        # create a list to store chunks of text
        text_chunks = []

        if pd.notnull(row['Text']):  # if 'Text' is not null, tokenize it
            text_chunks = [row['Text'][i:i + 512] for i in range(0, len(row['Text']), 512)]
            try:
                sentiment_results = sentiment_classifier(text_chunks)
            except:
                sentiment_results = []
            sentiments = [result['label'] for result in sentiment_results if 'label' in result]
            # get the most common sentiment
            common_sentiment = max(set(sentiments), key=sentiments.count) if sentiments else "Not classified"
            if common_sentiment == "Not classified":  # Duty cycle for not classified sentiment
                return get_sentiment({"Text": None, "Title": row['Title']})
            else:
                return common_sentiment
        elif pd.notnull(row['Title']):  # if 'Text' is null, tokenize 'Title'
            text_chunks = [row['Title'][i:i + 512] for i in range(0, len(row['Title']), 512)]
            try:
                sentiment_results = sentiment_classifier(text_chunks)
            except:
                return 'Not classified'
            sentiments = [result['label'] for result in sentiment_results if 'label' in result]
            # get the most common sentiment
            common_sentiment = max(set(sentiments), key=sentiments.count) if sentiments else "Not classified"
            return common_sentiment
        else:
            return 'Not classified'
    df['sentiment'] = df.apply(get_sentiment, axis = 1)
    sentiment_counts = df['sentiment'].value_counts()
    plt.figure(figsize=(8,6))
    plt.pie(sentiment_counts, labels = sentiment_counts.index, autopct ='%1.1f%%', startangle = 140)
    plt.title(f'Sentiment Distribution for "{keyword}"')
    if not os.path.exists('static'):
        os.makedirs('static')
    plt.savefig('static/sentiment_pie_chart.png')
    plt.close()

    return df, 'static/sentiment_pie_chart.png'

def draw_wordcloud(df):
    posts_in_year = df
    post_title_text_in_year = ' '.join(item for item in posts_in_year[~posts_in_year['Title'].isna()]['Title'])
    word_cloud = WordCloud(collocation_threshold = 2, width = 1000, height = 500, background_color ='white').generate(post_title_text_in_year)
    plt.figure(figsize=(10,5))
    plt.imshow(word_cloud, interpolation = "bilinear")
    plt.axis("off")
    plt.title(f'Word Cloud for {year}')
    plt.savefig(f'static/wordcloud_{year}.png')
    plt.close()

    return f'static/wordcloud_{year}.png'

#Streamlit app
st.title("Social Media Sentiment Analysis")
st.write("Analyze the sentiment of Reddit posts based on a given keyword.")
keyword = st.text_input('Enter a keyword: ')
subreddit = st.text_input('Enter a subreddit (optional):','all')

if st.button('Search'):
    with st.spinner('Fetching and analyzing data...'):
        df, pie_chart_path = draw_plot_for_keyword(keyword,subreddit)
        min_year = int(df['created_year'].min())
        max_year = int(df['created_year'].max())
        selected_year = st.slider('Select a year to generate word cloud:',min_year,max_year,min_year)
        wordcloud_path = draw_wordcloud_by_year(df,selected_year)
        st.image(pie_chart_path, wordcloud_path)
        st.success('Pie chart complete!')
        st.success(f'Word cloud for {selected_year} generated!')
