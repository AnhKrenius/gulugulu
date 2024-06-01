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
    data = []
    for submission in sorted_submissions:
      try:
        # Ensure consistent datetime conversion
        submission_time = datetime.utcfromtimestamp(submission.created_utc).strftime('%d-%m-%Y %H:%M:%S')
      except Exception as e:
        submission_time = ''
    
      data.append({
        'ID': submission.id,
        'Subreddit': submission.subreddit.display_name,
        'Title': submission.title,
        'Text': submission.selftext,
        'URL': submission.url,
        'Time': submission_time
      })

# Create DataFrame
    df = pd.DataFrame(data)
    df['Time'] = pd.to_datetime(df['Time'], errors ='coerce')
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
    #wordcloud
    post_title_text = ' '.join([title for title in df['Title'].str.lower()])
    word_cloud = WordCloud(collocation_threshold = 2, width = 1000, height=500,
                       background_color='white').generate(post_title_text)
    #Display
    plt.figure(figsize=(10,5))
    plt.imshow(word_cloud)
    plt.axis("off")
    plt.title(f'Word Cloud')
    plt.savefig(f'wordcloud.png')
    plt.close()

    return 'wordcloud.png'
# Sample posts
def get_sample_posts(df):
  pos_posts = df[df['sentiment']=='POS'].sample(3)
  neg_posts = df[df['sentiment']=='NEG'].sample(3)
  neu_posts = df[df['sentiment']=='NEU'].sample(3)
  return pos_posts, neg_posts, neu_posts
#Streamlit app
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
background-image: url("https://i.pinimg.com/originals/e4/52/99/e45299d660b601f029fe173f084feb42.jpg");
background-size: cover;
}
[data-testid="stHeader"]{
background:rgba(0,0,0,0);
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)
st.title("Social Media Sentiment Analysis")
st.write("Analyze the sentiment of Reddit posts based on a given keyword.")
keyword = st.text_input('Enter a keyword: ')
subreddit = st.text_input('Enter a subreddit (optional):','all')

if st.button('Search'):
    with st.spinner('Fetching and analyzing data...'):
        df, pie_chart_path = draw_plot_for_keyword(keyword,subreddit)
        wordcloud_path = draw_wordcloud(df)
        st.image(pie_chart_path)
        st.image(wordcloud_path)
        pos_posts, neg_posts, neu_posts = get_sample_posts(df)
        st.subheader("Sample Positive Posts:")
        st.table(pos_posts[['Title', 'Subreddit', 'Time']])

        st.subheader("Sample Negative Posts:")
        st.table(neg_posts[['Title', 'Subreddit', 'Time']])

        st.subheader("Sample Neutral Posts:")
        st.table(neu_posts[['Title', 'Subreddit', 'Time']])


