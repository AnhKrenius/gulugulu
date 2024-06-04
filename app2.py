import streamlit as st
import praw
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline
from datetime import datetime
from wordcloud import WordCloud
from concurrent.futures import ThreadPoolExecutor
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
        submissions = reddit.subreddit("all").search(keyword,sort=sort,limit=100)
    else:
        subreddit = reddit.subreddit(subreddit_name)
        submissions = subreddit.search(keyword,sort=sort, limit = 100)
    return submissions
def fetch_submissions(subreddit_name, keyword, sort_types):
    combined_submissions = {}
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(perform_search, subreddit_name, keyword, sort): sort for sort in sort_types}
        for future in futures:
            sort_type = futures[future]
            try:
                submissions = future.result()
                for submission in submissions:
                    combined_submissions[submission.id] = submission
            except Exception as e:
                st.warning(f"Error fetching submissions for sort type {sort_type}: {e}")
    return combined_submissions
def draw_plot_for_keyword(keyword,subreddit_name):
    sort_types = ['relevance','hot','top','new','comments']
    combined_submissions = fetch_submissions(subreddit_name, keyword, sort_types)
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
        'Time': submission_time,
        'isself': submission.is_self
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
    plt.figure(figsize=(8,6))
    plt.imshow(word_cloud)
    plt.axis("off")
    plt.title(f'Word Cloud')
    plt.savefig(f'wordcloud.png')
    plt.close()

    return 'wordcloud.png'
# Sample posts
def get_sample_posts(df):
  pos_posts = df[(df['sentiment']=='POS')&(df['isself']==True)].sample(3)
  neg_posts = df[(df['sentiment']=='NEG')&(df['isself']==True)].sample(3)
  neu_posts = df[(df['sentiment']=='NEU')&(df['isself']==True)].sample(3)
  return pos_posts, neg_posts, neu_posts
#Streamlit app
page_bg_img = """
<style>
section.main >div {max-width:70rem}
[data-testid="stAppViewContainer"] {
background-image: url("https://i.pinimg.com/originals/e4/52/99/e45299d660b601f029fe173f084feb42.jpg");
background-size: cover;
}
[data-testid="stHeader"]{
background:rgba(0,0,0,0);
}
[data-testid="column"] {
  /* using the shorthand property to set the border radius on all corners */
  top:50%;
  left:50%;
  border-radius: 50px;
  border-style: dotted;
  border-color: #bfbfbf;
  padding: 2.2rem 2.3rem;
  flex-direction:column;
  align-items:left;
  text-align: left;
  font-size: 32px;
  max-width: 320px;
  max-height: 300px;
}
</style>
"""
st.set_page_config(layout = "wide",
                   page_icon="☕",
                   page_title="GuLu")
st.markdown(page_bg_img, unsafe_allow_html=True)
st.title("Social Media Sentiment Analysis")
st.write("Analyze the sentiment of Reddit posts based on a given keyword.")
keyword = st.text_input('Enter a keyword: ')
subreddit = st.text_input('Enter a subreddit (optional):','all')

if st.button('Search'):
    with st.spinner('Fetching and analyzing data...'):
        df, pie_chart_path = draw_plot_for_keyword(keyword,subreddit)
        st.write(f"Total: {len(df)} posts")
        wordcloud_path = draw_wordcloud(df)
        images = [pie_chart_path, wordcloud_path]
        st.image(images,use_column_width=True)
        pos_posts, neg_posts, neu_posts = get_sample_posts(df)
        col1, col2, col3 = st.columns(3,gap = "medium")
        def short(text1):
          try:
            text = text1[:70]
          except:
            text = text1
          return text
        def short_title(title1):
          try:
            title = title1[:20]
          except:
            title = title1
          return title
        with col1:
            st.subheader("😊"+short_title(pos_posts.iloc[0]["Title"]))
            st.markdown(short(pos_posts.iloc[0]["Text"])+'...')
            url1 = pos_posts.iloc[0]["URL"]
            st.write("Read more [link](%s)" % url1)
        with col2:
            st.subheader("😭"+short_title(neg_posts.iloc[0]["Title"]))
            st.markdown(short(neg_posts.iloc[0]["Text"])+'...')
            url2 = neg_posts.iloc[0]["URL"]
            st.write("Read more [link](%s)" % url2)
        with col3:
            st.subheader("😶"+short_title(neu_posts.iloc[0]["Title"]))
            st.markdown(short(neu_posts.iloc[0]["Text"]) + '...')
            url3 = neu_posts.iloc[0]["URL"]
            st.write("Read more [link](%s)" % url3)


