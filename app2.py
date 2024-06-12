import streamlit as st
import praw
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline
from datetime import datetime
from wordcloud import WordCloud
from concurrent.futures import ThreadPoolExecutor
import os
# Initialize Reddit instance
reddit = praw.Reddit(client_id='yMQwdMYVS1J3wVfeb_3fuw',
                     client_secret='3sFTR2aigu8d0D2rXMFkNxJpyM3KLQ',
                     user_agent='testscript by u/sentiment',
                     username='Gulugulugulu1607',
                     password='gulugulugulu')
#Function to perform the search
def perform_search(subreddit_name, keyword, sort):
    if subreddit_name == "all":
        submissions = reddit.subreddit("all").search(keyword,sort=sort,limit=500)
    else:
        subreddit = reddit.subreddit(subreddit_name)
        submissions = subreddit.search(keyword,sort=sort, limit = 500)
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
def create_dataframe(keyword,subreddit_name):
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
        'Permalink': submission.permalink,
        'Time': submission_time,
        'isself': submission.is_self,
        'score':submission.score if submission.score else 0
      })

# Create DataFrame
    df = pd.DataFrame(data)
    df['URL']='https://www.reddit.com/' + df['Permalink']
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
    return df
  
def draw_pie_chart(df):  
    sentiment_counts = df['sentiment'].value_counts()
    plt.figure(figsize=(8,6))
    plt.pie(sentiment_counts, labels = sentiment_counts.index, autopct ='%1.1f%%', startangle = 140)
    plt.title(f'Sentiment Distribution for "{keyword}"')
    if not os.path.exists('static'):
        os.makedirs('static')
    plt.savefig('static/sentiment_pie_chart.png')
    plt.close()
    return 'static/sentiment_pie_chart.png'

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
def get_posts(df):
  pos_posts = df[df['sentiment']=='POS']
  neg_posts = df[df['sentiment']=='NEG']
  neu_posts = df[df['sentiment']=='NEU']
  return pos_posts, neg_posts, neu_posts
#Streamlit app
page_bg_img = """
<style>
section.main >div {max-width:90rem}
[data-testid="stAppViewContainer"] {
background-image: url("https://i.pinimg.com/originals/e4/52/99/e45299d660b601f029fe173f084feb42.jpg");
background-size: cover;
}
[data-testid="stHeader"]{
background:rgba(0,0,0,0);
}
[data-testid="stImage"]{
border: 1px solid #ddd;
border-radius: 4px;
padding: 5px;
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
if 'data_loaded' not in st.session_state:
  st.session_state.data_loaded = False
if not st.session_state.data_loaded:
  if st.button('Search'):
    with st.spinner('Collecting data...'):
        df = create_dataframe(keyword,subreddit)
        st.session_state.data_loaded = True
        st.session_state.df = df
if st.session_state.data_loaded:
    with st.spinner('Analyzing data...'):
        df=st.session_state.df
        if st.button('Search other keyword'):
          st.session_state.data_loaded = False
          st.rerun()
        st.write(f"Total: {len(df)} posts")
        min_year = int(df['created_year'].min())
        max_year = int(df['created_year'].max())
        chosen_min_year, chosen_max_year = st.slider('Choose a Year Range', min_value=min_year, max_value = max_year, value = (min_year, max_year))
        df_sub = df[(df['created_year']<=chosen_max_year)&(df['created_year']>=chosen_min_year)]
        pie_chart_path = draw_pie_chart(df_sub)
        wordcloud_path = draw_wordcloud(df_sub)
        images = [pie_chart_path, wordcloud_path]
        st.image(images,width=468)
        pos_posts, neg_posts, neu_posts = get_posts(df_sub)
        note = '<p style="font-family:Fira Sans; font-size: 16px;"><i>👉 Click on Score to sort as desired</i></p>'
        st.markdown(note, unsafe_allow_html=True)
        st.markdown('\n') # see #*
        col1,col2,col3=st.columns(3,gap='medium')
        with col1:
            st.dataframe(pos_posts[['Title','score','URL']],height =1000,hide_index=True,
                         column_config={
                             "Title": st.column_config.TextColumn(
                                 "😊 Positive Submissions",
                                 width=250
                             ),
                             "score": st.column_config.NumberColumn(
                                 "Score",
                                 help='A submission \'s score is simply the number of upvotes minus the number of downvotes.',
                                 format= "%d ⭐",
                                 width=80
                             ),
                             "URL": st.column_config.LinkColumn(
                                 "url",
                                 width=40
                             )
                             })
        with col2:
            st.dataframe(neg_posts[['Title','score','URL']],height =1000,hide_index=True,
                         column_config={
                             "Title": st.column_config.TextColumn(
                                 "😭 Negative Submissions",
                                 width=250
                             ),
                             "score": st.column_config.NumberColumn(
                                 "Score",
                                 help='A submission \'s score is simply the number of upvotes minus the number of downvotes.',
                                 format= "%d ⭐",
                                 width=80
                             ),
                             "URL": st.column_config.LinkColumn(
                                 "url",
                                 width=40
                             )
                             })
        with col3:
            st.dataframe(neu_posts[['Title','score','URL']],height=1000,hide_index=True,
                         column_config={
                             "Title": st.column_config.TextColumn(
                                 "😶 Neutral Submissions",
                                 width=250
                             ),
                             "score": st.column_config.NumberColumn(
                                 "Score",
                                 help = 'A submission \'s score is simply the number of upvotes minus the number of downvotes.',
                                 format = "%d ⭐",
                                 width=80
                             ),
                             'URL': st.column_config.LinkColumn(
                                 "url", width=40
                             )
                         })
        
        
