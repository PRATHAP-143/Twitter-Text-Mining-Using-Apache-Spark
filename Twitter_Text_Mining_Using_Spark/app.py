from flask import Flask, render_template
import pandas as pd
import os

app = Flask(__name__)

OUTPUT_FOLDER = "outputs"

@app.route("/")
def home():

    sentiment = pd.read_csv(
        os.path.join(OUTPUT_FOLDER, "sentiment_counts.csv")
    )

    words = pd.read_csv(
        os.path.join(OUTPUT_FOLDER, "word_frequency.csv")
    )

    tweets = pd.read_csv(
        os.path.join(OUTPUT_FOLDER, "cleaned_tweets.csv")
    )

    hashtags = pd.read_csv(
        os.path.join(OUTPUT_FOLDER, "top_hashtags.csv")
    )

    total_tweets = int(sentiment["tweet_count"].sum())

    sentiment_data = sentiment.to_dict(orient="records")
    word_data = words.head(10).to_dict(orient="records")
    tweet_data = tweets.head(10).to_dict(orient="records")
    hashtag_data = hashtags.to_dict(orient="records")

    return render_template(
        "index.html",
        total_tweets=total_tweets,
        sentiment_data=sentiment_data,
        word_data=word_data,
        tweet_data=tweet_data,
        hashtag_data=hashtag_data
    )

if __name__ == "__main__":
    app.run(debug=True)