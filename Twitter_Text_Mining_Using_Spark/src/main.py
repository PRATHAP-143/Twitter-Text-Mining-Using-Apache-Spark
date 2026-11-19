import matplotlib.pyplot as plt
import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, lower, regexp_replace, split,
    explode, trim, length, count, desc
)

# Project Directory
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_PATH = os.path.join(BASE_DIR, "data", "tweets.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# Stop Words
STOPWORDS = [
    "the", "is", "a", "an", "and", "or", "to",
    "of", "in", "on", "for", "with", "it",
    "this", "that", "was", "were", "are", "am",
    "i", "my", "we", "you", "by", "at", "as"
]


# Create Spark Session
spark = (
    SparkSession.builder
    .appName("Twitter Text Mining")
    .master("local[*]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")


print("\n====================================")
print(" TWITTER TEXT MINING USING APACHE SPARK")
print("====================================")


# MODULE 1: DATA COLLECTION

print("\n1. DATA COLLECTION")

df = spark.read.csv(
    DATA_PATH,
    header=True,
    inferSchema=True
)

df.show(10, truncate=False)

print("Total Tweets:", df.count())


# MODULE 2: DATA PREPROCESSING

print("\n2. DATA PREPROCESSING")

clean_df = df.dropna(subset=["text"])

clean_df = clean_df.withColumn(
    "clean_text",
    lower(col("text"))
)

# Remove URLs
clean_df = clean_df.withColumn(
    "clean_text",
    regexp_replace(
        col("clean_text"),
        r"https?://\S+|www\.\S+",
        " "
    )
)

# Remove special characters and numbers
clean_df = clean_df.withColumn(
    "clean_text",
    regexp_replace(
        col("clean_text"),
        r"[^a-zA-Z\s]",
        " "
    )
)

# Remove extra spaces
clean_df = clean_df.withColumn(
    "clean_text",
    regexp_replace(
        col("clean_text"),
        r"\s+",
        " "
    )
)

clean_df = clean_df.withColumn(
    "clean_text",
    trim(col("clean_text"))
)

clean_df = clean_df.filter(
    length(col("clean_text")) > 0
)

clean_df.select(
    "tweet_id", "clean_text"
).show(10, truncate=False)


# MODULE 3: WORD FREQUENCY ANALYSIS

print("\n3. WORD FREQUENCY ANALYSIS")

words = clean_df.select(
    explode(
        split(col("clean_text"), " ")
    ).alias("word")
)

words = words.filter(col("word") != "")

words = words.filter(
    ~col("word").isin(STOPWORDS)
)

word_frequency = words.groupBy("word").agg(
    count("*").alias("frequency")
)

word_frequency = word_frequency.orderBy(
    desc("frequency")
)

print("\nTOP 15 FREQUENT WORDS")

word_frequency.show(15)


# MODULE 4: SENTIMENT ANALYSIS

print("\n4. SENTIMENT ANALYSIS")

sentiment = clean_df.groupBy(
    "sentiment"
).agg(
    count("*").alias("tweet_count")
)

sentiment.show()


# MODULE 5: HASHTAG ANALYSIS

print("\n5. HASHTAG ANALYSIS")

hashtag_data = df.select(
    explode(
        split(col("text"), " ")
    ).alias("word")
)

hashtags = hashtag_data.filter(
    col("word").startswith("#")
)

hashtags = hashtags.withColumn(
    "word",
    lower(col("word"))
)

hashtag_frequency = hashtags.groupBy(
    "word"
).agg(
    count("*").alias("frequency")
)

hashtag_frequency = hashtag_frequency.orderBy(
    desc("frequency")
)

hashtag_frequency.show(10, truncate=False)


# MODULE 6: SAVE RESULTS
print("\n6. SAVING RESULTS")

import csv

def save_spark_df_as_csv(dataframe, file_path):
    rows = dataframe.collect()
    columns = dataframe.columns

    with open(file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)

        for row in rows:
            writer.writerow([row[column] for column in columns])


save_spark_df_as_csv(
    clean_df.select("tweet_id", "text", "clean_text", "sentiment"),
    os.path.join(OUTPUT_DIR, "cleaned_tweets.csv")
)

save_spark_df_as_csv(
    word_frequency,
    os.path.join(OUTPUT_DIR, "word_frequency.csv")
)

save_spark_df_as_csv(
    sentiment,
    os.path.join(OUTPUT_DIR, "sentiment_counts.csv")
)

save_spark_df_as_csv(
    hashtag_frequency,
    os.path.join(OUTPUT_DIR, "top_hashtags.csv")
)

print("All CSV results saved successfully!")
# MODULE 7: DATA VISUALIZATION
print("\n7. DATA VISUALIZATION")    

# Sentiment Distribution Chart
sentiment_rows = sentiment.collect()

sentiment_labels = [row["sentiment"] for row in sentiment_rows]
sentiment_counts = [row["tweet_count"] for row in sentiment_rows]

plt.figure(figsize=(8, 5))
plt.bar(sentiment_labels, sentiment_counts)
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of Tweets")
plt.tight_layout()

plt.savefig(os.path.join(OUTPUT_DIR, "sentiment_distribution.png"))
plt.close()


# Top 10 Words Chart
words_rows = word_frequency.limit(10).collect()

words_labels = [row["word"] for row in words_rows]
words_counts = [row["frequency"] for row in words_rows]

plt.figure(figsize=(10, 5))
plt.bar(words_labels, words_counts)
plt.title("Top 10 Frequent Words")
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(os.path.join(OUTPUT_DIR, "top_words.png"))
plt.close()

print("Charts saved successfully!")


# FINAL RESULT

print("\n====================================")
print(" PROJECT EXECUTED SUCCESSFULLY!")
print("====================================")

print("\nAll analysis results are saved in:")
print(OUTPUT_DIR)

spark.stop()