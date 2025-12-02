import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Download VADER lexicon
nltk.download('vader_lexicon')

# Load dataset
df = pd.read_csv("social_media_sentiments.csv")

print("\nFIRST 5 ROWS OF DATA:")
print(df.head())

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Function to get sentiment scores
def analyze_sentiment(text):
    if isinstance(text, str):
        return sia.polarity_scores(text)
    else:
        return {"neg": 0, "neu": 0, "pos": 0, "compound": 0}

# Apply sentiment analysis
df["sentiment"] = df["text"].apply(analyze_sentiment)

# Extract sentiment values into separate columns
df["neg"] = df["sentiment"].apply(lambda x: x["neg"])
df["neu"] = df["sentiment"].apply(lambda x: x["neu"])
df["pos"] = df["sentiment"].apply(lambda x: x["pos"])
df["compound"] = df["sentiment"].apply(lambda x: x["compound"])

# Assign final label (positive, negative, neutral)
def final_sentiment(c):
    if c >= 0.05:
        return "Positive"
    elif c <= -0.05:
        return "Negative"
    else:
        return "Neutral"

df["label"] = df["compound"].apply(final_sentiment)

print("\nVALUE COUNTS:")
print(df["label"].value_counts())

# ------ Visualization 1: Sentiment Count ------
plt.figure(figsize=(8, 5))
sns.countplot(df["label"], palette="viridis")
plt.title("Sentiment Category Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()

# ------ Visualization 2: Sentiment Score Distribution ------
plt.figure(figsize=(10, 5))
sns.histplot(df["compound"], bins=30, kde=True)
plt.title("Distribution of Compound Sentiment Scores")
plt.xlabel("Compound Score")
plt.ylabel("Frequency")
plt.show()

# Save processed data
df.to_csv("sentiment_results.csv", index=False)
print("\nProcessed sentiment results saved as sentiment_results.csv")
