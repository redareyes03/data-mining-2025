import pandas as pd
import numpy as np
import string
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS


df = pd.read_csv("Datasets/train.csv", sep=";")


text_columns = [
    col for col in df.columns
    if df[col].dtype == "object" or df[col].dtype == "string"
]

print("Detected text columns:", text_columns)

# -----------------------------
# Clean text function
# -----------------------------
def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove numbers
    text = ''.join([c for c in text if not c.isdigit()])

    return text

combined_text = ""

for col in text_columns:
    cleaned_col = df[col].astype(str).apply(clean_text)
    combined_text += " " + " ".join(cleaned_col)

# -----------------------------
# Generate Word Cloud
# -----------------------------
stopwords = set(STOPWORDS)

wordcloud = WordCloud(
    width=2000,
    height=1000,
    background_color="white",
    stopwords=stopwords,
    colormap="viridis"
).generate(combined_text)

# Plot
plt.figure(figsize=(18, 9))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud — All Text Columns Combined", fontsize=20)
plt.show()