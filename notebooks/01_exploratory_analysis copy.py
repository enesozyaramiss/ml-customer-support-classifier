# %%
# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import warnings
warnings.filterwarnings('ignore')

# NLP libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import re
from wordcloud import WordCloud

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Style settings
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

print("All libraries imported successfully!")



# %%
# Load datasets
ROOT = os.path.abspath(os.path.join(os.getcwd(), '..'))
train_path = os.path.join(ROOT, 'data', 'raw', 'customer_queries_data.csv')
test_path  = os.path.join(ROOT, 'data', 'raw', 'customer_queries_test.csv')
train_data = pd.read_csv(train_path)
test_data  = pd.read_csv(test_path)
print("Data Loading Summary")
print("=" * 50)
print(f"Training set shape: {train_data.shape}")
print(f"Test set shape: {test_data.shape}")
print(f"Training columns: {list(train_data.columns)}")
print(f"Test columns: {list(test_data.columns)}")

# %%
# Display first few rows
print("Training Data Sample:")
print(train_data.head())
print("\nTest Data Sample:")
print(test_data.head())

# %%
# Basic information about datasets
print("Training Data Info:")
print(train_data.info())
print("\nTest Data Info:")
print(test_data.info())

# %%
# Check for missing values
print("Missing Values Analysis:")
print("=" * 40)
print("Training Data:")
print(train_data.isnull().sum())
print("\nTest Data:")
print(test_data.isnull().sum())

# %%
# Check for duplicates
print("Duplicate Analysis:")
print("=" * 30)
train_duplicates = train_data.duplicated().sum()
test_duplicates = test_data.duplicated().sum()
print(f"Training set duplicates: {train_duplicates}")
print(f"Test set duplicates: {test_duplicates}")

if train_duplicates > 0:
    print("\n Sample duplicate queries:")
    duplicate_queries = train_data[train_data.duplicated(subset=['query'], keep=False)]
    print(duplicate_queries.head())

dup = test_data[test_data.duplicated(subset=['query'], keep=False)]
print(dup)

test_data = test_data.drop_duplicates(subset=['query'], keep='first').reset_index(drop=True)



# %%
# Analyze label distribution
label_counts = train_data['label'].value_counts()
print(" Label Distribution:")
print("=" * 30)
print(f"Total unique labels: {len(label_counts)}")
print(f"Most common label: {label_counts.index[0]} ({label_counts.iloc[0]} samples)")
print(f"Least common label: {label_counts.index[-1]} ({label_counts.iloc[-1]} samples)")
print(f"\nClass balance ratio (max/min): {label_counts.iloc[0] / label_counts.iloc[-1]:.2f}")

# %%
# Visualize label distribution
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

# Bar plot
label_counts.plot(kind='bar', ax=ax1, color='skyblue', edgecolor='black')
ax1.set_title('Distribution of Customer Query Categories', fontsize=16, fontweight='bold')
ax1.set_xlabel('Category', fontsize=12)
ax1.set_ylabel('Number of Queries', fontsize=12)
ax1.tick_params(axis='x', rotation=45)
ax1.grid(axis='y', alpha=0.3)

# Pie chart for top 10 categories
top_10_labels = label_counts.head(10)
ax2.pie(top_10_labels.values, labels=top_10_labels.index, autopct='%1.1f%%', startangle=90)
ax2.set_title('Top 10 Categories Distribution', fontsize=16, fontweight='bold')

plt.tight_layout()
plt.savefig('../reports/figures/label_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
# Statistical summary of label distribution
print("Label Distribution Statistics:")
print("=" * 40)
print(f"Mean samples per class: {label_counts.mean():.2f}")
print(f"Median samples per class: {label_counts.median():.2f}")
print(f"Standard deviation: {label_counts.std():.2f}")
print(f"Min samples: {label_counts.min()}")
print(f"Max samples: {label_counts.max()}")

# Calculate class imbalance
percentages = (label_counts / len(train_data) * 100).round(2)
print(f"\nClass percentage range: {percentages.min():.2f}% - {percentages.max():.2f}%")

# %%
# Text length analysis
train_data['query_length'] = train_data['query'].str.len()
train_data['word_count'] = train_data['query'].str.split().str.len()

print(" Text Length Statistics:")
print("=" * 35)
print("Character length:")
print(train_data['query_length'].describe())
print("\nWord count:")
print(train_data['word_count'].describe())

# %%
# Visualize text length distribution
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Character length distribution
ax1.hist(train_data['query_length'], bins=50, color='lightcoral', alpha=0.7, edgecolor='black')
ax1.set_title('Distribution of Query Character Length', fontsize=14, fontweight='bold')
ax1.set_xlabel('Character Length')
ax1.set_ylabel('Frequency')
ax1.axvline(train_data['query_length'].mean(), color='red', linestyle='--', label=f'Mean: {train_data["query_length"].mean():.1f}')
ax1.legend()
ax1.grid(alpha=0.3)

# Word count distribution
ax2.hist(train_data['word_count'], bins=50, color='lightgreen', alpha=0.7, edgecolor='black')
ax2.set_title('Distribution of Query Word Count', fontsize=14, fontweight='bold')
ax2.set_xlabel('Word Count')
ax2.set_ylabel('Frequency')
ax2.axvline(train_data['word_count'].mean(), color='green', linestyle='--', label=f'Mean: {train_data["word_count"].mean():.1f}')
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('../reports/figures/text_length_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
# Text length by category analysis
category_text_stats = train_data.groupby('label').agg({
    'query_length': ['mean', 'median', 'std'],
    'word_count': ['mean', 'median', 'std']
}).round(2)

print(" Text Statistics by Category (Top 10):")
print("=" * 50)
top_categories = label_counts.head(10).index
print(category_text_stats.loc[top_categories])

# %%
# Analyze common words across all queries
stop_words = set(stopwords.words('english'))

def get_common_words(text_series, n=20):
    """Extract most common words from text series."""
    all_text = ' '.join(text_series.astype(str))
    words = word_tokenize(all_text.lower())
    words = [word for word in words if word.isalpha() and word not in stop_words]
    return Counter(words).most_common(n)

common_words = get_common_words(train_data['query'])
print("🔤 Most Common Words (excluding stop words):")
print("=" * 45)
for word, count in common_words:
    print(f"{word}: {count}")

# %%
# Visualize common words
words, counts = zip(*common_words)
plt.figure(figsize=(12, 6))
plt.bar(words, counts, color='gold', edgecolor='black')
plt.title('Top 20 Most Common Words in Queries', fontsize=16, fontweight='bold')
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('../reports/figures/common_words.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
# Create word cloud
wordcloud_text = ' '.join(train_data['query'].astype(str))
wordcloud = WordCloud(width=800, height=400, background_color='white', 
                     stopwords=stop_words, max_words=100, colormap='viridis').generate(wordcloud_text)

plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Customer Queries', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('../reports/figures/wordcloud.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
# Common words by category (for top 5 categories)
print(" Common Words by Top Categories:")
print("=" * 40)
top_5_categories = label_counts.head(5).index

for category in top_5_categories:
    category_queries = train_data[train_data['label'] == category]['query']
    category_words = get_common_words(category_queries, n=10)
    print(f"\n{category}:")
    words_str = ', '.join([f"{word} ({count})" for word, count in category_words[:5]])
    print(f"  {words_str}")


# %%
# Check for very short or very long queries
print("Data Quality Issues:")
print("=" * 30)

very_short = train_data[train_data['word_count'] <= 2]
very_long = train_data[train_data['word_count'] >= 50]

print(f"Very short queries (≤2 words): {len(very_short)}")
print(f"Very long queries (≥50 words): {len(very_long)}")

if len(very_short) > 0:
    print("\nSample short queries:")
    print(very_short[['query', 'label']].head())

if len(very_long) > 0:
    print("\nSample long queries:")
    print(very_long[['query', 'label']].head())


# Check for potential data quality issues
print("🔍 Additional Quality Checks:")
print("=" * 35)

# Queries with only punctuation or numbers
pattern_issues = train_data[train_data['query'].str.match(r'^[^a-zA-Z]*$')]
print(f"Queries with no letters: {len(pattern_issues)}")

# Queries with unusual characters
special_chars = train_data[train_data['query'].str.contains(r'[^a-zA-Z0-9\s\.\,\?\!\'\"\-\(\)]')]
print(f"Queries with special characters: {len(special_chars)}")

# Empty or whitespace-only queries
empty_queries = train_data[train_data['query'].str.strip() == '']
print(f"Empty queries: {len(empty_queries)}")

# %%
print("KEY INSIGHTS FROM EDA:")
print("=" * 50)
print(f" Dataset Size: {len(train_data):,} training samples, {len(test_data):,} test samples")
print(f" Number of Categories: {len(label_counts)} distinct categories")
print(f" Class Balance: Relatively balanced (ratio: {label_counts.iloc[0] / label_counts.iloc[-1]:.2f}:1)")
print(f" Text Characteristics:")
print(f"   • Average query length: {train_data['query_length'].mean():.1f} characters")
print(f"   • Average word count: {train_data['word_count'].mean():.1f} words")
print(f"   • Text range: {train_data['word_count'].min()}-{train_data['word_count'].max()} words")
print(f" Data Quality: {len(train_data) - very_short.shape[0] - very_long.shape[0]:,} good quality samples")

# %%
# Save processed insights for next notebooks
insights_summary = {
    'total_samples': len(train_data),
    'num_categories': len(label_counts),
    'class_balance_ratio': label_counts.iloc[0] / label_counts.iloc[-1],
    'avg_query_length': train_data['query_length'].mean(),
    'avg_word_count': train_data['word_count'].mean(),
    'quality_issues': {
        'very_short': len(very_short),
        'very_long': len(very_long),
        'special_chars': len(special_chars)
    },
    'top_categories': label_counts.head(10).to_dict()
}

# Save insights to file
import json
with open('../reports/eda_insights.json', 'w') as f:
    json.dump(insights_summary, f, indent=2)

print("EDA insights saved to '../reports/eda_insights.json'")
print("\n Exploratory Data Analysis Complete!")
print("Next: Run 02_data_preprocessing.ipynb")