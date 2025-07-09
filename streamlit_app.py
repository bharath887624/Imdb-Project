import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import re
from sqlalchemy import create_engine

st.title('IMDB 2024 Data Scraping and Visualizations')

#Function to convert vote to numeric
def convert_votes_to_numeric(votes_str):
  if pd.isna(votes_str) or not isinstance(votes_str, str):
    return np.nan
  cleaned_str = votes_str.strip('() ').upper()
  if not cleaned_str:
        return np.nan
  try:
        if 'K' in cleaned_str:
            numeric_part = cleaned_str.replace('K', '')
            value = float(numeric_part) * 1000
        else:
            value = float(cleaned_str)

        return int(value)
  except ValueError:
      print(f"Warning: Could not convert '{votes_str}' (cleaned to '{cleaned_str}') to numeric. Returning NaN.")
      return np.nan

#Function to convert duration to minute
def convert_duration_to_minutes(duration_str):
    hours = 0
    minutes = 0
    duration_str = duration_str.lower()
    hour_match = re.search(r'(\d+)h', duration_str)
    if hour_match:
        hours = int(hour_match.group(1))
    minute_match = re.search(r'(\d+)m', duration_str)
    if minute_match:
        minutes = int(minute_match.group(1))
    return hours * 60 + minutes


#MySQL database
host = "localhost"
user = "root"
password = "root"
database = "imdb"

#SQLAlchemy connection
engine = create_engine(f"mysql+pymysql://{user}:{password}@{host}/{database}")

#Read data
df = pd.read_sql("SELECT * FROM imdb_table", con=engine)

df.info()

df['Rating'] = df['Rating'].astype(float)

df['Votes_numeric'] = df['Votes'].apply(convert_votes_to_numeric)

df['Duration_minutes'] = df['Duration'].apply(convert_duration_to_minutes)

print(df.info())

print(df.head())

st.markdown("<br>", unsafe_allow_html=True)

st.subheader('Top 10 Movies by Rating and Voting Counts')

top_10_movies = df.sort_values(by=['Votes_numeric', 'Rating'], ascending=[False, False]).head(10)
top_10 = top_10_movies[['Movie_name', 'Rating', 'Votes', 'Duration', 'genre']]

st.write("Movie Top 10")
st.dataframe(top_10)

st.subheader("Genre Distribution")
#st.info("Plot of count of movies for each genre in a bar chart")

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 5))
sns.countplot(
    data=df,
    x='genre',
    order=df['genre'].value_counts().index,
    hue='genre',
    legend=False,
    palette='Set1',
    ax=ax
)

for bar in ax.patches:
    height = bar.get_height()
    x_pos = bar.get_x() + bar.get_width() / 2
    ax.text(x_pos, height + 0.5, f'{int(height)}', ha='center', va='bottom', fontsize=9)

ax.set_title('Movie Count per Genre')
ax.set_xlabel('Genre')
ax.set_ylabel('Number of Movies')
plt.xticks(rotation=45)
plt.tight_layout()

st.pyplot(fig)

st.markdown("<br>", unsafe_allow_html=True)

st.subheader('Average Duration by Genre')

fig, ax = plt.subplots(figsize=(10, 6))

sns.barplot(
    data=df,
    y='Duration_minutes', 
    x='genre',
    errorbar=None,
    hue='genre',
    legend=False,
    palette='viridis',
    estimator='mean',
    ax=ax
)

# Add value annotations
for bar in ax.patches:
    height = bar.get_height()
    x_pos = bar.get_x() + bar.get_width() / 2
    ax.text(x_pos, height + 1, f'{height:.1f}', ha='center', va='bottom', fontsize=9)

# Set labels
ax.set_title('Average Duration per Genre')
ax.set_ylabel('Duration (minutes)')
ax.set_xlabel('Genre')
plt.xticks(rotation=45)
plt.tight_layout()

# Show plot
st.pyplot(fig)

st.markdown("<br>", unsafe_allow_html=True)

st.subheader('Rating Distribution')

fig, ax = plt.subplots(figsize=(10, 6))

sns.histplot(data=df, x='Rating', bins=10, kde=True, ax=ax)

ax.set_title('Distribution of Movie Ratings', fontsize=16)
ax.set_xlabel('Rating', fontsize=12)
ax.set_ylabel('Number of Movies', fontsize=12)
ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
st.pyplot(fig)

fig, ax = plt.subplots(figsize=(8, 5))

sns.boxplot(data=df, x='Rating', ax=ax)

ax.set_title('Boxplot of Movie Ratings', fontsize=16)
ax.set_xlabel('Rating', fontsize=12)
ax.grid(axis='x', linestyle='--', alpha=0.7)

plt.tight_layout()
st.pyplot(fig)

st.subheader('Genre-Based Rating Leaders')

#df_filtered = df

df_filtered = df[df['Votes_numeric'] > 1000]

top_rated_idx = df_filtered.groupby('genre')['Rating'].idxmax()

top_rated_movies = df_filtered.loc[top_rated_idx, ['Movie_name', 'Rating', 'Votes', 'Duration', 'genre']]

top_rated_movies = top_rated_movies.reset_index(drop=True)

st.dataframe(top_rated_movies)

st.markdown(
    "<div style='font-size: 12px; color: gray;'> <i>Note: The movies must have at least >1k in number of vote</i></div>",
    unsafe_allow_html=True
)

st.markdown("<br>", unsafe_allow_html=True)

st.subheader('Most Popular Genres by Voting')

# Compute total votes per genre
genre_votes = df.groupby('genre')['Votes_numeric'].sum().sort_values(ascending=False)

# Plot pie chart
fig, ax = plt.subplots(figsize=(5, 5))

ax.pie(
    genre_votes,
    labels=genre_votes.index,
    autopct='%1.1f%%',
    startangle=140
)

#ax.set_title('Most Popular Genres by Voting')
ax.axis('equal')  # Optional: keeps pie circle-shaped
plt.tight_layout()

st.markdown("<br>", unsafe_allow_html=True)

# Show in Streamlit
st.pyplot(fig)

st.subheader('Duration Extremes')

# Get shortest and longest movie
shortest = df.loc[df[df['Duration_minutes'].notna() & (df['Duration_minutes'] != 0)]['Duration_minutes'].idxmin()]
longest = df.loc[df['Duration_minutes'].idxmax()]

# Create figure with two text blocks
fig, axs = plt.subplots(1, 2, figsize=(14, 4))
#fig.suptitle("Duration Extremes: Shortest and Longest Movies", fontsize=16)

# Shortest Movie Block
axs[0].axis('off')
axs[0].set_title('Shortest Movie', fontsize=18, fontweight='bold')
axs[0].text(0, 0.8, f"{shortest['Movie_name']}", fontsize=14)
axs[0].text(0, 0.6, f"Duration: {shortest['Duration']}", fontsize=14)
axs[0].text(0, 0.4, f"Rating: {shortest['Rating']}", fontsize=14)
axs[0].text(0, 0.2, f"Votes: {shortest['Votes']}", fontsize=14)
axs[0].text(0, 0.0, f"Genre: {shortest['genre']}", fontsize=14)

# Longest Movie Block
axs[1].axis('off')
axs[1].set_title('Longest Movie', fontsize=18, fontweight='bold')
axs[1].text(0, 0.8, f"{longest['Movie_name']}", fontsize=14)
axs[1].text(0, 0.6, f"Duration: {longest['Duration']}", fontsize=14)
axs[1].text(0, 0.4, f"Rating: {longest['Rating']}", fontsize=14)
axs[1].text(0, 0.2, f"Votes: {longest['Votes']}", fontsize=14)
axs[1].text(0, 0.0, f"Genre: {longest['genre']}", fontsize=14)

plt.tight_layout()
st.pyplot(fig)

fig, ax = plt.subplots(figsize=(12, 6))
sns.violinplot(
    data=df,
    x='genre',
    y='Duration_minutes',
    palette='Set3',
    hue='genre',
    legend=False,
    ax=ax
)

st.markdown("<br>", unsafe_allow_html=True)

ax.set_title('Duration Distribution per Genre')
ax.set_xlabel('Genre')
ax.set_ylabel('Duration (minutes)')
plt.xticks(rotation=45)
plt.tight_layout()

st.pyplot(fig)

st.markdown("<br>", unsafe_allow_html=True)

st.subheader('Ratings by Genre')

numeric_df = df.select_dtypes(include=['number'])

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap='coolwarm',
    fmt=".2f",
    ax=ax
)

ax.set_title("Correlation Heatmap of DataFrame")
plt.tight_layout()

st.pyplot(fig)

# Calculate average rating per genre
genre_rating = df.groupby('genre')['Rating'].mean().sort_values(ascending=False)
genre_rating_df = genre_rating.to_frame().T  # Convert to 1-row dataframe

# Plot the heatmap
fig, ax = plt.subplots(figsize=(12, 2))
sns.heatmap(
    genre_rating_df,
    annot=True,
    fmt=".1f",
    cmap='coolwarm',
    cbar=True,
    linewidths=0.5,
    ax=ax
)

ax.set_title('Heatmap of Average Rating by Genre', fontsize=14)
ax.set_yticks([])  # Hide y-axis row label
plt.tight_layout()

st.pyplot(fig)

st.markdown("<br>", unsafe_allow_html=True)

st.subheader('Correlation Analysis')

# Create the relplot (note: relplot returns a FacetGrid)
import seaborn as sns

relplot = sns.relplot(
    data=df,
    y='Votes_numeric',
    x='Rating',
    alpha=0.7,
    height=6,
    aspect=1.5
)

relplot.fig.suptitle('Correlation Between Votes and Ratings', fontsize=14)
relplot.set_axis_labels("Rating", "Number of Votes")
relplot.ax.grid(True)
relplot.tight_layout()

# Display with Streamlit
st.pyplot(relplot)

st.markdown("<br>", unsafe_allow_html=True)

fig, ax = plt.subplots(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x='Rating',
    y='Votes_numeric',
    hue='genre',
    alpha=0.7,
    ax=ax
)

ax.set_title('Correlation Between Votes and Ratings')
ax.set_xlabel('Rating')
ax.set_ylabel('Number of Votes')
ax.grid(True)
plt.tight_layout()

st.pyplot(fig)

#st.dataframe(df.head())

st.markdown("<br>", unsafe_allow_html=True)

st.subheader("Interactive Filter")

# Duration Slider (0h to 7h)
dur_min, dur_max = st.slider(
    "Select Duration Range (in Hours)",
    0.0, 7.0, (0.0, 7.0),
    step=0.5
)
st.write(f"Selected duration: {dur_min}h to {dur_max}h")

# Rating Input Box
rt = st.number_input("Enter minimum rating upto 10", min_value=1, max_value=10, step=1)

if rt > 10:
    st.error("Invalid input, Please enter a value between 1 and 10.")
else:
    rate = rt

# vote count minimum
vote_k = st.number_input(
    "Enter minimum vote count (in K)",
    min_value=0,
    max_value=645,
    value=0,
    step=1
)

# Convert K to actual number
vote = vote_k * 1000

# Genre Select Box (Multi-select)
genre = st.multiselect(
    "Select Genres",
    options=['Action', 'Animation', 'Romance', 'Crime', 'Horror'],
    default=[]
)

if genre:
    genre_filter = df['genre'].str.lower().apply(
        lambda g: any(gen.lower() in g for gen in genre)
    )
else:
    genre_filter = pd.Series([True] * len(df))

df_filtered = df[
    (df['Duration_minutes'] >= dur_min * 60) &
    (df['Duration_minutes'] <= dur_max * 60) &
    (df['Rating'] >= rate) &
    (df['Votes_numeric'] >= vote) &
    genre_filter
]

st.dataframe(df_filtered)