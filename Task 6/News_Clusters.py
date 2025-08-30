import requests
from bs4 import BeautifulSoup
import feedparser
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


# Step 1: Scrape Articles


def scrape_dawn():
    url = "https://www.dawn.com/latest-news"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    articles = []
    for h2 in soup.find_all("h2", class_="story__title"):
        link = h2.find("a")
        if link:
            articles.append(link.get_text(strip=True))
    return articles

def scrape_bbc():
    rss_url = "http://feeds.bbci.co.uk/news/world/rss.xml"
    feed = feedparser.parse(rss_url)
    return [entry.title for entry in feed.entries]

def scrape_reuters():
    rss_url = "http://feeds.reuters.com/reuters/worldNews"
    feed = feedparser.parse(rss_url)
    return [entry.title for entry in feed.entries]

print("Scraping news sources...")
articles = scrape_dawn() + scrape_bbc() + scrape_reuters()
print(f"Collected {len(articles)} articles")


# Step 2: Preprocess


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text

cleaned_articles = [clean_text(a) for a in articles]


# Step 3: TF-IDF + Clustering


vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
X = vectorizer.fit_transform(cleaned_articles)

num_clusters = 6  # adjust based on dataset
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
kmeans.fit(X)

labels = kmeans.labels_


# Step 4: Extract Keywords per Cluster


terms = vectorizer.get_feature_names_out()
print("\n=== Cluster Keywords ===")
for i in range(num_clusters):
    cluster_terms = kmeans.cluster_centers_[i].argsort()[-10:][::-1]
    keywords = [terms[t] for t in cluster_terms]
    print(f"Cluster {i}: {', '.join(keywords)}")


# Step 5: Human-readable Cluster Names


cluster_names = {
    0: "Business / Sports",
    1: "International Politics & Trade",
    2: "Pakistan Politics / Cricket / Economy",
    3: "Human Rights & Media",
    4: "Crime / Terrorism / Violence",
    5: "Natural Disasters & Local Issues"
}


# Step 6: User Input → Assign Cluster


def predict_topic(user_text):
    cleaned = clean_text(user_text)
    vec = vectorizer.transform([cleaned])
    cluster = kmeans.predict(vec)[0]
    return cluster, cluster_names.get(cluster, "Unknown Topic")

print("\n=== Test with User Input ===")
while True:
    user_input = input("Enter a news article (or 'exit'): ")
    if user_input.lower() == "exit":
        break
    cluster_id, topic_name = predict_topic(user_input)
    print(f"Assigned to Cluster {cluster_id} → {topic_name}")
