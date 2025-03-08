import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import re

df = pd.read_csv("twitter_sentiment_data.csv")

sampled_df = df.sample(n=500,random_state=42)

def extract_mentions(text):
    return re.findall(r"@(\w+)", text)

def extract_retweet(text):
    match = re.search(r"RT @(\w+)", text)
    return match.group(1) if match else None

G = nx.DiGraph()

for _, row in sampled_df.iterrows():
    tweet_body = row["message"]
    author = f"user_{row['tweetid']}"

    mentions = extract_mentions(tweet_body)
    retweeted_user = extract_retweet(tweet_body)

    for mentioned_user in mentions:
        G.add_edge(author, mentioned_user)

    if retweeted_user:
        G.add_edge(author, retweeted_user)

pagerank_scores = nx.pagerank(G)
top_pagerank_users = sorted(pagerank_scores, key=pagerank_scores.get, reverse=True)[:3]

degree_centrality = nx.degree_centrality(G)
top_degree_users = sorted(degree_centrality, key=degree_centrality.get, reverse=True)[:3]

print("Top 3 Most Influential Users (PageRank):", top_pagerank_users)
print("Top 3 Highest Degree Users (Degree Centrality):", top_degree_users)

node_sizes = [G.degree(node) * 50 for node in G.nodes()]
plt.figure(figsize=(10, 6))
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=True, node_size=node_sizes, node_color="lightblue", font_size=8, edge_color="gray")
plt.title("Mapping Climate Change Discourse Network on Twitter/X")
plt.show()