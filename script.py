import asyncio
from telethon import TelegramClient
import pymongo
from datetime import datetime, timedelta
import re
from elasticsearch import Elasticsearch
import numpy as np
import json


class TelegramRecentAnalyzer:
    def __init__(self, api_id, api_hash, phone_number,
                 mongodb_uri='mongodb://localhost:27017/',
                 elasticsearch_host='localhost:9200'):
        """
        Initialize Telegram Client, MongoDB, and Elasticsearch connections

        :param api_id: Telegram API ID
        :param api_hash: Telegram API Hash
        :param phone_number: Your phone number
        :param mongodb_uri: MongoDB connection URI
        :param elasticsearch_host: Elasticsearch host
        """
        # Telegram Client Setup
        self.client = TelegramClient('session', api_id, api_hash)
        self.phone_number = phone_number

        # MongoDB Connection
        try:
            self.mongo_client = pymongo.MongoClient(mongodb_uri)
            self.db = self.mongo_client['telegram_analysis']
            self.raw_messages_collection = self.db['raw_messages']
            self.clustered_messages_collection = self.db['clustered_messages']
            print("MongoDB connection established")
        except Exception as e:
            print(f"MongoDB connection error: {e}")
            self.mongo_client = None

        # Elasticsearch Connection
        try:
            self.es = Elasticsearch(
        ['http://localhost:9200'],  # ציין את ה-URL עם סכימה (http:// או https://)
        timeout=30  # ניתן להוסיף עוד פרמטרים כמו timeout
        )

            # Create index if not exists
            if not self.es.indices.exists(index='telegram_messages'):
                self.es.indices.create(index='telegram_messages', body={
                    "settings": {
                        "analysis": {
                            "analyzer": {
                                "hebrew_analyzer": {
                                    "type": "custom",
                                    "tokenizer": "standard",
                                    "filter": ["lowercase", "hebrew_stopwords"]
                                }
                            },
                            "filter": {
                                "hebrew_stopwords": {
                                    "type": "stop",
                                    "stopwords": "_hebrew_"  # Use built-in Hebrew stop words
                                }
                            }
                        }
                    },
                    "mappings": {
                        "properties": {
                            "text": {"type": "text", "analyzer": "hebrew_analyzer"},
                            "channel": {"type": "keyword"},
                            "date": {"type": "date"}
                        }
                    }
                })
            print("Elasticsearch connection established")
        except Exception as e:
            print(f"Elasticsearch connection error: {e}")
            self.es = None

    async def connect(self):
        """
        Connect to Telegram
        """
        await self.client.start(phone=self.phone_number)
        print("Connected to Telegram successfully!")

    async def fetch_and_store_recent_messages(self, time_window=5):
        """
        Fetch messages from all dialogs in the last 5 minutes for Hebrew content
        and store them in MongoDB and Elasticsearch

        :param time_window: Time window in minutes
        :return: List of message dictionaries
        """
        from datetime import timezone

        # Calculate the time threshold (timezone-aware)
        time_threshold = datetime.now(timezone.utc) - timedelta(minutes=time_window)

        # Get all dialogs (groups and channels)
        dialogs = await self.client.get_dialogs()

        processed_messages = []
        for dialog in dialogs:
            # Skip if dialog is not a group or channel
            if not dialog.is_group and not dialog.is_channel:
                continue

            try:
                # Fetch messages from this dialog
                messages = await self.client.get_messages(dialog.entity, limit=100)

                for msg in messages:
                    # Ensure msg.date is timezone-aware
                    if msg.date and msg.date.tzinfo is None:
                        msg.date = msg.date.replace(tzinfo=timezone.utc)

                    # Skip messages older than the time threshold or without text
                    if not msg.text or msg.date < time_threshold:
                        continue

                    # Basic text cleaning and Hebrew content filtering
                    clean_text = re.sub(r'\s+', ' ', msg.text).strip()

                    # Basic Hebrew language detection (simple heuristic)
                    if not any('\u0590' <= char <= '\u05FF' for char in clean_text):
                        continue

                    message_data = {
                        'text': clean_text,
                        'channel': dialog.name or 'Unknown',
                        'date': msg.date
                    }
                    processed_messages.append(message_data)

                    # Store in MongoDB
                    if self.mongo_client:
                        self.raw_messages_collection.insert_one(message_data)

                    # Store in Elasticsearch
                    if self.es:
                        es_doc = {
                            'text': clean_text,
                            'channel': dialog.name or 'Unknown',
                            'date': msg.date
                        }
                        self.es.index(index='telegram_messages', body=es_doc)

            except Exception as e:
                print(f"Error fetching messages from {dialog.name or 'Unknown Dialog'}: {e}")

        return processed_messages

    def cluster_messages_by_similarity(self, similarity_threshold=0.6):
        """
        Cluster messages using Elasticsearch More Like This query

        :param similarity_threshold: Similarity threshold for clustering (0-1)
        :return: Clustered messages
        """
        # Retrieve all messages from Elasticsearch
        search_body = {
            "query": {
                "match_all": {}
            },
            "size": 1000  # Adjust as needed
        }
        search_results = self.es.search(index='telegram_messages', body=search_body)

        # Group messages into clusters
        clustered_messages = {}
        processed_indices = set()

        for idx, hit in enumerate(search_results['hits']['hits']):
            if idx in processed_indices:
                continue

            # Create a new cluster
            current_cluster = [hit]
            processed_indices.add(idx)

            # Find similar messages for this cluster
            for other_idx, other_hit in enumerate(search_results['hits']['hits']):
                if other_idx in processed_indices:
                    continue

                # Use More Like This query to check similarity
                mlt_query = {
                    "query": {
                        "more_like_this": {
                            "fields": ["text"],
                            "like": [
                                {
                                    "_index": "telegram_messages",
                                    "_id": hit['_id']
                                }
                            ],
                            "min_term_freq": 1,
                            "max_query_terms": 12,
                            "minimum_should_match": f"{int(similarity_threshold * 100)}%"
                        }
                    }
                }

                # Check if other document matches the cluster
                similar_search = self.es.search(index='telegram_messages', body=mlt_query)

                # If similar, add to cluster
                if any(other_hit['_id'] == similar_hit['_id'] for similar_hit in similar_search['hits']['hits']):
                    current_cluster.append(other_hit)
                    processed_indices.add(other_idx)

            # Store cluster
            clustered_messages[len(clustered_messages)] = current_cluster

        return clustered_messages

    def save_clustered_messages_to_mongodb(self, clustered_messages):
        """
        Save clustered messages to a new MongoDB collection

        :param clustered_messages: Dictionary of clustered messages
        """
        if not self.mongo_client:
            print("MongoDB not connected")
            return

        # Clear previous clusters
        self.clustered_messages_collection.delete_many({})

        # Save new clusters
        for cluster_id, cluster_messages in clustered_messages.items():
            cluster_data = {
                'cluster_id': cluster_id,
                'messages': [
                    {
                        'text': msg['_source']['text'],
                        'channel': msg['_source']['channel'],
                        'date': msg['_source']['date']
                    } for msg in cluster_messages
                ],
                'created_at': datetime.utcnow(),
                'cluster_size': len(cluster_messages)
            }
            self.clustered_messages_collection.insert_one(cluster_data)

        print(f"Saved {len(clustered_messages)} message clusters to MongoDB")

    async def analyze_recent_messages(self):
        """
        Main method to analyze recent messages across all dialogs
        """
        # Connect to Telegram
        await self.connect()

        # Fetch and store recent messages
        messages = await self.fetch_and_store_recent_messages()

        # Check if any messages found
        if not messages:
            print("No recent Hebrew messages found.")
            return

        # Cluster messages by similarity
        clustered_messages = self.cluster_messages_by_similarity()

        # Print cluster summaries
        print("\n--- Message Clusters ---")
        for cluster_id, cluster_messages in clustered_messages.items():
            print(f"\nCluster {cluster_id}:")
            print(f"Number of messages: {len(cluster_messages)}")
            print("Sample messages:")
            for msg in cluster_messages[:3]:  # Print first 3 messages as sample
                print(f"- {msg['_source']['text'][:100]}... (from {msg['_source']['channel']})")
            print("Channels:", set(msg['_source']['channel'] for msg in cluster_messages))

        # Save to MongoDB
        self.save_clustered_messages_to_mongodb(clustered_messages)

        # Close the client
        await self.client.disconnect()


async def main():
    # Replace with your actual credentials
    API_ID = 29348829  # Your Telegram API ID
    API_HASH = '3b65d709e533edaf7dda6c903e27575a'  # Your API Hash
    PHONE_NUMBER = '+972555577104'  # Your phone number

    # Initialize and run analyzer
    analyzer = TelegramRecentAnalyzer(API_ID, API_HASH, PHONE_NUMBER)

    try:
        await analyzer.analyze_recent_messages()
    except Exception as e:
        print(f"An error occurred: {e}")


# Run the script
if __name__ == '__main__':
    asyncio.run(main())

"""
Prerequisites and Setup Instructions:

1. Install Required Python Packages:
```bash
pip install telethon pymongo elasticsearch
```

2. Install Elasticsearch:
- For Ubuntu/Debian:
```bash
# Import Elasticsearch GPG key
wget -qO - https://artifacts.elastic.co/GPG-KEY-elasticsearch | sudo apt-key add -

# Install apt-transport-https
sudo apt-get install apt-transport-https

# Add Elasticsearch repository
echo "deb https://artifacts.elastic.co/packages/7.x/apt stable main" | sudo tee -a /etc/apt/sources.list.d/elastic-7.x.list

# Update and install
sudo apt-get update
sudo apt-get install elasticsearch

# Start Elasticsearch service
sudo systemctl start elasticsearch
sudo systemctl enable elasticsearch
```

3. Telegram API Setup:
- Go to https://my.telegram.org/apps
- Create an application to get API ID and Hash

4. Configuration Notes:
- Update API_ID, API_HASH, and PHONE_NUMBER in the script
- Ensure MongoDB and Elasticsearch are running
- First login will require phone verification

5. Customize similarity threshold in cluster_messages_by_similarity method
"""