import asyncio
from telethon import TelegramClient
import pymongo
from datetime import datetime, timedelta
import re
from elasticsearch import Elasticsearch
import numpy as np
import json
from collections import Counter


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

            # Clear existing collections at start
            print("🗑️ Clearing existing MongoDB collections...")
            self.db.raw_messages.drop()
            self.db.clean_messages.drop()
            self.db.clustered_messages.drop()

            # Initialize three collections
            self.raw_messages_collection = self.db['raw_messages']
            self.clean_messages_collection = self.db['clean_messages']
            self.clustered_messages_collection = self.db['clustered_messages']
            print("✅ MongoDB collections reset successfully")
        except Exception as e:
            print(f"❌ MongoDB connection error: {e}")
            self.mongo_client = None

        # Elasticsearch Connection
        try:
            self.es = Elasticsearch(["http://localhost:9200"])
            print(f"🔍 Elasticsearch connection: {self.es}")

            # Delete existing index if exists
            print("🗑️ Clearing existing Elasticsearch index...")
            if self.es.indices.exists(index='telegram_messages'):
                self.es.indices.delete(index='telegram_messages')

            # Create new index
            self.es.indices.create(index='telegram_messages', body={
                "settings": {
                    "analysis": {
                        "analyzer": {
                            "hebrew_analyzer": {
                                "type": "custom",
                                "tokenizer": "standard",
                                "filter": ["lowercase"]
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
            print("✅ Elasticsearch index reset successfully")
        except Exception as e:
            print(f"❌ Elasticsearch connection error: {e}")
            self.es = None

    def find_constant_texts(self, messages):
        """
        Find texts that appear 3 or more times in a channel's messages

        :param messages: List of messages
        :return: Set of constant texts to be removed
        """
        # Count text occurrences
        text_counter = Counter()
        for msg in messages:
            text_counter[msg['text']] += 1

        # Return texts appearing 3 or more times
        return {text for text, count in text_counter.items() if count >= 3}

    async def connect(self):
        """
        Connect to Telegram
        """
        await self.client.start(phone=self.phone_number)
        print("🔗 Connected to Telegram successfully!")

    async def fetch_and_store_recent_messages(self, time_window=10):
        def clean_text(text):
            """
            Function to clean text by removing non-Hebrew words and non-alphabetic characters (except spaces).

            :param text: The input text to clean
            :return: Cleaned text with only Hebrew characters and spaces
            """
            # Keep only Hebrew characters and spaces (remove non-Hebrew letters and punctuation)
            hebrew_text = re.sub(r'[^א-ת\s]', '', text)

            # Filter out words that contain only non-Hebrew characters
            hebrew_words = [word for word in hebrew_text.split() if re.match(r'^[א-ת]+$', word)]

            # Join words back into a cleaned text
            return ' '.join(hebrew_words)

        """
        Fetch messages from all dialogs in the last time_window minutes for Hebrew content
        and store them in MongoDB and Elasticsearch

        :param time_window: Time window in minutes
        :return: List of message dictionaries
        """
        from datetime import timezone

        # Calculate the time threshold (timezone-aware)
        time_threshold = datetime.now(timezone.utc) - timedelta(minutes=time_window)

        # Print time window details
        print(f"🕒 Analyzing messages from {time_threshold.strftime('%Y-%m-%d %H:%M:%S')} to now")

        # Get all dialogs (groups and channels)
        dialogs = await self.client.get_dialogs()

        raw_processed_messages = []
        clean_processed_messages = []

        for dialog in dialogs:
            # Skip if dialog is not a group or channel
            if not dialog.is_group and not dialog.is_channel:
                continue

            try:
                print(f"📬 Fetching messages from: {dialog.name or 'Unknown Dialog'}")
                # Fetch messages from this dialog
                messages = await self.client.get_messages(dialog.entity, limit=50)

                # Track messages in this dialog for finding constant texts
                dialog_messages = []

                for msg in messages:
                    # Ensure msg.date is timezone-aware
                    if msg.date and msg.date.tzinfo is None:
                        msg.date = msg.date.replace(tzinfo=timezone.utc)

                    # Skip messages older than the time threshold or without text
                    if not msg.text or msg.date < time_threshold:
                        continue

                    # Raw message storage
                    raw_message_data = {
                        'text': msg.text,
                        'channel': dialog.name or 'Unknown',
                        'date': msg.date
                    }
                    raw_processed_messages.append(raw_message_data)

                    # Store raw messages in MongoDB
                    if self.mongo_client:
                        self.raw_messages_collection.insert_one(raw_message_data)

                    # Clean the text by removing non-Hebrew words and non-alphabetical characters
                    clean_text_content = clean_text(msg.text)

                    # Skip empty cleaned texts
                    if not clean_text_content:
                        continue

                    dialog_messages.append({'text': clean_text_content})

                # Find constant texts in this dialog
                constant_texts = self.find_constant_texts(dialog_messages)

                # Filter out constant texts
                dialog_messages = [
                    msg for msg in dialog_messages
                    if msg['text'] not in constant_texts
                ]

                # Process filtered (clean) messages
                for msg_data in dialog_messages:
                    clean_message_data = {
                        'text': msg_data['text'],
                        'channel': dialog.name or 'Unknown',
                        'date': msg.date
                    }
                    clean_processed_messages.append(clean_message_data)

                    # Store clean messages in MongoDB
                    if self.mongo_client:
                        self.clean_messages_collection.insert_one(clean_message_data)

                    # Store clean messages in Elasticsearch
                    if self.es:
                        es_doc = {
                            'text': msg_data['text'],
                            'channel': dialog.name or 'Unknown',
                            'date': msg.date
                        }
                        self.es.index(index='telegram_messages', body=es_doc)

                # Print number of messages fetched from this dialog
                print(f"📊 Raw messages from {dialog.name or 'Unknown Dialog'}: {len(raw_processed_messages)}")
                print(f"📊 Clean messages from {dialog.name or 'Unknown Dialog'}: {len(clean_processed_messages)}")

            except Exception as e:
                print(f"❌ Error fetching messages from {dialog.name or 'Unknown Dialog'}: {e}")

        print(f"📊 Total processed raw messages: {len(raw_processed_messages)}")
        print(f"📊 Total processed clean messages: {len(clean_processed_messages)}")
        return clean_processed_messages

    def cluster_messages_by_word_overlap(self, overlap_threshold=0.5, max_messages=1000):
        """
        Cluster messages using simple word overlap similarity

        :param overlap_threshold: Percentage of words that must overlap to be considered similar (0-1)
        :param max_messages: Maximum number of messages to process
        :return: Clustered messages
        """
        # Retrieve base messages
        search_body = {
            "query": {"match_all": {}},
            "size": max_messages
        }
        search_results = self.es.search(index='telegram_messages', body=search_body)

        # Function to calculate word overlap similarity
        def calculate_word_overlap(text1, text2):
            """
            Calculate the percentage of words that overlap between two texts

            :param text1: First text
            :param text2: Second text
            :return: Percentage of overlapping words (0-1)
            """
            # Split texts into words
            words1 = set(text1.split())
            words2 = set(text2.split())

            # Calculate overlap
            if not words1 or not words2:
                return 0

            # Calculate overlap percentage based on the smaller text
            min_words = min(len(words1), len(words2))
            overlap_count = len(words1.intersection(words2))

            return overlap_count / min_words

        # Dictionary to store clusters
        clustered_messages = {}
        processed_ids = set()

        # Iterate through messages
        for idx, base_hit in enumerate(search_results['hits']['hits']):
            # Skip if already processed
            if base_hit['_id'] in processed_ids:
                continue

            # Start a new cluster
            current_cluster = [base_hit]
            processed_ids.add(base_hit['_id'])

            # Initialize first channel and date
            first_message_channel = base_hit['_source']['channel']
            first_message_date = base_hit['_source']['date']

            # Compare with other messages
            for other_hit in search_results['hits']['hits'][idx + 1:]:
                # Skip if already processed
                if other_hit['_id'] in processed_ids:
                    continue

                other_text = other_hit['_source']['text']
                base_text = base_hit['_source']['text']
                other_date = other_hit['_source']['date']

                # Calculate word overlap
                overlap = calculate_word_overlap(base_text, other_text)

                # If overlap is above threshold, add to cluster
                if overlap >= overlap_threshold:
                    current_cluster.append(other_hit)
                    processed_ids.add(other_hit['_id'])

                    # Update initial channel if this message is earlier
                    if datetime.fromisoformat(other_date) < datetime.fromisoformat(first_message_date):
                        first_message_channel = other_hit['_source']['channel']
                        first_message_date = other_hit['_source']['date']

            # Store cluster if it has multiple messages
            if len(current_cluster) > 1:
                cluster_id = len(clustered_messages)
                clustered_messages[cluster_id] = {
                    'messages': current_cluster,
                    'first_message_channel': first_message_channel
                }

        # Print cluster summary
        print(f"\n🌐 Message Clusters Summary:")
        for cluster_id, cluster_data in clustered_messages.items():
            cluster_messages = cluster_data['messages']
            print(f"\n📦 Cluster {cluster_id}:")
            print(f"🔢 Number of messages: {len(cluster_messages)}")
            print(f"📝 First message channel: {cluster_data['first_message_channel']}")
            print("📝 Sample messages:")
            for msg in cluster_messages[:3]:  # Print first 3 messages as sample
                print(f"- {msg['_source']['text'][:100]}... (from {msg['_source']['channel']})")
            print("📍 Channels:", set(msg['_source']['channel'] for msg in cluster_messages))
            print(f"🕒 Initial Report: {msg['_source']['date']}")

        return clustered_messages

    def save_clustered_messages_to_mongodb(self, clustered_messages):
        """
        Save clustered messages to a new MongoDB collection

        :param clustered_messages: Dictionary of clustered messages
        """
        if not self.mongo_client:
            print("❌ MongoDB not connected")
            return

        # Save new clusters
        for cluster_id, cluster_data in clustered_messages.items():
            cluster_messages = cluster_data['messages']
            cluster_doc = {
                'cluster_id': cluster_id,
                'first_message_channel': cluster_data['first_message_channel'],
                'messages': [
                    {
                        'text': msg['_source']['text'],
                        'channel': msg['_source']['channel'],
                        'date': msg['_source']['date']
                    } for msg in cluster_messages
                ],
                'created_at': datetime.utcnow(),
                'cluster_size': len(cluster_messages),
                'initial_report_timestamp': min(
                    datetime.fromisoformat(msg['_source']['date']) for msg in cluster_messages)
            }
            self.clustered_messages_collection.insert_one(cluster_doc)

        print(f"💾 Saved {len(clustered_messages)} message clusters to MongoDB")

    async def analyze_recent_messages(self):
        """
        Main method to analyze recent messages across all dialogs
        """
        # Connect to Telegram
        print("🚀 Starting Telegram Message Analysis...")
        await self.connect()

        # Fetch and store recent messages (returns clean messages)
        messages = await self.fetch_and_store_recent_messages()

        # Check if any messages found
        if not messages:
            print("❗ No recent Hebrew messages found.")
            return

        # Cluster messages by similarity
        print("🔗 Starting message clustering...")
        clustered_messages = self.cluster_messages_by_word_overlap()

        # Save to MongoDB
        self.save_clustered_messages_to_mongodb(clustered_messages)

        # Close the client
        await self.client.disconnect()
        print("🏁 Telegram Message Analysis Complete!")


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
        print(f"❌ An error occurred: {e}")


# Run the script
if __name__ == '__main__':
    asyncio.run(main())