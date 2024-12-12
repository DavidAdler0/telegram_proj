import asyncio
from telethon import TelegramClient
import pymongo
from datetime import datetime, timezone, timedelta
import re
from elasticsearch import Elasticsearch
import numpy as np
from collections import Counter


class TelegramRecentAnalyzer:
    def __init__(self, api_id, api_hash, phone_number,
                 mongodb_uri='mongodb://localhost:27017/',
                 elasticsearch_host='localhost:9200'):
        """
        Initialize Telegram Client, MongoDB, and Elasticsearch connections
        """
        # Telegram Client Setup
        self.client = TelegramClient('session', api_id, api_hash)
        self.phone_number = phone_number

        # MongoDB Connection
        self.mongo_client = pymongo.MongoClient(mongodb_uri)
        self.db = self.mongo_client['telegram_analysis']

        # Clear existing collections
        print("🗑️ Clearing existing MongoDB collections...")
        self.db.raw_messages.drop()
        self.db.clean_messages.drop()
        self.db.clustered_messages.drop()

        # Initialize collections
        self.raw_messages_collection = self.db['raw_messages']
        self.clean_messages_collection = self.db['clean_messages']
        self.clustered_messages_collection = self.db['clustered_messages']

        # Elasticsearch Connection
        self.es = Elasticsearch(["http://localhost:9200"])
        if self.es.indices.exists(index='telegram_messages'):
            self.es.indices.delete(index='telegram_messages')

        # Create Elasticsearch index
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
        print("✅ Initialization complete")

    def find_recurring_channel_texts(self, messages):
        """
        Find texts appearing more than 5 times in the 50 messages,
        supporting sequences longer than 3 words.

        :param messages: List of messages
        :return: Set of recurring texts
        """
        # Create a mapping of text sequences
        text_sequences = Counter()

        for msg in messages:
            if not hasattr(msg, 'text') or not msg.text:
                continue

            # Split message into word sequences of 3+ words
            words = msg.text.split()

            # Check for sequences of varying lengths (from 3 to full message length)
            for length in range(3, len(words) + 1):
                for i in range(len(words) - length + 1):
                    sequence = ' '.join(words[i:i + length])
                    text_sequences[sequence] += 1

        # Filter recurring sequences (more than 5 occurrences)
        recurring_texts = {seq for seq, count in text_sequences.items() if count > 5}

        print(f"🔁 Recurring text sequences found: {recurring_texts}")
        return recurring_texts

    def clean_hebrew_text(self, text, recurring_texts=None):
        """
        Clean text by:
        1. Removing non-Hebrew words
        2. Removing recurring texts
        3. Keeping only Hebrew characters and spaces
        """
        if not isinstance(text, str):
            return ''

        # Remove non-Hebrew and non-space characters
        hebrew_text = re.sub(r'[^א-ת\s]', '', text)

        # Remove recurring texts
        if recurring_texts:
            for rec_text in recurring_texts:
                hebrew_text = hebrew_text.replace(rec_text, '')

        # Split and filter words
        words = [
            word for word in hebrew_text.split()
            if re.match(r'^[א-ת]+$', word)
        ]

        return ' '.join(words)

    async def connect(self):
        """Connect to Telegram"""
        await self.client.start(phone=self.phone_number)
        print("🔗 Connected to Telegram successfully!")

    def calculate_text_similarity(self, text1, text2):
        """
        Calculate text similarity using Jaccard index

        :param text1: First text
        :param text2: Second text
        :return: Similarity score (0-1)
        """
        words1 = set(text1.split())
        words2 = set(text2.split())

        if not words1 or not words2:
            return 0

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0

    async def analyze_recent_messages(self):
        """
        Improved clustering method with strict similarity requirements
        """
        await self.connect()

        # Get all dialogs
        dialogs = await self.client.get_dialogs()

        for dialog in dialogs:
            # Skip if not a group or channel
            if not dialog.is_group and not dialog.is_channel:
                continue

            print(f"\n📬 Analyzing dialog: {dialog.name or 'Unknown Dialog'}")

            try:
                # Fetch 50 messages
                messages = []
                async for msg in self.client.iter_messages(dialog.entity, limit=50):
                    if hasattr(msg, 'text') and msg.text:
                        messages.append(msg)

                if not messages:
                    print(f"⏩ No valid messages found in dialog: {dialog.name or 'Unknown Dialog'}")
                    continue

                # 1. Find recurring texts in the channel
                recurring_texts = self.find_recurring_channel_texts(messages)

                # 2. Store raw messages
                raw_messages = []
                for msg in messages:
                    raw_message_data = {
                        'text': msg.text,
                        'channel': dialog.name or 'Unknown',
                        'date': msg.date or datetime.now(timezone.utc)
                    }
                    raw_messages.append(raw_message_data)

                # Bulk insert raw messages
                if raw_messages:
                    self.raw_messages_collection.insert_many(raw_messages)

                # 3. Clean messages
                clean_messages = []
                for msg in messages:
                    cleaned_text = self.clean_hebrew_text(msg.text, recurring_texts)

                    if cleaned_text:
                        clean_message_data = {
                            'text': cleaned_text,
                            'channel': dialog.name or 'Unknown',
                            'date': msg.date or datetime.now(timezone.utc)
                        }
                        clean_messages.append(clean_message_data)

                # Bulk insert clean messages
                if clean_messages:
                    self.clean_messages_collection.insert_many(clean_messages)

                # 4. Advanced Clustering with Strict Similarity
                existing_clusters = list(self.clustered_messages_collection.find())

                new_clusters = []

                for msg in clean_messages:
                    best_cluster = None
                    best_similarity = 0
                    best_cluster_index = None

                    # Check against all existing clusters from Mongo
                    for cluster_index, existing_cluster in enumerate(existing_clusters):
                        # Compare with the first message of the cluster
                        first_cluster_message = existing_cluster.get('first_message', {})

                        # Calculate similarity with the first message of the cluster
                        similarity = self.calculate_text_similarity(
                            msg['text'],
                            first_cluster_message.get('text', '')
                        )

                        if similarity >= 0.5 and similarity > best_similarity:
                            best_cluster = existing_cluster
                            best_similarity = similarity
                            best_cluster_index = cluster_index

                    if best_cluster:
                        # Update existing cluster in Mongo
                        update_query = {'_id': best_cluster['_id']}
                        update_operation = {
                            '$push': {'all_messages': msg},
                            '$inc': {'cluster_size': 1},
                            '$min': {'first_message.date': msg['date']}
                        }
                        self.clustered_messages_collection.update_one(update_query, update_operation)

                        # Update Elasticsearch index
                        es_doc = {
                            'cluster_id': best_cluster.get('cluster_id'),
                            'text': msg['text'],
                            'channel': msg['channel'],
                            'date': msg['date']
                        }
                        self.es.index(index='telegram_messages', body=es_doc)
                    else:
                        # Create new cluster
                        new_cluster = {
                            'first_message': msg,
                            'all_messages': [msg],
                            'cluster_size': 1,
                            'first_channel': msg['channel'],
                            'created_at': datetime.now(timezone.utc)
                        }
                        # Insert to Mongo
                        inserted_cluster = self.clustered_messages_collection.insert_one(new_cluster)

                        # Index in Elasticsearch
                        es_doc = {
                            'cluster_id': str(inserted_cluster.inserted_id),
                            'text': msg['text'],
                            'channel': msg['channel'],
                            'date': msg['date']
                        }
                        self.es.index(index='telegram_messages', body=es_doc)

                print(f"✅ Processed {len(messages)} messages from {dialog.name or 'Unknown Dialog'}")

            except Exception as e:
                print(f"❌ Error processing dialog {dialog.name or 'Unknown Dialog'}: {e}")
                import traceback
                traceback.print_exc()

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
        import traceback
        traceback.print_exc()


# Run the script
if __name__ == '__main__':
    asyncio.run(main())