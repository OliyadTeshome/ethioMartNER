"""
Enhanced Telegram data scraper for EthioMart NER pipeline.

This module handles data ingestion from Telegram channels containing
Ethiopian e-commerce information with comprehensive metadata collection.
"""

import asyncio
import json
import logging
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
from telethon import TelegramClient
from telethon.tl.types import Channel, Message, DocumentAttributeFilename
from telethon.tl.functions.messages import GetHistoryRequest
from dotenv import load_dotenv
import os
import hashlib

logger = logging.getLogger(__name__)


class EnhancedTelegramScraper:
    """Enhanced scraper for Telegram channels to collect comprehensive e-commerce data."""

    def __init__(
        self,
        api_id: Optional[str] = None,
        api_hash: Optional[str] = None,
        phone_number: Optional[str] = None,
        session_name: str = "ethiomart_session",
    ):
        """Initialize the enhanced Telegram scraper.

        Args:
            api_id: Telegram API ID
            api_hash: Telegram API Hash
            phone_number: Phone number for authentication
            session_name: Session file name
        """
        load_dotenv()
        
        self.api_id = api_id or os.getenv("TELEGRAM_API_ID")
        self.api_hash = api_hash or os.getenv("TELEGRAM_API_HASH")
        self.phone_number = phone_number or os.getenv("TELEGRAM_PHONE_NUMBER")
        self.session_name = session_name
        
        if not all([self.api_id, self.api_hash, self.phone_number]):
            raise ValueError("Missing required Telegram API credentials")
        
        self.client = TelegramClient(session_name, self.api_id, self.api_hash)
        self.data_dir = Path("data/raw")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Amharic e-commerce keywords for filtering
        self.amharic_keywords = [
            "ሽያጭ", "ግዛ", "ዋጋ", "ቅናሽ", "አዲስ", "ተጠቃሚ", "ድምጽ", "ስልክ", "ኮምፒዩተር",
            "ማሽን", "መጠጥ", "ምግብ", "ልብስ", "ጫማ", "ቤት", "መኪና", "ሞተር", "ቁርስ", "ጠረጴዛ"
        ]
        
        self.english_keywords = [
            "sale", "buy", "price", "discount", "new", "used", "offer", "phone", "laptop",
            "computer", "car", "house", "rent", "sell", "purchase", "deal", "bargain"
        ]

    async def __aenter__(self):
        """Async context manager entry."""
        await self.client.start(phone=self.phone_number)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.client.disconnect()

    def extract_media_info(self, message: Message) -> Dict[str, Any]:
        """Extract media information from message.
        
        Args:
            message: Telegram message object
            
        Returns:
            Dictionary with media information
        """
        media_info = {
            "has_media": False,
            "media_type": None,
            "file_name": None,
            "file_size": None,
            "media_url": None
        }
        
        if message.media:
            media_info["has_media"] = True
            
            # Determine media type
            if hasattr(message.media, 'photo'):
                media_info["media_type"] = "photo"
            elif hasattr(message.media, 'document'):
                media_info["media_type"] = "document"
                # Extract filename if available
                for attr in message.media.document.attributes:
                    if isinstance(attr, DocumentAttributeFilename):
                        media_info["file_name"] = attr.file_name
                        break
                media_info["file_size"] = message.media.document.size
            elif hasattr(message.media, 'webpage'):
                media_info["media_type"] = "webpage"
                media_info["media_url"] = message.media.webpage.url
        
        return media_info

    def extract_sender_info(self, message: Message) -> Dict[str, Any]:
        """Extract sender information from message.
        
        Args:
            message: Telegram message object
            
        Returns:
            Dictionary with sender information
        """
        sender_info = {
            "sender_id": None,
            "sender_username": None,
            "sender_first_name": None,
            "sender_last_name": None,
            "is_bot": False
        }
        
        if message.sender_id:
            sender_info["sender_id"] = message.sender_id
            
        if hasattr(message, 'sender') and message.sender:
            sender = message.sender
            sender_info["sender_username"] = getattr(sender, 'username', None)
            sender_info["sender_first_name"] = getattr(sender, 'first_name', None)
            sender_info["sender_last_name"] = getattr(sender, 'last_name', None)
            sender_info["is_bot"] = getattr(sender, 'bot', False)
        
        return sender_info

    def clean_amharic_text(self, text: str) -> str:
        """Clean and normalize Amharic text.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned Amharic text
        """
        if not isinstance(text, str):
            return ""
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove phone numbers (Ethiopian format)
        text = re.sub(r'(\+251|0)?[0-9]{9}', '', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep Amharic and basic punctuation
        text = re.sub(r'[^\w\s\u1200-\u137F\u1380-\u139F\u2D80-\u2DDF\uAB00-\uAB2F.,!?]', '', text)
        
        return text.strip()

    def is_ecommerce_message(self, text: str) -> bool:
        """Check if message contains e-commerce related content.
        
        Args:
            text: Message text to check
            
        Returns:
            True if message is e-commerce related
        """
        if not text:
            return False
        
        text_lower = text.lower()
        
        # Check for Amharic keywords
        for keyword in self.amharic_keywords:
            if keyword in text:
                return True
        
        # Check for English keywords
        for keyword in self.english_keywords:
            if keyword in text_lower:
                return True
        
        # Check for price patterns (Ethiopian Birr)
        price_patterns = [
            r'\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:ብር|birr|ETB)',
            r'(?:ብር|birr|ETB)\s*\d+(?:,\d{3})*(?:\.\d{2})?'
        ]
        
        for pattern in price_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        return False

    async def get_channel_messages(
        self,
        channel_username: str,
        limit: int = 1000,
        offset_date: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch comprehensive messages from a Telegram channel.

        Args:
            channel_username: Channel username or ID
            limit: Maximum number of messages to fetch
            offset_date: Start date for message fetching

        Returns:
            List of comprehensive message dictionaries
        """
        messages = []
        
        try:
            async with self.client:
                channel = await self.client.get_entity(channel_username)
                
                async for message in self.client.iter_messages(
                    channel, limit=limit, offset_date=offset_date
                ):
                    if message.text:
                        # Extract basic message info
                        message_data = {
                            "message_id": message.id,
                            "date": message.date.isoformat(),
                            "text": message.text,
                            "cleaned_text": self.clean_amharic_text(message.text),
                            "channel": channel_username,
                            "views": getattr(message, "views", 0),
                            "forwards": getattr(message, "forwards", 0),
                            "replies": getattr(message, "replies", 0),
                            "is_ecommerce": self.is_ecommerce_message(message.text),
                            "message_hash": hashlib.md5(f"{message.id}_{channel_username}".encode()).hexdigest()
                        }
                        
                        # Extract media info
                        media_info = self.extract_media_info(message)
                        message_data.update(media_info)
                        
                        # Extract sender info
                        sender_info = self.extract_sender_info(message)
                        message_data.update(sender_info)
                        
                        messages.append(message_data)
                        
                        if len(messages) % 100 == 0:
                            logger.info(f"Fetched {len(messages)} messages from {channel_username}")
        
        except Exception as e:
            logger.error(f"Error fetching messages from {channel_username}: {e}")
            
        return messages

    async def scrape_channels(
        self,
        channels: List[str],
        limit_per_channel: int = 1000,
        days_back: int = 30,
    ) -> pd.DataFrame:
        """Scrape multiple Telegram channels with comprehensive data.

        Args:
            channels: List of channel usernames
            limit_per_channel: Maximum messages per channel
            days_back: Number of days back to fetch messages

        Returns:
            DataFrame containing all scraped messages with comprehensive metadata
        """
        all_messages = []
        offset_date = datetime.now() - timedelta(days=days_back)
        
        for channel in channels:
            logger.info(f"Scraping channel: {channel}")
            messages = await self.get_channel_messages(
                channel, limit_per_channel, offset_date
            )
            all_messages.extend(messages)
            
            # Rate limiting to avoid API restrictions
            await asyncio.sleep(3)
        
        df = pd.DataFrame(all_messages)
        
        if not df.empty:
            # Add metadata
            df['scraped_at'] = datetime.now().isoformat()
            df['source'] = 'telegram'
            
            # Save raw data
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.data_dir / f"telegram_comprehensive_{timestamp}.json"
            
            df.to_json(output_file, orient="records", indent=2)
            logger.info(f"Saved {len(df)} messages to {output_file}")
        
        return df

    def filter_and_clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter and clean the scraped data for NER processing.
        
        Args:
            df: DataFrame with scraped messages
            
        Returns:
            Cleaned and filtered DataFrame
        """
        # Filter e-commerce messages
        ecommerce_df = df[df['is_ecommerce'] == True].copy()
        
        # Remove duplicates based on message hash
        ecommerce_df = ecommerce_df.drop_duplicates(subset=['message_hash'])
        
        # Remove messages with very short text
        ecommerce_df = ecommerce_df[ecommerce_df['cleaned_text'].str.len() > 10]
        
        # Sort by date
        ecommerce_df = ecommerce_df.sort_values('date', ascending=False)
        
        logger.info(f"Filtered to {len(ecommerce_df)} e-commerce messages from {len(df)} total")
        
        return ecommerce_df

    async def run_comprehensive_scraping_pipeline(
        self,
        channels: List[str],
        limit_per_channel: int = 1000,
        days_back: int = 30,
        save_filtered: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Run the complete comprehensive scraping pipeline.

        Args:
            channels: List of channel usernames
            limit_per_channel: Maximum messages per channel
            days_back: Number of days back to fetch messages
            save_filtered: Whether to save filtered data

        Returns:
            Tuple of (raw_data, filtered_data) DataFrames
        """
        logger.info("Starting comprehensive Telegram scraping pipeline")
        
        # Scrape all channels
        raw_df = await self.scrape_channels(channels, limit_per_channel, days_back)
        
        if raw_df.empty:
            logger.warning("No messages were scraped")
            return raw_df, pd.DataFrame()
        
        # Filter and clean data
        filtered_df = self.filter_and_clean_data(raw_df)
        
        if save_filtered and not filtered_df.empty:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.data_dir / f"ecommerce_filtered_{timestamp}.json"
            filtered_df.to_json(output_file, orient="records", indent=2)
            logger.info(f"Saved filtered data to {output_file}")
        
        return raw_df, filtered_df


async def main():
    """Main function for testing the enhanced scraper."""
    # Ethiopian e-commerce channels (replace with actual channels)
    channels = [
        "ethiopian_marketplace",
        "addis_deals", 
        "ethio_shopping",
        "addis_tech_market",
        "ethiopia_electronics"
    ]
    
    async with EnhancedTelegramScraper() as scraper:
        raw_df, filtered_df = await scraper.run_comprehensive_scraping_pipeline(
            channels, limit_per_channel=100
        )
        print(f"Scraped {len(raw_df)} total messages")
        print(f"Filtered to {len(filtered_df)} e-commerce messages")


if __name__ == "__main__":
    asyncio.run(main()) 