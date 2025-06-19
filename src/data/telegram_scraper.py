"""
Telegram data scraper for EthioMart NER pipeline.

This module handles data ingestion from Telegram channels containing
Ethiopian e-commerce information.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd
from telethon import TelegramClient
from telethon.tl.types import Channel, Message
from dotenv import load_dotenv
import os

logger = logging.getLogger(__name__)


class TelegramScraper:
    """Scraper for Telegram channels to collect e-commerce data."""

    def __init__(
        self,
        api_id: Optional[str] = None,
        api_hash: Optional[str] = None,
        phone_number: Optional[str] = None,
        session_name: str = "ethiomart_session",
    ):
        """Initialize the Telegram scraper.

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

    async def __aenter__(self):
        """Async context manager entry."""
        await self.client.start(phone=self.phone_number)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.client.disconnect()

    async def get_channel_messages(
        self,
        channel_username: str,
        limit: int = 1000,
        offset_date: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch messages from a Telegram channel.

        Args:
            channel_username: Channel username or ID
            limit: Maximum number of messages to fetch
            offset_date: Start date for message fetching

        Returns:
            List of message dictionaries
        """
        messages = []
        
        try:
            async with self.client:
                channel = await self.client.get_entity(channel_username)
                
                async for message in self.client.iter_messages(
                    channel, limit=limit, offset_date=offset_date
                ):
                    if message.text:
                        message_data = {
                            "id": message.id,
                            "date": message.date.isoformat(),
                            "text": message.text,
                            "channel": channel_username,
                            "views": getattr(message, "views", 0),
                            "forwards": getattr(message, "forwards", 0),
                        }
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
        """Scrape multiple Telegram channels.

        Args:
            channels: List of channel usernames
            limit_per_channel: Maximum messages per channel
            days_back: Number of days back to fetch messages

        Returns:
            DataFrame containing all scraped messages
        """
        all_messages = []
        offset_date = datetime.now() - timedelta(days=days_back)
        
        for channel in channels:
            logger.info(f"Scraping channel: {channel}")
            messages = await self.get_channel_messages(
                channel, limit_per_channel, offset_date
            )
            all_messages.extend(messages)
            
            # Rate limiting
            await asyncio.sleep(2)
        
        df = pd.DataFrame(all_messages)
        
        if not df.empty:
            # Save raw data
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.data_dir / f"telegram_data_{timestamp}.json"
            
            df.to_json(output_file, orient="records", indent=2)
            logger.info(f"Saved {len(df)} messages to {output_file}")
        
        return df

    def filter_ecommerce_messages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter messages that contain e-commerce related content.

        Args:
            df: DataFrame with scraped messages

        Returns:
            Filtered DataFrame with e-commerce messages
        """
        # Amharic keywords for e-commerce
        ecommerce_keywords = [
            "ሽያጭ", "ግዛ", "ዋጋ", "ቅናሽ", "አዲስ", "ተጠቃሚ", "ድምጽ",
            "sale", "buy", "price", "discount", "new", "used", "offer"
        ]
        
        # Filter messages containing e-commerce keywords
        mask = df["text"].str.contains("|".join(ecommerce_keywords), case=False, na=False)
        filtered_df = df[mask].copy()
        
        logger.info(f"Filtered {len(filtered_df)} e-commerce messages from {len(df)} total")
        return filtered_df

    async def run_scraping_pipeline(
        self,
        channels: List[str],
        limit_per_channel: int = 1000,
        days_back: int = 30,
        save_filtered: bool = True,
    ) -> pd.DataFrame:
        """Run the complete scraping pipeline.

        Args:
            channels: List of channel usernames
            limit_per_channel: Maximum messages per channel
            days_back: Number of days back to fetch messages
            save_filtered: Whether to save filtered data

        Returns:
            DataFrame with filtered e-commerce messages
        """
        logger.info("Starting Telegram scraping pipeline")
        
        # Scrape all channels
        raw_df = await self.scrape_channels(channels, limit_per_channel, days_back)
        
        if raw_df.empty:
            logger.warning("No messages were scraped")
            return raw_df
        
        # Filter e-commerce messages
        filtered_df = self.filter_ecommerce_messages(raw_df)
        
        if save_filtered and not filtered_df.empty:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.data_dir / f"ecommerce_data_{timestamp}.json"
            filtered_df.to_json(output_file, orient="records", indent=2)
            logger.info(f"Saved filtered data to {output_file}")
        
        return filtered_df


async def main():
    """Main function for testing the scraper."""
    # Example channels (replace with actual Ethiopian e-commerce channels)
    channels = [
        "ethiopian_marketplace",
        "addis_deals",
        "ethio_shopping"
    ]
    
    async with TelegramScraper() as scraper:
        df = await scraper.run_scraping_pipeline(channels, limit_per_channel=100)
        print(f"Scraped {len(df)} e-commerce messages")


if __name__ == "__main__":
    asyncio.run(main()) 