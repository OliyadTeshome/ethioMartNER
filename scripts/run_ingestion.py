#!/usr/bin/env python3
"""
Data ingestion script for EthioMart NER pipeline.
"""

import asyncio
import logging
from pathlib import Path
import click
from src.data.telegram_scraper import TelegramScraper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.option("--channels", "-c", multiple=True, help="Telegram channels to scrape")
@click.option("--limit", "-l", default=1000, help="Messages per channel")
@click.option("--days", "-d", default=30, help="Days back to scrape")
def main(channels, limit, days):
    """Run data ingestion from Telegram channels."""
    if not channels:
        channels = ["ethiopian_marketplace", "addis_deals"]
    
    async def run_scraping():
        async with TelegramScraper() as scraper:
            df = await scraper.run_scraping_pipeline(
                list(channels), limit_per_channel=limit, days_back=days
            )
            logger.info(f"Scraped {len(df)} messages from {len(channels)} channels")
    
    asyncio.run(run_scraping())


if __name__ == "__main__":
    main() 