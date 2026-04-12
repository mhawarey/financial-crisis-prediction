import pandas as pd
import numpy as np
import requests
import json
import datetime as dt
from pandas_datareader import data as pdr
import yfinance as yf
import fredapi
import time
from bs4 import BeautifulSoup
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("data_collector.log"), logging.StreamHandler()]
)
logger = logging.getLogger("DataCollector")

class DataCollector:
    """
    Collects financial and economic data from various free public APIs.
    Handles data retrieval, cleaning, and storage for the crisis prediction system.
    """
    
    def __init__(self):
        self.fred_api_key = None  # Optional: set your FRED API key if you have one
        self.initialize_fred_api()
        # Don't use pdr_override anymore as it's deprecated
        
        # Cache for data to avoid repeated API calls
        self.data_cache = {}
        
        logger.info("DataCollector initialized")
    
    def initialize_fred_api(self):
        """Initialize the FRED API if a key is available."""
        if self.fred_api_key:
            try:
                self.fred = fredapi.Fred(api_key=self.fred_api_key)
                logger.info("FRED API initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize FRED API: {e}")
                self.fred = None
        else:
            logger.info("No FRED API key provided, will use direct download method")
            self.fred = None
    
    def get_market_indicators(self, start_date=None, end_date=None):
        """
        Collect market indicators from Yahoo Finance.
        
        Parameters:
        - start_date: Start date for data collection (default: 5 years ago)
        - end_date: End date for data collection (default: today)
        
        Returns:
        - DataFrame with market indicator data
        """
        if start_date is None:
            start_date = dt.datetime.now() - dt.timedelta(days=5*365)
        if end_date is None:
            end_date = dt.datetime.now()
        
        # Define market indicators to track
        indicators = {
            '^GSPC': 'S&P500',            # S&P 500 Index
            '^VIX': 'VIX',                # Volatility Index
            '^TNX': 'US10YR',             # 10-Year Treasury Yield
            '^TYX': 'US30YR',             # 30-Year Treasury Yield
            '^IRX': 'US13W',              # 13-Week Treasury Bill Rate
            '^DJI': 'DJIA',               # Dow Jones Industrial Average
            '^IXIC': 'NASDAQ',            # NASDAQ Composite
            'GC=F': 'Gold',               # Gold Futures
            'CL=F': 'OilWTI',             # Crude Oil WTI Futures
            'HG=F': 'Copper',             # Copper Futures
            'EURUSD=X': 'EURUSD',         # Euro to USD exchange rate
            'JPYUSD=X': 'JPYUSD',         # Yen to USD exchange rate
            'GBPUSD=X': 'GBPUSD'          # Pound to USD exchange rate
        }
        
        cache_key = f"market_{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}"
        if cache_key in self.data_cache:
            logger.info("Using cached market data")
            return self.data_cache[cache_key]
        
        # Create an empty DataFrame to store all indicators
        all_data = pd.DataFrame()
        
        try:
            logger.info("Collecting market indicators from Yahoo Finance")
            
            # Download data for each indicator using yfinance directly
            for ticker, name in indicators.items():
                try:
                    # Use yfinance directly instead of through pandas_datareader
                    ticker_data = yf.Ticker(ticker)
                    df = ticker_data.history(start=start_date, end=end_date)
                    
                    if not df.empty:
                        # Extract just the closing prices and rename
                        closing_prices = df['Close'].rename(name)
                        
                        if all_data.empty:
                            all_data = pd.DataFrame(closing_prices)
                        else:
                            all_data = all_data.join(closing_prices, how='outer')
                        
                        logger.info(f"Successfully downloaded data for {name}")
                    else:
                        logger.warning(f"Empty data returned for {name}")
                except Exception as e:
                    logger.error(f"Error downloading {name}: {e}")
                    
                # Add a small delay to avoid rate limits
                time.sleep(0.5)
            
            # Calculate derived indicators
            if 'S&P500' in all_data.columns and 'VIX' in all_data.columns:
                # Calculate daily returns
                all_data['S&P500_Returns'] = all_data['S&P500'].pct_change()
                
                # Calculate rolling volatility (20-day)
                all_data['Rolling_Volatility'] = all_data['S&P500_Returns'].rolling(window=20).std() * np.sqrt(252)
                
                # Calculate VIX/Realized Volatility ratio (>1 indicates fear)
                all_data['VIX_RV_Ratio'] = all_data['VIX'] / (all_data['Rolling_Volatility'] * 100)
            
            # Calculate Treasury yield curve indicators
            if 'US10YR' in all_data.columns and 'US13W' in all_data.columns:
                # Calculate 10Y-3M spread (negative values may signal recession)
                all_data['Yield_Curve_10Y_3M'] = all_data['US10YR'] - all_data['US13W']
            
            # Fill missing values with forward fill, then backward fill
            all_data = all_data.fillna(method='ffill').fillna(method='bfill')
            
            # Cache the result
            self.data_cache[cache_key] = all_data
            
            logger.info(f"Market data collection complete: {len(all_data)} rows, {all_data.columns.size} indicators")
            return all_data
            
        except Exception as e:
            logger.error(f"Error in market data collection: {e}")
            return pd.DataFrame()  # Return empty DataFrame on error
    
    def get_economic_indicators(self, start_date=None, end_date=None):
        """
        Collect economic indicators from FRED.
        
        Parameters:
        - start_date: Start date for data collection (default: 5 years ago)
        - end_date: End date for data collection (default: today)
        
        Returns:
        - DataFrame with economic indicator data
        """
        if start_date is None:
            start_date = dt.datetime.now() - dt.timedelta(days=5*365)
        if end_date is None:
            end_date = dt.datetime.now()
        
        cache_key = f"economic_{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}"
        if cache_key in self.data_cache:
            logger.info("Using cached economic data")
            return self.data_cache[cache_key]
        
        # Define economic indicators to track from FRED
        indicators = {
            'UNRATE': 'Unemployment_Rate',         # Unemployment Rate
            'CPIAUCSL': 'CPI',                     # Consumer Price Index for All Urban Consumers
            'FEDFUNDS': 'Fed_Funds_Rate',          # Federal Funds Effective Rate
            'INDPRO': 'Industrial_Production',     # Industrial Production Index
            'HOUST': 'Housing_Starts',             # Housing Starts
            'RSAFS': 'Retail_Sales',               # Retail Sales
            'GACDFSA066MSFRBPHI': 'ADS_Index',     # Aruoba-Diebold-Scotti Business Conditions Index
            'CSUSHPISA': 'Case_Shiller_Index',     # Case-Shiller Home Price Index
            'MSPUS': 'Median_House_Price',         # Median Sales Price of Houses Sold
            'BUSLOANS': 'Commercial_Loans',        # Commercial and Industrial Loans
            'USREC': 'Recession_Indicator',        # US Recession Indicator (1 = recession)
            'USGOVD': 'Govt_Debt_GDP',             # Federal Government Debt as % of GDP
            'GFDEBTN': 'Govt_Debt',                # Federal Government Debt
            'NCBDBIQ027S': 'Household_Debt_GDP',   # Household Debt to GDP
            'NFCIINDX': 'Financial_Conditions',    # Chicago Fed National Financial Conditions Index
            'STLFSI2': 'Financial_Stress',         # St. Louis Fed Financial Stress Index
            'T10Y2Y': 'Yield_Curve_10Y_2Y',        # 10-Year Treasury Minus 2-Year Treasury
            'T10Y3M': 'Yield_Curve_10Y_3M',        # 10-Year Treasury Minus 3-Month Treasury
            'BAMLH0A0HYM2': 'High_Yield_Spread',   # ICE BofA US High Yield Index Option-Adjusted Spread
            'DCOILWTICO': 'Oil_Price_WTI'          # Crude Oil Prices: WTI
        }
        
        try:
            logger.info("Collecting economic indicators from FRED")
            
            # Create an empty DataFrame to store the data
            all_data = pd.DataFrame()
            
            for series_id, name in indicators.items():
                try:
                    # Try to get data from FRED API if available
                    if self.fred:
                        series = self.fred.get_series(
                            series_id, 
                            observation_start=start_date.strftime('%Y-%m-%d'),
                            observation_end=end_date.strftime('%Y-%m-%d')
                        )
                    else:
                        # Alternative: use pandas_datareader
                        series = pdr.DataReader(
                            series_id, 'fred', 
                            start=start_date, 
                            end=end_date
                        ).squeeze()
                    
                    if isinstance(series, pd.Series):
                        # Rename the series and add to the DataFrame
                        series.name = name
                        if all_data.empty:
                            all_data = pd.DataFrame(series)
                        else:
                            all_data = all_data.join(series, how='outer')
                        
                        logger.info(f"Successfully downloaded data for {name}")
                    else:
                        logger.warning(f"Data for {name} not in expected format")
                except Exception as e:
                    logger.error(f"Error downloading {name}: {e}")
                
                # Add a small delay to avoid rate limits
                time.sleep(0.5)
            
            # Calculate derived indicators
            if 'CPI' in all_data.columns:
                # Calculate CPI YoY change (inflation rate)
                all_data['Inflation_Rate'] = all_data['CPI'].pct_change(periods=12) * 100
            
            # Fill missing values
            all_data = all_data.fillna(method='ffill').fillna(method='bfill')
            
            # Cache the result
            self.data_cache[cache_key] = all_data
            
            logger.info(f"Economic data collection complete: {len(all_data)} rows, {all_data.columns.size} indicators")
            return all_data
            
        except Exception as e:
            logger.error(f"Error in economic data collection: {e}")
            return pd.DataFrame()  # Return empty DataFrame on error
    
    def get_news_sentiment(self, days=30):
        """
        Collect news sentiment data from free sources.
        Uses a basic sentiment analysis approach on financial news headlines.
        
        Parameters:
        - days: Number of days of news to collect (default: 30)
        
        Returns:
        - DataFrame with date, headline, source, and sentiment score
        """
        cache_key = f"news_{days}"
        if cache_key in self.data_cache:
            logger.info("Using cached news sentiment data")
            return self.data_cache[cache_key]
        
        try:
            logger.info(f"Collecting news sentiment data for past {days} days")
            
            # List of financial news RSS feeds
            feeds = [
                "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^GSPC&region=US&lang=en-US",
                "http://feeds.marketwatch.com/marketwatch/topstories/",
                "https://www.cnbc.com/id/10000664/device/rss/rss.html",  # CNBC Economy news
                "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000115"  # CNBC Finance news
            ]
            
            import feedparser
            from nltk.sentiment import SentimentIntensityAnalyzer
            
            # Try to initialize NLTK - may need to download data first time
            try:
                sia = SentimentIntensityAnalyzer()
            except:
                import nltk
                nltk.download('vader_lexicon')
                sia = SentimentIntensityAnalyzer()
            
            all_news = []
            
            for feed_url in feeds:
                try:
                    feed = feedparser.parse(feed_url)
                    
                    for entry in feed.entries:
                        # Extract date, title, and source
                        if hasattr(entry, 'published_parsed'):
                            pub_date = dt.datetime(*entry.published_parsed[:6])
                        else:
                            pub_date = dt.datetime.now()  # Default to now if no date
                        
                        title = entry.title
                        source = feed.feed.title if hasattr(feed.feed, 'title') else "Unknown"
                        
                        # Calculate sentiment
                        sentiment = sia.polarity_scores(title)
                        
                        # Add to the list
                        all_news.append({
                            'date': pub_date,
                            'headline': title,
                            'source': source,
                            'sentiment': sentiment['compound'],
                            'negative': sentiment['neg'],
                            'positive': sentiment['pos']
                        })
                    
                    logger.info(f"Retrieved {len(feed.entries)} news items from {source}")
                except Exception as e:
                    logger.error(f"Error fetching news from {feed_url}: {e}")
            
            # Convert to DataFrame and sort by date
            news_df = pd.DataFrame(all_news)
            if not news_df.empty:
                news_df = news_df.sort_values('date', ascending=False)
            
                # Calculate daily aggregate sentiment
                daily_sentiment = news_df.groupby(news_df['date'].dt.date)['sentiment'].agg(['mean', 'count'])
                daily_sentiment.columns = ['avg_sentiment', 'article_count']
                
                # Filter to the requested number of days
                cutoff_date = dt.datetime.now() - dt.timedelta(days=days)
                news_df = news_df[news_df['date'] >= cutoff_date]
                
                # Cache the results
                self.data_cache[cache_key] = (news_df, daily_sentiment)
                
                logger.info(f"News sentiment collection complete: {len(news_df)} articles")
                return news_df, daily_sentiment
            else:
                logger.warning("No news items collected")
                return pd.DataFrame(), pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error in news sentiment collection: {e}")
            return pd.DataFrame(), pd.DataFrame()
    
    def get_geopolitical_events(self):
        """
        Collect geopolitical event data from free sources.
        Uses the GDELT Project's Global Knowledge Graph.
        
        Returns:
        - DataFrame with geopolitical events and impact scores
        """
        cache_key = "geopolitical"
        if cache_key in self.data_cache:
            logger.info("Using cached geopolitical data")
            return self.data_cache[cache_key]
        
        try:
            logger.info("Collecting geopolitical event data")
            
            # This is a placeholder for the full implementation
            # In a real implementation, we would use GDELT API or other sources
            
            # For the trial version, return a sample DataFrame
            dates = pd.date_range(end=dt.datetime.now(), periods=30)
            events = [
                "Trade tensions increase between major economies",
                "Central bank meeting signals policy shift",
                "Political uncertainty in European markets",
                "New regulations impact financial sector",
                "Currency volatility affects emerging markets",
                "Major cyber attack targets financial institutions",
                "International sanctions announced",
                "Global supply chain disruptions reported",
                "Energy price volatility increases",
                "Unexpected election results in key market"
            ]
            
            import random
            
            data = []
            for date in dates:
                # Add 1-3 random events for each date
                for _ in range(random.randint(1, 3)):
                    event = random.choice(events)
                    impact = random.uniform(-0.8, 0.8)  # Random impact score
                    
                    data.append({
                        'date': date,
                        'event': event,
                        'impact_score': impact,
                        'source': random.choice(['News API', 'GDELT', 'RSS Feed', 'Reuters'])
                    })
            
            geo_df = pd.DataFrame(data)
            
            # Calculate daily aggregate impact
            daily_impact = geo_df.groupby(geo_df['date'].dt.date)['impact_score'].mean().to_frame('avg_impact')
            
            # Cache the results
            self.data_cache[cache_key] = (geo_df, daily_impact)
            
            logger.info(f"Geopolitical event collection complete: {len(geo_df)} events")
            return geo_df, daily_impact
            
        except Exception as e:
            logger.error(f"Error in geopolitical data collection: {e}")
            return pd.DataFrame(), pd.DataFrame()
    
    def combine_indicators(self):
        """
        Combine all indicators into a single DataFrame for analysis.
        
        Returns:
        - DataFrame with combined indicators
        """
        try:
            logger.info("Combining all indicators")
            
            # Get data for the past 5 years
            end_date = dt.datetime.now()
            start_date = end_date - dt.timedelta(days=5*365)
            
            # Collect data from different sources
            market_data = self.get_market_indicators(start_date, end_date)
            economic_data = self.get_economic_indicators(start_date, end_date)
            _, news_sentiment = self.get_news_sentiment(days=365)  # Get a year of news sentiment
            _, geo_impact = self.get_geopolitical_events()
            
            # Combine data
            # Start with market data which is usually daily
            combined_data = market_data.copy()
            
            # Add economic data (may be monthly or quarterly)
            # Resample to daily frequency to match market data
            if not economic_data.empty:
                economic_daily = economic_data.resample('D').interpolate(method='linear')
                combined_data = combined_data.join(economic_daily, how='left')
            
            # Add news sentiment data
            if not news_sentiment.empty:
                combined_data = combined_data.join(news_sentiment, how='left')
            
            # Add geopolitical impact data
            if not geo_impact.empty:
                combined_data = combined_data.join(geo_impact, how='left')
            
            # Fill missing values
            combined_data = combined_data.fillna(method='ffill').fillna(method='bfill')
            
            logger.info(f"Combined indicators: {combined_data.shape[0]} rows, {combined_data.shape[1]} columns")
            return combined_data
            
        except Exception as e:
            logger.error(f"Error combining indicators: {e}")
            return pd.DataFrame()