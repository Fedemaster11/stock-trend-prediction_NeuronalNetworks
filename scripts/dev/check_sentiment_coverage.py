# scripts/check_sentiment_coverage.py
import pandas as pd

def check_sentiment_coverage(file_path):
    """
    Loads a dataset and checks which tickers have non-zero sentiment data.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return

    # Check if the required sentiment columns exist
    sentiment_cols = ['n_tweets', 'mean_sent', 'pos_share', 'neg_share']
    if not all(col in df.columns for col in sentiment_cols):
        print(f"Error: The dataset does not contain all required sentiment columns: {sentiment_cols}.")
        return

    print("Checking sentiment data coverage per ticker...")

    # Group by ticker and check the sum of n_tweets
    coverage = df.groupby('ticker')['n_tweets'].sum().reset_index()
    
    # Filter for tickers with at least one tweet
    tickers_with_sentiment = coverage[coverage['n_tweets'] > 0]
    tickers_without_sentiment = coverage[coverage['n_tweets'] == 0]

    if not tickers_with_sentiment.empty:
        print("\n✅ Tickers with available sentiment data (n_tweets > 0):")
        print(tickers_with_sentiment)
    else:
        print("\n❌ No tickers with sentiment data were found in the dataset.")

    if not tickers_without_sentiment.empty:
        print("\n⚠️ Tickers with price data but NO sentiment data:")
        print(tickers_without_sentiment)

if __name__ == "__main__":
    file_path = "data/processed/merged_dataset.csv"
    check_sentiment_coverage(file_path)