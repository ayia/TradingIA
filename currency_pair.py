import requests
import sys

def fetch_data(currency_pair):
    # Construct the URL dynamically
    url = f"https://forex-bestshots-black-meadow-4133.fly.dev/api/FetchPairsData/Lats10IndicatorsBarHistoricaldata?currencyPairs={currency_pair}&interval=OneHour"

    # Make the GET request
    try:
        response = requests.get(url, headers={"accept": "text/plain"})
        response.raise_for_status()  # Raise an error for HTTP errors

        # Print the result
        print("Response from the server:")
        print(response.text)
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Ensure an argument is passed
    if len(sys.argv) < 2:
        print("Usage: python script.py <currency_pair>")
        sys.exit(1)

    # Get the currency pair from the command-line arguments
    currency_pair = sys.argv[1]
    fetch_data(currency_pair)