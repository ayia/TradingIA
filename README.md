# Currency Pair Prediction Project

## Project Overview

This project is designed for predicting future price bars for currency pairs using machine learning models, specifically Long Short-Term Memory (LSTM) neural networks. The project is divided into two main scripts:

- **`currency_pair.py`**: Predicts the next bar for a given currency pair based on historical data and outputs trading signals with Stop Loss (SL) and Take Profit (TP) calculations.
- **`train_all_pairs.py`**: Trains LSTM models for multiple currency pairs using historical data and saves the models for prediction.

## Features

- Fetch historical data for currency pairs from an API.
- Preprocess data, including scaling and normalization.
- Train LSTM models on time-series data.
- Predict future price bars and generate trading signals (Buy/Sell).
- Calculate pips, risk-reward ratio, and Stop Loss/Take Profit levels.

## Prerequisites

Ensure the following libraries are installed:

```bash
pip install tensorflow numpy pandas joblib matplotlib requests scikit-learn
```

## Script Details

### `currency_pair.py`

This script is used to predict the next price bar for a specific currency pair.

#### Workflow

1. **Fetch Data**: Fetch the last 10 data points from the API for the specified currency pair.
2. **Preprocess Data**: Scale the features using previously saved scalers.
3. **Load Model**: Load the pre-trained LSTM model and scalers for the given currency pair.
4. **Make Predictions**: Predict the next open, close, high, and low prices.
5. **Generate Trading Signals**:
   - Determine if the signal is "Buy" or "Sell" based on predicted open and close prices.
   - Calculate TP and SL pips and risk-reward ratio.

#### Usage

Run the script with the currency pair name as an argument:

```bash
python currency_pair.py <currency_pair>
```

Example:

```bash
python currency_pair.py EURUSD
```

#### Outputs

- Predicted prices (Open, Close, High, Low)
- Trading direction (Buy/Sell)
- TP and SL pips
- Risk-reward ratio

### `train_all_pairs.py`

This script trains LSTM models for all currency pairs provided as `.json` files in a specified folder.

#### Workflow

1. **Load Data**: Read historical data from JSON files.
2. **Preprocess Data**:
   - Normalize feature and target values using MinMaxScaler.
   - Create sequences (sliding windows) of data for time-series modeling.
3. **Train LSTM Model**:
   - Define a model with LSTM layers and dropout for regularization.
   - Train the model using training data and validate it.
4. **Save Model and Scalers**:
   - Save the trained model in `.h5` format.
   - Save the scalers used for preprocessing.
   - Generate learning curve plots for visualization.

#### Usage

Run the script to train models for all `.json` files in the `training.Data` folder:

```bash
python train_all_pairs.py
```

#### Outputs

- Trained LSTM model for each currency pair in the `Models/<pair_name>` directory.
- Scalers used for preprocessing in `.pkl` format.
- Learning curve plots for each currency pair.

## File Structure

```plaintext
.
├── currency_pair.py       # Prediction script
├── train_all_pairs.py     # Training script
├── Models/                # Directory for saving models and scalers
├── training.Data/         # Directory for storing training data in JSON format
```

## Customization

- **API Endpoint**: Modify the API endpoint in the `fetch_data` function in `currency_pair.py` if needed.
- **Training Parameters**: Adjust hyperparameters like `window_size`, `epochs`, and `batch_size` in `train_all_pairs.py` to improve model performance.
- **Logging**: Add additional logging for better debugging if required.

## Notes

- Ensure that the JSON files for training are correctly formatted with the necessary columns (`SMA`, `EMA`, `RSI`, `MACD`, etc.).
- Regularly update the training data and retrain the models to maintain accuracy.

## Troubleshooting

- **Data Issues**: If sequences cannot be created due to insufficient data, ensure your JSON files contain at least `(window_size + horizon)` rows.
- **API Errors**: Check the API response and headers if the data fetching fails.
- **Model Performance**: Experiment with different hyperparameters and architectures if the predictions are inaccurate.

## License

This project is licensed under the MIT License.

