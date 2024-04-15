import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from prophet import Prophet
from prophet.plot import add_changepoints_to_plot
from sklearn.metrics import mean_absolute_error


def display_data(df: pd.DataFrame) -> None:
    """
    Display data from a DataFrame using a subplot layout.

    Parameters:
    - df (DataFrame): The DataFrame containing the data to be displayed. It is expected
                      to have a 'Date' column and additional columns containing the data
                      to be plotted.

    Returns:
    - None: The function displays the data using plotly subplots and does not return any value.
    """

    columns = df.columns[1:]
    fig = make_subplots(rows=3, cols=2, subplot_titles=columns)
    for i, column in enumerate(columns, start=1):
        row = (i - 1) // 2 + 1
        col = (i - 1) % 2 + 1
        fig.add_trace(go.Scatter(x=df["Date"], y=df[column]), row=row, col=col)
    fig.update_layout(height=1500, width=1000, title_text="Twitter Data", showlegend=False)
    fig.show()


def display_volume_by_year(df: pd.DataFrame) -> None:
    """
    Display a pie chart showing the sum of volume data for each year.

    Parameters:
    - df (DataFrame): The DataFrame containing the volume data. It is expected
                      to have a 'Date' column from which the year is extracted,
                      and a 'Volume' column containing the volume data.

    Returns:
    - None: The function displays the pie chart and does not return any value.
    """
    df["Year"] = df["Date"].dt.year
    df_pie = df.groupby("Year")["Volume"].sum()
    layout = {"title": "Pie Chart for Sum of Volume Data against Each Year"}
    fig = go.Figure(data=[go.Pie(labels=df_pie.index, values=df_pie.values, textinfo="label")], layout=layout)
    fig.show()


def display_key_points(df: pd.DataFrame) -> None:
    """
    Display key points in Twitter stock data.

    Parameters:
    - df (DataFrame): The DataFrame containing the Twitter stock data with columns:
                      'Date', 'Open', 'High', 'Low', 'Close'. 'Date' should be in
                      datetime format.

    Returns:
    - None: The function displays a plot showing the key points in the Twitter
            stock data and does not return any value.
    """
    data = go.Ohlc(
        x=df["Date"],
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        increasing=dict(line=dict(color="#58FA58")),
        decreasing=dict(line=dict(color="#FA5858"))
    )

    layout = {
        "title": "Twitter Stocks",
        "xaxis": {"title": "Date", "rangeslider": {"visible": False}},
        "yaxis": {"title": "Stock Price (USD$)"},
        "shapes": [
            {
                "x0": "2015-10-05",
                "x1": "2015-10-05",
                "y0": 0,
                "y1": 1,
                "xref": "x",
                "yref": "paper",
                "line": {"color": "rgb(30,30,30)", "width": 1}
            },
            {
                "x0": "2020-03-15", "x1": "2020-03-15",
                "y0": 0, "y1": 1, "xref": "x", "yref": "paper",
                "line": {"color": "rgb(30,30,30)", "width": 1}
            }
        ],
        "annotations": [
            {
                "x": "2015-10-05",
                "y": 0.6,
                "xref": "x",
                "yref": "paper",
                "showarrow": False,
                "xanchor": "left",
                "text": "Jack Dorsey becomes CEO of Twitter."
            },
            {
                "x": "2020-03-15",
                "y": 0.05,
                "xref": "x",
                "yref": "paper",
                "showarrow": False,
                "xanchor": "left",
                "text": "Lockdown started in USA due to Covid19."
            }
        ]
    }

    fig = go.Figure(data=[data], layout=layout)
    fig.update(layout_xaxis_rangeslider_visible=True)
    fig.show()


def dispaly_after_covid(df: pd.DataFrame) -> None:
    """
    Display candlestick plot for Twitter stocks after the Covid-19 outbreak.

    Parameters:
    - df (DataFrame): The DataFrame containing the Twitter stock data. It is expected
                      to have a 'Date' column in datetime format, and columns for 'Open',
                      'High', 'Low', and 'Close' prices.

    Returns:
    - None: The function displays a candlestick plot showing the Twitter stock data after
            the Covid-19 outbreak and does not return any value.
    """

    after_covid = df.loc[df["Date"] > "2020-03-15"]
    max_value = after_covid.iloc[:,1:-2].max().max()

    group = after_covid.groupby(["Date"])
    monthly_averages = group.aggregate({"Open": np.mean, "High": np.mean, "Low": np.mean, "Close":np.mean})
    monthly_averages.reset_index(level=0, inplace=True)

    trace = go.Candlestick(
        x=monthly_averages["Date"],
        open=monthly_averages["Open"].values.tolist(),
        high=monthly_averages["High"].values.tolist(),
        low=monthly_averages["Low"].values.tolist(),
        close=monthly_averages["Close"].values.tolist(),
        increasing=dict(line=dict(color="red")),
        decreasing=dict(line=dict(color="lightgreen"))
    )

    layout = {
        'title': 'Twitter Stocks <br> <i> After Covid </i>',
        'xaxis': {'title': 'Date', 'rangeslider': {'visible': False}},
        'yaxis': {'title': 'Stock Price (USD$)'},
        'shapes': [
            {
                'x0': 0,
                'x1': 1,
                'y0': max_value,
                'y1': max_value,
                'xref': 'paper',
                'line': {'color': 'rgb(30,30,30)', 'width': 1}
            }
        ],
        'annotations': [
            {
                'x': '2020-03-15',
                'y': 0.95,
                'xref': 'x',
                'yref': 'paper',
                'showarrow': False,
                'text': 'Peak Value = %f' %max_value
            }
        ]
    }

    data = [trace]
    fig = go.Figure(data=data, layout=layout)
    fig.update(layout_xaxis_rangeslider_visible=True)
    fig.show()


def display_mas(df: pd.DataFrame) -> None:
    """
    Display Moving Averages (MAs) and closing price relationship plot.

    Parameters:
    - df (DataFrame): The DataFrame containing the stock data. It is expected to have
                      a 'Date' column in datetime format and a 'Close' column containing
                      the closing prices.

    Returns:
    - None: The function displays a plot showing the relationship between Moving Averages
            (MAs) and closing price and does not return any value.
    """

    df["MA10"] = df.Close.rolling(window=10).mean()
    df["MA50"] = df.Close.rolling(window=50).mean()
    df["MA200"] = df.Close.rolling(window=200).mean()

    colors = ["#ffF500", "#00308F", "#DC143C"]
    avgs = ["MA10", "MA50", "MA200"]

    fig = make_subplots()
    fig.append_trace(
    {
        'x': df["Date"],
        'y': df["Close"],
        'type': 'scatter',
        'name': 'Close',
        'line': {'color': 'green'}
    }, 1, 1)

    for col, color in zip(avgs, colors):
        fig.append_trace(
        {
            'x': df["Date"],
            'y': df[col],
            'type': 'scatter',
            'name': col,
            'line': {'color': color}
        }, 1, 1)

    fig["layout"].update(height=800, title='Relationship between MAs and Closing Price')
    fig.show()


def dayli_forecast(df: pd.DataFrame) -> None:
    """
    Perform daily forecast using Facebook Prophet.

    Parameters:
    - df (DataFrame): The DataFrame containing the historical data for forecasting.
                       It should have a 'ds' column for dates and a 'y' column for values
                       to be forecasted.

    Returns:
    - None: The function generates and displays plots showing the daily prediction
            and its components using Facebook Prophet, and does not return any value.
    """

    # Fit the Prophet model
    model = Prophet().fit(df)

    # Create future dataframe for forecasting
    future_prices = model.make_future_dataframe(periods=365, freq="D")

    # Predict Prices
    forecast = model.predict(future_prices)
    forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail()

    fig = model.plot(forecast)
    add_changepoints_to_plot(fig.gca(), model, forecast)
    plt.title("Daily Prediction \n 1 year time frame")
    plt.show()

    fig = model.plot_components(forecast)
    plt.show()


def monthly_forecast(df: pd.DataFrame) -> None:
    """
    Perform monthly forecast using Facebook Prophet.

    Parameters:
    - df (DataFrame): The DataFrame containing the historical data for forecasting.
                      It should have a 'ds' column for dates and a 'y' column for values
                      to be forecasted.

    Returns:
    - None: The function generates and displays plots showing the monthly prediction
            and its components using Facebook Prophet, and does not return any value.
    """

    model = Prophet(changepoint_prior_scale=0.03).fit(df)

    # Create future dataframe for forecasting
    future_prices = model.make_future_dataframe(periods=12, freq="ME")

    # Predict prices
    forecast = model.predict(future_prices)

    fig = model.plot(forecast)
    add_changepoints_to_plot(fig.gca(), model, forecast)
    plt.title("Monthly Prediction \n 1 year time frame")
    plt.show()

    fig = model.plot_components(forecast)
    plt.show()

    evaluate_model(df, forecast)


def evaluate_model(df: pd.DataFrame, forecast: pd.DataFrame) -> None:
    """
    Evaluate the performance of a forecasted model.

    Parameters:
    - df (DataFrame): The DataFrame containing the historical data used for forecasting.
                       It should have a 'ds' column for dates and a 'y' column for actual values.
    - forecast (DataFrame): The DataFrame containing the forecasted values. It should have a 'yhat' column
                            representing the predicted values.

    Returns:
    - None: The function calculates and prints the mean absolute error (MAE) between the actual and
            predicted values, and displays a line chart comparing the actual and predicted values.
    """

    y_true = df["y"].values
    y_pred = forecast["yhat"][:-12].values
    mae = mean_absolute_error(y_true, y_pred)
    print("Mean absolute error:", mae)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["ds"],
            y=y_true,
            mode="lines",
            name="Actual"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["ds"],
            y=y_pred,
            mode='lines',
            name='Predicted'
        )
    )
    fig["layout"].update(title='Line chart for Actual and Predicted values')
    fig.show()


def main():
    df = pd.read_csv("stock_data.csv")
    df["Date"] = pd.to_datetime(df["Date"])

    forecast_df = df[["Date", "Close"]]
    forecast_df.rename(columns={"Close": "y", "Date": "ds"}, inplace=True)

    monthly_forecast(forecast_df)


if __name__ == "__main__":
    main()
