import pandas as pd
import datetime as dt
import yfinance as yf
import numpy as np
import plotly.graph_objects as go
import hvplot.pandas
import plotly.express as px
import dash
import dash_html_components as html
import dash_core_components as dcc


start = dt.datetime.today()-dt.timedelta(1825)
end = dt.datetime.today()
ticker = 'AAPL'

stock_data = yf.download(ticker, start, end)
stock_data

def MACD(df,a,b,c):
    df = stock_data.copy()
    df['Fast_EMA']=df['Adj Close'].ewm(span = a, min_periods = a).mean()
    df['Slow_EMA']=df['Adj Close'].ewm(span = b, min_periods = b).mean()
    df['MACD'] = df['Fast_EMA']-df['Slow_EMA']
    df['Signal'] = df['MACD'].ewm(span = c, min_periods = c).mean()
    df['Histogram'] = df['MACD'] - df['Signal']
    df.dropna(inplace = True)
    return df

df_1 = MACD(stock_data, 12,26,9)
Price_df = df_1.reset_index()
Price_df


fig = go.Figure(data=[go.Candlestick(x=Price_df['Date'],
open=Price_df['Open'],
high=Price_df['High'],
low=Price_df['Low'],
close=Price_df['Close'])])

fig.update_layout(
title='Stock Price',
yaxis_title='Stock',
shapes = [dict(
x0='2016-12-09', x1='2016-12-09', y0=0, y1=1, xref='x', yref='paper',
line_width=2)],
annotations=[dict(
x='2016-12-09', y=0.05, xref='x', yref='paper',
showarrow=False, xanchor='left', text='Candlestick Chart')]
    )

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
dcc.Graph(
        id='example-graph',
        figure=fig
    )
])
