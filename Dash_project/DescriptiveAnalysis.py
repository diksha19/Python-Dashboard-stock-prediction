import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import statistics
import pandas as pd

# Calculating the descriptive part for the data selected
def stats(df,type):
    if type=='Mean':
        return np.mean(df.Close)
    if type=='Median':
        return np.median(df.Close)
    if type=='Mode':
        return statistics.mode(df.Close)
    if type=='STD':
        return np.std(df.Close)


'''Plotting the descriptive graphs for the data selected'''
def get_desc(df,window, type):

    if type == 'WMA':
        weights = np.arange(1, window + 1)
        qwerty = df['Close'].rolling(window).apply(lambda prices: np.dot(prices, weights) / weights.sum())
        print(qwerty)
        # plot the moving avg.
        fig = px.line(x=df.Date, y=df.Close,title="Weighted Moving Average")
        fig.add_scatter(x=df.Date, y=qwerty, mode='lines',name= 'WMA')
    elif type == 'MA':
        df['MA'] = df.Close.rolling(window).mean()
        fig = px.line(x=df.Date, y=df.Close,title="Moving Average")
        fig.add_scatter(x=df.Date, y=df.MA, mode='lines',name='Moving average')
    elif type == 'LT':
        fig = px.scatter(df, y=df.Close, trendline='ols', trendline_scope='overall',title='Linear Trend Line')
    elif type == 'MACD':
        slow, fast, smooth = 26,12,9
        exp1 = df['Close'].ewm(span=fast, adjust=False).mean()
        exp2 = df['Close'].ewm(span=slow, adjust=False).mean()
        macd = exp2 - exp1
        signal_exp3 = macd.ewm(span=smooth, adjust=False).mean()
        fig = px.line(y=macd,title="MACD")
        fig.add_scatter(y=signal_exp3, mode='lines',name='MACD')

    return fig


