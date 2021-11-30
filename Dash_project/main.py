import statistics

import dash
import dash_html_components as html  # Contains components for every HTML tag
import dash_core_components as dcc
import numpy as np
import pandas as pd
import plotly.express as px  # Plotly line chart into the dashboard
from dash.dependencies import Input, Output, State   # To use callbacks
import datetime
import time
import DescriptiveAnalysis
from predicion import testfunc,StockMarket


ticker = pd.read_excel('TickerSymbols.xlsx', usecols="A")
list_of_ticker = list(ticker.Symbol)

app = dash.Dash()
app.layout= html.Div([
    html.H1('STOCK PRICE DATA ANALYSIS AND PREDICTION'),
    html.P(["This dashboard allows users to select the stock and give prediction",
            html.Br(),
            html.A("STOCK PRICE DATA ANALYSIS",href='https://finance.yahoo.com/',
                   target="_blank")
           ]),
    dcc.Input(id='ticker', value='Select ticker', type='text'),
    html.Div(id='Company Name'),
    dcc.Input(id='Date',value='Enter start date',type='text'),
    html.Div(id='Show_date'),
    dcc.Input(id='Date_end',value='Enter end date',type='text'),
    html.Div(id='show_end_date'),
    html.Br(),
    html.H2('Select number of days to predict'),
    dcc.Slider('slider',min=2,max=30,value=10,marks={i:str(i)for i in range(2,31)}),
    html.Button('Predict and wait for 10 sec', id='Predict_button_State', n_clicks=0),
    dcc.Graph(id='Plot'),
    html.Div(id='mean',children = 'mean'),
    html.Div(id='median',children = 'median'),
    html.Div(id='mode',children = 'mode'),
    html.Div(id='std_dev',children = 'std dev'),
    dcc.Graph(id='Plot_pred')
])




@app.callback(
    Output(component_id='Plot', component_property='figure'),
    Output(component_id='Company Name', component_property='children'),
    Output(component_id='Show_date', component_property='children'),
    Output(component_id='show_end_date', component_property='children'),
    Output(component_id='Plot_pred', component_property='figure'),
    Output(component_id='mean', component_property='children'),
    Output(component_id='median', component_property='children'),
    Output(component_id='mode', component_property='children'),
    Output(component_id='std_dev', component_property='children'),
    Input('Predict_button_State','n_clicks'),
    State(component_id='ticker', component_property='value'),
    State(component_id='Date', component_property='value'),
    State(component_id='Date_end', component_property='value'),
    State(component_id='slider', component_property='value')

)
def check_ticker(button_click, input_text, input_date, input_end_date,no_of_days):
    start_date = 1606520423
    end_date = 1638056423

    def chkDate(date_text):
        try:
            x = bool(datetime.datetime.strptime(date_text, "%Y-%m-%d"))
            return x
        except ValueError:
            x = False
            return x

    if chkDate(input_date):
        s_date = f'Start Date is set to : {input_date}'
        syear, smonth, sday = map(int, input_date.split('-'))
        start_date = int(time.mktime(datetime.datetime(syear, smonth, sday, 23, 59).timetuple()))

    else:
        s_date = "Invalid Date, please use YYYY-MM-DD Format,Plotting for default DATE"
    if chkDate(input_end_date):
        e_date = f'End Date is set to : {input_end_date}'
        eyear, emonth, eday = map(int, input_end_date.split('-'))
        end_date = int(time.mktime(datetime.datetime(eyear, emonth, eday, 23, 59).timetuple()))


    else:
        e_date = "Invalid Date, please use YYYY-MM-DD Format,Plotting for default DATE"

    if (start_date >= end_date):
        s_date = 'Start date should be before end date,Plotting for default DATE.'
        e_date = 'End date should be after start date,Plotting for default DATE.'
        start_date = 1606520423
        end_date = 1638056423

    if input_text in list_of_ticker:
        ticker_api = f'https://query1.finance.yahoo.com/v7/finance/download/{input_text}?period1={start_date}&period2={end_date}&interval=1d&events=history&includeAdjustedClose=true'
        df = pd.read_csv(ticker_api)
        stock_fig = px.line(x=df.Date, y=df.Close)
        mn = np.mean(df.Close)
        md = np.median(df.Close)
        mdo = statistics.mode(df.Close)
        stdv = np.std(df.Close)
        s= StockMarket(df,timestep=100)
        predict_output= s.stockMarketPred(user_date='2020-01-31',number_of_days=no_of_days+1)
        print(no_of_days)
        list_predict = []
        for i in predict_output:
            list_predict.append(i[0])
        #predict_output = testfunc(df,start_date)
        return stock_fig, f'Ticker of stock selected {input_text} is valid', s_date,e_date, px.line(predict_output),f'Mean is : {mn}',f'Median is:{md}',f'Mode is : {mdo}',f'Standard deviation is : {stdv}'
    else:
        return px.line(), f'Ticiker selected {input_text} is not a valid Ticker', s_date, e_date, px.line(),'mean','median','mode','std dev'


if __name__ == '__main__':
    app.run_server(debug=True)  # Debug argument is true for updating the dashboard automatically
