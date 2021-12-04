##### Importing all modules required.
import dash ## Dash is used which is python framework created by plotly for creating interactive web applications.
import dash_html_components as html ## Contains components for every HTML tag from dash.
import dash_core_components as dcc  ## Contains components for every HTML tag from dash.
import numpy as np  ## Numpy for mathematical function library.
import pandas as pd ## Pandas for data analysis and manipulation.
import plotly.express as px  # Plotly line chart into the dashboard
from dash.dependencies import Input, Output, State   # To use callbacks
import datetime ## Deals with all datetime functions in python
import time ## time module is principally for working with Unix time stamps
from DescriptiveAnalysis import get_desc,stats ## Module calling from another .py file
from predicion import StockMarket ## Module calling from another .py file

'''
All the ticker related functionality (Selecting stock,start date and end date)with html layout '''

ticker = pd.read_excel('TickerSymbols.xlsx', usecols="A")
list_of_ticker = list(ticker.Symbol)
external_stylesheet = ['https://codepen.io/chriddyp/pen/bWLwgP.css'] ##  Applying the external stylesheet for the webpage.
app = dash.Dash(external_stylesheets=external_stylesheet)

########################################################################################################################################################################################


app.layout= html.Div([
    html.H1('STOCK PRICE DATA ANALYSIS AND PREDICTION',style={'textAlign':'center','background-image': 'https://miro.medium.com/max/2563/1*PDPybdiyzspH8YpUINdl4w.png'}),
    html.P(["This dashboard allows users to select the stock and see the prediction.",
            html.Br(),
            html.A("STOCK PRICE DATA ANALYSIS",href='https://finance.yahoo.com/', ## Website used for stocks
                   target="_blank")
           ]),
    html.H4('Select Ticker:'),
    dcc.Input(id='ticker', value='Select ticker', type='text'),
    html.H5('Selected date range should more than 16 months!'), # Handling error : for best results the user must select 16 months data-range to get model trained and predict the correct value.

    html.Div(id='Company Name'),
    html.H4('Select Starting Date:'),
    dcc.Input(id='Date',value='2017-01-01',type='text'),

    html.Div(id='Show_date',children='Please use YYYY-MM-DD Format for date'), ## Error handling for time
    html.H4('Select End Date:'),
    dcc.Input(id='Date_end',value='2021-11-29',type='text'),

    html.Div(id='show_end_date', children='Please use YYYY-MM-DD Format for date'), ## Error handling for  time
    html.Br(),
    html.H2('Select number of days to predict',style={'textAlign':'center'}),
    dcc.Slider('slider',min=2,max=100,value=10,marks={i:str(i)for i in range(2,101)}),
    html.H2('Select window for moving average or weighted moving average',style={'textAlign':'center'}),
    dcc.Slider('slider_ma',min=10,max=100,value=50,marks={i:str(i)for i in range(10,100)}),
    html.Br(),
    html.H3('Select Dropdown for descriptive graphs:'),
    dcc.Dropdown(   ## Dropdowns
        id='ma_wma',
        options=[
            {'label': 'Moving Average', 'value': 'MA'},
            {'label': 'Weighted Moving Average', 'value': 'WMA'},
            {'label': 'Linear Trend', 'value': 'LT'},
            {'label': 'MACD', 'value': 'MACD'}

        ],
        value='MA'
    ),
    html.Br(),
    html.Button('Predict/Plot selected graph and wait for 10 sec', id='Predict_button_State', n_clicks=0),
    html.Br(),
    html.H2('Graph for selected Stock',style={'textAlign':'center'}),
    dcc.Graph(id='Plot'),
    html.Br(),
    html.H2('Prediction for next selected days',style={'textAlign':'center'}),
    html.Div(id='rmse',children = '\t\t\t\tRMSE for validation: '),
    dcc.Graph(id='Plot_pred'),
    html.Br(),
    html.H2('Descriptive Statistics',style={'textAlign':'center'}),
    html.H4('Summary Statistics',style={'textAlign':'center'}),
    html.Div(id='mean',children = 'mean',style={'textAlign':'center'}),
    html.Div(id='median',children = 'median',style={'textAlign':'center'}),
    html.Div(id='mode',children = 'mode',style={'textAlign':'center'}),
    html.Div(id='std_dev',children = 'std dev',style={'textAlign':'center'}),
    dcc.Graph(id='Plot_ma'),

], style={'background-image': 'url(https://previews.123rf.com/images/poommy105/poommy1051802/poommy105180200028/95847357-abstract-financial-chart-with-uptrend-line-graph-candle-stick-graph-of-investment-trading-on-world-m.jpg)'})


#######################################################################################################################################################################################




''' Call back function '''

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
    Output(component_id='Plot_ma', component_property='figure'),
    Output(component_id='rmse', component_property='children'),
    Input('Predict_button_State','n_clicks'),
    State(component_id='ticker', component_property='value'),
    State(component_id='Date', component_property='value'),
    State(component_id='Date_end', component_property='value'),
    State(component_id='slider', component_property='value'),
    State(component_id='slider_ma', component_property='value'),
    State(component_id='ma_wma', component_property='value'),

)



###############################################################################################################################################################################


def check_ticker(button_click, input_text, input_date, input_end_date,no_of_days,ma_window,ma_wma): ## Input in parameters
    start_date = 1606520423 # Default start date
    end_date = 1638056423   # Default end date

    def chkDate(date_text):   ## Function for checking date and year month and date, x is variable to store date
        try:
            x = bool(datetime.datetime.strptime(date_text, "%Y-%m-%d"))
            return x
        except ValueError: ## Handling error for invalid date format
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
        start_date = 1420156740
        end_date = 1638056423

    if input_text in list_of_ticker:   ###  To check if the ticker is present in the list of tickers
        ticker_api = f'https://query1.finance.yahoo.com/v7/finance/download/{input_text}?period1={start_date}&period2={end_date}&interval=1d&events=history&includeAdjustedClose=true'
        df = pd.read_csv(ticker_api)
      ####  df.to_csv(f'{input_text}_Data.csv',index=False)   ## To save the user input date range into csv file
        stock_fig = px.line(x=df.Date, y=df.Close, )
        mn = stats(df,'Mean')
        md = stats(df,'Median')
        mdo = stats(df,'Mode')
        stdv= stats(df,'STD')
        desc_fig = get_desc(df, ma_window,ma_wma)
        #ma_fig = get_macd(df,26,12,9)
        s= StockMarket(df,timestep=100)
        predict_output, rmse= s.stockMarketPred(user_date='2020-01-31',number_of_days=no_of_days+1)
        print(no_of_days)
        list_predict = []
        for i in predict_output:
            list_predict.append(i[0])
        #predict_output = testfunc(df,start_date)
        return stock_fig, f'Ticker of stock selected {input_text} is valid', s_date,e_date, px.line(predict_output, title='Linear Trend Line'),f'Mean is : {mn}',f'Median is: {md}',f'Mode is : {mdo}',f'Standard deviation is : {stdv}',desc_fig,rmse
    else:
        return px.line(), f'Ticiker selected {input_text} is not a valid Ticker', s_date, e_date, px.line(),'Mean','Median','Mode','Standard deviation',px.line(), ''


if __name__ == '__main__':
    app.run_server(debug=True)  # Debug argument is true for updating the dashboard automatically



###############################################################################################################################################################################
