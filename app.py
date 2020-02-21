import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Output, Input, State
import plotly.graph_objs as go
import datetime as dt
import pandas as pd
import numpy as np
from pmprophet.model import PMProphet, Sampler
import requests
from newsapi import NewsApiClient
import json



urlConfirmed = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv"
urlRecovered = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Recovered.csv"
urlDeceased = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Deaths.csv"




def getData(url):
    '''Get the recovered numbers from Johns Hopkins.
    Input:
    ======
        url: string with url address of raw data
    Output:
    =======
    recovered:
        pandas dataframe
        columns = ds, Mainland China, Outside Mainland China
    '''
    # data time series
    df = pd.read_csv(url)
    # groupby country/region
    df = df.groupby(["Country/Region"]).sum()
    # drop latitude and longitude
    df.drop(['Lat', 'Long'], axis=1, inplace=True)
    # take transpose so that the columns correspond to country/region
    df = df.T
    # convert index to datetime
    df.index = pd.to_datetime(df.index)
    # convert datetime index to column
    df['ds']=df.index
    df.reset_index(inplace=True)
    # Mainland China and its complement
    df["Outside Mainland China"] = df.sum(axis=1) - df["Mainland China"]
    return df[["ds", "Mainland China", "Outside Mainland China"]]


def getTotal(url):
    '''Get latest figures of confirmed cases, recovered and deceased.
    Input:
    ======
        url: string url of raw data
    Output:
    =======
        totalConfirmed, totalRecovered, totalDeceased: int, int, int
    '''
    df = getData(url)
    total = df['Mainland China'].iloc[-1] + df['Outside Mainland China'].iloc[-1]
    return total


def generatePredictions(df, N=7):
    '''Generate forecast using Generalized Additive Models (GAM) which generalizes FBProphet.
    Input:
    ======
        df: dataframe
            columns:
                ds: date
                y:  observed data points (typically log of values in order to model a multiplicative process)
        N:  int
            number of days of prediction into the future (default value 7)
    Output:
    =======
        ddf: dataframe
            columns:
                ds:     date
                y_hat:  predicted value
                y_low:  predicted low
                y_high: predicted high
    '''
    # GAM
    m = PMProphet(df, growth=True, intercept=True, n_changepoints=25, changepoints_prior_scale=.01, name='model')
    # Fit the model (using NUTS)
    m.fit(method=Sampler.NUTS, draws=2500)
    # make N predictions into the future
    ddf = m.predict(N, alpha=0.2, include_history=True, plot=False)
    return ddf




def getNews():
    '''Get the latest news from https://newsapi.org/.'''
    # headers
    headers = {'Authorization': '<Enter News API token here.>'}

    # news articles
    everythingNewsUrl = 'https://newsapi.org/v2/everything'

    # payloads
    everythingPayload = {'q': 'coronavirus', 'language': 'en', 'sortBy': 'relevancy'}

    try:
        # fetch every news article
        response = requests.get(url=everythingNewsUrl, headers=headers, params=everythingPayload)

        # info only about articles
        articlesList = response.json()['articles']

        # convert articles list to json string and then convert json string to pandas dataframe
        df = pd.read_json(json.dumps(articlesList))

        # choose the title and the url
        df = pd.DataFrame(df[['title', 'url']])
        return df
    except:
        print("**News did not load properly! Please verify the News API authentication.**")
        return None

def generateNewsTable(max=10):
    '''Generate news HTML table with a maximum of 10 entries.'''
    df = getNews()

    return html.Div(
        [
            html.Div(
                html.Table(
                    # Header
                    [html.Tr([html.Th()])]
                    +
                    # Body
                    [
                        html.Tr(
                            [
                                html.Td(
                                    html.A(
                                        df.iloc[i]['title'],
                                        href=df.iloc[i]['url'],
                                        target='_blank'
                                    )
                                )
                            ]
                        )
                        for i in range(min(len(df),max))
                    ]
                ),
                style={'height': '300px', 'overflowY': 'scroll'},
            ),
        ],
        style={'height': '100%'},)

def generateLatestTable():
    '''Generate table with the latest updated total numbers.'''
    con = getTotal(urlConfirmed)
    rec = getTotal(urlRecovered)
    dec = getTotal(urlDeceased)
    df = pd.DataFrame({'Confirmed' : str(con), 'Recovered' : str(rec), 'Deceased': str(dec), 'Recovery Rate': str(round(100*rec/(rec+dec), 2))}, index=[0])
    return dash_table.DataTable(
        id='table',
        columns=[{"name": str(i), "id": str(i)} for i in df.columns],
        style_header={'backgroundColor': 'rgb(30, 30, 30)',
                        'color':'#f7370E',
                        },
        style_cell={
         'textAlign': 'center',
         'backgroundColor': 'rgb(50, 50, 50)',
         'color': 'white',
         'max_width':'50%'
    },
        data=df.to_dict('records'),)


def plotData(url, mainland_china=True):
    '''Generate figure from data.'''
    df = getData(url)
    if mainland_china:
        df['y']=df['Mainland China']
        title = 'Confirmed Cases in Mainland China'
    else:
        df['y']=df['Outside Mainland China']
        title = 'Confirmed Cases Outside of Mainland China'
    trace_observed = go.Scatter(
        x=df['ds'],
        y=df['y'],
        name = "Confirmed Cases",
        line = dict(color = '#17BECF'),
        opacity = 0.8)


    data = [trace_observed]
    layout= {
            'title': title
        }
    fig = dict(data=data, layout=layout)
    return fig


def plotPrediction(url, mainland_china=True):
    '''Generate figure for log of prediction.'''
    df = getData(url)
    if mainland_china:
        # note taking the log of the data!
        df['y']=np.log(df['Mainland China'])
        title = 'Forecast (Mainland China)'
    else:
        df['y']=np.log(df['Outside Mainland China'])
        title = 'Forecast (Outside of Mainland China)'
    ddf = df[['ds','y']]
    ddf = generatePredictions(ddf)
    trace_observed = go.Scatter(
        x=df['ds'],
        y=df['y'],
        name = "Log of Confirmed Cases",
        line = dict(color = 'blue'),
        opacity = 0.9)
    trace_predicted = go.Scatter(
        x=ddf['ds'],
        y=ddf['y_hat'],
        name = "Log of Predicted Cases",
        line = dict(color = 'dimgray'),
        opacity = 0.8)

    trace_high = go.Scatter(
            x=ddf['ds'],
            y=ddf['y_high'],
            name = "Log of Predicted High",
            line = dict(color = 'dimgray'),
            opacity = 0.4)

    trace_low = go.Scatter(
        x=ddf['ds'],
        y=ddf['y_low'],
        name = "Log of Predicted Low",
        line = dict(color = 'dimgray'),
        opacity = 0.4)


    data = [trace_observed, trace_predicted, trace_high, trace_low]
    layout= {
            'title': title
        }
    fig = dict(data=data, layout=layout)
    return fig





# Dash app with external css

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# app layout

colors = {
    'background': '#041C7C',
    'text': '#7FDBFF',
    'main': '#f7370E'
}



app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
    html.H1(
        children='COVID-19 Update',
        style={
            'textAlign': 'center',
            'color': colors['main']
        }
    ),
    html.H6("Last Updated on "+ dt.datetime.now().strftime("%d/%m/%Y, %H:%M"),
        style={
            'textAlign': 'center',
            'color': 'dimgray'
        }
    ),
        html.Div([

                    html.Div([
                        dcc.Graph(id='my-graph_china', figure=plotData(urlConfirmed)),
                        dcc.Graph(id='my-graph_predict_china', figure=plotPrediction(urlConfirmed))

                        ], className="four columns"),

                    html.Div([
                        dcc.Graph(id='my-graph_outside', figure=plotData(urlConfirmed, mainland_china=False)),
                        dcc.Graph(id='my-graph_predict_outside_china', figure=plotPrediction(urlConfirmed, mainland_china=False))

                        ], className="four columns"),


                    html.Div([
                        html.H2(children='Data',  style={'color':colors['text']}), generateLatestTable(),
                        html.Small("(source: Johns Hopkins)", style={'color':'dimgray'}),
                        html.Br(),
                        html.H2(children = 'News', style={'color':colors['text']} ),
                        generateNewsTable(),
                        html.Small("(source: newsapi.org)", style={'color':'dimgray'}),
                        ], className='four columns'),



                    ],className="row")]



)



if __name__=="__main__":
    app.run_server(debug=False, port=3000)
