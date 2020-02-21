# COVID-19-Forecast

COVID-19-Forecast is a Dash application written in Python that summarizes the latest coronavirus data and news and displays a time series forecast of new cases using Generalized Additive Models (GAM). 

The source of the data is Johns Hopkins (https://github.com/CSSEGISandData/COVID-19/tree/master/csse_covid_19_data), and the source of the news is https://newsapi.org.

It might take few minutes to load since the forecasting is done using a non-parametric model. It only works using Python 3, and the user needs to specify the News API token in the app file. 

![Image description](https://github.com/ws1001/COVID-19-Forecast/blob/master/covid-19.png)
