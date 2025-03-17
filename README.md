# Phase 5 Project : Time-Series 
# Predictive Modeling for The Income Fund of America (AMECX)
# Fund Price Forecasting
# Author : Jhonathan David H. Shaikh
# Date : August 21st, 2023

![awesome](https://www.inventiva.co.in/wp-content/uploads/2023/02/how-ai-and-ml-can-fight-against-money-laundering-in-investment-firms-1-780x470.webp)

### Business Understanding
This dataset represents a mutual fund facility called the The Income Fund of America, traded under the ticker name AMECX. In context, this fund is one representative from the more than 8,000 U.S. mutual funds available for investment in this asset class. The combined assets in US mutual funds was estimated $22.11 trillion approximately as of the end of 2022. but there is significant concentration of assets in a relatively small number of mutual fund families (with 50% of all assets held by Top 10 mutual fund families). Understanding mutual funds and having an idea on wether or not to invest in one can be a good advantage for the average or business investor. This analysis aims to help those individuals and provide a ground base for future analysis of other funds using ARIMA modeling.

### Data Understanding
Type of file : CSV

Source: Yahoo Finance

### Feature Engeneering and Technical Indicators
Columns in our dataframe are explained as below and represent technical indicator in our dataset:

Date : Index in our time series that specifies the date associated with the price. (USD)
Open Price: The first price of AMECX was purchased on the trading day (USD)
Close Price: The last price of AMECX was purchased at the end of trading day (USD)
High: The maximum price of AMECX was purchased on trading day (USD)
Low: The minimum price of AMECX was purchased on the trading day (USD)
Adjusted Closing Price: Stock exchanges witness buying and selling of millions of shares every minute. When the exchanges close, the last trading price of the stock is recorded as the closing price of the share (USD)
Volume: The sum of actual trades made during the trading day (USD)

### Forecasting Methodology

Autoregressive moving average model will be used in Time Series (TS). In statistics and mathematics, TS is a series of data points indexed in time order. A time series is a sequence take at successive equally spaced points in time.Thus, this project is aimed at two very targeted goal using Time Series Modeling :
Jupyter Notebook A brief note on this Jupyter notebook. This notebook presents story line following this structure:
* Part 1: Overview, Business Understanding & Objective 
* Part 2 : EDA
* Part 3 : Modeling
* Part 4 : Conclusion

### Business Modeling Objectives

1. Understand Fund  - including composition, set up , prices behavior and stats
2. Project Fund Growth withing one year

### Important Modeling Sucess Considerations

We are conducting a Time Series Analysis over the The Income Fund of America (AMECX):A data set that tracks a sample of investment products, prices over time. In particular, a time series allows one to see what factors influence certain variables from period to period. Time series analysis can be useful to see how a given asset, in this case Fund Price, or economic variable, in this case Price, changes over time.

## Map of Jupyter Notebook ( A Summary of What the Reader will find in Notebook)

**Set Up** - **Data Preparation for Time Series:**
 
 - Exploratory Data Analysis
 
 - Creating Time Series Ready Datasets: 
     - Dropping unwanted columns
     - Setting index
     - Creating Time Series Data 
     - Visualizations
      - Stats, Density plots,Histograms
     

**Phase 1** **- Decomposition & Stationarity Testing**
- Import of all relevant libraries
- Assesing Trends 
    - Components of Time Series
        - Data, Trends, Seasonality, Random
- Decomposition 
- Rolling Stats
- Visualizations 
- Stationarity Testing- Dickey-Fuller

**Phase 2** -**Differencing -Auto Correlation and Partial Auto Correlation**
- Differencing
- Auto-Regressive (ACF) Explanations
- Partial Auto(PACF) Explanations
- ACF Visualizations -pre differencing
- PACF Visualizations - predifferencing
- Differencing Explained
- Differencing Code Executed

**Phase 3** **-Models**
- Approach 1 Baseline Models
- Approach 2 Other Models
- AIC 
    -AIC explained
- Cross Validations
- RMSE
 - Model 1 AR
 - Model 2 MA
 - Model 3 ARIMA
 - Model 4 SARIMAX
 
**Phase 4** **- Forecasting**
- Forecasting

**Phase 5** - **Conclusion & Next Steps** 
- Conclusion
- Future Next Steps

## PHASE 1

### Assesing Trends 

Assesing trends is the next step to determine what I need to do with the Dataset and prepare it for modeling, I'll go into these details:
- Decomposition (Visualizing Seasonality, Trends, Noise)
- Rolling Mean
- Dickey Fuller Testing


#### Components of Time Series Trends
Time series is affected by four components. They can be separated from the observed data and in include : Trend, Seasonailty, Cyclical, and Irregular Components.

- **Trends**: The long term movement of the time series. For example, series relating to growth of stock, show upward trend


- **Seasonality**: Fluctuations in the data set that follow a regular pattern, cause by outside influences. For example, ice cream sales up in summer months.  


- **Cyclical**: When the data rises or falls at non fixed periods. For example, For example, business cycles, in stock, people selling their loosing stock in the year end might drive down always the price of stock.


- **Irregular**: Caused by unpredicatable differences. For example, a surprised announcement of a merger and acquisition might drive up or down the price of a stock. 


####  Decomposition of Time Series

Time series data can exhibit a variety of patterns, and it is often helpful to split a time series into several components, each representing an underlying pattern category.
These TS major components are : Seasonal component. Cyclical component. Irregular (noise) component. These components are important because they can affect the time lags or how the lags are shown.



![image](https://github.com/JhonathanDavid/phase5-repo/assets/102439898/45151c94-fbe7-4fa1-af01-7e8d3a062561)



#### Assesing Stationarity : Dickey-Fuller Test

A Dickey-Fullet Test, tests if the the data is stationary or not.
The null hypothesis of this test is that time series is not stationary. I'll asses with p.value <.05 , if that holds true the hypothesis is rejected and I say TS is stationary otherwise I'll fail to reject the null hypothesis and the data is not stationary. 


## PHASE 2- DIFFERENCING 

Differencing is a technique to transform a non-stationary time series into a stationary one. It involves subtracting the current value of the series from the previous one, or from a lagged value. It can be used to remove the series dependence on time like trends and seasonality. This is an important step in preparing the data used in ARIMA Modeling. To do this we can code a new plot showing the differencing applied. Let's also understand the sub-components of Auto Correlation and Partial Autocorrelation.


The value of time gap being considered and is called the lag. A lag 1 autocorrelation is the correlation between values that are one time period apart. More generally, a lag k autocorrelation is the correlation between values that are k time periods apart.


Now that I know that the data is not stationary, I'll do one of the most common methods of dealing with both trend and seasonality, called differencing.  In this technique, I take the difference of an observation at a particular time instant with that at the previous instant (also known as a "lag").


#### Coding for ACF and PACF  AutoCorrelation and Partial Autocorrelation

**ACF**
Autocorrelation is a measure of how much the data sets at one point in time influences data sets at a later point in time- ACF seeks to identify how correlated the values in a time series are with each other.
The ACF starts at a lag of 0, which is the correlation of the time series with itself and therefore results in a correlation of 1. The ACF plots the correlation coefficient against the lag, which is measured in terms of a number of periods or units. In essence, its a measure of the link between the present and the past, therefore it helps us identify the moving average.


**PACF**
Partial Autocorrelation (PACF) is a measure, that can plot the partial correlation coefficients between the series and lags of itself. In general, the "partial" correlation between two variables is the amount of correlation between them, which is not explained by their mutual correlations with a specified set of other variables.
In general, the "partial" correlation between two variables is the amount of correlation between them which is not explained by their mutual correlations with a specified set of other variables. PACF therefore helps us identify the Auto regressive order. PACF measures directs effects a.k.a Auto Regressive.

**WHAT IS THE DIFFERENCE BETWEEN ACF and PACF ?**

Partial autocorrelation function (PACF) gives the partial correlation of a stationary time series with its own lagged values, regressed the values of the time series at all shorter lags. It contrasts with the autocorrelation function, which does not control for other lags.

Both, ACF and PACF can provide valuable insights into the behaviour of time series data. They are often used to decide the number of Autoregressive (AR) and Moving Average (MA) lags for the ARIMA models. Moreover, they can also help detect any seasonality within the data. The correct application and interpretation are essential in extracting useful information from the ACF and PACF plots.

ACF and PACF can provide valuable insights into the behaviour of time series data. They are often used to decide the number of Autoregressive (AR) and Moving Average (MA) lags for the ARIMA models. Moreover, they can also help detect any seasonality within the data. The correct application and interpretation are essential in extracting useful information from the ACF and PACF plots.


![image](https://github.com/JhonathanDavid/phase5-repo/assets/102439898/b9c708b1-4792-442c-9781-b65eebde6f1a)



## PHASE 3 - MODELING

#### Train Test Split

I'll be conducting a train/test split to start modeling.
(Please see Jupyter Notebook for code)

### Base Model 
Naive model, train data shifted by 1 .
(Please see Jupyter noteboo for detailed code)

### Various models asses and tried
(Please see Jupyter Notebook for detailed code)

### SARIMAX

**SARIMAX** is an extension of the ARIMA class of models. ARIMA models compose 2 parts: the autoregressive term (AR) and the moving-average term (MA). AR views the value at one time just as a weighted sum of past values. The MA model takes that same value also as a weighted sum but of past residuals. Overall, ARIMA is a very good model. However, it cannot handle seasonality, thus SARIMAX is used in this model.

![image](https://github.com/JhonathanDavid/phase5-repo/assets/102439898/421d959e-b466-4950-91d3-634263a24709)


**Understanding Charts Above**

- Quantile Plots:
Commonly known as Q-Q Plots, It helps answer the question: "if the set of observations approximately normally distributed?". It is a plot of the quantiles of the first data set against the quantiles of the second data set (Sample vs. Theoritical in this case). Shows you how reliable predictions are within standard deviations. Our Mean Price, is fairly good at predictions within value.


- Histogram plus Estimated Density (KDE)
Undelying distribution for this data. Created bins for the data, and count the number of values creating a histogram. The KDE is the smooth out continous version of that data distribution. Allowing to estimate the probability density function. And the PDF, allows us to find the chances that the value of a random variable will occur within a range of values that you specify. More specifically, a PDF is a function where its integral for an interval provides the probability of a value occurring in that interval.


- Correlogram
A correlogram is a plot of autocorrelations . In time series data, looking at correlations between succesive correlations over time, that are periods apart (it can be 1 period or several periods apart)/For example a data group or point that you observe a month ago or a point you observed two months ago. The horizontal axis is the timeline. The blue shadows are the thresholds. The bars above the shadows are autocorrelations that are statistically significant it is not 0 and they are
It answers the question: 1) Is that Data Random? It is when not all points are above threshold. 2) Is there a trend in the data? There will be a trend, when the autocorrelations coeffiecient do not fall below the critical upper limit (upper limit) at any lag . If there is a trend the data is not stationary.

## PHASE 4- Forecast 

### Prediction Performant Model Recommended: SARIMAX

![image](https://github.com/JhonathanDavid/phase5-repo/assets/102439898/8e703e56-8d73-4657-9230-f0a52e4003ef)


### Conclusions

* Best Performant Model recommended SARIMAX Model 
* In the next year, the price will likely drop to 21 dollars in the next year. Oscillating between $ 21  - $ 22 dollars

## Summary
```
├── README.md                           <- The top-level README for reviewers of this project
├── Data                                <- Folder containing the dataset used for Modeling
├── Notebooks                           <- Folder containing pptx and other code
├── Nontechnical Presentation.pdf       <- PDF version of Business Power Point
├── index_p5.pdf                        <- PDF version of the Jupyter notebook, containing data preparation and modeling combined
├── index.ipynb                         <- Jupyter notebook version of ipynb project containing Exploratory Data Analysis & Data Preparation
└── Modeling.ipynb                      <- Jupyter notebook contains ARIMA modeling
```
