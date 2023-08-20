# Phase 5 Project : Time-Series DataSet
# Predictive Modeling 
# Fund Price Forecasting

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

Autoregressive moving average model will be used in Time Series (TS). In statistics and mathematics, TS is a series of data points indexed in time order. A time series is a sequence take at successive equally spaced points in time.This project will look into a Real Estate database to find valuable information about the value of homes and return on investment. My company, Xabios data international, has been hired to determind the change of prices in this market over time and to determine which areas have the most potential to increase. Thus, this project is aimed at two very targeted goal using Time Series Modeling :
Jupyter Notebook A brief note on this Jupyter notebook. This notebook presents story line following this structure:
* Part 1: Overview, Business Understanding & Objective 
* Part 2 : EDA
* Part 3 : Modeling
* Part 4 : Conclusion

### Business Modeling Objectives

1. Understand Fund  - how they have changed overtime 
2. Project Fund Growth withing one year


### Business Understanding & Business Problem
As important to Telecommunications companies as it is to earn new customers to their services, is the need to retain current customers and preventing them from going to competitors. The act of leaving customers or loss of customers are referred to in this industry as "Churn", from a business perspective. The business would like to understand exactly what factors or situations contribute to churning, and most importantly, post identification of those factors, define precises strategic initiatives to retain customers. This project, therefore aims to help the business first in identifying the attributes and factors that cause the churn, and in turn, building models that can help the business predict it, so that in fact strategic business initiatives can be outlined for solution.


### Important Modeling Sucess Considerations

We are conducting a Time Series Analysis over the Zillow Home Value Index (ZHVI):A data set that tracks a sample of real estate prices over time. In particular, a time series allows one to see what factors influence certain variables from period to period. Time series analysis can be useful to see how a given asset, in this case Real Estate, or economic variable, in this case ROI, changes over time.

### Conclusions
Modeling derived results to invest in these ZIP Codes due to Higher ROI
Highest projected ROI:
* 6069 = 24.12%
* 6610 = 31%
* 6330 = 21%
* 6039 = 19%
* 6058 = 19%.

## Summary
```
├── README.md                      <- The top-level README for reviewers of this project
├── Folder                         <- Folder containing 2 ipynbs one for EDA and second for Arima Modeling
├── PDF1_Nontechnical Presentation <- PDF version of Business Power Point 
├── PDF2_Jupyter Notebook.pdf      <- PDF version of ipynb project containing EDA and Arima modeling
└── time series folder             <- contains initial instructions
```
