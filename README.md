Currency Exchange Rate Prediction System
========================================

### About ###
-----------------------------
This project is about creating machine learning models to predict the currency exchange rate between two countries. The motivation behind this project is to identify which day in future would allow the transaction to cost as minimal as possible.

### Technology Stack ### 
-----------------------------
1. Python
2. PyCharm IDE
3. Machine Learning Packages
4. Microsoft Excel

### Steps To Use ### 
-----------------------------
1. Make sure you have following files in the same folder,
    - arima.py [first model]
	- rnn.py [seconde model]
	- currency_prediction.py [Main driver file]
	- currency_prediction_data_set.csv [Data set]
2. Make sure the following dependencies are installed in your machine,
    - python 3.5
	- statsmodels
    - sklearn
	- pandas
	- matplotlib
	- keras
	- numpy
3. Open the terminal to the location where files described in the step 1 are present (using cd command).
4. Type the following command in the terminal
	- python3 currency_prediction.py
then enter the currency for which you want to predict the exchange rate from the listed currencies.

Note: 
1. The prediction would be for 2/18/2017 as we have collected data until 2/17/2017.
2. The predicted value/s would be on the console. The plotted graphs and corresponding mean square errors would be saved in a pdf and txt files respectively. 

### References ###
-----------------------------

Python building predictive models
- https://www.analyticsvidhya.com/blog/2015/09/build-predictive-model-10-minutes-python/

Review and description:
- http://www.investopedia.com/articles/forex/11/4-ways-to-forecast-exchange-rates.asp

Data:
- http://investexcel.net/automatically-download-historical-forex-data-into-excel/

ARIMA Model using python: 
- http://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/
- https://www.analyticsvidhya.com/blog/2016/02/time-series-forecasting-codes-python/

Paper for using ANN to perform prediction:
- https://pdfs.semanticscholar.org/88d3/dfe825c07dec8362509d0dfc64fed2a43d28.pdf
