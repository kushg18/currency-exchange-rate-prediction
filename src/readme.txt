Follow the below steps in order to run the codes.

1. make sure you have following files in the same folder,
	arima.py [first model]
	rnn.py [seconde model]
	currency_prediction.py [Main driver file]
	currency_prediction_data_set.csv [Data set]

2. make sure the following dependencies are installed in your machine,
	python 3.5
	statsmodels
	sklearn
	pandas
	matplotlib
	keras
	numpy

3. Open the terminal to the location where files described in the step 1 are present (using cd command)

4. type the following command in the terminal
	python3 currency_prediction.py
then enter the currency for which you want to predict the exchange rate from the listed currencies.

Note: 
1. The prediction would be for 2/18/2017 as we have collected data until 2/17/2017.
2. The predicted value/s would be on the console. The plotted graphs and corresponding mean square errors would be saved in a pdf and txt files respectively. 