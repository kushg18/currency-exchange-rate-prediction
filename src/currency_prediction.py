import warnings

from arima import *
from matplotlib import pyplot
from pandas import read_csv
from rnn import *

TRAINING_PERCENTAGE = 0.7
TESTING_PERCENTAGE = 1 - TRAINING_PERCENTAGE
NUMBER_OF_PREVIOUS_DATA_POINTS = 3
LENGTH_DATA_SET = 0
TRAINING_SET_LENGTH = 0
TESTING_SET_LENGTH = 0


def plot(currency, raw_data, training_predict, testing_predict, testing_predict_arima, file_name):
    global LENGTH_DATA_SET
    LENGTH_DATA_SET = len(raw_data)
    training_data_trend = [None] * LENGTH_DATA_SET
    testing_data_trend = [None] * LENGTH_DATA_SET

    training_data_trend[NUMBER_OF_PREVIOUS_DATA_POINTS:len(training_predict) + NUMBER_OF_PREVIOUS_DATA_POINTS] = \
        list(training_predict[:, 0])
    testing_data_trend[NUMBER_OF_PREVIOUS_DATA_POINTS - 1:len(training_predict) + NUMBER_OF_PREVIOUS_DATA_POINTS] = \
        list(testing_predict[:, 0])

    actual = pyplot.plot(raw_data[int(TRAINING_PERCENTAGE * LENGTH_DATA_SET):], label="Actual data points",
                         color="blue")
    testing_rnn = pyplot.plot(testing_data_trend, label="Testing prediction RNN", color="red")
    testing_arima = pyplot.plot(testing_predict_arima, label="Testing prediction ARIMA", color="green")

    pyplot.ylabel('currency values for 1 USD')
    pyplot.xlabel('number of days')
    pyplot.title('USD/' + currency + ' : actual vs predicted')

    pyplot.legend()
    # pyplot.show()
    pyplot.savefig(file_name)
    pyplot.clf()


def main():
    data_set_frame = read_csv('currency_prediction_data_set.csv', header=0,
                              index_col=0, squeeze=True)
    column_headers = str([cur[4:] for cur in data_set_frame.columns.values.tolist()])
    currency = input('Enter any one of ' + column_headers + ' currencies \n').strip()

    raw_data, testing_predict_arima = arima_model(currency)
    training_predict, testing_predict = rnn_model(currency)
    print('Plotting combined graph of both the models...')
    plot(currency, raw_data, training_predict, testing_predict, testing_predict_arima,
         "testing_prediction_arima_and_rnn.pdf")


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    main()  # setting the entry point
