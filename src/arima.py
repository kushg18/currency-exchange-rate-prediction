import warnings

from matplotlib import pyplot
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMA

TRAINING_PERCENTAGE = 0.7
TESTING_PERCENTAGE = 1 - TRAINING_PERCENTAGE
NUMBER_OF_PREVIOUS_DATA_POINTS = 3
LENGTH_DATA_SET = 0
TRAINING_SET_LENGTH = 0
TESTING_SET_LENGTH = 0


def training_testing_buckets(raw_data, training_percentage, testing_percentage):
    global TRAINING_SET_LENGTH, TESTING_SET_LENGTH
    TRAINING_SET_LENGTH = int(LENGTH_DATA_SET * training_percentage)
    TESTING_SET_LENGTH = LENGTH_DATA_SET - TRAINING_SET_LENGTH
    training_set, testing_set = raw_data[0:TRAINING_SET_LENGTH], raw_data[TRAINING_SET_LENGTH:LENGTH_DATA_SET]
    return training_set, testing_set


def evaluate_performance_arima(testing_actual, testing_predict):
    return mean_squared_error(testing_actual, testing_predict)


def plot_arima(currency, testing_actual, testing_predict, file_name):
    actual = pyplot.plot(testing_actual, label="Actual data points", color="blue")
    testing = pyplot.plot(testing_predict, label="Testing prediction", color="green")

    pyplot.ylabel('currency values for 1 USD')
    pyplot.xlabel('number of days')
    pyplot.title('USD/' + currency + ' : actual vs predicted using ARIMA')

    pyplot.legend()
    # pyplot.show()
    pyplot.savefig(file_name)
    pyplot.clf()


def load_data_set(currency):
    data_set_frame = read_csv('currency_prediction_data_set.csv', header=0,
                              index_col=0, squeeze=True)
    column_headers = data_set_frame.columns.values.tolist()
    currency_index = column_headers.index('USD/' + currency.upper()) + 1

    data_file = read_csv("currency_prediction_data_set.csv", usecols=[currency_index], engine='python')
    # the type of data_file is a matrix, as returned by pandas
    raw_data = []  # need to convert a matrix values to a simple list of values
    for data_point in data_file.values.tolist():
        raw_data.append(data_point[0])
    global LENGTH_DATA_SET
    LENGTH_DATA_SET = len(raw_data)
    return raw_data


def build_model_predict_arima(training_set, testing_set):
    testing_predict = list()
    training_predict = list(training_set)
    for testing_set_index in range(TESTING_SET_LENGTH):
        arima = ARIMA(training_predict, order=(5, 1, 0))
        arima_model = arima.fit(disp=0)
        forecasting = arima_model.forecast()[0].tolist()[0]
        testing_predict.append(forecasting)
        training_predict.append(testing_set[testing_set_index])
        # print("Predicted = ", testing_predict[-1], "Expected = ", testing_set[testing_set_index])

    print('predicting...')
    print('\t The prediction for the next day:', arima_model.forecast()[0])
    # for future_day_i in range(7):
    #     # if future_day_i == 0 or future_day_i == 2 or future_day_i == 7:
    #     forecasting = arima_model.forecast()[0]
    #     print('day', future_day_i + 1, forecasting)
    #     training_predict.append(forecasting)
    #     arima = ARIMA(training_predict, order=(3, 1, 1))
    #     arima_model = arima.fit(disp=0)

    return testing_predict


def arima_model(currency):
    print('\nARIMA Model')

    print('loading the dataset...')
    raw_data = load_data_set(currency)

    print('splitting training and testing set...')
    training_actual_arima, testing_actual_arima = training_testing_buckets(raw_data, TRAINING_PERCENTAGE,
                                                                           TESTING_PERCENTAGE)

    print('building and training model...')
    testing_predict_arima = build_model_predict_arima(training_actual_arima, testing_actual_arima)

    print('evaluating performance...')
    mse_arima = evaluate_performance_arima(testing_actual_arima, testing_predict_arima)
    print('\t Testing Mean Square Error:', mse_arima)

    with open("mse_arima.txt", 'w') as mse_file:
        mse_file.write(str(mse_arima) + '\n')

    print('plotting the graph...')
    plot_arima(currency, testing_actual_arima, testing_predict_arima, "testing_prediction_arima.pdf")

    print('done...')
    return raw_data, testing_predict_arima


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    data_set_frame = read_csv('currency_prediction_data_set.csv', header=0,
                              index_col=0, squeeze=True)
    column_headers = str([cur[4:] for cur in data_set_frame.columns.values.tolist()])
    currency = input('Enter any one of ' + column_headers + ' currencies\n').strip()
    arima_model(currency)  # setting the entry point
