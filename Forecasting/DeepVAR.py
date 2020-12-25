import pandas as pd
import numpy as np
import mxnet as mx
import matplotlib.pyplot as plt
from gluonts.dataset.common import ListDataset
from gluonts.model.deepvar import DeepVAREstimator
from gluonts.model.gpvar import GPVAREstimator
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.mx.trainer import Trainer
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.dataset.field_names import FieldName
from itertools import islice
from gluonts.evaluation.backtest import backtest_metrics
np.random.seed(123)
mx.random.seed(123)

def create_dataset(file_path):
    df = pd.read_csv(file_path, header=0, index_col=0)
    df = df.set_index('Time')
    df['Diff'] = np.log((df['Open']/df['Close']))
    #df['Diff'] = df['Close'].sub(df['Open'], axis=0)
    df = df[['Diff',"fear","anger","anticipation","trust","suprise","positive","negative",
             "sadness","disgust","joy","Volume_of_tweets","Retweet","Replies","Likes"]]
    values = df.values
    values = values.astype('float32')
    target = np.transpose(values) #(5,433)
    return target, df


def train(file_path, P,frac):
    target, df = create_dataset(file_path)
    i = 0
    rolling_test = []
    train_size = int(frac * df.shape[0])
    starts = [pd.Timestamp(df.index[0]) for _ in range(len(target))]
    delay = 0
    grouper_train = MultivariateGrouper(max_target_dim=df.shape[0])
    grouper_test = MultivariateGrouper(max_target_dim=df.shape[0])

    train_ds = ListDataset([{FieldName.TARGET: targets, FieldName.START: start}
                            for (targets, start) in zip(target[:, 0:train_size - P], starts)],freq='1B')
    train_ds = grouper_train(train_ds)

    while train_size + delay< df.shape[0]:
        delay = int(P) * i
        test_ds = ListDataset([{FieldName.TARGET: targets, FieldName.START: start}
                               for (targets, start) in zip(target[:, 0:train_size+delay], starts)],
                              freq='1B')
        test_ds = grouper_test(test_ds)
        rolling_test.append(test_ds)
        i+=1
    estimator = GPVAREstimator(prediction_length=pred_len, context_length=6, freq='1B', target_dim=df.shape[1],
        trainer=Trainer(ctx="cpu", epochs=200))
    return train_ds, rolling_test, estimator,train_size


def plot_forecasts(test_ds, predictor):
    forecast_it, ts_it = make_evaluation_predictions(test_ds, predictor=predictor, num_samples=100)
    forecasts = list(forecast_it)
    tss = list(ts_it)
    for target, forecast in islice(zip(tss, forecasts), 1):
        return forecast.copy_dim(0).mean_ts, target[0]

pred_len = 3
frac = 0.8
file_path = '/Users/gabriel/PycharmProjects/Finance/Data/5B- grouped final results no ML/FB_1d_final_results_no_ML.csv'
train_ds, rolling_test, estimator,train_size = train(file_path, pred_len, frac)
train_output = estimator.train_model(train_ds,num_workers = None)
predictor = train_output.predictor


for i, test_set in enumerate(rolling_test):
    predict_stock, real_stock = plot_forecasts(test_set, predictor)
    if i == 0:
        predicted_change_stock = predict_stock
    else:
        predicted_change_stock = predicted_change_stock.append(predict_stock)
prediction = predicted_change_stock.to_frame()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
fig, ax = plt.subplots(2, 1,figsize=(10, 7))
#First PLOT: Training and Test Set seperated by a red line, test set has the predictions overlayed
real_stock.plot(color = 'b', ax = ax[0])
prediction.plot(color = 'g', ax = ax[0])
ax[0].axvline(real_stock.index[train_size-pred_len-1], color='r') # end of train dataset
ax[0].grid(which="both") #grid lines
ax[0].legend(["Change in Stock", 'Predicted Change in stock'], loc="upper left")
#second Plot: zoomed in version of the test set from the first plot
real_stock[train_size:].plot(color = 'b', ax = ax[1])
prediction.plot(color = 'r', ax = ax[1])
ax[1].grid(which="both")
plt.show()