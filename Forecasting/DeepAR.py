import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mxnet as mx

from gluonts.dataset.common import ListDataset
#from gluonts.mx.distribution.gaussian import GaussianOutput
from gluonts.model.deepar import DeepAREstimator
from gluonts.mx.trainer import Trainer
from itertools import islice
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation.backtest import backtest_metrics
mx.random.seed(0)
np.random.seed(0)

def create_dataset(file_path):
    df = pd.read_csv(file_path, header=0, index_col=0)
    df = df.set_index('Time')
    #print(df.shape)
    df['Diff'] = np.log((df['Open'] / df['Close']))
    #df['Diff'] = df['Close'].sub(df['Open'],axis =0)
    #print(df.shape)
    return df


def prepare(df, P, frac, ep):
    rolling_test = []
    train_size = int(frac * df.shape[0])
    i = 0
    delay = 0
    train_ds = ListDataset([{"start": pd.Timestamp(df.index[0]), "target": df.Diff[0:train_size - P],
                             'feat_dynamic_real': [df.fear[0:train_size - P],
                                                   df.anger[0:train_size - P],
                                                   df.anticipation[0:train_size - P],
                                                   df.trust[0:train_size - P],
                                                   df.suprise[0:train_size - P],
                                                   df.positive[0:train_size - P],
                                                   df.negative[0:train_size - P],
                                                   df.sadness[0:train_size - P],
                                                   df.disgust[0:train_size - P],
                                                   df.joy[0:train_size - P],
                                                   df.Volume_of_tweets[0:train_size - P],
                                                   df.Retweet[0:train_size - P],
                                                   df.Replies[0:train_size - P],
                                                   df.Likes[0:train_size - P]
                                                   ]}], freq='1B')
    while train_size + delay < df.shape[0]:
        delay = int(P) * i
        test_ds = ListDataset([dict(start=pd.Timestamp(df.index[0]), target=df.Diff[0:train_size + delay],
                                feat_dynamic_real=[df.fear[0:train_size + delay],
                                                   df.anger[0:train_size + delay],
                                                   df.anticipation[0:train_size + delay],
                                                   df.trust[0:train_size + delay],
                                                   df.suprise[0:train_size + delay],
                                                   df.positive[0:train_size + delay],
                                                   df.negative[0:train_size + delay],
                                                   df.sadness[0:train_size + delay],
                                                   df.disgust[0:train_size + delay],
                                                   df.joy[0:train_size + delay],
                                                   df.Volume_of_tweets[0:train_size + delay],
                                                   df.Retweet[0:train_size + delay],
                                                   df.Replies[0:train_size + delay],
                                                   df.Likes[0:train_size + delay]
                                                   ])], freq='1B')
        i += 1
        rolling_test.append(test_ds)

    print("We have 1 training set of", train_size, "days and then ",len(rolling_test),"testing sets of ",delay," days total")
    estimator = DeepAREstimator(prediction_length=P, context_length=5, freq='1B',use_feat_dynamic_real=True,
                                trainer=Trainer(ctx="cpu", epochs=ep,)) #hybridize=False, ), )
    return train_ds, rolling_test, estimator, train_size, i

def evaluate(predictor, rolling_test,train_ds):
    for i, test_ds in enumerate(rolling_test):
        predict_stock, real_stock = evaluate_forecast(test_ds, predictor,train_ds)
        if i == 0:
            predicted_change_stock = predict_stock
        else:
            predicted_change_stock = predicted_change_stock.append(predict_stock)
    prediction = predicted_change_stock.to_frame()
    return prediction, real_stock

def evaluate_forecast(test_ds, predictor,train_ds):
    forecast_it, ts_it = make_evaluation_predictions(test_ds, predictor=predictor, num_samples=100)
    forecasts = list(forecast_it)
    tss = list(ts_it)
    for target, forecast in islice(zip(tss, forecasts), 1):
        return forecast.mean_ts, target


def modified_plot(prediction, true_val, pred_len, train_end):
    fig2 =plt.figure()
    ax_list = []
    widths = [8]
    heights_2 = [10, 10]
    spec2 = fig2.add_gridspec(ncols =1, nrows=len(heights_2), width_ratios = widths, height_ratios = heights_2)
    for row in range(len(heights_2)):
        ax_list.append(fig2.add_subplot(spec2[row, 0]))
    # First PLOT: Training and Test Set seperated by a red line, test set has the predictions overlayed
    true_val.plot(color='b', ax=ax_list[0])
    prediction.plot(color='g', ax=ax_list[0])
    ax_list[0].axvline(true_val.index[train_end - pred_len - 1], color='r')  # end of train dataset
    ax_list[0].grid(which="both")  # grid lines
    ax_list[0].legend(["Change in Stock", 'Predicted Change in stock'], loc="upper left")
    # Second Plot: zoomed in version of the test set from the first plot
    true_val[train_end - pred_len - 1:].plot(color='b', ax=ax_list[1])
    prediction.plot(color='r', linestyle='-', ax=ax_list[1])
    ax_list[1].legend(["Truth", 'Prediction', 'Adv Prediction'], loc="upper left")
    ax_list[1].grid(which="both")
    plt.show()



P = 3
frac = 0.8
epochs = 200
num_features = 14
normalize = 'normalize'
file_path = '/Users/gabriel/PycharmProjects/Finance/Data/5B- grouped final results no ML/FB_1d_final_results_no_ML.csv'


df = create_dataset(file_path)
train_ds, rolling_test, estimator, train_end, i = prepare(df, P, frac, epochs)
train_output = estimator.train_model(train_ds,num_workers = None)
print('--------- done -----------')
predictor = train_output.predictor
print('------- wow-------')
prediction, true_val = evaluate(predictor, rolling_test, train_ds)
#return prediction, true_val, metrics, gradients, rolling_test, train_end,i, predictor


#prediction, true_val, metrics, gradients, rolling_test, train_end,i, train_output = train_normally(P,frac, epochs, file_path,normalize)
modified_plot(prediction, true_val, P, train_end)
# Printing Metrics
