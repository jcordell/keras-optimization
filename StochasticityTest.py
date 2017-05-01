from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import sys
import numpy as np
from sklearn.metrics import mean_squared_error
import data_parser

def getData():
    data = data_parser.parse("DBTT_Data22.csv")
    data_lwr = data_parser.parse("CD_LWR_clean8.csv")
    X = ["N_log(eff fl p =.05)", "N_log(eff fl p =.4)", "N_log(eff fl p =.5)", "N(Cu)", "N(Ni)", "N(Mn)", "N(P)", "N(Si)","N( C )", "N_log(eff fl p =.1)", "N_log(eff fl p =.2)", "N_log(eff fl p =.3)",  "N(Temp)"]
    Y = "CD delta sigma"
    data.set_x_features(X)
    data.set_y_feature(Y)
    data_lwr.set_y_feature(Y)
    data_lwr.set_x_features(X)
    data.add_exclusive_filter("Alloy",'=', 29)
    data.add_exclusive_filter("Alloy",'=', 8)
    data.add_exclusive_filter("Alloy", '=', 1)
    data.add_exclusive_filter("Alloy", '=', 2)
    data.add_exclusive_filter("Alloy", '=', 14)

    data_lwr.add_exclusive_filter("Alloy",'=', 29)
    data_lwr.add_exclusive_filter("Alloy", '=', 14)
    x_test = np.array(data_lwr.get_x_data())
    y_test = np.array(data_lwr.get_y_data())
    x_train = np.array(data.get_x_data())
    y_train = np.array(data.get_y_data())
    print("Training with", np.shape(y_train)[0], "data points")
    return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = getData()

# normalize y column
y_train = y_train/758.92
# gridsearch parameters
learning_rate = [.005, .05]
hidden1 = [5]
hidden2 = [5, 40]

num_ensembles = 2

# intializing rms_list
columns = 3
rows = len(learning_rate) * len(hidden1) * len(hidden2)
rms_list = [[0 for i in range(rows)] for j in range(columns)]

for i in range(0, columns):
    j = 0
    for lr in learning_rate:
        for h1 in hidden1:
            for h2 in hidden2:
                for n in range(num_ensembles):
                    from keras.models import Sequential
                    from keras.layers.core import Dense, Dropout, Activation
                    from keras.optimizers import Adadelta, Adam, rmsprop

                    model = Sequential()
                    model.add(Dense(units=h1, input_dim=np.shape(x_train)[1]))
                    model.add(Activation('sigmoid'))

                    model.add(Dense(h2))
                    model.add(Activation('sigmoid'))

                    model.add(Dense(1))
                    model.add(Activation('linear'))

                    ADAM = Adam(lr=lr)
                    model.compile(loss='mean_squared_error', metrics=['accuracy'], optimizer=ADAM)
                    # model.compile(loss='binary_crossentropy', optimizer=params['optimizer'])

                    model.fit(x_train, y_train, nb_epoch=2000, batch_size=1400, verbose=0)
                    y_predict = model.predict(x_test) * 758.92  # todo way to not hardcode this?

                    if(n == 0):
                        y_predict_all = y_predict
                    else:
                        np.column_stack((y_predict, y_predict_all))

                # find average prediction
                y_predict_avg = np.mean(y_predict_all, axis = 1)
                rms = np.sqrt(mean_squared_error(y_test, y_predict))
                print('RMS:', rms)
                rms_list[i][j] = rms
                j+=1

print(rms_list)
print(np.shape(rms_list)[0])
rank_final = []
for i in range(np.shape(rms_list)[0]):
    temp = np.array(rms_list[i]).argsort()
    ranks = np.empty(len(rms_list[i]), int)
    ranks[temp] = np.arange(len(rms_list[i]))
    rank_final.append(ranks)
print("Average rank:")
print(np.mean(rank_final, axis=0))
print(" ")
print("StDev:")
print(np.std(rank_final, axis=0))
print(" ")
print("Average StDev:")
print(np.mean(np.std(rank_final, axis=0)))
