import os
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from tensorflow.keras.layers import LSTM, Dense, Dropout, TimeDistributed, Conv1D, MaxPooling1D, Flatten
from readInput import prepare_dataset, load_files_heracleia, load_files_object_size
from utils import createDir

conv = False  # True if using 1D conv layers for the input - conv filters through timesteps.
summary = False  # print models summary

train_timesteps = 3
test_timesteps = 3

fold = 5

n_length = 1
filters_num = 20
kernel_size = 2
pooling_size = 2
rnn_hidden_size = 200
dense_hidden_size = 100

verbose, epochs, batch_size = 0, 300, 200


# Definition of the LSTM Model model
def create_model(n_outputs, n_features, n_length):
    model = tf.keras.Sequential()
    if conv:
        model.add(
            TimeDistributed(Conv1D(filters=filters_num,
                                   kernel_size=kernel_size, activation='relu'),
                            input_shape=(None, n_length, n_features)))
        model.add(TimeDistributed(MaxPooling1D(pool_size=pooling_size)))
        model.add(TimeDistributed(Conv1D(filters=filters_num, kernel_size=kernel_size, activation='relu')))
        # model.add(TimeDistributed(Conv1D(filters=filters_num, kernel_size=kernel_size, activation='relu')))
        model.add(TimeDistributed(Dropout(0.5)))
        model.add(TimeDistributed(Flatten()))
        model.add(LSTM(rnn_hidden_size))
    else:
        model.add(LSTM(rnn_hidden_size, return_sequences=True, input_shape=(None, n_features)))
        model.add(LSTM(rnn_hidden_size))
    model.add(Dropout(0.5))
    model.add(Dense(dense_hidden_size, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def train_model(train_x, train_y, checkpoint_path=None):
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
    train_x = train_x[:, :train_timesteps, :]
    # reshape data into time steps of sub-sequences
    if conv:
        n_steps = train_x.shape[1] / n_length
        train_x = train_x.reshape((train_x.shape[0], int(n_steps), n_length, train_x.shape[2]))
        # testX = test_x.reshape((test_x.shape[0], int(n_steps), n_length, trainX.shape[2]))

    model = create_model(n_outputs, n_features, n_length)
    # Display the model's architecture
    if summary:
        model.summary()

    if os.path.exists(checkpoint_path + ".index"):
        model.load_weights(checkpoint_path)
        print("Previous Model Loaded")
    else:
        createDir(checkpoint_path)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=verbose)
    # fit network
    model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks=[cp_callback],
              validation_split=1 / fold)

    return model


#  evaluate a model
def evaluate_model(model, test_x, test_y):
    if conv:
        n_steps = test_x.shape[1] / n_length
        test_x = test_x.reshape((test_x.shape[0], int(n_steps), n_length, test_x.shape[2]))
    # evaluate model
    # eval_result = model.evaluate(test_x[0][:30].reshape(test_x.shape[0], 30, test_x.shape[2], test_x.shape[3]), test_y, batch_size=1, verbose=0)
    eval_result = model.evaluate(test_x, test_y,
                                 batch_size=5, verbose=0)
    # print(eval_result)
    return eval_result[1]


def summarize_results(scores):
    print(scores)
    m, s = np.mean(scores), np.std(scores)
    print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))


# run an experiment
def run_experiment():
    # load data
    scores = list()
    x_data, labels = load_files_object_size()

    # one hot encode y
    labels = tf.keras.utils.to_categorical(labels)

    # shuffle dataset
    x_data, labels = shuffle(x_data, labels)

    # split data to train and test
    trainX, test_X = x_data[:int(x_data.shape[0] * 0.9)], x_data[int(x_data.shape[0] * 0.9):]
    trainy, test_y = labels[:int(x_data.shape[0] * 0.9)], labels[int(x_data.shape[0] * 0.9):]

    # create dir of models if not existent
    if not os.path.exists("models/"):
        createDir("models/")

    # train 10 different models
    for r in range(10):
        checkpoint_path = "models/training_" + str(r + 1) + "/cp.ckpt"

        model = train_model(trainX, trainy, checkpoint_path)

        test_X = test_X[:, :test_timesteps, :]
        score = evaluate_model(model, test_X, test_y)

        score = score * 100.0
        print('>#%d: %.3f' % (r + 1, score))
        scores.append(score)

    # summarize results
    summarize_results(scores)


physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

# run the experiment
run_experiment()
