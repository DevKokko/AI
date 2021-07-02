import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from os import write
import tensorflow.keras.callbacks as callbacks
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.utils as utils
from tensorflow.keras.metrics import MeanSquaredError, MeanAbsoluteError, MeanAbsolutePercentageError
from data_processing import loadData, setListsToEqualLen
import neural_networks as NN
import matplotlib.pyplot as plt
from tensorflow import config as tf_config
import argparse
import sys

ES_patience = 5
ES_threshold = 0.0000005

def trainNN(datasetNP, outputFn, NNtype, epochs, batchSize, lr, filters, depth, plotModelFlag, plotStatsFlag):
    createLogFile(outputFn, epochs, lr, NNtype, batchSize, filters, depth, ES_threshold, ES_patience)

    x_train, y_train = loadData(datasetNP)

    print("-> Successfully loaded training data")
    print("-> X train shape: ", x_train.shape)
    print("-> Y train shape: ", y_train.shape)

    if NNtype == "CNN":
        print("-> Building a Convolution Neural Network model")
        model = NN.build3dModel(filters, depth)
    else: # RNN
        print("-> Building a Residual Neural Network model")
        model = NN.build_model_residual(filters, depth)

    print("-> Neural Network was build")

    if plotModelFlag:
        utils.plot_model(model, to_file=f'{outputFn}_model_schema.png', show_shapes=True, show_layer_names=False)

    model.compile(
        optimizer=optimizers.Adam(lr), 
        loss='mean_squared_error',
        # metrics=[MeanSquaredError(), MeanAbsoluteError(), MeanAbsolutePercentageError(), Accuracy()]
        metrics=[MeanAbsolutePercentageError()]
    )
    model.summary()
    print("-> Training started..")
    history = model.fit(x_train, y_train,
            shuffle=True,
            batch_size=batchSize,
            epochs=epochs,
            verbose=2,
            validation_split=0.1,
            callbacks=[callbacks.EarlyStopping(monitor='val_loss', patience=ES_patience, min_delta=ES_threshold)])

    model.save(f'{outputFn}_model.h5')

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    loss, val_loss = setListsToEqualLen(loss, val_loss)
    print("loss", loss)
    print("val_loss", val_loss)

    MAPE = history.history['mean_absolute_percentage_error']
    val_MAPE = history.history['val_mean_absolute_percentage_error']
    MAPE, val_MAPE = setListsToEqualLen(MAPE, val_MAPE)
    print("MAPE", MAPE)
    print("val_MAPE", val_MAPE)

    if plotStatsFlag:
        epochs_range = range(len(loss)) # to also include early stopping cases
        plt.figure(figsize=(20, 20))

        plt.subplot(2, 2, 1)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')

        plt.subplot(2, 2, 2)
        plt.plot(epochs_range, MAPE, label='Training MAPE')
        plt.plot(epochs_range, val_MAPE, label='Validation MAPE')
        plt.legend(loc='upper right')
        plt.title('Training and Validation MAPE')

        plt.savefig(f'{outputFn}_stats_plot.png', bbox_inches='tight')
        plt.show()


def createLogFile(outputFn, epochs, lr, type, batchSize, filters, depth, ESthreshold=None, ESpatience=None):
    with open(f"{outputFn}.log", "w") as f:
        f.write(f"- Network type: {type}\n")
        f.write(f"- Epochs: {epochs}\n")
        f.write(f"- Learning Rate (LR): {lr}\n")
        f.write(f"- Batch Size: {batchSize}\n")
        f.write(f"- Number of filters: {filters}\n")
        f.write(f"- Network depth: {depth}\n")

        if ESthreshold:
            f.write(f"- Early Stopping Min Delta: {ESthreshold}\n")
        if ESpatience:
            f.write(f"- Early Stopping Patience: {ESpatience} epochs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Neural Networks')
    parser.add_argument('-i', '--input', type=str, dest='input', help='The input numpy dataset filename', required=True)
    parser.add_argument('-o', '--output', type=str, dest='output', help='The output filename', required=True)
    parser.add_argument('-g', '--gpu', dest='GPUflag', help='Flag whether to make use of the GPU or not', action='store_true')
    parser.add_argument('-t', '--type', type=str, dest='type', help='Neural Network type', choices=['CNN', 'RNN'], default="CNN")
    parser.add_argument('-e', '--epochs', type=int, dest='epochs', help='Epochs number', default=10)
    parser.add_argument('-b', '--batch_size', type=int, dest='batchSize', help='Batch Size', default=256)
    parser.add_argument('-lr', '--learning_rate', type=float, dest='lr', help='Learning Rate', default=0.0005)
    parser.add_argument('-s', '--stats', dest='plotStatsFlag', help='Flag whether to plot training statistics or not', action='store_true')
    parser.add_argument('-m', '--model', dest='plotModelFlag', help='Flag whether to plot model schema or not', action='store_true')
    parser.add_argument('-f', '--filters', type=int, dest='filters', help="Number of layers' filters", default=32)
    parser.add_argument('-d', '--depth', type=int, dest='depth', help="Neural Network's depth", default=6)
    parser.set_defaults(GPUflag=False)
    parser.set_defaults(plotStatsFlag=False)
    parser.set_defaults(plotModelFlag=False)
    args = parser.parse_args()

    if args.GPUflag:
        physical_devices = tf_config.experimental.list_physical_devices('GPU')
        assert len(physical_devices) > 0, "-> ERROR: Not enough GPU hardware devices available.."
        tf_config.experimental.set_memory_growth(physical_devices[0], True)

    trainNN(args.input, args.output, args.type, args.epochs, args.batchSize, args.lr, args.filters, args.depth, args.plotModelFlag, args.plotStatsFlag)
