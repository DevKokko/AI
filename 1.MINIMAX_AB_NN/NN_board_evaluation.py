
from tensorflow.keras.models import load_model as load_NN_model
import data_processing
import numpy as np


def get_NN_board_evaluation(board):
    npBoard = data_processing.convertBoardPossitionToNP(board)
    npBoard = np.expand_dims(npBoard, 0)
    try:
        return get_NN_board_evaluation.reconstructed_model(npBoard)
    except AttributeError:
        get_NN_board_evaluation.reconstructed_model = load_NN_model("model.h5")
        return get_NN_board_evaluation.reconstructed_model(npBoard)
