import numpy as np
from constraints import empty_marker, getPlayerMadeMillsIndexes, getParticipatingMills
import AB_pruning
from visualization import printBoard
import heuristics
import argparse
import math
from sys import exit
import ray

coordsMap = {
    0: 0,
    1: 1,
    2: 2,
    3: 8,
    4: 9,
    5: 10,
    6: 16,
    7: 17,
    8: 18,
    9: 3,
    10: 11,
    11: 19,
    12: 20,
    13: 12,
    14: 4,
    15: 21,
    16: 22,
    17: 23,
    18: 13,
    19: 14,
    20: 15,
    21: 5,
    22: 6,
    23: 7
}


markersMap = {
    "M": 'X',
    "E": 'O',
    "O": empty_marker
}


markersToIntMap = {
    'X': 2,
    'O': 1,
    empty_marker: 0
}


def setListsToEqualLen(list1, list2):
    if len(list1) == len(list2):
        return list1, list2
    
    smallerLenList = None
    biggerLenList = None

    if len(list1) < len(list2):
        smallerLenList = list1
        biggerLenList = list2
    else:
        smallerLenList = list2
        biggerLenList = list1

    return smallerLenList, biggerLenList[:len(smallerLenList)]


def extractBoardPossition(dataset_line):
    board = [None] * 24 
    for char_i in range(24):
        parsedMarker = dataset_line[char_i]
        board[coordsMap[char_i]] = markersMap[parsedMarker]

    return board


def extractIsPlacementPhase(dataset_line):
    return not (dataset_line[24] == '0' and dataset_line[25] == '0')


def extractMenRemaining(dataset_line, placementPhase):
    if placementPhase:
        return {
            'X': int(dataset_line[24]) + int(dataset_line[26]),
            'O': int(dataset_line[25]) + int(dataset_line[27])
        }
    else: # movement phase: the number of remaining men is the number of the placed men on the board
        return {
            'X': int(dataset_line[26]),
            'O': int(dataset_line[27])
        }


def getMovesCounter(dataset_line, placementPhase):
    if not placementPhase:
        return 19
    
    return  (9 - int(dataset_line[24])) + (9 - int(dataset_line[25]))


def getPlayersMadeMills(board):
    return {
        'X': getPlayerMadeMillsIndexes(board, 'X'),
        'O': getPlayerMadeMillsIndexes(board, 'O')
    }


def extractGameStatesFromDataset(filePATH):
    gameStates = []
    with open(filePATH) as f:
        lines = f.readlines()
        for line in lines:
            gameState = {
                "board": extractBoardPossition(line),
                "placement_phase": extractIsPlacementPhase(line)
            }
            gameState["men_remaining"] = extractMenRemaining(line, gameState["placement_phase"])
            gameState["made_mills_indexes"] = getPlayersMadeMills(gameState["board"])
            gameState["moves_counter"] = getMovesCounter(line, gameState["placement_phase"])
            gameStates.append(gameState)

    return gameStates


def evaluateBoardPossition(gameState, participating_mills, heuristicType="hard"):
    if heuristicType == "easy":
        evaluation = heuristics.easy_heuristic(gameState["board"], gameState["placement_phase"], 'X', 'X', gameState["made_mills_indexes"], gameState["men_remaining"], participating_mills)
    elif heuristicType == "normal":
        evaluation = heuristics.normal_heuristic(gameState["board"], gameState["placement_phase"], 'X', 'X', gameState["made_mills_indexes"], gameState["men_remaining"], participating_mills)
    elif heuristicType == "hard":
        evaluation = heuristics.hard_heuristic(gameState["board"], gameState["placement_phase"], 'X', 'X', gameState["made_mills_indexes"], gameState["men_remaining"], participating_mills)
    else:
        print("-> ERROR: invalid heuristic type..")
        exit(1)
    
    return evaluation


def evaluateBoardPossitionDeep(gameState, participating_mills, heuristicType="hard"):
    if heuristicType == "easy":
        evaluation_fun = heuristics.easy_heuristic
    elif heuristicType == "normal":
        evaluation_fun = heuristics.normal_heuristic
    elif heuristicType == "hard":
        evaluation_fun = heuristics.hard_heuristic
    else:
        print("-> ERROR: invalid heuristic type..")
        exit(1)
        
    result = AB_pruning.alphaBetaPruning(gameState["board"], 3, 'X', -math.inf, math.inf, gameState["moves_counter"], evaluation_fun, 'X', participating_mills, gameState["made_mills_indexes"], gameState["men_remaining"])
    return result.evaluation


def normalizeVector(v):
    return np.asarray(v / abs(v).max() / 2 + 0.5, dtype=np.float32) # normalization (0 - 1)


def convertBoardPossitionToNP2(board):
    npBoard = np.empty((24, ), dtype=np.int8)
    for i, marker in enumerate(board):
        npBoard[i] = markersToIntMap[marker]

    return npBoard


def convertBoardPossitionToNP(board):
    npBoard = np.zeros((2, 3, 8), dtype=np.int8) # one for each player, one for each square, 8 for each square
    for i, marker in enumerate(board):

        if marker == 'X': 
            playerIndex = 0 # first player
        elif marker == 'O':
            playerIndex = 1 # second player
        else: # empty spot
            continue
        
        if i < 8: # outer square
            squareIndex = 0
            indexInSquare = i
        elif i < 16: # middle square [8, 15]
            squareIndex = 1
            indexInSquare = i - 8
        else: # inner square [16, 23]
            squareIndex = 2
            indexInSquare = i - 16

        npBoard[playerIndex][squareIndex][indexInSquare] = 1

    return npBoard


@ray.remote
def parallel(gameState, participating_mills):
    evaluation = evaluateBoardPossitionDeep(gameState, participating_mills)
    npBoard = convertBoardPossitionToNP(gameState["board"])
    return evaluation, npBoard


def parseDataset(boardPossitions, boardPossitionsEvaluation, gameStates, participating_mills, printBoardFlag):
    ids = []
    for gameState in gameStates:
        id = parallel.remote(gameState, participating_mills)
        ids.append(id)

    results = ray.get(ids)
    for evaluation, npBoard in results:
        boardPossitions.append(npBoard)
        boardPossitionsEvaluation.append(evaluation)

    print("-> Number of game states parsed: ", len(gameStates))
    

def getData(datasets, printBoardFlag=False):
    participating_mills = getParticipatingMills()
    boardPossitions = []
    boardPossitionsEvaluation = []

    gameStates = []
    for dataset in datasets:
        gameStatesTmp = extractGameStatesFromDataset(dataset)
        gameStates.extend(gameStatesTmp)

    ray.init()
    parseDataset(boardPossitions, boardPossitionsEvaluation, gameStates, participating_mills, printBoardFlag)
    print("=" * 50)

    return np.array(boardPossitions), normalizeVector(np.array(boardPossitionsEvaluation))
    

def loadData(filename):
    data = np.load(filename)
    return data['x'], data['y']


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parsing and saving training data in a numpy file')
    parser.add_argument('-i', '--input', type=str, dest='inputs', nargs='+', help='The input dataset filenames', required=True)
    parser.add_argument('-o', '--output', type=str, dest='output', help='The output numpy filename', required=True)
    args = parser.parse_args()

    x_train, y_train = getData(args.inputs)
    np.savez(args.output, x=x_train, y=y_train)
    print("-> Datasets successfully parsed and saved as numpy file.")