
import gym
import numpy as np

import config

from stable_baselines import logger

class Player():
    def __init__(self, id, token, score, triads, pieces, phase, piece_to_move, in_a_row):
        self.id = id
        self.token = token
        self.score = score
        self.triads = triads
        self.pieces = pieces
        self.phase = phase #0 = placing piece, 1 = removing enemy piece, 2 = choosing piece to move, 3 = new piece position
        self.piece_to_move = piece_to_move
        self.in_a_row = in_a_row


class Token():
    def __init__(self, symbol, number):
        self.number = number
        self.symbol = symbol


class NineMensMorrisEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, verbose = False, manual = False):
        super(NineMensMorrisEnv, self).__init__()
        self.name = 'ninemensmorris'
        self.manual = manual

        self.rows = 3
        self.cols = 8
        self.n_players = 2
        self.grid_shape = (self.rows, self.cols)
        self.num_squares = self.rows * self.cols
        self.action_space = gym.spaces.Discrete(self.num_squares)
        self.observation_space = gym.spaces.Box(-1, 1, self.grid_shape + (3, ) )
        self.verbose = verbose
        self.current_player_num = 0
        self.done = False
        self.players = []

    @property
    def observation(self):
        if self.current_player.token.number == 1:
            position_1 = np.array([1 if x.number == 1 else 0  for x in self.board]).reshape(self.grid_shape)
            position_2 = np.array([1 if x.number == -1 else 0 for x in self.board]).reshape(self.grid_shape)
        else:
            position_1 = np.array([1 if x.number == -1 else 0 for x in self.board]).reshape(self.grid_shape)
            position_2 = np.array([1 if x.number == 1 else 0 for x in self.board]).reshape(self.grid_shape)

        legal_actions = self.legal_actions.reshape(self.grid_shape)
        out = np.stack([position_1, position_2, legal_actions], axis = -1)
        return out

    @property
    def legal_actions(self):
        legal_actions = []
        for action_num in range(self.action_space.n):
            legal = self.is_legal(action_num)
            legal_actions.append(legal)

        return np.array(legal_actions)


    def is_legal(self, action_num):
        if self.players[self.current_player_num].phase == 0:
            if self.board[action_num].number==0:# and action_num in self.legal:
                return 1
            else:
                return 0
        elif self.players[self.current_player_num].phase == 1:
            if self.board[action_num].number!=0 and self.square_is_player(self.board, action_num, self.current_player_num)==False:# and action_num in self.legal:
                return 1
            else:
                return 0
        elif self.players[self.current_player_num].phase == 2:
            if self.board[action_num].number!=0 and self.square_is_player(self.board, action_num, self.current_player_num)==True and self.has_empty_neighbours(action_num):# and action_num in self.legal:
                return 1
            else:
                return 0
        elif self.players[self.current_player_num].phase == 3:
            if self.board[action_num].number==0 and action_num in NEIGHBOURS[self.players[self.current_player_num].piece_to_move]:
                return 1
            else:
                return 0

    def has_empty_neighbours(self, square):
        has_empty = False
        for neighbour in NEIGHBOURS[square]:
            if self.board[neighbour].number==0:
                has_empty = True
                break
        return has_empty

    def square_is_player(self, board, square, player):
        return board[square].number == self.players[player].token.number

    def check_game_over(self, board = None , player = None):

        if board is None:
            board = self.board

        if player is None:
            player = self.current_player_num
		
        empty = 0
        for x in self.board:
            if x.number == 0:
                empty = empty+1

        for x,y,z in WINNERS:
            if self.square_is_player(board, x, player) and self.square_is_player(board, y, player) and self.square_is_player(board, z, player):
                triad = [x,y,z]
                if triad not in self.players[player].triads:
                    self.players[player].triads.append(triad)
                    return 1, (empty == 0), True

        if empty == 0:
            logger.debug("Board full")
            return  0, True, False

		#enforcing it to make good moves
        return -0.01, False, False

    def check_if_blocked(self, board, space):
        enemy_player = 0
        if self.current_player_num == 0:
            enemy_player = 1

        reward = 0

        for x,y,z in WINNERS:
            if(x == space or y == space or z == space):
                if self.square_is_player(board, x, self.current_player_num) and self.square_is_player(board, y, enemy_player) and self.square_is_player(board, z, enemy_player) or \
                self.square_is_player(board, x, enemy_player) and self.square_is_player(board, y, self.current_player_num) and self.square_is_player(board, z, enemy_player) or \
                self.square_is_player(board, x, enemy_player) and self.square_is_player(board, y, enemy_player) and self.square_is_player(board, z, self.current_player_num):
                    reward = reward + 0.05

        return reward

    @property
    def current_player(self):
        return self.players[self.current_player_num]

    def step(self, action):

        reward = [0.0] * self.n_players

        # check move legality
        board = self.board
        go_to_next = False
        got_point = False
        force_done = False
        done = True
        blocked = 0

        if not self.is_legal(action):
            force_done = True
            done = True
            reward = [1.0] * self.n_players
            reward[self.current_player_num] = -1.0
        else:
            square = action#self.get_square(board, action)

            if self.players[self.current_player_num].phase == 0: #placing piece
                board[square] = self.current_player.token
                blocked = self.check_if_blocked(board, square)
                reward[self.current_player_num] += blocked
                self.players[self.current_player_num].in_a_row += blocked/0.05 #adding a multiplier for each triad that the player blocked

                self.players[self.current_player_num].pieces = self.players[self.current_player_num].pieces-1
                if self.players[self.current_player_num].pieces == 0:
                    self.players[self.current_player_num].phase = 2
                go_to_next = True
            elif self.players[self.current_player_num].phase == 1: #removing opponent piece
                board[square] = Token('.', 0)
				
                if self.players[self.current_player_num].pieces == 0:
                    self.players[self.current_player_num].phase = 2
                else:
                    self.players[self.current_player_num].phase = 0

                reward[self.current_player_num] += 0.05

                enemy_player = 0
                if self.current_player_num == 0:
                    enemy_player = 1

                reward[enemy_player] -= 0.05
				
				#removing any opponent triads that contain this piece
                tempTriads = []
                for triad in self.players[enemy_player].triads:
                    if square not in triad:
                        tempTriads.append(triad)
                self.players[enemy_player].triads = tempTriads
				
                enemy_pieces = 0
                for x in self.board:
                    if x == self.players[enemy_player].token:
                        enemy_pieces = enemy_pieces+1

                if enemy_pieces == 2 and self.players[enemy_player].pieces == 0: #if the enemy only has 2 pieces and he has already placed all 9
                    force_done = True
                    reward[self.current_player_num] = 1.0
                    reward[enemy_player] = -1.0

                go_to_next = True
            elif self.players[self.current_player_num].phase == 2: #selecting piece to move
                board[square] = Token('.', 0)

                #removing any triads that contain this piece
                tempTriads = []
                for triad in self.players[self.current_player_num].triads:
                    if square not in triad:
                        tempTriads.append(triad)
                self.players[self.current_player_num].triads = tempTriads

                self.players[self.current_player_num].phase = 3
                self.players[self.current_player_num].piece_to_move = square
                go_to_next = False
            elif self.players[self.current_player_num].phase == 3: #moving piece
                board[square] = self.current_player.token
                self.players[self.current_player_num].phase = 2
                self.players[self.current_player_num].piece_to_move = -1
                go_to_next = True

            self.turns_taken += 1
            r, done, did3 = self.check_game_over()
            if force_done == True:
                done = True
            if did3 == True:
                reward[self.current_player_num] += 0.05 + 0.05*self.players[self.current_player_num].in_a_row
                self.players[self.current_player_num].in_a_row += 1
                self.players[self.current_player_num].phase = 1
                go_to_next = False
            else:
                if blocked != 0 and (self.players[self.current_player_num].phase == 0 or self.players[self.current_player_num].phase == 2):
                    self.players[self.current_player_num].in_a_row = 0

        self.done = done

        if not done and go_to_next:
            self.current_player_num = (self.current_player_num + 1) % 2

        return self.observation, reward, done, {}

    def reset(self):
        self.board = [Token('.', 0)] * self.num_squares
        self.players = [Player('1', Token('X', 1), 0, [], 9, 0, -1, 0), Player('2', Token('O', -1), 0, [], 9, 0, -1, 0)]
        self.current_player_num = 0
        self.turns_taken = 0
        self.done = False
        logger.debug(f'\n\n---- NEW GAME ----')
        return self.observation

	#self, id, token, score, triads, pieces, phase, piece_to_move
    def init(self, state):
        board_state = state.split("-")[0]
        player_state = state.split("-")[1]
        turns_taken = int(state.split("-")[2])
        self.board = [Token('.', 0)] * self.num_squares
        for idx, c in enumerate(board_state):
            if c == "0":
                self.board[idx] = Token('.', 0)
            if c == "1":
                self.board[idx] = Token('X', 1)
            elif c == "2":
                self.board[idx] = Token('O', -1)
        self.players = [Player('1', Token('X', 1), 0, [], int(player_state[0]), (2 if int(player_state[0])==0 else 0), -1, 0), Player('2', Token('O', -1), 0, [], int(player_state[1]), (2 if int(player_state[1])==0 else 0), -1, 0)]
        
        for player in [0,1]:
            for x,y,z in WINNERS:
                if self.square_is_player(self.board, x, player) and self.square_is_player(self.board, y, player) and self.square_is_player(self.board, z, player):
                    triad = [x,y,z]
                    if triad not in self.players[player].triads:
                        self.players[player].triads.append(triad)
		
        self.current_player_num = 1
        self.turns_taken = turns_taken
        self.done = False
        #logger.debug(f'\n\n---- NEW CUSTOM GAME ----')
        return self.observation

    def api(self):
        result = ""
        for token in self.board:
            result += str(2 if token.number==-1 else token.number)
        result += "-"
        result += str(self.players[0].pieces)
        result += str(self.players[1].pieces)
        result += str(self.players[0].phase)
        result += str(self.players[1].phase)
        result += "-"
        result += str(self.turns_taken)
        result += "."
        #if self.current_player_num == 1:
        #    return None
        return result
		
    def render(self, mode='human', close=False):
        logger.debug('')
        if close:
            return
        if self.done:
            logger.debug(f'GAME OVER')
        else:
            action = ""
            if self.players[self.current_player_num].phase == 0:
                action = "place a piece"
            elif self.players[self.current_player_num].phase == 1:
                action = "select an opponent's piece to remove it"
            elif self.players[self.current_player_num].phase == 2:
                action = "select a piece to move"
            elif self.players[self.current_player_num].phase == 3:
                action = "move the selected piece"

            logger.debug(f"It is Player {self.current_player.id}'s turn to {action}")

        whole_row = ""
        logger.debug(self.board[0].symbol + "     " + self.board[1].symbol + "     " + self.board[2].symbol)
        #logger.debug("")
        logger.debug("  " + self.board[3].symbol + "   " + self.board[4].symbol + "   " + self.board[5].symbol)
        #logger.debug("")
        logger.debug("    " + self.board[6].symbol + " " + self.board[7].symbol + " " + self.board[8].symbol)
        #logger.debug("")
        logger.debug(self.board[9].symbol + " " + self.board[10].symbol + " " + self.board[11].symbol + "   " + self.board[12].symbol + " " + self.board[13].symbol + " " + self.board[14].symbol)
        logger.debug("    " + self.board[15].symbol + " " + self.board[16].symbol + " " + self.board[17].symbol)
        #logger.debug("")
        logger.debug("  " + self.board[18].symbol + "   " + self.board[19].symbol + "   " + self.board[20].symbol)
        #logger.debug("")
        logger.debug(self.board[21].symbol + "     " + self.board[22].symbol + "     " + self.board[23].symbol)
        
        #for row in range(0,self.rows):
        #    whole_row = ""
        #    for col in range(0,self.cols):
        #        idx = (row*self.cols)+col
        #        symb = self.board[idx].symbol
        #        if idx not in self.legal:
        #            symb = " "
        #        whole_row += symb + " "
        #        #logger.debug(' '.join([x.symbol for x in self.board[i:(i+self.cols)]]))
        #    logger.debug(whole_row)

        if self.verbose:
            logger.debug(f'\nObservation: \n{self.observation}')

        if not self.done:
            logger.debug(f'\nLegal actions: {[i for i,o in enumerate(self.legal_actions) if o != 0]}')



    #def rules_move(self):
    #    WRONG_MOVE_PROB = 0.01
    #    player = self.current_player_num
    #    print(self.action_space.n)
    #    for action in range(self.action_space.n):
    #        if self.is_legal(action):
    #            new_board = self.board.copy()
    #            square = self.get_square(new_board, action)
    #            new_board[square] = self.players[player].token
    #            _, done = self.check_game_over(new_board, player)
    #            if done:
    #                action_probs = [WRONG_MOVE_PROB] * self.action_space.n
    #                action_probs[action] = 1 - WRONG_MOVE_PROB * (self.action_space.n - 1)
    #                return action_probs

    #    player = (self.current_player_num + 1) % 2

    #    for action in range(self.action_space.n):
    #        if self.is_legal(action):
    #            new_board = self.board.copy()
    #            square = self.get_square(new_board, action)
    #            new_board[square] = self.players[player].token
    #            _, done = self.check_game_over(new_board, player)
    #            if done:
    #                action_probs = [0] * self.action_space.n
    #                action_probs[action] = 1 - WRONG_MOVE_PROB * (self.action_space.n - 1)
    #                return action_probs


    #    action, masked_action_probs = self.sample_masked_action([1] * self.action_space.n)
    #    return masked_action_probs


WINNERS = [
			[0,1,2],
            [3,4,5],
            [6,7,8],
            [9,10,11],
            [12,13,14],
            [15,16,17],
            [18,19,20],
            [21,22,23],

            [0,9,21],
            [3,10,18],
            [6,11,15],
            [1,4,7],
            [16,19,22],
            [8,12,17],
            [5,13,20],
            [2,14,23]
		]


NEIGHBOURS = [
                [1,9],
                [0,2,4],
                [1,14],
                [4,10],
                [1,3,5,7],
                [4,13],
                [7,11],
                [4,6,8],
                [7,12],
                [0,10,21],
                [3,9,11,18],
                [6,10,15],
                [8,13,17],
                [5,12,14,20],
                [2,13,23],
                [11,16],
                [15,17,19],
                [12,16],
                [10,19],
                [16,18,20,22],
                [13,19],
                [9,22],
                [19,21,23],
                [14,22]
            ]