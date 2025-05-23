import numpy as np
import random 
import pdb
import copy

ROW_COUNT = 6 
COLUMN_COUNT = 7 

### Game set up ###
def create_board(trivial = False):
    if trivial:
        return np.array([
        [2, 2, 1, 2, 0, 1, 0],
        [1, 1, 1, 2, 0, 2, 0],
        [2, 1, 1, 1, 0, 2, 0],
        [1, 2, 2, 2, 0, 1, 0],
        [2, 1, 1, 1, 0, 2, 0],
        [2, 2, 2, 1, 0, 1, 0]])
        
    else:
        return np.zeros((ROW_COUNT, COLUMN_COUNT))

def drop_piece(board, action, piece):
    row, col = next_open_row(board, action), action
    board[row][col] = piece
    return board
    
def is_valid_location(board, col):
    # wheter or not the position is valid, if the to is empty
    return board[ROW_COUNT - 1][col] == 0

def next_open_row(board,col):
    for r in range(ROW_COUNT):
        if board[r][col] == 0:
            return r
        
def print_board(board):
    print(np.flip(board, 0))

def winning_move(board,piece):
    # Check horizontal locations for win
    # subtract 3 
    for c in range(COLUMN_COUNT-3):
        for r in range(ROW_COUNT):
            if board[r][c] == piece and board[r][c+1] == piece and board[r][c+2] == piece and board[r][c+3] == piece:
                return True
    # Check vertical locations for win
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT-3):
            if board[r][c] == piece and board[r+1][c] == piece and board[r+2][c] == piece and board[r+3][c] == piece:
                return True
    # Check positively sloped diagonals
    for c in range(COLUMN_COUNT-3):
        for r in range(ROW_COUNT-3):
            if board[r][c] == piece and board[r+1][c+1] == piece and board[r+2][c+2] == piece and board[r+3][c+3] == piece:
                return True
    # Check negatively sloped diagonals
    for c in range(COLUMN_COUNT-3):
        for r in range(3, ROW_COUNT):
            if board[r][c] == piece and board[r-1][c+1] == piece and board[r-2][c+2] == piece and board[r-3][c+3] == piece:
                return True
            

def tie(board):
    # if empty entry then not tie 
    for row in range(ROW_COUNT):
        for col in range(ROW_COUNT):
            if board[row, col] == 0:
                return False
    # no empty cell, hence tie 
    return True 

def valid_actions(board):
    valid = []
    for col in range(COLUMN_COUNT):
        if is_valid_location(board, col):
            valid.append(col)
    return valid


def terminal_state(board, p1, p2):
    return tie(board) or winning_move(board,p1) or winning_move(board,p2)

### mcts implementation ###

class Node:
    def __init__(self, state, parent = None):
        self._state = state                                # game state
        self._state.setflags(write=False)                  # make the array immutable
        self.parent = parent                               # parent node 
        self.children = []                                 # Child node of current state
        self.unexp_actions = valid_actions(self.state)     # possible unexplored actions in node 
        self.N = 0                                         # number of node visits 
        self.Q = 0                                         # rewards propagated
        self.W = 0                                         # number of total wins from this node              

    @property
    def state(self):
        return self._state  # read-only access to the state

class MCTS():
    
    def __init__(self):
        self.state_to_node = {}
        self.p1 = 1
        self.p2 = 2

    def reset_memory(self):
        self.state_to_node.clear()

    def get_or_create_node(self, state, parent = None):
        """Avoid duplicate node states"""
        # return node state if existing
        key = tuple(state.flatten())

        if key in self.state_to_node:
            return self.state_to_node[key]
        else:
            # create new node 
            new_node = Node(state, parent)
            # map state to node 
            self.state_to_node[key] = new_node
            # create a new node 
            return new_node
    
    def random_move(self, board):
        """Generate random move for simulations"""
        col = np.random.choice(valid_actions(board))
        return col

    def new_enemy_state(self, board):
        """Given that we took an action we generate a new state after opponent moved"""
        # generate enemy move and then state resulting from it
        enemy_move = self.random_move(board) # returns (row, col)

        new_state = drop_piece(board.copy(), enemy_move, self.p2)
        return new_state

    def terminal_state(self, board, p1, p2):
        """Check if termianl state has been reached"""
        return tie(board) or winning_move(board, self.p1) or winning_move(board, self.p2)

    def utc(self, node):
        """return UCT value"""
        return node.Q/node.N + np.sqrt(2)*np.sqrt(np.log(node.parent.N)/node.N)

    def get_rand_node(self, node):
        """Select random node for expansion or begining of UCT"""
        # select action at random 
        action = np.random.choice(node.unexp_actions)
        node.unexp_actions.remove(action) # remove action from unexplored actions

        # create 
        next_state = drop_piece(node.state.copy(), action, self.p1)
        child = self.get_or_create_node(next_state, node)

        return child 

    def selection(self, node):
        """Selection mechanism"""
        current_node = node 

        while True:

            # check if we reached a terminal state 
            if self.terminal_state(current_node.state, self.p1, self.p2):
                return current_node

            # if selection is not yet possible pass node to expand
            if len(current_node.unexp_actions) != 0:
                return current_node
           
            # run uct when it is possible to do selection
            selected_node, highest_utc = None, -float('inf')

            # iterate over nodes and do selection using UTC
            for child in current_node.children:
                
                # compute utc value for node
                child_utc = self.utc(child)

                # select action with highest utc score 
                if  child_utc > highest_utc:
                    selected_node = child
                    highest_utc = child_utc
            
            # gerate random move from slected node
            inter_state = self.new_enemy_state(selected_node.state.copy())
            current_node = self.get_or_create_node(inter_state, selected_node)
            # update  children relationship 
            selected_node.children.append(current_node)


    def expansion(self, node):
        """Expand search tree for unexplorede nodes"""
        # select a random move out of the unexplored moves and generate node
        selected_node = self.get_rand_node(node)
        
        # update child parent relationship
        node.children.append(selected_node)
        
        return selected_node


    def get_reward(self, node):
        """Get reward"""
        # check if current node is in terminal state
        if winning_move(node.state, self.p1):
            return 1
        elif winning_move(node.state, self.p2):
            return -1
        else:
            return 0

    
    def simulate(self, node):
        """Given a action we simulate the terminal state"""
        board = node.state.copy()
        
        # enemy 2 always start in simulations 
        players = [2, 1]
        turn = 0
        # simulate the rest of the game
        while not terminal_state(board, self.p1, self.p2):
            # generate random action
            action = np.random.choice(valid_actions(board))
            # update board 
            board = drop_piece(board, action, players[turn])
            
            # update turn 
            turn = 1 - turn

        # check for reward
        if winning_move(board, self.p1):
            return 1
        elif winning_move(board, self.p2):
            return -1
        else:
            return 0 # no reward  of tie 


    def backpropagate(self, node, reward):
        """backpropagate result upwards through trajectory"""
        while node is not None:
            # propagate reward and visit 
            node.Q += reward
            node.N += 1 
            if reward == 1:
                node.W += 1

            # remove parent from node (clean trail history)
            parent = node.parent 
            # node.parent = None
            node = parent
            
    def best_action(self, node):
        """Select best action after perfoming mcst"""
        best_state, success_prob, select_value = None, None, -float('inf')

        # loop over the root child and choose best performer
        for child in node.children:
            
            # compute selection value for each children 
            child_value = child.Q / child.N
            
            # select best perfoming child best on selection value
            if best_state is None or child_value > select_value:
                best_state = child.state.copy()
                select_value = child_value
                # compute success probability of move 
                success_prob = child.W / child.N
        
        return best_state, success_prob
        
    def mcts(self, root, n_iter):
        """msct algorithm"""
        
        # convergce criteria setup 
        conv_criteria = 1e-4
        root_exp_val = 0

        # iterate for n iterations or termiante due to convergence
        for i in range(n_iter):
            # select node based on uct, if unexplored then selection is random 
            node = self.selection(root)
            # check if terminal state have been reached 
            if not self.terminal_state(node.state, self.p1, self.p2):
                child = self.expansion(node)
                reward = self.simulate(child)
                self.backpropagate(child, reward)
            # propagate reward if selection is done until termnate state
            else:
                reward = self.get_reward(node)
                self.backpropagate(node, reward)
        
            # compute current expected return and check convergence
            if root.N > 0:
                current_exp_val = root.Q / root.N

                # check for convergence criteria
                if abs(current_exp_val - root_exp_val) < conv_criteria and i > n_iter/2:
                    break
                else:
                    root_exp_val = current_exp_val

        # return best action based on expected average reward 
        return self.best_action(root)
    

##### game function #####

def random_move(board,):
    """Generate random move for simulations"""
    col = np.random.choice(valid_actions(board))
    row = next_open_row(board,col)
    return  col

def new_enemy_state(board, player):
    """Given that we took an action we generate a new state after opponent moved"""
    # generate enemy move and then state resulting from it
    board = board.copy()
    enemy_move = random_move(board)
    new_state = drop_piece(board, enemy_move, piece = player)
    return new_state
        

def play_games(n_games, mcst_iter, enemy= "random", display = False, trivial = False):
    # set players 
    p1, p2 = 1, 2

    # score_board
    score = 0

    # init mcst
    player = MCTS()
    
    for _ in range(n_games):

        # create board and set it as root 
        board = create_board(trivial = trivial)
        root = Node(board, parent = None)

        # set turn
        turn = 0

        # if display show initial state
        if display:
            print("Starting board")
            print_board(board)
            print()

        #  reset mcts memory
        player.reset_memory()


        while not terminal_state(board, p1, p2):

            if turn == 0:
                # run mcst and get state of choosen action
                board, success_prob = player.mcts(root, mcst_iter)

                # if we cant to display the progression the game 
                if display:
                    print(f"Curent move was made with {success_prob} success prob.")
                    print_board(board)
                    print()
            else:
                # enemy move ahead 
                # if playing user
                if enemy != "random":
                    # print board
                    print_board(board)
                    # get board move from user 
                    input_move = input("Enter move (row, column): ").int()
                    # transform move to legal action and generate new board
                    move = tuple( x - 1 for x in (map(int, input_move.strip("()").split(","))))
                    board = drop_piece(board, move, p2)
                # generate random move for opponent 
                else:
                    board = new_enemy_state(board, p2)
                
                # if display show 
                if display:
                    print("Enemy move")
                    print_board(board)
                    print()
                
                # reset root node for next iteration
                root = Node(board, parent = None)

            # change turns 
            turn = 1 - turn 

        # check end state of game
        if winning_move(board, p1):
            if display: print("Player 1 Won!\n")
            score += 1
        elif winning_move(board, p2):
            if display: print("Player 2 Won!\n")
        else:
            if display: print("It's a tie!\n")
        
        if display: print("____New game stated____\n")

    print(f"Rate of success: {(score/n_games) * 100}%")
    return 0 

### play game ###

mcts_iter = 1000
n_games = int(input("Number of games to be played: "))
display = True if (input("Display success prob. and board? (y,n) ").lower() == "y") else False
trivial = True if (input("Use assignment (trivial) stating board? (y,n) ").lower() == "y") else False
print("______Start game_____\n")
play_games(n_games, mcts_iter, display = display, trivial = trivial) 