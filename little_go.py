import copy

import numpy as np

#Global variables 
BLACK_PIECE = 1
WHITE_PIECE = 2


KOMI_POINTS = 2.5

#Input files
FILE_INPUT = 'input.txt'
#Output files
FILE_OUTPUT = 'output.txt'

#File for counting the steps that has occured

FILE_STEP = 'step_num.txt'
EMPTY = 0
GRID = 5

# For movements Left,Right,Up,Down
DX = [1, -1, 0, 0]
DY = [0, 0, -1, 1]


def readInput(file_input=FILE_INPUT):
    with open(file_input) as input_file:
        input_file_lines = [line.strip() for line in input_file.readlines()]
        player = int(input_file_lines[0])
        prev_state = np.array([[int(digit) for digit in line] for line in input_file_lines[1:6]], dtype=int)
        curr_state = np.array([[int(digit) for digit in line] for line in input_file_lines[6:11]], dtype=int)
        return player, prev_state, curr_state



def writeOutput(next_move):
    with open(FILE_OUTPUT, 'w') as output_file:
        output_file.write('PASS' if next_move is None or next_move == (-1, -1) else f'{next_move[0]},{next_move[1]}')

def steps_count(prev_state, curr_state):
    initial_prev_state = all(all(cell == EMPTY for cell in row) for row in prev_state)
    initial_current_state = all(all(cell == EMPTY for cell in row) for row in curr_state)

    step_mapping = {
        (True, True): 0,
        (True, False): 1,
    }

    current_step = step_mapping.get((initial_prev_state, initial_current_state))

    if current_step is None:
        with open(FILE_STEP) as current_step_file:
            current_step = int(current_step_file.readline())
            current_step += 2

    with open(FILE_STEP, 'w') as current_step_file:
        current_step_file.write(f'{current_step}')

    return current_step

class MyPlayer:
    def __init__(self, player, prev_state, curr_state):
        # The player who is playing(BLACK_PIECE/WHITE_PIECE)
        self.player = player
        #The Opponent Player
        self.rival_player = self.get_rival(self.player)
        #The State of the board BEFORE the opponent made his/her move.
        self.prev_state = prev_state
        #The State of the board AFTER the opponent made his/her move.
        self.curr_state = curr_state

    def minimax_search(self, search_max_depth, b_factor, current_step):
        # The search uses alpha beta pruning in minimax search for returning the best move that could be performed by the player in each step.
        best_move, best_score = self.maximum(self.curr_state, self.player, search_max_depth, 0, b_factor,-np.inf, np.inf, None, current_step, False)
        writeOutput(best_move)

    def maximum(self, curr_board_state, player, search_max_depth, curr_depth, b_factor, alpha_value, beta_value, prev_move, current_step, cons_moves):
        if search_max_depth == curr_depth or current_step + curr_depth == 24:
            return self.board_state_evaluation(curr_board_state, player)
        if cons_moves:
            return self.board_state_evaluation(curr_board_state, player)
        best_move = None
        best_score = -np.inf
        cons_moves = False
        potential_moves = self.find_potential_moves(curr_board_state, player)
        # -1,-1 means that the when the player uses a pass 
        # it is also appended in the potential move just if making no move has more evaluating value than making any move
        potential_moves.append((-1, -1))
        if prev_move == (-1, -1):
            cons_moves = True
            
        #checking on all branches
        for potential_move in potential_moves[:b_factor]:
            # get the rival player  
            rival_player = self.get_rival(player)
            if potential_move != (-1, -1):
                new_board_state = self.play(curr_board_state, player, potential_move)
            else:
                new_board_state = copy.deepcopy(curr_board_state)
            worst_score = self.minimum(new_board_state, rival_player, search_max_depth, curr_depth + 1,b_factor, alpha_value, beta_value, potential_move, current_step, cons_moves)
            if best_score < worst_score:
                best_score = worst_score
                best_move = potential_move
            if best_score >= beta_value:
                if curr_depth == 0:
                    return best_move, best_score
                else:
                    return best_score
            alpha_value = max(alpha_value, best_score)
        #If the current depth the uppermost return the best move too
        if curr_depth == 0:
            return best_move, best_score
        else:
            return best_score

    def minimum(self, curr_board_state, player, search_max_depth, curr_depth, b_factor, alpha_value, beta_value, prev_move,
                  current_step, cons_moves):
        #if the search depth has reached maximum
        if search_max_depth == curr_depth:
            return self.board_state_evaluation(curr_board_state, player)
        
        if current_step + curr_depth == 24 or cons_moves:
            return self.board_state_evaluation(curr_board_state, self.player)
        
        potential_moves = self.find_potential_moves(curr_board_state, player)
        worst_score = np.inf
        cons_moves = False
        potential_moves.append((-1, -1))
        
        if prev_move == (-1, -1):
            cons_moves = True
            
        for potential_move in potential_moves[:b_factor]:
            rival_player = self.get_rival(player)
            if potential_move == (-1, -1):
                new_board_state = copy.deepcopy(curr_board_state)
            else:
                new_board_state = self.play(curr_board_state, player, potential_move)
            best_score = self.maximum(new_board_state, rival_player, search_max_depth, curr_depth + 1,b_factor, alpha_value, beta_value, potential_move, current_step, cons_moves)
            if best_score < worst_score:
                worst_score = best_score
            if worst_score <= alpha_value:
                return worst_score
            beta_value = min(beta_value, worst_score)
        return worst_score
   #done
    def board_state_evaluation(self, curr_board_state, player):
        # Evaluation of game state includes
        # Varibles used for evaluation
        # Player pieces count, Opponent pieces count
        # Player liberty count, Oppenent Liberty count
        # No of player pieces on the edge, No of opponent player pieces on the edge
        player_count = 0
        rival_count = 0
        player_liberty = set()
        rival_liberty = set()
        rival_player = self.get_rival(player)
        i = 0

        while i < GRID:
            j = 0
            while j < GRID:
                if curr_board_state[i][j] == player:
                    player_count += 1
                elif curr_board_state[i][j] == rival_player:
                    rival_count += 1
                else:
                    index = 0
                    while index < len(DX):
                        nx = i + DX[index]
                        ny = j + DY[index]
                        if 0 <= nx < GRID and 0 <= ny < GRID:
                            if curr_board_state[nx][ny] == player:
                                player_liberty.add((i, j))
                            elif curr_board_state[nx][ny] == rival_player:
                                rival_liberty.add((i, j))
                        index += 1
                j += 1
            i += 1

        player_edge_count = 0
        rival_edge_count = 0
        j = 0

        while j < GRID:
            if curr_board_state[0][j] == player or curr_board_state[GRID - 1][j] == player:
                player_edge_count += 1
            if curr_board_state[0][j] == rival_player or curr_board_state[GRID - 1][j] == rival_player:
                rival_edge_count += 1
            j += 1

        j = 1
        while j < GRID - 1:
            if curr_board_state[j][0] == player or curr_board_state[j][GRID - 1] == player:
                player_edge_count += 1
            if curr_board_state[j][0] == rival_player or curr_board_state[j][GRID - 1] == rival_player:
                rival_edge_count += 1
            j += 1

        empty_center_count = 0
        i = 1

        while i < GRID - 1:
            j = 1
            while j < GRID - 1:
                if curr_board_state[i][j] == EMPTY:
                    empty_center_count += 1
                j += 1
            i += 1

        sum_of_points = min(max((len(player_liberty) - len(rival_liberty)), -8), 8) - (9 * player_edge_count * (empty_center_count / 9))+ (
            -4 * self.euler_no_calculation(curr_board_state, player)) + (
            5 * (player_count - rival_count)) 
        
        if self.player == WHITE_PIECE:
            sum_of_points += KOMI_POINTS

        return sum_of_points
    #done
    def is_valid(x, y):
            return 0 <= x < GRID and 0 <= y < GRID
        
        
    def play(self, curr_board_state, player, play):
        new_board_state = copy.deepcopy(curr_board_state)
        
        new_board_state[play[0]][play[1]] = player

        rival_player = self.get_rival(player)

        

        def remove_connected_opponents(x, y):
            st = [(x, y)]
            vis = set()
            delete_opponent = True

            while st:
                node = st.pop()
                vis.add(node)

                for index in range(len(DX)):
                    new_nx = node[0] + DX[index]
                    new_ny = node[1] + DY[index]

                    if 0 <= new_nx < GRID and 0 <= new_ny < GRID:
                        if (new_nx, new_ny) in vis:
                            continue
                        elif new_board_state[new_nx][new_ny] == EMPTY:
                            delete_opponent = False
                            break
                        elif new_board_state[new_nx][new_ny] == rival_player and (new_nx, new_ny) not in vis:
                            st.append((new_nx, new_ny))

            if delete_opponent:
                for coin in vis:
                    new_board_state[coin[0]][coin[1]] = EMPTY

        for index in range(len(DX)):
            nx = play[0] + DX[index]
            ny = play[1] + DY[index]

            if 0 <= nx < GRID and 0 <= ny < GRID and new_board_state[nx][ny] == rival_player:
                remove_connected_opponents(nx, ny)

        return new_board_state
    
    #done
    def euler_no_calculation(self, curr_board_state, player):
        rival_player = self.get_rival(player)
        new_board_state = np.zeros((GRID + 2, GRID + 2), dtype=int)

        i = 0
        while i < GRID:
            j = 0
            while j < GRID:
                new_board_state[i + 1][j + 1] = curr_board_state[i][j]
                j += 1
            i += 1

        z1_player,z2_player,z3_player= 0,0,0
        z1_rival,z2_rival,z3_rival = 0,0,0

        i = 0
        while i < GRID:
            j = 0
            while j < GRID:
                new_sub_board_state = new_board_state[i: i + 2, j: j + 2]
                #Counting the euler values for the player for all 3 conditions
                z1_player += self.z1_count(new_sub_board_state, player)
                z2_player += self.z2_count(new_sub_board_state, player)
                z3_player += self.z3_count(new_sub_board_state, player)
                #Counting the euler values for the opponent for all 3 conditions
                z1_rival += self.z1_count(new_sub_board_state, rival_player)
                z2_rival += self.z2_count(new_sub_board_state, rival_player)
                z3_rival += self.z3_count(new_sub_board_state, rival_player)
                j += 1
            i += 1

        return (z1_player - z3_player + 2 * z2_player - (z1_rival - z3_rival + 2 * z2_rival)) / 4


    def z1_count(self, sub_board_state, player):
       # Pattern one player surrounded by three oppenents
       # 0 X
       # X X
       
        count = 0

       
        for i in range(2):
            for j in range(2):
                if sub_board_state[i][j] == player:
                    if (sub_board_state[1 - i][j] != player and sub_board_state[i][1 - j] != player and sub_board_state[1 - i][1 - j] != player):
                        count += 1

        return count


    def z2_count(self, sub_board_state, player):
        # Pattern 1: player in top-left and bottom-right, others not player
        # 0 X
        # X 0
        pattern1 = (
            sub_board_state[0][0] == player
            and sub_board_state[1][1] == player
            and sub_board_state[0][1] != player
            and sub_board_state[1][0] != player
        )

        # Pattern 2: player in top-right and bottom-left, others not player
        # X 0
        # 0 X
        pattern2 = (
            sub_board_state[0][1] == player
            and sub_board_state[1][0] == player
            and sub_board_state[0][0] != player
            and sub_board_state[1][1] != player
        )

        # If either pattern is found, return 1; otherwise, return 0
        if pattern1 or pattern2:
            return 1
        else:
            return 0


    def z3_count(self, sub_board_state, player):
        # Pattern 1: All cells contain 'player' except the bottom-right cell
        pattern1 = (
            sub_board_state[0][0] == player
            and sub_board_state[0][1] == player
            and sub_board_state[1][0] == player
            and sub_board_state[1][1] != player
        )

        # Pattern 2: All cells contain 'player' except the top-left cell
        pattern2 = (
            sub_board_state[0][0] != player
            and sub_board_state[0][1] == player
            and sub_board_state[1][0] == player
            and sub_board_state[1][1] == player
        )

        # Pattern 3: All cells contain 'player' except the top-right cell
        pattern3 = (
            sub_board_state[0][0] == player
            and sub_board_state[0][1] != player
            and sub_board_state[1][0] == player
            and sub_board_state[1][1] == player
        )

        # Pattern 4: All cells contain 'player' except the bottom-left cell
        pattern4 = (
            sub_board_state[0][0] != player
            and sub_board_state[0][1] == player
            and sub_board_state[1][0] == player
            and sub_board_state[1][1] == player
        )

        # If any of the patterns is found, return 1; otherwise, return 0
        if pattern1 or pattern2 or pattern3 or pattern4:
            return 1
        else:
            return 0

    def find_potential_moves(self, curr_board_state, player):
        #make a list of all the available moves in the current board
        potential_moves = {'s3': [], 'c1': [], 'r2': []}
        i = 0
        while i < GRID:
            j = 0
            while j < GRID:
                if curr_board_state[i][j] == EMPTY:
                    if self.liberty_checker(curr_board_state, i, j, player):
                        if not self.ko_checker(i, j):
                            if i == 0 or j == 0 or i == GRID - 1 or j == GRID - 1:
                                potential_moves.get('s3').append((i, j))
                            else:
                                potential_moves.get('r2').append((i, j))
                    else:
                        index = 0
                        while index < len(DX):
                            nx = i + DX[index]
                            ny = j + DY[index]
                            if 0 <= nx < GRID and 0 <= ny < GRID:
                                rival_player = self.get_rival(player)
                                if curr_board_state[nx][ny] == rival_player:
                                    new_board_state = copy.deepcopy(curr_board_state)
                                    new_board_state[i][j] = player
                                    if not self.liberty_checker(new_board_state, nx, ny, rival_player):
                                        if not self.ko_checker(i, j):
                                            potential_moves.get('c1').append((i, j))
                                        break
                            index += 1
                j += 1
            i += 1

        potential_moves_list = []
        
        for potential_move in potential_moves.get('c1'):
            potential_moves_list.append(potential_move)
            
        for potential_move in potential_moves.get('r2'):
            potential_moves_list.append(potential_move)
            
        for potential_move in potential_moves.get('s3'):
            potential_moves_list.append(potential_move)

        return potential_moves_list


    def liberty_checker(self, curr_board_state, i, j, player):
        
        st = [(i, j)]
        #for visited nodes
        vis = set()
        
        while st:
            node = st.pop()
            vis.add(node)
            for index in range(len(DX)):
                nx = node[0] + DX[index]
                ny = node[1] + DY[index]
                if 0 <= nx and nx < GRID and 0 <= ny and ny< GRID:
                    if (nx, ny) in vis:
                        continue
                    elif curr_board_state[nx][ny] == EMPTY:
                        return True
                    elif curr_board_state[nx][ny] == player and (nx, ny) not in vis:
                        st.append((nx, ny))
        return False

    def get_rival(self, player):
        if player == BLACK_PIECE:
            return WHITE_PIECE
        else:
            return BLACK_PIECE

    def ko_checker(self, i, j):
        if self.prev_state[i][j] != self.player:
            return False
        new_board_state = copy.deepcopy(self.curr_state)
        new_board_state[i][j] = self.player
        opponent_i, opponent_j = self.rival_play()
        for index in range(len(DX)):
            nx = i + DX[index]
            ny = j + DY[index]
            if nx == opponent_i and ny == opponent_j:
                
                if not self.liberty_checker(new_board_state, nx, ny, self.rival_player):
                   
                    self.group_eliminate(new_board_state, nx, ny, self.rival_player)
        
        return np.array_equal(new_board_state, self.prev_state)

    def rival_play(self):
        return next((i, j) for i in range(GRID) for j in range(GRID) if self.curr_state[i][j] != self.prev_state[i][j] and self.curr_state[i][j] != EMPTY), None


    def group_eliminate(self, curr_board_state, i, j, player):
        st = [(i, j)]
        vis = set()
        while st:
            # using DFS
            node = st.pop()
            
            vis.add(node)
            
            curr_board_state[node[0]][node[1]] = EMPTY
            for index in range(len(DX)):
                nx = node[0] + DX[index]
                ny = node[1] + DY[index]
                if self.is_valid(nx,ny):
                    if (nx, ny) in vis:
                        continue
                    elif curr_board_state[nx][ny] == player:
                        st.append((nx, ny))
        return curr_board_state



if __name__ == '__main__':
    player, prev_state, curr_state = readInput()
    
    current_step = steps_count(prev_state, curr_state)
    
    my_player = MyPlayer(player, prev_state, curr_state)
    
    my_player.minimax_search(search_max_depth=4, b_factor=20, current_step=current_step)