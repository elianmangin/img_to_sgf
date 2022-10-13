##CODE FROM https://gist.github.com/FirefoxMetzger/bbc7e14a777dd529942d3e68ba919a9c
import numpy as np
import sgf # pip install sgf -- simple parser for the file format
import timeit

def decode_position(pos_string):
    # position in .sgf is char[2] and row-first, e.g. "fd"
    positions = "abcdefghijklmnopqrs"
    x = positions.index(pos_string[0])
    y = positions.index(pos_string[1])
    return y, x

def render(black_board, white_board):
    for y in range(black_board.shape[0]):
        if y < 10:
            row = "0"+str(y)
        else:
            row = str(y)
        for x in range(black_board.shape[1]):
            if white_board[y,x]:
                row += "W"
            elif black_board[y,x]:
                row += "B"
            else:
                row +="-"
        print(row)

def get_neighboors(y,x,board_shape):
    neighboors = list()

    if y > 0:
        neighboors.append((y-1,x))
    if y < board_shape[0] - 1:
        neighboors.append((y+1,x))
    if x > 0:
        neighboors.append((y,x-1))
    if x < board_shape[1] - 1:
        neighboors.append((y,x+1))

    return neighboors

def test_group(board,opponent_board,y,x, current_group):
    """ Assume the current group is captured. Find it via flood fill
    and if an empty neighboor is encountered, break (group is alive).
    board - 19x19 array of player's stones
    opponent_board - 19x19 array of opponent's stones
    x,y - position to test
    current_group - tested stones in player's color
    """

    pos = (y,x)

    if current_group[pos]:
        # already tested stones are no liberties
        return False

    if opponent_board[pos]:
        current_group[pos] = True

        neighboors = get_neighboors(y,x,board.shape)
        for yn, xn in neighboors:
            has_liberties = test_group(board,opponent_board,yn,xn,current_group)
            if has_liberties:
                return True
        return False

    return not board[pos]

def fast_capture_pieces(black_board, white_board, turn_white, y,x):
    """Remove all pieces from the board that have 
    no liberties. This function modifies the input variables in place.
    black_board is a 19x19 np.array with value 1.0 if a black stone is
    present and 0.0 otherwise.
    white_board is a 19x19 np.array similar to black_board.
    active_player - the player that made a move
    (x,y) - position of the move
    """

    # only test neighboors of current move (other's will have unchanged
    # liberties)
    neighboors = get_neighboors(y,x,black_board.shape)

    board = white_board if turn_white else black_board
    opponent_board = black_board if turn_white else white_board

    # to test suicidal moves
    original_pos = (y,x)

    # only test adjacent stones in opponent's color
    for pos in neighboors:
        if not opponent_board[pos]:
            continue

        current_group = np.zeros((19,19), dtype=bool)
        has_liberties = test_group(board, opponent_board, *pos, current_group)
        if not has_liberties:
            opponent_board[current_group] = False

    current_group = np.zeros((19,19), dtype=bool)
    has_liberties = test_group(opponent_board, board, *original_pos, current_group)
    if not has_liberties:
        board[current_group] = False

def floodfill(liberties,y,x):
    """
    flood fill a region that is now known to have liberties. 1.0 signals a liberty, 0.0 signals
    undecided and -1.0 a known non-liberty (black stone)
    liberties is an np.array of currently known liberties and non-liberties
    """
    
    #"hidden" stop clause - not reinvoking for "liberty" or "non-liberty", only for "unknown".
    if not liberties[y][x]:
        liberties[y][x] = 1.0 
        if y > 0:
            floodfill(liberties,y-1,x)
        if y < liberties.shape[0] - 1:
            floodfill(liberties,y+1,x)
        if x > 0:
            floodfill(liberties,y,x-1)
        if x < liberties.shape[1] - 1:
            floodfill(liberties,y,x+1)

def capture_pieces(black_board, white_board, y,x):
    """Remove all pieces from the board that have 
    no liberties. This function modifies the input variables in place.
    black_board is a 19x19 np.array with value 1.0 if a black stone is
    present and 0.0 otherwise.
    white_board is a 19x19 np.array similar to black_board.
    """

    has_stone = np.logical_or(black_board,white_board)
    white_liberties = np.zeros((19,19))
    black_liberties = np.zeros((19,19))

    # stones in opposite color have no liberties
    white_liberties[black_board] = -1.0
    black_liberties[white_board] = -1.0

    for y in range(has_stone.shape[0]):
        for x in range(has_stone.shape[1]):
            if not has_stone[y,x]:
                floodfill(white_liberties,y,x)
                floodfill(black_liberties,y,x)

    white_liberties[white_liberties == 0.0] = -1.0
    black_liberties[black_liberties == 0.0] = -1.0

    white_board[white_liberties == -1.0] = 0.0
    black_board[black_liberties == -1.0] = 0.0

def parse(location, capture_pieces):
    with open(location, "r") as f:
        collection = sgf.parse(f.read())

    # assume only a single game per file
    game = collection[0]

    black_board = np.zeros((19,19), dtype=bool)
    white_board = np.zeros((19,19), dtype=bool)

    # set initial stones for black
    turn_white = False # black starts the game
    try:
        stones = game.root.properties["AB"]
        for position in stones:
            pos = decode_position(position)
            black_board[pos] = 1.0
        # treat the last initial stone as black's first move
        yield black_board, white_board, pos
        turn_white = True
    except KeyError:
        pass # no initial stones

    for move in game.rest:
        tag = "W" if turn_white else "B"
        move_string = move.properties[tag][0]

        if not move_string: # pass move
            turn_white = not turn_white
            yield black_board, white_board, (None, None)
            continue

        pos = decode_position(move_string)
        if white_board[pos] or black_board[pos]:
            print(render(black_board, white_board))
            raise Exception("Can't move on top of another stone at %d , %d" % pos)

        if turn_white:
            white_board[pos] = True
        else:
            black_board[pos] = True

        capture_pieces(black_board,white_board,turn_white, *pos)

        turn_white = not turn_white
        yield black_board, white_board, pos


def test(capture_pieces):
        for black, white, last_move  in parse(replay, capture_pieces):
            pass

if __name__ == "__main__":
    replay = "test.sgf"
    runs = 1000
    fast = timeit.timeit(lambda: test(fast_capture_pieces), number=runs)
    print("Fast capture %d runs: %.2f" % (runs, fast))
    slow = timeit.timeit(lambda: test(capture_pieces), number=runs)
    print("Slow capture %d runs: %.2f" % (runs, slow))
    print("Relative gain: %.2f" % (slow/fast))

    for reply1, reply2  in zip(parse(replay, fast_capture_pieces),parse(replay, capture_pieces)):
        fast_black, fast_white, fast_move = reply1
        black, white, move = reply2

        assert move == fast_move

        # doesn't apply here due to the old version violating the rules
        #assert np.all(fast_black == black)
        #assert np.all(fast_white == white)