import math
import chess.lib
from chess.lib.utils import encode, decode
from chess.lib.heuristics import evaluate
from chess.lib.core import makeMove


###########################################################################################
# Utility function: Determine all the legal moves available for the side.
# This is modified from chess.lib.core.legalMoves:
#  each move has a third element specifying whether the move ends in pawn promotion
def generateMoves(side, board, flags):
    for piece in board[side]:
        fro = piece[:2]
        for to in chess.lib.availableMoves(side, board, piece, flags):
            promote = chess.lib.getPromote(None, side, board, fro, to, single=True)
            yield [fro, to, promote]
            
###########################################################################################
# Example of a move-generating function:
# Randomly choose a move.
def random(side, board, flags, chooser):
    '''
    Return a random move, resulting board, and value of the resulting board.
    Return: (value, moveList, boardList)
      value (int or float): value of the board after making the chosen move
      moveList (list): list with one element, the chosen move
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
    Input:
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      chooser: a function similar to random.choice, but during autograding, might not be random.
    '''
    moves = [ move for move in generateMoves(side, board, flags) ]
    if len(moves) > 0:
        move = chooser(moves)
        newside, newboard, newflags = makeMove(side, board, move[0], move[1], flags, move[2])
        value = evaluate(newboard)
        return (value, [ move ], { encode(*move): {} })
    else:
        return (evaluate(board), [], {})

###########################################################################################
# Stuff you need to write:
# Move-generating functions using minimax, alphabeta, and stochastic search.
def minimax(side, board, flags, depth):
    '''
    Return a minimax-optimal move sequence, tree of all boards evaluated, and value of best path.
    Return: (value, moveList, moveTree)
      value (float): value of the final board in the minimax-optimal move sequence
      moveList (list): the minimax-optimal move sequence, as a list of moves
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
    Input:
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      depth (int >=0): depth of the search (number of moves)
    '''
    
    moves = [ move for move in generateMoves(side, board, flags) ]

    # If depth is 0, return the evaluation of the current board state
    if depth == 0:
      value = chess.lib.heuristics.evaluate(board)
      moveList = []
      moveTree = dict()
      return (value, moveList, moveTree)

    # Max player's turn
    elif (side == False):
      value = -math.inf
      moveList = []
      moveTree = dict()
      for move in generateMoves(side,board,flags):
        newside, newboard, newflags = chess.lib.makeMove(side, board, move[0], move[1], flags, move[2])
        temp_val, temp_moveList, temp_moveTree = minimax(newside, newboard, newflags, depth-1)
        moveTree[encode(*move)] = temp_moveTree
        if(temp_val > value):
          value = temp_val
          moveList.clear()
          temp_moveList.insert(0, move)
          moveList = temp_moveList.copy()
      return(value, moveList, moveTree)

    # Min player's turn
    else:
      value = math.inf
      moveList = []
      moveTree = dict()
      for move in generateMoves(side,board,flags):
        newside, newboard, newflags = chess.lib.makeMove(side, board, move[0], move[1], flags, move[2])
        temp_val, temp_moveList, temp_moveTree = minimax(newside, newboard, newflags, depth-1)
        moveTree[encode(*move)] = temp_moveTree
        if(temp_val < value):
          value = temp_val
          moveList.clear()
          temp_moveList.insert(0 ,move)
          moveList= temp_moveList.copy()
      return(value, moveList, moveTree)




def alphabeta(side, board, flags, depth, alpha=-math.inf, beta=math.inf):
    '''
    Return minimax-optimal move sequence, and a tree that exhibits alphabeta pruning.
    Return: (value, moveList, moveTree)
      value (float): value of the final board in the minimax-optimal move sequence
      moveList (list): the minimax-optimal move sequence, as a list of moves
      moveTree (dict: encode(*move)->dict): a tree of moves that were evaluated in the search process
    Input:
      side (boolean): True if player1 (Min) plays next, otherwise False
      board (2-tuple of lists): current board layout, used by generateMoves and makeMove
      flags (list of flags): list of flags, used by generateMoves and makeMove
      depth (int >=0): depth of the search (number of moves)
    '''

    moves = [ move for move in generateMoves(side, board, flags) ]

    # If depth is 0, return the evaluation of the current board state
    if depth == 0:
      value = chess.lib.heuristics.evaluate(board)
      moveList = []
      moveTree = dict()
      return (value, moveList, moveTree)

    # Max player's turn
    elif (side == False):
      value = -math.inf
      moveList = []
      moveTree = dict()
      for move in generateMoves(side,board,flags):
        newside, newboard, newflags = chess.lib.makeMove(side, board, move[0], move[1], flags, move[2])
        temp_val, temp_moveList, temp_moveTree = alphabeta(newside, newboard, newflags, depth-1, alpha = alpha, beta= beta)
        moveTree[encode(*move)] = temp_moveTree
        if(temp_val > value):
          value = temp_val
          moveList.clear()
          temp_moveList.insert(0, move)
          moveList = temp_moveList.copy()
          if temp_val >= beta:
            break
          alpha = max(alpha, temp_val)
      return(value, moveList, moveTree)

    # Min player's turn
    else:
      value = math.inf
      moveList = []
      moveTree = dict()
      for move in generateMoves(side,board,flags):
        newside, newboard, newflags = chess.lib.makeMove(side, board, move[0], move[1], flags, move[2])
        temp_val, temp_moveList, temp_moveTree = alphabeta(newside, newboard, newflags, depth-1, alpha = alpha, beta = beta)
        moveTree[encode(*move)] = temp_moveTree
        if(temp_val < value):
          value = temp_val
          moveList.clear()
          temp_moveList.insert(0 ,move)
          moveList= temp_moveList.copy()
          if temp_val <= alpha:
            break
          beta = min(beta, temp_val)
      return(value, moveList, moveTree)


