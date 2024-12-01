from numpy import array
from enum import Flag

from graphics import *


PIECES = dict(zip(
    "KQRBNPkqrbnp",
    (King, Queen, Rook, Bishop, Knight, Pawn) * 2
))

CHORDS = {chr(x+96) + str(z):(x, z) for x in range(1, 9) for z in range(1, 9)}
ALPHNUM = {(x, z):chr(x+96) + str(z) for x in range(1, 9) for z in range(1, 9)}


class CastleRights(Flag):
    K = 1
    Q = 2
    k = 4
    q = 8


class Board:
    pieces: list[Piece]
    board: list[int] # array of indices into 'pieces' (-1 is empty)
    turn: bool # True -> white, False -> black
    castling: CastleRights # which sides are castling available on
    epsq: tuple[int, int] | None # square where en passant is possible
    halfmoves: int # half moves since capture or pawn move (50 move rule)
    moves: int # full moves since start of game (starts at 1)

    def __init__(self,
                 pieces: list[Piece],
                 board: list[int],
                 turn: bool=True,
                 castling: CastleRights=CastleRights(0),
                 epsq: tuple[int, int] | None=None,
                 halfmoves: int=0,
                 moves: int=1):

        self.pieces = pieces
        self.board = board
        self.turn = turn
        self.castling = castling
        self.epsq = epsq
        self.halfmoves = halfmoves
        self.moves = moves

    @classmethod
    def fromFEN(cls, fen: str) -> "Board":
        pcs, side, cstl, ep, hm, fm = fen.split(" ")
        pieces = []
        board = [-1] * 64
        i = 0

        for y, rnk in enumerate(pcs.split("/")):
            if rnk[0] == "8":
                continue

            x = 0
            for c in rnk:
                if c in "1234567":
                    x += int(c)
                else:
                    p = PIECES[c](
                        (x - 3.5, 0, 3.5 - y),
                        c.isupper()
                    )
                    pieces.append(p)
                    board[x + (7 - y) * 8] = i
                    i += 1

                x += 1

        c = 0
        if cstl != "-":
            c = ("K" in cstl)     +\
                ("Q" in cstl) * 2 +\
                ("k" in cstl) * 4 +\
                ("q" in cstl) * 8

        turn = side == "w"
        castling = CastleRights(c)
        epsq = None if ep == "-" else CHORDS[ep]
        halfmoves = int(hm)
        moves = int(fm)

        return cls(pieces, board, turn, castling, epsq, halfmoves, moves)
