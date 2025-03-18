from dataclasses import dataclass
from enum import Enum, Flag, IntEnum, IntFlag
from typing import overload

from config import *
from graphics import *


PIECES = dict(zip(
    "KQRBNPkqrbnp",
    (King, Queen, Rook, Bishop, Knight, Pawn) * 2
))

COORDS =    {chr(x+96) + str(z):(x,z) for x in range(1, 9) for z in range(1, 9)}
POSITIONS = {chr(x+97) + str(z+1):(z*8) + x for x in range(8) for z in range(8)}
ALPHNUM =   {(x,z):chr(x+96) + str(z) for x in range(1, 9) for z in range(1, 9)}
ALPHNUM2 =  {(x + z*8):chr(x+97) + str(z+1) for x in range(8) for z in range(8)}

ORTHOGS = [(-1, 0), (0, -1), (0, 1), (1, 0)]
DIAGS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
KNIGHTDIRS = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]


class Pieces(Enum):
    """
    Enum of piece types
    """

    NONE = 0
    KING = 1
    QUEEN = 2
    ROOK = 3
    BISHOP = 4
    KNIGHT = 5
    PAWN = 6

    @classmethod
    def fromclass(cls, p: PieceType | type[PieceType]) -> "Pieces":
        return {
            King: Pieces.KING,
            Queen: Pieces.QUEEN,
            Rook: Pieces.ROOK,
            Bishop: Pieces.BISHOP,
            Knight: Pieces.KNIGHT,
            Pawn: Pieces.PAWN,

        }[p.__class__] # type: ignore

    @classmethod
    def fromlabel(cls, p: str) -> "Pieces":
        return {
            "k": Pieces.KING,
            "q": Pieces.QUEEN,
            "r": Pieces.ROOK,
            "b": Pieces.BISHOP,
            "n": Pieces.KNIGHT,
            "p": Pieces.PAWN,

        }[p.lower()]

class Square(IntEnum):
    BLACK = 0
    WHITE = 1
    EMPTY = 2
    INVALID = 3

class Side(IntEnum):
    BLACK = 0
    WHITE = 1

    def __invert__(self) -> "Side":
        if self.value:
            return Side.BLACK
        else:
            return Side.WHITE

class CastleRights(Flag):
    K = 1
    Q = 2
    k = 4
    q = 8

    WHITE = K | Q
    BLACK = k | q

CASTLEALL = CastleRights(15)

class MoveFlags(IntFlag):
    """
        P   T   2   1
     0: 0   0   0   0   quiet moves
     1: 0   0   0   1   double pawn push
     2: 0   0   1   0   king castle
     3: 0   0   1   1   queen castle
     4: 0   1   0   0   capture
     5: 0   1   0   1   en-passant capture
     8: 1   0   0   0   queen-promotion
     9: 1   0   0   1   rook-promotion
    10: 1   0   1   0   bishop-promotion
    11: 1   0   1   1   knight-promotion
    12: 1   1   0   0   queen-promotion capture
    13: 1   1   0   1   rook-promotion capture
    14: 1   1   1   0   bishop-promotion capture
    15: 1   1   1   1   knight-promotion capture
    """

    NONE = 0
    DATA1 = 1
    DATA2 = 2
    CAPTURE = 4
    PROMO = 8

    QUIET     = NONE
    DBLPAWN   = DATA1
    CASTLE    = DATA2
    KCASTLE   = DATA2
    QCASTLE   = DATA2 | DATA1
    ENPASSANT = CAPTURE | DATA1
    QPROMO    = PROMO
    RPROMO    = PROMO | DATA1
    BPROMO    = PROMO | DATA2
    NPROMO    = PROMO | DATA2 | DATA1
    QCAPPROM  = PROMO | CAPTURE
    RCAPPROM  = PROMO | CAPTURE | DATA1
    BCAPPROM  = PROMO | CAPTURE | DATA2
    NCAPPROM  = PROMO | CAPTURE | DATA2 | DATA1


@dataclass(frozen=True)
class Move:
    turn: Side
    origin: int # 0-63
    target: int # 0-63
    flags: MoveFlags

    def __post_init__(self) -> None:
        object.__setattr__(self, "captures", self.flags & MoveFlags.CAPTURE)
        object.__setattr__(self, "bin", hash(self))
        object.__setattr__(self, "__hash__", lambda s: self.bin) # type: ignore

    def __hash__(self) -> int:
        return self.turn << 16 + self.flags << 12 + self.origin << 6 + self.target

    def __repr__(self) -> str:
        return f"{ALPHNUM2[self.origin]}{'x' if self.flags & MoveFlags.CAPTURE else ''}{ALPHNUM2[self.target]}"


class Board:
    pieces: list[PieceType] # array of pieces objects on board to be rendered
    board: list[Pieces] # array of piece types
    squares: list[Square] # array of square states
    objmap: list[int] # mapping (index) between squares and pieces (-1 = empty)
    turn: Side
    castling: CastleRights # which sides are castling available on
    epsq: int # square behind en-passantable pawn (-1 = none)
    halfmoves: int # half moves since capture or pawn move (50 move rule)
    nmoves: int # full moves since start of game (starts at 1)

    def __init__(self,
                 pieces: list[PieceType],
                 board: list[Pieces],
                 squares: list[Square],
                 objmap: list[int],
                 turn: Side=Side.WHITE,
                 castling: CastleRights=CASTLEALL,
                 epsq: int=-1,
                 halfmoves: int=0,
                 nmoves: int=1):

        self.pieces = pieces
        self.board = board
        self.squares = squares
        self.objmap = objmap
        self.turn = turn
        self.castling = castling
        self.epsq = epsq
        self.halfmoves = halfmoves
        self.nmoves = nmoves

        self.redraw = True # flag to signal renderer to redraw

    def __repr__(self) -> str:
        out = ""

        for r in range(7, -1, -1):
            for f in range(8):
                i = r * 8 + f

                c = self.squares[i].name[0] + self.board[i].name[0]

                if c == "EN":
                    c = "  "

                out += c + " "

            out += "\n"

        return out + "\n"

    def toFEN(self) -> str: ... # TODO: Implement me

    @classmethod
    def fromFEN(cls, fen: str) -> "Board":
        pcs, side, cstl, ep, hm, fm = fen.split(" ")
        pieces = []
        board = [Pieces.NONE] * 64
        squares = [Square.EMPTY] * 64
        objmap = [-1] * 64
        i = 0

        for y, rnk in enumerate(pcs.split("/")):
            if rnk[0] == "8":
                continue

            x = 0
            for c in rnk:
                if c in "1234567":
                    x += int(c)
                else:
                    ps = c.isupper()

                    p = PIECES[c](
                        (x - 3.5, 0, 3.5 - y),
                        ps
                    )
                    pieces.append(p)
                    objmap[x + (7 - y) * 8] = i

                    board[x + (7 - y) * 8] = Pieces.fromlabel(c)
                    squares[x + (7 - y) * 8] = (Square.WHITE if ps else Square.BLACK)

                    i += 1

                x += 1

        c = 0
        if cstl != "-":
            c = ("K" in cstl)     +\
                ("Q" in cstl) * 2 +\
                ("k" in cstl) * 4 +\
                ("q" in cstl) * 8

        turn = (Side.WHITE if side == "w" else Side.BLACK)
        castling = CastleRights(c)
        epsq = -1 if ep == "-" else POSITIONS[ep]
        halfmoves = int(hm)
        nmoves = int(fm)

        return cls(pieces, board, squares, objmap, turn, castling, epsq, halfmoves, nmoves)

    # pass in either index or rank&file
    @overload
    def get(self, sqr: int, /) -> tuple[Square, Pieces]: ...

    @overload
    def get(self, rank: int, file: int, /) -> tuple[Square, Pieces]: ...

    def get(self, rank: int, file: int=-1, /) -> tuple[Square, Pieces]:
        if file == -1:
            sqr = rank
            rank, file = divmod(sqr, 8)
        else:
            sqr = rank * 8 + file

        if -1 < rank < 8 and -1 < file < 8:
            return self.squares[sqr], self.board[sqr]
        else:
            return Square.INVALID, Pieces.NONE

    def isattacked(self, sqr: int, side: Side) -> bool:
        "does opposite of side control this square?"

        rank, file = divmod(sqr, 8)

        for dr, df in ORTHOGS + DIAGS:
            r, f = rank - dr, file - df
            state, piece = self.get(r, f)

            if state == ~side and piece == Pieces.KING:
                return True

        for dr, df in ORTHOGS:
            r, f = rank, file
            for d in range(7):
                r -= dr
                f -= df

                if not (-1 < r < 8 and -1 < f < 8):
                    break

                state, piece = self.get(r, f)

                if state == side:
                    break
                elif state == ~side and (piece == Pieces.ROOK or piece == Pieces.QUEEN):
                    return True

        for dr, df in DIAGS:
            r, f = rank, file
            for d in range(7):
                r -= dr
                f -= df

                if not (-1 < r < 8 and -1 < f < 8):
                    break

                state, piece = self.get(r, f)

                if state == side:
                    break
                elif state == ~side and (piece == Pieces.BISHOP or piece == Pieces.QUEEN):
                    return True

        for dr, df in KNIGHTDIRS:
            r, f = rank - dr, file - df
            state, piece = self.get(r, f)

            if state == ~side and piece == Pieces.KNIGHT:
                return True

        for dr, df in ([(1, -1), (1, 1)] if side else [(-1, -1), (-1, 1)]):
            r, f = rank - dr, file - df
            state, piece = self.get(r, f)

            if state == ~side and piece == Pieces.PAWN:
                return True

        return False

    def isattackedafter(self, sqr: int, move: Move) -> bool:
        side = move.turn
        ro, fo = divmod(move.origin, 8)
        rt, ft = divmod(move.target, 8)
        rank, file = divmod(sqr, 8)

        for dr, df in ORTHOGS + DIAGS:
            r, f = rank - dr, file - df
            state, piece = self.get(r, f) # will prevent overflow

            if state == ~side and piece == Pieces.KING:
                return True

        for dr, df in ORTHOGS:
            r, f = rank, file
            for d in range(7):
                r -= dr
                f -= df

                if not (-1 < r < 8 and -1 < f < 8):
                    break

                state, piece = self.get(r, f)

                if (r == ro and f == fo) or (r + 1 - 2 * side) * 8 + f == self.epsq: # move has unblocked rank/file
                    continue
                elif state == side or (r == rt and f == ft): # move has blocked rank/file
                    break
                elif state == ~side and (piece == Pieces.ROOK or piece == Pieces.QUEEN):
                    return True

        for dr, df in DIAGS:
            r, f = rank, file
            for d in range(7):
                r -= dr
                f -= df

                if not (-1 < r < 8 and -1 < f < 8):
                    break

                state, piece = self.get(r, f)

                if (r == ro and f == fo) or (r + 1 - 2 * side) * 8 + f == self.epsq: # move has unblocked diagonal
                    continue
                elif state == side or (r == rt and f == ft): # move has blocked diagonal
                    break
                elif ~side and (piece == Pieces.BISHOP or piece == Pieces.QUEEN):
                    return True

        for dr, df in KNIGHTDIRS:
            r, f = rank - dr, file - df
            state, piece = self.get(r, f)

            if r == rt and f == ft: # move may have captured knight
                continue
            elif state == ~side and piece == Pieces.KNIGHT:
                return True

        for dr, df in ([(1, -1), (1, 1)] if side else [(-1, -1), (-1, 1)]):
            r, f = rank - dr, file - df
            state, piece = self.get(r, f)

            if (r == rt and f == ft) or (r == ro and f == fo):
                continue
            elif state == ~side and piece == Pieces.PAWN:
                return True

        return False

    def verify(self, move: Move) -> bool:
        turn = move.turn
        target = move.target
        origin = move.origin
        flags = move.flags

        state, piece = self.get(origin)
        assert state == turn, move
        assert piece != Pieces.NONE, move

        if piece == Pieces.KING:
            if self.isattacked(origin, turn) or self.isattacked(target, turn): # can't move into check
                return False

            elif flags & 0b1110 == MoveFlags.CASTLE:
                # TODO: We should probably check if rooks are in the right place

                if flags == MoveFlags.KCASTLE:
                    if turn:
                        if CastleRights.K not in self.castling:
                            return False
                        elif not self.squares[5] == self.squares[6] == Square.EMPTY:
                            return False
                        elif self.isattacked(5, turn): # can't castle through check
                            return False
                    else:
                        if CastleRights.k not in self.castling:
                            return False
                        elif not self.squares[61] == self.squares[62] == Square.EMPTY:
                            return False
                        elif self.isattacked(61, turn): # can't castle through check
                            return False
                else:
                    if turn:
                        if CastleRights.Q not in self.castling:
                            return False
                        elif not self.squares[1] == self.squares[2] == self.squares[3] == Square.EMPTY:
                            return False
                        elif self.isattacked(3, turn): # can't castle through check
                            return False
                    else:
                        if CastleRights.q not in self.castling:
                            return False
                        elif not self.squares[57] == self.squares[58] == self.squares[59] == Square.EMPTY:
                            return False
                        elif self.isattacked(59, turn): # can't castle through check
                            return False

        else:
            kingsqr = self.board.index(Pieces.KING)
            if self.squares[kingsqr] != turn: # wrong king has been picked
                kingsqr = self.board.index(Pieces.KING, kingsqr + 1)

            if self.isattackedafter(kingsqr, move): # can't stay in check
                return False

        return True

    def _gen_king(self, sqr: int, turn: Side) -> list[Move]:
        rank, file = divmod(sqr, 8)
        moves = []

        for dr, df in ORTHOGS + DIAGS:
            if df or dr: # must move
                r, f = rank + dr, file + df

                if not (-1 < r < 8 and -1 < f < 8):
                    continue

                target = (8 * r) + f
                state, _ = self.get(r, f)

                if state == Square.INVALID or state == turn:
                    continue
                elif state == Square.EMPTY:
                    moves.append(Move(turn, sqr, target, MoveFlags.QUIET))
                else:
                    moves.append(Move(turn, sqr, target, MoveFlags.CAPTURE))

        if turn:
            if CastleRights.K in self.castling and self.squares[5] == self.squares[6] == Square.EMPTY:
                moves.append(Move(turn, sqr, 6, MoveFlags.KCASTLE))
            elif CastleRights.Q in self.castling and self.squares[1] == self.squares[2] == self.squares[3] == Square.EMPTY:
                moves.append(Move(turn, sqr, 2, MoveFlags.QCASTLE))
        elif CastleRights.k in self.castling and self.squares[61] == self.squares[62] == Square.EMPTY:
            moves.append(Move(turn, sqr, 62, MoveFlags.KCASTLE))
        elif CastleRights.q in self.castling and self.squares[57] == self.squares[58] == self.squares[59] == Square.EMPTY:
            moves.append(Move(turn, sqr, 58, MoveFlags.QCASTLE))

        return moves

    def _gen_queen(self, sqr: int, turn: Side) -> list[Move]:
        rank, file = divmod(sqr, 8)
        moves = []

        for dr, df in ORTHOGS + DIAGS:
            r, f = rank, file

            for d in range(7):
                r += dr
                f += df

                if not (-1 < r < 8 and -1 < f < 8):
                    break

                target = (8 * r) + f
                state, _ = self.get(r, f)

                if state == Square.INVALID or state == turn:
                    break
                elif state == Square.EMPTY:
                    moves.append(Move(turn, sqr, target, MoveFlags.QUIET))
                else:
                    moves.append(Move(turn, sqr, target, MoveFlags.CAPTURE))

                    break

        return moves

    def _gen_rook(self, sqr: int, turn: Side) -> list[Move]:
        rank, file = divmod(sqr, 8)
        moves = []

        for dr, df in ORTHOGS:
            r, f = rank, file

            for d in range(7):
                r += dr
                f += df

                if not (-1 < r < 8 and -1 < f < 8):
                    break

                target = (8 * r) + f
                state, _ = self.get(r, f)

                if state == Square.INVALID or state == turn:
                    break
                elif state == Square.EMPTY:
                    moves.append(Move(turn, sqr, target, MoveFlags.QUIET))
                else:
                    moves.append(Move(turn, sqr, target, MoveFlags.CAPTURE))

                    break

        return moves

    def _gen_bishop(self, sqr: int, turn: Side) -> list[Move]:
        rank, file = divmod(sqr, 8)
        moves = []

        for dr, df in DIAGS:
            r, f = rank, file

            for d in range(7):
                r += dr
                f += df

                if not (-1 < r < 8 and -1 < f < 8):
                    break

                target = (8 * r) + f
                state, _ = self.get(r, f)

                if state == Square.INVALID or state == turn:
                    break
                elif state == Square.EMPTY:
                    moves.append(Move(turn, sqr, target, MoveFlags.QUIET))
                else:
                    moves.append(Move(turn, sqr, target, MoveFlags.CAPTURE))

                    break

        return moves

    def _gen_knight(self, sqr: int, turn: Side) -> list[Move]:
        rank, file = divmod(sqr, 8)
        moves = []

        for dr, df in KNIGHTDIRS:
            r, f = rank + dr, file + df

            if not (-1 < r < 8 and -1 < f < 8):
                continue

            target = (8 * r) + f
            state, _ = self.get(r, f)

            if state == Square.INVALID or state == turn:
                continue
            elif state == Square.EMPTY:
                moves.append(Move(turn, sqr, target, MoveFlags.QUIET))
            else:
                moves.append(Move(turn, sqr, target, MoveFlags.CAPTURE))

        return moves

    def _gen_pawn(self, sqr: int, turn: Side) -> list[Move]:
        rank, file = divmod(sqr, 8)
        moves = []

        if turn:
            if rank == 6: # ready for promotion
                if self.get(7, file)[0] == Square.EMPTY: # push promotion
                    moves.append(Move(turn, sqr, sqr + 8, MoveFlags.QPROMO))
                    moves.append(Move(turn, sqr, sqr + 8, MoveFlags.RPROMO))
                    moves.append(Move(turn, sqr, sqr + 8, MoveFlags.BPROMO))
                    moves.append(Move(turn, sqr, sqr + 8, MoveFlags.NPROMO))

                if 0 < file and self.get(7, file - 1)[0] == Square.BLACK: # capture left promotion
                    moves.append(Move(turn, sqr, sqr + 7, MoveFlags.QCAPPROM))
                    moves.append(Move(turn, sqr, sqr + 7, MoveFlags.RCAPPROM))
                    moves.append(Move(turn, sqr, sqr + 7, MoveFlags.BCAPPROM))
                    moves.append(Move(turn, sqr, sqr + 7, MoveFlags.NCAPPROM))

                if file < 7 and self.get(7, file + 1)[0] == Square.BLACK: # capture right promotion
                    moves.append(Move(turn, sqr, sqr + 9, MoveFlags.QCAPPROM))
                    moves.append(Move(turn, sqr, sqr + 9, MoveFlags.RCAPPROM))
                    moves.append(Move(turn, sqr, sqr + 9, MoveFlags.BCAPPROM))
                    moves.append(Move(turn, sqr, sqr + 9, MoveFlags.NCAPPROM))
            else:
                if self.get(rank + 1, file)[0] == Square.EMPTY: # push
                    moves.append(Move(turn, sqr, sqr + 8, MoveFlags.QUIET))

                    if rank == 1 and self.get(3, file)[0] == Square.EMPTY: # double push
                        moves.append(Move(turn, sqr, sqr + 16, MoveFlags.DBLPAWN))

                if 0 < file and self.get(rank + 1, file - 1)[0] == Square.BLACK: # capture left
                    moves.append(Move(turn, sqr, sqr + 7, MoveFlags.CAPTURE))

                if file < 7 and self.get(rank + 1, file + 1)[0] == Square.BLACK: # capture right
                    moves.append(Move(turn, sqr, sqr + 9, MoveFlags.CAPTURE))

                target = self.epsq

                if target != -1 and rank == 4: # en passant
                    if target == sqr + 7: # en passant left
                        moves.append(Move(turn, sqr, target, MoveFlags.ENPASSANT))

                    elif target == sqr + 9: # en passant right
                        moves.append(Move(turn, sqr, target, MoveFlags.ENPASSANT))
        else:
            if rank == 1: # ready for promotion
                if self.get(0, file)[0] == Square.EMPTY: # push promotion
                    moves.append(Move(turn, sqr, sqr - 8, MoveFlags.QPROMO))
                    moves.append(Move(turn, sqr, sqr - 8, MoveFlags.RPROMO))
                    moves.append(Move(turn, sqr, sqr - 8, MoveFlags.BPROMO))
                    moves.append(Move(turn, sqr, sqr - 8, MoveFlags.NPROMO))

                if file < 7 and self.get(0, file + 1)[0] == Square.WHITE: # capture left promotion
                    moves.append(Move(turn, sqr, sqr - 7, MoveFlags.QCAPPROM))
                    moves.append(Move(turn, sqr, sqr - 7, MoveFlags.RCAPPROM))
                    moves.append(Move(turn, sqr, sqr - 7, MoveFlags.BCAPPROM))
                    moves.append(Move(turn, sqr, sqr - 7, MoveFlags.NCAPPROM))

                if 0 < file and self.get(0, file - 1)[0] == Square.WHITE: # capture right promotion
                    moves.append(Move(turn, sqr, sqr - 9, MoveFlags.QCAPPROM))
                    moves.append(Move(turn, sqr, sqr - 9, MoveFlags.RCAPPROM))
                    moves.append(Move(turn, sqr, sqr - 9, MoveFlags.BCAPPROM))
                    moves.append(Move(turn, sqr, sqr - 9, MoveFlags.NCAPPROM))
            else:
                if self.get(rank - 1, file)[0] == Square.EMPTY: # push
                    moves.append(Move(turn, sqr, sqr - 8, MoveFlags.QUIET))

                    if rank == 6 and self.get(4, file)[0] == Square.EMPTY: # double push
                        moves.append(Move(turn, sqr, sqr - 16, MoveFlags.DBLPAWN))

                if file < 7 and self.get(rank - 1, file + 1)[0] == Square.WHITE: # capture left
                    moves.append(Move(turn, sqr, sqr - 7, MoveFlags.CAPTURE))

                if 0 < file and self.get(rank - 1, file - 1)[0] == Square.WHITE: # capture right
                    moves.append(Move(turn, sqr, sqr - 9, MoveFlags.CAPTURE))

                target = self.epsq

                if target != -1 and rank == 3: # en passant
                    if target == sqr - 7: # en passant left
                        moves.append(Move(turn, sqr, target, MoveFlags.ENPASSANT))

                    elif target == sqr - 9: # en passant right
                        moves.append(Move(turn, sqr, target, MoveFlags.ENPASSANT))

        return moves

    def gen_moves(self) -> list[Move]:
        turn = self.turn
        moves: list[Move] = []

        for i in range(64):
            if self.squares[i] != turn:
                continue

            match self.board[i]:
                case Pieces.KING:
                    moves.extend(self._gen_king(i, turn))
                case Pieces.QUEEN:
                    moves.extend(self._gen_queen(i, turn))
                case Pieces.ROOK:
                    moves.extend(self._gen_rook(i, turn))
                case Pieces.BISHOP:
                    moves.extend(self._gen_bishop(i, turn))
                case Pieces.KNIGHT:
                    moves.extend(self._gen_knight(i, turn))
                case Pieces.PAWN:
                    moves.extend(self._gen_pawn(i, turn))

        moves = [m for m in moves if self.verify(m)]

        return moves

    def make_move(self, move: Move) -> None:
        turn = move.turn
        target = move.target
        origin = move.origin
        flags = move.flags

        oindex = self.objmap[origin]
        assert oindex != -1, move

        if isinstance(self.pieces[oindex], Pawn):
            self.halfmoves = -1

        if MoveFlags.CAPTURE not in flags:
            assert self.board[target] == Pieces.NONE, move
            assert self.squares[target] == Square.EMPTY, move
            assert self.objmap[target] == -1, move
        elif flags != MoveFlags.ENPASSANT:
            assert self.board[target] != Pieces.NONE, move
            assert self.squares[target] == ~turn, move
            assert self.objmap[target] != -1, move

        if MoveFlags.PROMO in flags:
            if flags & 0b111 == MoveFlags.QPROMO:
                obj = self.pieces[oindex]
                self.pieces[oindex] = Queen(side=turn)
                del obj

            elif flags & 0b111 == MoveFlags.RPROMO:
                obj = self.pieces[oindex]
                self.pieces[oindex] = Rook(side=turn)
                del obj

            elif flags & 0b111 == MoveFlags.BPROMO:
                obj = self.pieces[oindex]
                self.pieces[oindex] = Bishop(side=turn)
                del obj

            elif flags & 0b111 == MoveFlags.NPROMO:
                obj = self.pieces[oindex]
                self.pieces[oindex] = Knight(side=turn)
                del obj

        elif self.board[origin] == Pieces.KING:
            if turn:
                self.castling &= ~CastleRights.WHITE
            else:
                self.castling &= ~CastleRights.BLACK

        elif self.board[origin] == Pieces.ROOK or self.board[target] == Pieces.ROOK:
            if origin == 0 or target == 0:
                self.castling &= ~CastleRights.Q
            elif origin == 7 or target == 7:
                self.castling &= ~CastleRights.K
            elif origin == 56 or target == 56:
                    self.castling &= ~CastleRights.q
            elif origin == 63 or target == 63:
                    self.castling &= ~CastleRights.k

        self.board[origin] = Pieces.NONE
        self.squares[origin] = Square.EMPTY

        self.board[target] = Pieces.fromclass(self.pieces[oindex])
        self.squares[target] = Square(turn)

        self.pieces[oindex].torankfile(*divmod(target, 8))

        if flags == MoveFlags.ENPASSANT:
            i = target - 16 * self.turn + 8

            tindex = self.objmap[i]
            assert tindex != -1, move

            taken = self.pieces.pop(tindex)
            del taken

            for i in range(len(self.objmap)):
                if self.objmap[i] > tindex:
                    self.objmap[i] -= 1

            self.objmap[i] = -1
            self.board[i] = Pieces.NONE
            self.squares[i] = Square.EMPTY

        elif MoveFlags.CAPTURE in flags:
            tindex = self.objmap[target]
            assert tindex != -1, move

            taken = self.pieces.pop(tindex)
            del taken

            for i in range(len(self.objmap)):
                if self.objmap[i] > tindex:
                    self.objmap[i] -= 1

        elif flags & 0b1110 == MoveFlags.CASTLE:
            if turn:
                if flags == MoveFlags.KCASTLE:
                    self.castling &= ~CastleRights.K
                    ro = 7
                    rt = 5
                else:
                    self.castling &= ~CastleRights.Q
                    ro = 0
                    rt = 3
            else:
                if flags == MoveFlags.KCASTLE:
                    self.castling &= ~CastleRights.k
                    ro = 63
                    rt = 61
                else:
                    self.castling &= ~CastleRights.q
                    ro = 56
                    rt = 59

            rindex = self.objmap[ro]
            assert rindex != -1, move

            self.board[ro] = Pieces.NONE
            self.squares[ro] = Square.EMPTY

            self.board[rt] = Pieces.ROOK
            self.squares[rt] = Square(turn)

            self.pieces[rindex].torankfile(*divmod(rt, 8))

            self.objmap[rt] = self.objmap[ro]
            self.objmap[ro] = -1

        if flags == MoveFlags.DBLPAWN:
            epsq = target - 16 * turn + 8
        else:
            epsq = -1

        self.objmap[target] = self.objmap[origin]
        self.objmap[origin] = -1

        self.halfmoves += 1
        self.nmoves += 1 - turn
        self.turn = ~turn
        self.epsq = epsq