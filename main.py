from time import sleep

from input import getch, isansitty, kbhit
from graphics import *


def handlein(camera: Camera) -> bool:
    redraw = False
    key = ord(getch())

    if key in [0, 224]:
        key = ord(getch())

        match key:
            case 72: # up
                camera.rot += [-5, 0, 0]
            case 75: # left
                camera.rot += [0, -5, 0]
            case 77: # right
                camera.rot += [0, 5, 0]
            case 80: # down
                camera.rot += [5, 0, 0]

            case 83: # delete
                pass

            case 115: # ctrl+left
                pass

            case 116: # ctrl+right
                pass

            case _:
                pass

        if key in [75, 77]:
            redraw = True

        elif key in [72, 80]:
            redraw = True

    else:
        match key:
            case 1: # ctrl+a
                pass

            case 8: # backspace
                pass

            case 9: # tab:
                camera.pos += [0, -0.2, 0]

            case 10 | 13: # newline / carriage return
                pass

            case 14: # ctrl+n:
                pass

            case 15: # ctrl+o:
                pass

            case 17: # ctrl+q
                pass

            case 19: # ctrl+s:
                pass

            case 22: # ctrl+v, sometimes?
                pass

            case 23: # ctrl+w:
                pass

            case 27: # escape:
                pass

            case 31: # ctrl+/
                pass

            case k if k < 32: # any other weird sequence
                pass

            case 32: # space
                camera.pos += [0, 0.2, 0]
            case 113: # q
                camera.pos += [-0.14, 0, 0.14] @ camera.yrotmat
            case 119: # w
                camera.pos += [0, 0, 0.2] @ camera.yrotmat
            case 101: # e
                camera.pos += [0.14, 0, 0.14] @ camera.yrotmat
            case 97: # a
                camera.pos += [-0.2, 0, 0] @ camera.yrotmat
            case 115: # s
                camera.pos += [0, 0, -0.2] @ camera.yrotmat
            case 100: # d
                camera.pos += [0.2, 0, 0]  @ camera.yrotmat

            case _:
                pass

        if key in [9, 32, 97, 100, 101, 113, 115, 119]:
            redraw = True

    return redraw

def main() -> None:
    camera = Camera((0, 6, -7), (40, 0, 0))

    # xgrid = [(Line(Point((x, -1, -4)), Point((x, -1, 4))), {"overdraw": True}) for x in range(-3, 4)]
    # zgrid = [(Line(Point((-4, -1, z)), Point((4, -1, z))), {"overdraw": True}) for z in range(-3, 4)]
    border = [
        (Line(Point((-4, -1, -4)), Point((-4, -1, 4))), {"overdraw": True}),
        (Line(Point((4, -1, -4)), Point((4, -1, 4))), {"overdraw": True}),
        (Line(Point((-4, -1, -4)), Point((4, -1, -4))), {"overdraw": True}),
        (Line(Point((-4, -1, 4)), Point((4, -1, 4))), {"overdraw": True})
    ]

    checkerboard = []
    cbpoints = [Point((x, -1, z)) for z in range(-4, 5) for x in range(-4, 5) if -8 != x + z != 8]

    for i in range(0, 70, 2):
        if i % 18 >= 16:
            continue

        p1 = cbpoints[i]
        p2 = cbpoints[i + 10]
        p3 = cbpoints[i + 9]
        p4 = cbpoints[i + 1]

        checkerboard.append((Triangle(p3, p1, p2), {"char": ":"}))
        checkerboard.append((Triangle(p1, p4, p2), {"char": ":"}))

    # points = [Point((x, y, z)) for x in (-1,1) for y in (-1,1) for z in (-1,1)]
    # lines = {(0,4),(1,5,4,6),(5,7,6,2,3),(7,3,1,0,2)}
    # cube = Mesh(points, lines)

    WKing = King((0.5, 0, -3.5))
    WQueen = Queen((-0.5, 0, -3.5))
    WKRook = Rook((-3.5, 0, -3.5))
    WQRook = Rook((3.5, 0, -3.5))
    WKBishop = Bishop((-1.5, 0, -3.5))
    WQBishop = Bishop((1.5, 0, -3.5))
    WKKnight = Knight((-2.5, 0, -3.5))
    WQKnight = Knight((2.5, 0, -3.5))
    WPawns = [(Pawn((x - 3.5, 0, -2.5)), {"char": "#"}) for x in range(8)]

    BKing = King((0.5, 0, 3.5))
    BQueen = Queen((-0.5, 0, 3.5))
    BKRook = Rook((-3.5, 0, 3.5))
    BQRook = Rook((3.5, 0, 3.5))
    BKBishop = Bishop((-1.5, 0, 3.5))
    BQBishop = Bishop((1.5, 0, 3.5))
    BKKnight = Knight((-2.5, 0, 3.5))
    BQKnight = Knight((2.5, 0, 3.5))
    BPawns = [(Pawn((x - 3.5, 0, 2.5)), {"char": "'"}) for x in range(8)]

    objects = [
        *checkerboard,
        # *xgrid,
        # *zgrid,
        *border,
        (WKing, {"char": "#"}),
        (WQueen, {"char": "#"}),
        (WKRook, {"char": "#"}),
        (WQRook, {"char": "#"}),
        (WKBishop, {"char": "#"}),
        (WQBishop, {"char": "#"}),
        (WKKnight, {"char": "#"}),
        (WQKnight, {"char": "#"}),
        *WPawns,
        (BKing, {"char": "'"}),
        (BQueen, {"char": "'"}),
        (BKRook, {"char": "'"}),
        (BQRook, {"char": "'"}),
        (BKBishop, {"char": "'"}),
        (BQBishop, {"char": "'"}),
        (BKKnight, {"char": "'"}),
        (BQKnight, {"char": "'"}),
        *BPawns,
    ]

    redraw = True

    while True:
        if redraw:
            render(objects, camera)

            print(end="\x1b[%d;%dH\x1b[31m+\x1b[0m" % (VH/2, VW/2), flush=True)

        while not kbhit():
            sleep(0.01)

        redraw = handlein(camera)


if __name__ == "__main__":
    if not isansitty():
        exit("This system doesn't fully support ANSI escape codes.")

    try:
        print(end="\x1b7\x1b[?1049h\x1b[?25l")

        main()
    finally:
        print(end="\x1b[?1049l\x1b8\x1b[?25h")
