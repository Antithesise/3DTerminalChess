from time import sleep, time

from engine import *
from graphics import *
from input import getch, isansitty, kbhit


def handlein(camera: Camera, dt: float) -> bool:
    redraw = False
    key = ord(getch())

    if key in [0, 224]:
        key = ord(getch())

        match key:
            case 72: # up
                camera.rot += [-100*dt, 0, 0]
            case 75: # left
                camera.rot += [0, -100*dt, 0]
            case 77: # right
                camera.rot += [0, 100*dt, 0]
            case 80: # down
                camera.rot += [100*dt, 0, 0]

            # case 83: # delete
            #     pass

            # case 115: # ctrl+left
            #     pass

            # case 116: # ctrl+right
            #     pass

            # case _:
            #     pass

        if key in [72, 75, 77, 80]:
            redraw = True

    else:
        match key:
            # case 1: # ctrl+a
            #     pass

            # case 8: # backspace
            #     pass

            case 9: # tab:
                camera.pos += [0, -4*dt, 0]

            # case 10 | 13: # newline / carriage return
            #     pass

            # case 14: # ctrl+n:
            #     pass

            # case 15: # ctrl+o:
            #     pass

            # case 17: # ctrl+q
            #     pass

            # case 19: # ctrl+s:
            #     pass

            # case 22: # ctrl+v, sometimes?
            #     pass

            # case 23: # ctrl+w:
            #     pass

            # case 27: # escape:
            #     pass

            # case 31: # ctrl+/
            #     pass

            # case k if k < 32: # any other weird sequence
            #     pass

            case 32: # space
                camera.pos += [0, 4*dt, 0]
            case 97: # a
                camera.pos += [-4*dt, 0, 0] @ camera.yrotmat
            case 100: # d
                camera.pos += [4*dt, 0, 0]  @ camera.yrotmat
            case 101: # e
                camera.pos += [2.8*dt, 0, 2.8*dt] @ camera.yrotmat
            case 113: # q
                camera.pos += [-2.8*dt, 0, 2.8*dt] @ camera.yrotmat
            case 115: # s
                camera.pos += [0, 0, -4*dt] @ camera.yrotmat
            case 119: # w
                camera.pos += [0, 0, 4*dt] @ camera.yrotmat
            case 120: # x
                camera.pos = array((0, 6, -7), float64)
                camera.rot = array((40, 0, 0), float64)

            # case _:
            #     pass

        if key in [9, 32, 97, 100, 101, 113, 115, 119, 120]:
            redraw = True

    return redraw


def main() -> None:
    board = Board.fromFEN(STARTFEN)
    camera = Camera((0, 6, -7), (40, 0, 0))

    cbpoints = tuple((x, -1, z) for z in range(-4, 5) for x in range(-4, 5) if -8 != x + z != 8)
    cbtris   = tuple((i, i + 10, i + 9) for i in range(0, 70, 2) if i % 18 < 16) +\
               tuple((i, i + 1, i + 10) for i in range(0, 70, 2) if i % 18 < 16)
    cb       = Mesh(cbpoints, cbtris, (), fill=".", worldloc=True)

    # xgrid = [(Line((x, -1, -4), (x, -1, 4)), {"overdraw": True}) for x in range(-3, 4)]
    # zgrid = [(Line((-4, -1, z), (4, -1, z)), {"overdraw": True}) for z in range(-3, 4)]
    # bpoints = [
    #     (-4, -1, -4),
    #     (-4, -1, 4),
    #     (4, -1, 4),
    #     (4, -1, -4)
    # ]
    # border = [(Line(bpoints[i], bpoints[(i+1) % 4]), {"overdraw": True}) for i in range(4)]

    objects = [
        cb,
        # *xgrid,
        # *zgrid,
        # *border,
    ]

    redraw = True
    t = time()

    tt = 0
    n = 0

    while True:
        if redraw:
            redraw = False

            t = time()
            res = render(objects + board.pieces, camera)
            dt = time() - t or FMIN

            if DEBUG:
                fps = 1/dt
                tt += fps
                n += 1

                res += (
                    f"\n\nDelta Time: {dt*1000:3.0f} ms"
                    f"\nExtrap FPS: {fps:6.2f}"
                    f"\n  Mean FPS: {tt/n:6.2f}"
                )

            print(end=res, flush=True)

        while not kbhit():
            sleep(0.0001)

        redraw = handlein(camera, dt + 0.02) # account for render/input time etc.


if __name__ == "__main__":
    if not isansitty():
        exit("This system doesn't fully support ANSI escape codes.")

    try:
        print(end="\x1b7\x1b[?1049h\x1b[?25l")

        main()
    finally:
        print(end="\x1b[?1049l\x1b8\x1b[?25h")
