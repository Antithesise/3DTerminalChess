from sys import stdin, stdout
from platform import system


if system() == "Windows":
    from msvcrt import getch, kbhit # type: ignore

else:
    from termios import TCSADRAIN, tcgetattr, tcsetattr # type: ignore
    from select import select # type: ignore
    from tty import setraw # type: ignore
    from sys import stdin # type: ignore


    fd = stdin.fileno()
    old_settings = tcgetattr(fd)

    def getch() -> bytes:
        try:
            setraw(fd)

            return stdin.read(1).encode()
        finally:
            tcsetattr(fd, TCSADRAIN, old_settings)

    def kbhit() -> bool:
        return bool(select([stdin], [], [], 0)[0])


def isansitty() -> bool:
    """
    Checks if stdout supports ANSI escape codes and is a tty.
    """

    while kbhit():
        getch()

    stdout.write("\x1b[6n")
    stdout.flush()

    stdin.flush()
    if kbhit():
        if ord(getch()) == 27 and kbhit():
            if getch() == b"[":
                while kbhit():
                    getch()

                return stdout.isatty()

    return False


if __name__ == "__main__":
    while True:
        c = getch()
        print(f"{c}: {ord(c)}")
