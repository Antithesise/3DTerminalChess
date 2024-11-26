from abc import ABC, abstractmethod
from bisect import bisect_left
from functools import lru_cache
from math import copysign, cos, radians, sin
# from multiprocessing import Pool
from os import get_terminal_size
from sys import float_info
from time import sleep
from typing import Collection, NamedTuple, Never, NoReturn, Optional, Self, overload
from numpy import typing as npt
import numpy as np

from input import getch, kbhit


CACHE_SIZE = 1024

VW, VH = get_terminal_size()
SCALE = 200
CRATIO = 1 / 2.32

FMIN = (5e-324 or float_info.min) # smallest floating point value

Vec3 = tuple[float, float, float] | npt.ArrayLike

class Vec2(NamedTuple):
    x: float
    y: float

    def __add__(self, o: "Vec2 | tuple[float, float]") -> "Vec2":
        return Vec2(self.x + o[0], self.y + o[1])

    def __sub__(self, o: "Vec2 | tuple[float, float]") -> "Vec2":
        return Vec2(self.x - o[0], self.y - o[1])

    def __mul__(self, s: float) -> "Vec2":
        return Vec2(self.x * s, self.y * s)

    def __div__(self, s: float) -> "Vec2":
        return Vec2(self.x / s, self.y / s)

    def __round__(self) -> "Vec2":
        return Vec2(round(self.x), round(self.y))

    def __eq__(self, o: "Vec2 | tuple[float, float]") -> bool:
        return self.x == o[0] and self.y == o[1]


global screenbuffer
screenbuffer = "\x1b[2J"

@lru_cache(CACHE_SIZE)
def gradmap(grad: float):
    chars = {
        -4: "|",
        -2: "\\",
        -1: "`",
        -0.5: "^",
        -0.2: "~",
        0: "-",
        0.2: "~",
        0.5: "^",
        1: ",",
        2: "/",
        4: "|"
    }

    keys = tuple(chars.keys())

    pos = bisect_left(keys, grad)
    if pos == 0:
        return chars[keys[0]]
    if pos == len(chars):
        return chars[keys[-1]]
    before = keys[pos - 1]
    after = keys[pos]
    if after - grad < grad - before:
        return chars[after]
    else:
        return chars[before]

@lru_cache(CACHE_SIZE)
def rotmat(rx: float, ry: float, rz: float) -> npt.NDArray:
    rx, ry, rz = radians(rx), radians(ry), radians(rz)

    return np.array([
        [ 1,       0,       0      ],
        [ 0,       cos(rx), sin(rx)],
        [ 0,      -sin(rx), cos(rx)]]) @ np.array([
        [ cos(ry), 0,      -sin(ry)],
        [ 0,       1,       0      ],
        [ sin(ry), 0,       cos(ry)]]) @ np.array([
        [ cos(rz), sin(rz), 0      ],
        [-sin(rz), cos(rz), 0      ],
        [ 0,       0,       1      ]])

@lru_cache(CACHE_SIZE)
def transform(x: float, y: float) -> Vec2:
    return Vec2(VW/2 + x * SCALE, VH/2 - y * SCALE * CRATIO)

def xyclip(p1: Vec2, p2: Vec2) -> tuple[Vec2, Vec2] | None:
    """
    liang-barsky for the win
    """

    dx, dy = p2 - p1

    p = [-dx, dx, -dy, dy]
    q = [p1.x - 1, VW - p1.x, p1.y - 1, VH - p1.y]
    start = 0
    stop = 1

    for i in range(4):
        if p[i] == 0:
            if q[i] < 0:
                return
        else:
            if p[i] < 0:
                start = max(q[i] / p[i], start)
            else:
                stop = min(q[i] / p[i], stop)

    if start > stop:
        return

    return p1 + (start * dx, start * dy), \
           p1 + (stop * dx, stop * dy)


class AbstractObject(ABC):
    pos: npt.NDArray
    rot: npt.NDArray

    def __new__(cls, *args, **kwards) -> Self:
        obj = super().__new__(cls)

        obj.pos = np.empty((3,), np.float64)
        obj.rot = np.empty((3,), np.float64)

        return obj

    @property
    def x(self) -> float:
        return self.pos[0]

    @property
    def y(self) -> float:
        return self.pos[1]

    @property
    def z(self) -> float:
        return self.pos[2]

    @property
    def rx(self) -> float:
        return self.rot[0]

    @property
    def ry(self) -> float:
        return self.rot[1]

    @property
    def rz(self) -> float:
        return self.rot[2]

    @property
    def rotmat(self) -> npt.NDArray:
        """
        Useful for transforming/aligning children
        """

        return rotmat(self.rx, self.ry, self.rz)

    @abstractmethod
    def project(self, obj: "AbstractObject") -> tuple[tuple[Vec2, ...], tuple[bool, ...]]: ...

    @classmethod
    @abstractmethod
    def render(cls, proj: tuple[Vec2, ...], zclip: tuple[bool, ...], *args, **kwargs) -> None: ...


class Camera(AbstractObject):
    foclen: float

    @lru_cache(CACHE_SIZE)
    def __yrotmat(self, ry: float) -> npt.NDArray:
        return np.array([
            [cos(radians(ry)), 0, -sin(radians(ry))],
            [0, 1, 0 ],
            [sin(radians(ry)), 0, cos(radians(ry))]
        ])

    @property
    def yrotmat(self) -> npt.NDArray:
        return self.__yrotmat(self.ry)

    def __new__(cls, *args, **kwards) -> Self:
        obj = super().__new__(cls)

        return obj

    def __init__(self,
                 pos: Vec3=(0,0,0),
                 rot: Vec3=(0,0,0),
                 foclen: float=0.1) -> None:

        self.pos = np.array(pos, np.float64)
        self.rot = np.array(rot, np.float64)
        self.foclen = foclen

    @lru_cache(CACHE_SIZE)
    def __project(self, rot: tuple[float, float, float], pos: tuple[float, float, float], fl: float) -> tuple[tuple[Vec2, ...], tuple[bool, ...]]:
        dx, dy, dz = (self.rotmat @ (np.array(pos) + [0, 0, fl]))

        if dz == 0: dz = FMIN

        bx = dx / abs(dz)
        by = dy / abs(dz)

        return (transform(bx, by),), (dz <= 0,)

    def project(self, obj: AbstractObject) -> tuple[tuple[Vec2, ...], tuple[bool, ...]]:
        res, clip = self.__project(tuple(self.rot), tuple(obj.pos - self.pos), self.foclen)

        return tuple(Vec2(*v) for v in res), clip # deepcopy to prevent side effects

    @classmethod
    def render(cls, proj: Never, zclip: Never) -> NoReturn:
        raise NotImplementedError("No Escherian shenanigans here please.")

class Point(AbstractObject):
    def __init__(self,
                 pos: Vec3=(0,0,0),
                 rot: Vec3=(0,0,0)) -> None:

        self.pos = np.array(pos, np.float64)
        self.rot = np.array(rot, np.float64)

    def project(self, camera: Camera) -> tuple[tuple[Vec2, ...], tuple[bool, ...]]:
        return camera.project(self)

    @classmethod
    def render(cls, proj: tuple[Vec2], zclip: tuple[bool]) -> None:
        global screenbuffer

        for p, c in zip(proj, zclip):
            if 0 < p.x < VW and 0 < p.y < VH and not c:
                screenbuffer += "\x1b[%d%dH#" % (p.y, p.x)

class Line(AbstractObject):
    p1: Point
    p2: Point
    worldloc: bool

    @overload
    def __init__(self,
                 p1: Point, p2: Point,
                 /, *,
                 pos: Vec3=...,
                 rot: Vec3=...,
                 worldloc: bool=True) -> None: ...

    @overload
    def __init__(self,
                 *p: Point,
                 pos: Vec3=...,
                 rot: Vec3=...,
                 worldloc: bool=True) -> None: ...

    def __init__(self,
                 *p,
                 pos: Vec3=(0,0,0),
                 rot: Vec3=(0,0,0),
                 worldloc: bool=True) -> None:

        self.p1, self.p2 = p[:2]
        self.worldloc = worldloc

        self.pos = np.array(pos, np.float64)
        self.rot = np.array(rot, np.float64)

    def project(self, camera: Camera) -> tuple[tuple[Vec2, ...], tuple[bool, ...]]:
        res = []
        zclip = []

        if self.worldloc:
            res, zclip = camera.project(self.p1)
            r, c = camera.project(self.p2)
            res += r
            zclip += c

        else:
            res, zclip = camera.project(Point(
                self.pos + (self.rotmat @ self.p1.pos.T).T,
                self.rot + self.p1.rot
            ))
            r, c = camera.project(Point(
                self.pos + (self.rotmat @ self.p2.pos.T).T,
                self.rot + self.p2.rot
            ))

            res += r
            zclip += c

        return res, zclip

    @classmethod
    def render(cls, proj: tuple[Vec2, Vec2], zclip: tuple[bool, bool], char: str="") -> None:
        """
        Good old Bresenham
        """

        global screenbuffer

        if all(zclip):
            return

        clipped = xyclip(*proj[:2])
        if clipped is None:
            return

        p1, p2 = (round(p) for p in clipped)

        if  p1.x < 1 > p2.x or p1.x > VW - 1 < p2.x or \
            p1.y < 1 > p2.y or p1.y > VH - 1 < p2.y:
            return

        dx, dy = p2 - p1
        sx = int(copysign(1, dx))
        sy = int(copysign(1, dy))

        char = char or gradmap(-dy / (dx or FMIN))

        if dx == 0:
            x = p1.x
            y = int(max(min(p1.y, p2.y), 0))
            dy = min(max(p1.y, p2.y), VH) + 1 - y

            for y in range(y, y + int(dy)):
                screenbuffer += "\x1b[%d;%dH%s" % (y, x, char)

            return

        elif dy == 0:
            x = max(min(p1.x, p2.x), 0)
            dx = min(max(p1.x, p2.x), VW) + 1 - x
            y = p1.y

            # s = "\x1b[%d;" % y
            # screenbuffer += s + s.join("%dH-" % (x + i) for i in range(int(dx)))
            screenbuffer += "\x1b[%d;%dH%s" % (y, x, char * int(dx))

            return

        else:
            dx = abs(dx)
            dy = -abs(dy)
            error = dx + dy

            x, y = p1

            if 0 < x < VW and 0 < y < VH:
                screenbuffer += "\x1b[%d;%dH%s" % (y, x, char)

            if sx > 0:
                x2 = min(VW, p2.x)
            else:
                x2 = max(1, p2.x)

            if sy > 0:
                y2 = min(VH, p2.y)
            else:
                y2 = max(1, p2.y)

            while True:
                e2 = 2 * error

                if e2 >= dy:
                    error += dy
                    x += sx

                if e2 <= dx:
                    error += dx
                    y += sy

                if  (sx > 0 and x > x2) or (sx < 0 and x < x2) or \
                    (sy > 0 and y > y2) or (sy < 0 and y < y2):
                    break

                if 0 < x < VW and 0 < y < VH:
                    screenbuffer += "\x1b[%d;%dH%s" % (y, x, char)

class Triangle(AbstractObject):
    p1: Point
    p2: Point
    p3: Point
    worldloc: bool

    @overload
    def __init__(self,
                 p1: Point, p2: Point, p3: Point,
                 /, *,
                 pos: Vec3=...,
                 rot: Vec3=...,
                 worldloc: bool=True) -> None: ...

    @overload
    def __init__(self,
                 *p: Point,
                 pos: Vec3=...,
                 rot: Vec3=...,
                 worldloc: bool=True) -> None: ...

    def __init__(self,
                 *p,
                 pos: Vec3=(0,0,0),
                 rot: Vec3=(0,0,0),
                 worldloc: bool=True) -> None:

        self.p1, self.p2, self.p3 = p[:3]
        self.worldloc = worldloc

        self.pos = np.array(pos, np.float64)
        self.rot = np.array(rot, np.float64)

    def project(self, camera: Camera) -> tuple[tuple[Vec2, ...], tuple[bool, ...]]:
        res = []
        zclip = []

        if self.worldloc:
            res, zclip = camera.project(self.p1)
            r, c = camera.project(self.p2)
            res += r
            zclip += c
            r, c = camera.project(self.p3)
            res += r
            zclip += c

        else:
            res, zclip = camera.project(Point(
                self.pos + (self.rotmat @ self.p1.pos.T).T,
                self.rot + self.p1.rot
            ))
            r, c = camera.project(Point(
                self.pos + (self.rotmat @ self.p2.pos.T).T,
                self.rot + self.p2.rot
            ))
            res += r
            zclip += c
            r, c = camera.project(Point(
                self.pos + (self.rotmat @ self.p3.pos.T).T,
                self.rot + self.p3.rot
            ))
            res += r
            zclip += c

        return res, zclip

    @classmethod
    def render(cls, proj: tuple[Vec2, ...], zclip: tuple[bool, ...], char: str="") -> None:
        """
        Half space rasterisation algorithm based on Nicolas Capens'
        https://web.archive.org/web/20050408192410/http://sw-shader.sourceforge.net/rasterizer.html
        """

        global screenbuffer

        if all(zclip):
            return

        (x1, y1), (x2, y2), (x3, y3) = proj[:3]

        # viewport culling
        if  (x1 < 1 and x2 < 1 and x3 < 1) or (x1 > VW and x2 > VW and x3 > VW) or\
            (y1 < 1 and y2 < 1 and y3 < 1) or (y1 > VH and y2 > VH and y3 > VH):
                return

        # backface culling
        if x2*y1 + x3*y2 + x1*y3 < x1*y2 + x2*y3 + x3*y1:
            return

            # (x2, y2), (x1, y1), (x3, y3) = proj[:3] # reverse chirality

        char = char or "#"

        dx12 = x1 - x2
        dx23 = x2 - x3
        dx31 = x3 - x1

        dy12 = y1 - y2
        dy23 = y2 - y3
        dy31 = y3 - y1

        minx = round(min(x1, x2, x3, VW))
        maxx = round(max(x1, x2, x3, 0))
        miny = round(min(y1, y2, y3, VH))
        maxy = round(max(y1, y2, y3, 0))

        c1 = dy12 * x1 - dx12 * y1
        c2 = dy23 * x2 - dx23 * y2
        c3 = dy31 * x3 - dx31 * y3

        c1 += (dy12 < 0 or (dy12 == 0 and dx12 > 0))
        c2 += (dy23 < 0 or (dy23 == 0 and dx23 > 0))
        c3 += (dy31 < 0 or (dy31 == 0 and dx31 > 0))

        cy1 = c1 + dx12 * miny - dy12 * minx
        cy2 = c2 + dx23 * miny - dy23 * minx
        cy3 = c3 + dx31 * miny - dy31 * minx

        for y in range(miny, maxy):
            cx1 = cy1
            cx2 = cy2
            cx3 = cy3

            for x in range(minx, maxx):
                if 0 < x < VW and 0 < y < VH:
                    if cx1 >= 0 and cx2 >= 0 and cx3 >= 0:
                        screenbuffer += "\x1b[%d;%dH%s" % (y, x, char)

                cx1 -= dy12
                cx2 -= dy23
                cx3 -= dy31

            cy1 += dx12
            cy2 += dx23
            cy3 += dx31

class Mesh(AbstractObject):
    points: tuple[Point, ...]
    lines: set[tuple[int, ...]]
    worldloc: bool

    def __init__(self,
                 points: tuple[Point, ...],
                 chains: set[tuple[int, ...]],
                 pos: Vec3=(0,0,0),
                 rot: Vec3=(0,0,0),
                 worldloc: bool=True) -> None:

        self.points = points
        self.lines = chains
        self.pos = np.array(pos, np.float64)
        self.rot = np.array(rot, np.float64)
        self.worldloc = worldloc

    def project(self, camera: Camera) -> tuple[tuple[Vec2, ...], tuple[bool, ...]]:
        res = ()
        clip = ()

        if self.worldloc:
            for p in self.points:
                r, c = camera.project(p)
                res += r
                clip += c

        else:
            for p in self.points:
                r, c = camera.project(Point(
                    self.pos + (self.rotmat @ p.pos),
                    self.rot + p.rot
                ))

                res += r
                clip += c

        return res, clip

    @classmethod
    def render(cls, proj: tuple[Vec2, ...], zclip: tuple[bool, ...], lines: Optional[set[tuple[int, ...]]]=None) -> None:
        if lines is None:
            for i in range(len(proj)):
                for j in range(i):
                    Line.render(
                        (proj[i], proj[j]),
                        (zclip[i], zclip[j])
                    )
        else:
            for p in lines:
                for i in range(len(p) - 1):
                    Line.render(
                        (proj[p[i]], proj[p[i+1]]),
                        (zclip[p[i]], zclip[p[i+1]])
                    )

class Piece(Mesh):
    points: tuple[Point, ...]
    triangles: set[tuple[int, int, int]]
    lines: set[tuple[int, int]]
    worldloc = False

    def __init__(self,
                 pos: Vec3=(0,0,0),
                 rot: Vec3=(0,0,0)) -> None:

        self.pos = np.array(pos, np.float64)
        self.rot = np.array(rot, np.float64)

    @classmethod
    def render(cls, proj: tuple[Vec2, ...], zclip: tuple[bool, ...], char: str="") -> None:
        for i, j, k in cls.triangles:
            Triangle.render(
                (proj[i], proj[j], proj[k]),
                (zclip[i], zclip[j], zclip[k]),
                char
            )

        for i, j in cls.lines:
            Line.render(
                (proj[i], proj[j]),
                (zclip[i], zclip[j]),
                char
            )

    @classmethod
    @abstractmethod
    def gen_mesh(cls) -> None: ...

class King(Piece):
    @classmethod
    def gen_mesh(cls) -> None:
        divs = 8 # 4 times a factor of 90
        offset = 17 * divs

        sintab = tuple(sin(radians(t)) for t in range(0, 360, 360 // divs))
        points = []
        triangles = set()
        lines = set()

        for (y, m) in [
            (-1, 0.4),      # 0 * 8
            (-0.9, 0.4),    # 1 * 8
            (-0.8, 0.4),    # 2 * 8
            (-0.7, 0.3),    # 3 * 8
            (-0.6, 0.3),    # 4 * 8
            (-0.5, 0.2),    # 5 * 8
            (-0.4, 0.15),   # 6 * 8
            (-0.3, 0.125),  # 7 * 8
            (-0.2, 0.1125), # 8 * 8
            (-0.1, 0.1),    # 9 * 8
            (0, 0.1),       # 10 * 8
            (0.1, 0.1),     # 11 * 8
            (0.2, 0.2),     # 12 * 8
            (0.3, 0.1),     # 13 * 8
            (0.4, 0.1333),  # 14 * 8
            (0.5, 0.1666),  # 15 * 8
            (0.6, 0.2),     # 16 * 8
        ]:
            points += [(m * sintab[i], y, m * sintab[i - (divs>>2)]) for i in range(divs)]

        points += [
            (0, -1, 0),     # 136
            (0, 0.6, 0),    # 137
            (0, 0.8, 0),    # 138
            (-0.1, 0.7, 0), # 139
            (0.1, 0.7, 0)   # 140
        ]

        triangles.update({
            ((i + 1) % divs, i, offset) for i in range(divs)
        })

        for j in range(0, offset, divs):
            triangles.update({
                (j + i, j + (i + 1) % divs, j + divs + 1) for i in range(divs)
            })

        lines = {
            (offset + 1, offset + 2),
            (offset + 3, offset + 4)
        }

        cls.points = tuple(Point(p) for p in points)
        cls.triangles = triangles
        cls.lines = lines

class Queen(Piece):
    @classmethod
    def gen_mesh(cls) -> None:
        divs = 8 # 4 times a factor of 90
        offset = 17 * divs

        sintab = tuple(sin(radians(t)) for t in range(0, 360, 360 // divs))
        points = []
        triangles = set()
        lines = set()

        for (y, m) in [
            (-1, 0.4),      # 0 * 8
            (-0.9, 0.4),    # 1 * 8
            (-0.8, 0.4),    # 2 * 8
            (-0.7, 0.3),    # 3 * 8
            (-0.6, 0.3),    # 4 * 8
            (-0.5, 0.2),    # 5 * 8
            (-0.4, 0.15),   # 6 * 8
            (-0.3, 0.125),  # 7 * 8
            (-0.2, 0.1125), # 8 * 8
            (-0.1, 0.1),    # 9 * 8
            (0, 0.1),       # 10 * 8
            (0.1, 0.1),     # 11 * 8
            (0.2, 0.2),     # 12 * 8
            (0.3, 0.1),     # 13 * 8
            (0.4, 0.1333),  # 14 * 8
            (0.5, 0.1666),  # 15 * 8
            (0.6, 0.2),     # 16 * 8
        ]:
            points += [(m * sintab[i], y, m * sintab[i - (divs>>2)]) for i in range(divs)]

        points += [
            (0, -1, 0),     # 136
            (0, 0.6, 0),    # 137
        ]

        triangles.update({
            ((i + 1) % divs + j, i + j, offset) for i in range(divs) for j in (0, 17)
        })

        for j in range(0, offset, divs):
            triangles.update({
                (j + i, j + (i + 1) % divs, j + divs + 1) for i in range(divs)
            })

        cls.points = tuple(Point(p) for p in points)
        cls.triangles = triangles
        cls.lines = lines


def handlein(camera: Camera) -> bool:
    redraw = False
    key = ord(getch())

    if key in [0, 224]:
        key = ord(getch())

        match key:
            case 72: # up
                camera.rot += [-10, 0, 0]
            case 75: # left
                camera.rot += [0, -10, 0]
            case 77: # right
                camera.rot += [0, 10, 0]
            case 80: # down
                camera.rot += [10, 0, 0]

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


def aproject(obj: AbstractObject, camera: Camera, a, kw):
    p = obj.project(camera)

    return obj, p, a, kw

def render(objs: Collection[tuple[AbstractObject, tuple, dict]], camera: Camera) -> None:
    global screenbuffer
    screenbuffer = "\x1b[2J\x1b[H3D Terminal Chess\n\nWASD + QE \tmove\n ← ↑ ↓ →  \trotate\nspace/tab  \tup/down"

    # with Pool(5) as p:
    #     proj = p.starmap(aproject, [(o, camera, a, kw) for o, a, kw in objs])

    # for o, p, a, kw in proj:
    #     o.render(*p, *a, **kw)

    for o, a, kw in objs:
        o.render(*o.project(camera), *a, **kw)

    # cells = list(enumerate(screenbuffer.split("\x1b[")))
    # cells.sort(key=lambda c: int((q := c[1].split(";", 1))[0]) * VW + int(q[1].split("H", 1)[0]) + 1/c[0])

    # res = []

    # py = px = 0

    # for c in cells:
    #     y, x = c[1].split(";", 1)
    #     y, x = int(y), int(x.split("H", 1)[0])

    #     if py == y and x == px + 1:
    #         res[-1] += c
    #     elif py != y or py != x:
    #         res += c

    #     py, px = y, x

    #

    print(end=screenbuffer, flush=True)

def main() -> None:
    King.gen_mesh()
    Queen.gen_mesh()

    camera = Camera((0, 6, -7), (40, 0, 0))

    xgrid = [(Line(Point((x, -1, -4)), Point((x, -1, 4))), (), {}) for x in range(-4, 5)]
    zgrid = [(Line(Point((-4, -1, z)), Point((4, -1, z))), (), {}) for z in range(-4, 5)]

    checkerboard = []
    cbpoints = [Point((x, -1, z)) for z in range(-4, 5) for x in range(-4, 5) if -8 != x + z != 8]

    for i in range(0, 70, 2):
        if i % 18 >= 16:
            continue

        p1 = cbpoints[i]
        p2 = cbpoints[i + 10]
        p3 = cbpoints[i + 9]
        p4 = cbpoints[i + 1]

        checkerboard.append((Triangle(p3, p1, p2), (":",), {}))
        checkerboard.append((Triangle(p1, p4, p2), (":",), {}))

    # points = [Point((x, y, z)) for x in (-1,1) for y in (-1,1) for z in (-1,1)]
    # lines = {(0,4),(1,5,4,6),(5,7,6,2,3),(7,3,1,0,2)}
    # cube = Mesh(points, lines)

    WKing = King((0.5, 0, -3.5))
    WQueen = Queen((-0.5, 0, -3.5))

    BKing = King((0.5, 0, 3.5))
    BQueen = Queen((-0.5, 0, 3.5))

    objects = [
        *checkerboard,
        *xgrid,
        *zgrid,
        (WKing, ("#",), {}),
        (WQueen, ("#",), {}),
        (BKing, (".",), {}),
        (BQueen, (".",), {}),
    ]

    redraw = True

    while True:
        if redraw:
            render(objects, camera)

        while not kbhit():
            sleep(0.01)

        redraw = handlein(camera)


if __name__ == "__main__":
    try:
        print(end="\x1b7\x1b[?1049h\x1b[?25l")

        main()
    finally:
        print(end="\x1b[?1049l\x1b8\x1b[?25h")
