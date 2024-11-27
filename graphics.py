from abc import ABC, abstractmethod
from bisect import bisect_left
from functools import lru_cache
from math import ceil, copysign, cos, floor, radians, sin, sqrt
# from multiprocessing import Pool
from os import get_terminal_size
# from random import randint
from sys import float_info
from typing import Any, Collection, NamedTuple, Never, NoReturn, Self, TypeVar, overload
from numpy import typing as npt
import numpy as np


CACHE_SIZE = 1024
MESH_DIVS = 8 # 4 times a factor of 90

VW, VH = get_terminal_size()
SCALE = 200
CRATIO = 1 / 2.32

FMIN = (5e-324 or float_info.min) # smallest floating point value

Vec3 = tuple[float, float, float] | npt.ArrayLike

class Vec2[T: (float, int)](NamedTuple):
    x: T
    y: T

    def __add__(self, o: "Vec2[T] | tuple[T, T]") -> "Vec2[T]":
        return Vec2(self.x + o[0], self.y + o[1])

    def __sub__(self, o: "Vec2[T] | tuple[T, T]") -> "Vec2[T]":
        return Vec2(self.x - o[0], self.y - o[1])

    def __mul__(self, s: T) -> "Vec2[T]":
        return Vec2(self.x * s, self.y * s)

    def __div__(self, s: T) -> "Vec2[T]":
        return Vec2(self.x / s, self.y / s)

    def __round__(self) -> "Vec2[int]":
        return Vec2(round(self.x), round(self.y))

    def __eq__(self, o: "Vec2[T] | tuple[T, T]") -> bool:
        return self.x == o[0] and self.y == o[1]


global screenbuffer, zbuffer
screenbuffer = "\x1b[2J"
zbuffer: dict[int, float] = {}

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
def normal(p1: tuple[float, float, float],
           p2: tuple[float, float, float],
           p3: tuple[float, float, float]) -> npt.NDArray:
    p = np.array(p1, np.float64)
    N = np.cross(p2 - p, p3 - p)

    return N / np.sqrt((N ** 2).sum())

@lru_cache(CACHE_SIZE)
def rotmat(rx: float, ry: float, rz: float) -> npt.NDArray:
    r = radians(rx), radians(ry), radians(rz)
    sx, sy, sz = map(sin, r)
    cx, cy, cz = map(cos, r)

    return np.array([
        [ 1,  0,  0 ],
        [ 0,  cx, sx],
        [ 0, -sx, cx]
    ]) @ np.array([
        [ cy, 0, -sy],
        [ 0,  1, 0  ],
        [ sy, 0,  cy]
    ]) @ np.array([
        [ cz, sz, 0 ],
        [-sz, cz, 0 ],
        [ 0,  0,  1 ]
    ])

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
    def project(self, obj: "AbstractObject") -> tuple[tuple[Vec2, ...], tuple[float, ...]]: ...

    @classmethod
    @abstractmethod
    def render(cls, proj: tuple[Vec2, ...], z: tuple[float, ...], *args, **kwargs) -> None: ...


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
    def __project(self, rot: tuple[float, float, float], pos: tuple[float, float, float], fl: float) -> tuple[tuple[Vec2, ...], tuple[float, ...]]:
        dx, dy, dz = (self.rotmat @ (np.array(pos) + [0, 0, fl]))

        if dz == 0: dz = FMIN

        bx = dx / abs(dz)
        by = dy / abs(dz)

        return (transform(bx, by),), (dz,)

    def project(self, obj: AbstractObject) -> tuple[tuple[Vec2, ...], tuple[float, ...]]:
        res, clip = self.__project(tuple(self.rot), tuple(obj.pos - self.pos), self.foclen)

        return tuple(Vec2(*v) for v in res), clip # deepcopy to prevent side effects

    @classmethod
    def render(cls, proj: Never, z: Never) -> NoReturn:
        raise NotImplementedError("No Escherian shenanigans here please.")

class Point(AbstractObject):
    def __init__(self,
                 pos: Vec3=(0,0,0),
                 rot: Vec3=(0,0,0)) -> None:

        self.pos = np.array(pos, np.float64)
        self.rot = np.array(rot, np.float64)

    def project(self, camera: Camera) -> tuple[tuple[Vec2, ...], tuple[float, ...]]:
        return camera.project(self)

    @classmethod
    def render(cls, proj: tuple[Vec2], z: tuple[float], char: str="#") -> None:
        global screenbuffer, zbuffer

        for p, c in zip(proj, z):
            if c <= 0:
                continue

            if 0 < p.x < VW and 0 < p.y < VH and not c:
                if zbuffer.setdefault(i := round(p.y-1) * VW + round(p.x-1), c) >= c:
                    screenbuffer += "\x1b[%d;%dH%c" % (p.y, p.x, char)
                    zbuffer[i] = c

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

    def project(self, camera: Camera) -> tuple[tuple[Vec2, ...], tuple[float, ...]]:
        res = []
        z = []

        if self.worldloc:
            res, z = camera.project(self.p1)
            r, c = camera.project(self.p2)
            res += r
            z += c

        else:
            res, z = camera.project(Point(
                self.pos + (self.rotmat @ self.p1.pos.T).T,
                self.rot + self.p1.rot
            ))
            r, c = camera.project(Point(
                self.pos + (self.rotmat @ self.p2.pos.T).T,
                self.rot + self.p2.rot
            ))

            res += r
            z += c

        return res, z

    @classmethod
    def render(cls,
               proj: tuple[Vec2, Vec2],
               z: tuple[float, float],
               *,
               char: str="",
               overdraw: bool=False) -> None:
        """
        Good old Bresenham
        """

        global screenbuffer, zbuffer

        if z[0] <= 0 and z[1] <= 0:
            return

        clipped = xyclip(*proj[:2])
        if clipped is None:
            return

        p1, p2 = (round(p) for p in clipped)

        if  p1.x < 1 > p2.x or p1.x > VW - 1 < p2.x or \
            p1.y < 1 > p2.y or p1.y > VH - 1 < p2.y:
            return

        c = sum(z) / 2

        dx, dy = p2 - p1
        sx = int(copysign(1, dx))
        sy = int(copysign(1, dy))

        char = char or gradmap(-dy / (dx or FMIN))

        if dx == 0:
            x = round(p1.x)
            y = round(max(min(p1.y, p2.y), 0))
            dy = round(min(max(p1.y, p2.y), VH) + 1 - y)

            for y in range(y, y + dy):
                if zbuffer.setdefault(i := (y-1) * VW + x - 1, c) >= c or overdraw:
                    screenbuffer += "\x1b[%d;%dH%c" % (y, x, char)
                    zbuffer[i] = c

            return

        elif dy == 0:
            x = round(max(min(p1.x, p2.x), 0))
            dx = round(min(max(p1.x, p2.x), VW)) + 1 - x
            y = round(p1.y)

            for x in range(x, x + dx):
                if zbuffer.setdefault(i := (y-1) * VW + x - 1, c) >= c or overdraw:
                    screenbuffer += "\x1b[%d;%dH%c" % (y, x, char)
                    zbuffer[i] = c

        else:
            dx = abs(dx)
            dy = -abs(dy)
            error = dx + dy

            x, y = round(p1)

            if 0 < x < VW and 0 < y < VH:
                if zbuffer.setdefault((i := (y-1) * VW + x - 1), c) >= c or overdraw:
                    screenbuffer += "\x1b[%d;%dH%c" % (y, x, char)
                    zbuffer[i] = c

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
                    if zbuffer.setdefault(i := (y-1) * VW + x - 1, c) >= c or overdraw:
                        screenbuffer += "\x1b[%d;%dH%c" % (y, x, char)
                        zbuffer[i] = c


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

    def project(self, camera: Camera) -> tuple[tuple[Vec2, ...], tuple[float, ...]]:
        res = []
        z = []

        if self.worldloc:
            res, z = camera.project(self.p1)
            r, c = camera.project(self.p2)
            res += r
            z += c
            r, c = camera.project(self.p3)
            res += r
            z += c

        else:
            res, z = camera.project(Point(
                self.pos + (self.rotmat @ self.p1.pos.T).T,
                self.rot + self.p1.rot
            ))
            r, c = camera.project(Point(
                self.pos + (self.rotmat @ self.p2.pos.T).T,
                self.rot + self.p2.rot
            ))
            res += r
            z += c
            r, c = camera.project(Point(
                self.pos + (self.rotmat @ self.p3.pos.T).T,
                self.rot + self.p3.rot
            ))
            res += r
            z += c

        return res, z

    @classmethod
    def render(cls,
               proj: tuple[Vec2, Vec2, Vec2],
               z: tuple[float, float, float],
               shading: float=-1,
               *,
               char: str="") -> None:
        """
        Half space rasterisation algorithm based on Nicolas Capens'
        https://web.archive.org/web/20050408192410/http://sw-shader.sourceforge.net/rasterizer.html
        """

        global screenbuffer, zbuffer

        if z[0] <= 0 and z[1] <= 0 and z[2] <= 0:
            return

        (x1, y1), (x2, y2), (x3, y3) = proj[:3]

        # viewport culling
        if  (x1 < 1 and x2 < 1 and x3 < 1) or (x1 > VW and x2 > VW and x3 > VW) or\
            (y1 < 1 and y2 < 1 and y3 < 1) or (y1 > VH and y2 > VH and y3 > VH):
                return

        # backface culling
        if x2*y1 + x3*y2 + x1*y3 < x1*y2 + x2*y3 + x3*y1:
            return

            # (x2, y2), (x1, y1), (x3, y3) = proj[:3] # reverse winding

        # screenbuffer += "\x1b[38;5;%dm" % randint(0, 255)

        if shading >= 0:
            screenbuffer += "\x1b[38;5;%dm" % round(255 - 23 * sqrt(shading**3))

        char = char or "#"
        c = max(z)

        dx12 = x1 - x2
        dx23 = x2 - x3
        dx31 = x3 - x1

        dy12 = y1 - y2
        dy23 = y2 - y3
        dy31 = y3 - y1

        minx = floor(min(x1, x2, x3, VW))
        maxx = ceil(max(x1, x2, x3, 0))
        miny = floor(min(y1, y2, y3, VH))
        maxy = ceil(max(y1, y2, y3, 0))

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
                        if zbuffer.setdefault(i := (y-1) * VW + x - 1, c) >= c:
                            screenbuffer += "\x1b[%d;%dH%c" % (y, x, char)
                            zbuffer[i] = c

                cx1 -= dy12
                cx2 -= dy23
                cx3 -= dy31

            cy1 += dx12
            cy2 += dx23
            cy3 += dx31

        if shading >= 0:
            screenbuffer += "\x1b[0m"

class Mesh(AbstractObject):
    points: tuple[Point, ...]
    triangles: set[tuple[int, int, int]]
    lines: set[tuple[int, ...]]
    worldloc: bool

    def __init__(self,
                 points: tuple[Point, ...],
                 triangles: set[tuple[int, int, int]],
                 lines: set[tuple[int, ...]],
                 pos: Vec3=(0,0,0),
                 rot: Vec3=(0,0,0),
                 worldloc: bool=True) -> None:

        self.points = points
        self.triangles = triangles
        self.lines = lines
        self.pos = np.array(pos, np.float64)
        self.rot = np.array(rot, np.float64)
        self.worldloc = worldloc

    def project(self, camera: Camera) -> tuple[tuple[Vec2, ...], tuple[float, ...], tuple[float, ...]]:
        res = ()
        clip = ()
        shading = ()

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

        for i, j, k in self.triangles:
            shading += 0.5 + np.dot(normal(
                tuple(self.points[i].pos),
                tuple(self.points[j].pos),
                tuple(self.points[k].pos)
            ), (sqrt(0.5), sqrt(0.5), 0)) / 2,

        return res, clip, shading

    @classmethod
    def render(cls,
               proj: tuple[Vec2, ...],
               z: tuple[float, ...],
               shading: tuple[float, ...],
               *,
               triangles: set[tuple[int, int, int]],
               lines: set[tuple[int, ...]],
               char: str="") -> None:

        for i, (j, k, l) in enumerate(triangles):
            Triangle.render(
                (proj[j], proj[k], proj[l]),
                (z[j], z[k], z[l]),
                shading[i],
                char=char
            )

        for p in lines:
            for i in range(len(p) - 1):
                Line.render(
                    (proj[p[i]], proj[p[i+1]]),
                    (z[p[i]], z[p[i+1]]),
                    char=char
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
    def render(cls,
               proj: tuple[Vec2, ...],
               z: tuple[float, ...],
               shading: tuple[float, ...],
               *,
               char: str="") -> None:
        for i, (j, k, l) in enumerate(cls.triangles):
            Triangle.render(
                (proj[j], proj[k], proj[l]),
                (z[j], z[k], z[l]),
                shading[i],
                char=char
            )

        for i, j in cls.lines:
            Line.render(
                (proj[i], proj[j]),
                (z[i], z[j]),
                char=char
            )

    @classmethod
    @abstractmethod
    def gen_mesh(cls) -> None: ...

class King(Piece):
    @classmethod
    def gen_mesh(cls) -> None:
        offset = 11 * MESH_DIVS

        sintab = tuple(sin(radians(t)) for t in range(0, 360, 360 // MESH_DIVS))
        points = []
        triangles = set()
        lines = set()

        for (y, m) in [
            (-1, 0.4),      # 0 * MESH_DIVS
            (-0.8, 0.4),    # 1 * MESH_DIVS
            (-0.7, 0.3),    # 2 * MESH_DIVS
            (-0.6, 0.3),    # 3 * MESH_DIVS
            (-0.5, 0.2),    # 4 * MESH_DIVS
            (-0.4, 0.15),   # 5 * MESH_DIVS
            (-0.3, 0.125),  # 6 * MESH_DIVS
            (0.1, 0.1),     # 7 * MESH_DIVS
            (0.2, 0.2),     # 8 * MESH_DIVS
            (0.3, 0.1),     # 9 * MESH_DIVS
            (0.55, 0.2),    # 10 * MESH_DIVS
        ]:
            points += [(
                m * sintab[i],
                y,
                m * sintab[i - (MESH_DIVS>>2)]
            ) for i in range(MESH_DIVS)]

        points += [
            (0, -1, 0),     # offset
            (0, 0.6, 0),    # offset + 1
            (0, 0.8, 0),    # offset + 2
            (-0.1, 0.7, 0), # offset + 3
            (0.1, 0.7, 0)   # offset + 4
        ]

        triangles.update({
            ((i + 1) % MESH_DIVS, i, offset) for i in range(MESH_DIVS)
        })

        for j in range(0, offset - MESH_DIVS, MESH_DIVS):
            triangles.update({
                (j + i,
                 j + (i + 1) % MESH_DIVS,
                 j + MESH_DIVS + i) for i in range(MESH_DIVS)
            })

        for j in range(0, offset - MESH_DIVS, MESH_DIVS):
            triangles.update({
                (j + (i + 1) % MESH_DIVS,
                 j + MESH_DIVS + (i + 1) % MESH_DIVS,
                 j + MESH_DIVS + i) for i in range(MESH_DIVS)
            })

        triangles.update({
            (offset - 1 - (i + 1) % MESH_DIVS,
             offset - 1 - i,
             offset + 1) for i in range(MESH_DIVS)
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
        offset = 11 * MESH_DIVS

        sintab = tuple(sin(radians(t)) for t in range(0, 360, 360 // MESH_DIVS))
        points = []
        triangles = set()
        lines = set()

        for (y, m) in [
            (-1, 0.4),      # 0 * MESH_DIVS
            (-0.8, 0.4),    # 1 * MESH_DIVS
            (-0.7, 0.3),    # 2 * MESH_DIVS
            (-0.6, 0.3),    # 3 * MESH_DIVS
            (-0.5, 0.2),    # 4 * MESH_DIVS
            (-0.4, 0.15),   # 5 * MESH_DIVS
            (-0.3, 0.125),  # 6 * MESH_DIVS
            (0.1, 0.1),     # 7 * MESH_DIVS
            (0.2, 0.2),     # 8 * MESH_DIVS
            (0.3, 0.1),     # 9 * MESH_DIVS
            (0.6, 0.2),     # 10 * MESH_DIVS
        ]:
            points += [(
                m * sintab[i],
                y,
                m * sintab[i - (MESH_DIVS>>2)]
            ) for i in range(MESH_DIVS)]

        points += [
            (0, -1, 0),     # offset
            (0, 0.7, 0),    # offset + 1
        ]

        triangles.update({
            ((i + 1) % MESH_DIVS, i, offset) for i in range(MESH_DIVS)
        })

        for j in range(0, offset - MESH_DIVS, MESH_DIVS):
            triangles.update({
                (j + i,
                 j + (i + 1) % MESH_DIVS,
                 j + MESH_DIVS + i) for i in range(MESH_DIVS)
            })

        for j in range(0, offset - MESH_DIVS, MESH_DIVS):
            triangles.update({
                (j + (i + 1) % MESH_DIVS,
                 j + MESH_DIVS + (i + 1) % MESH_DIVS,
                 j + MESH_DIVS + i) for i in range(MESH_DIVS)
            })

        triangles.update({
            (offset - 1 - (i + 1) % MESH_DIVS,
             offset - 1 - i,
             offset + 1) for i in range(MESH_DIVS)
        })

        cls.points = tuple(Point(p) for p in points)
        cls.triangles = triangles
        cls.lines = lines

class Rook(Piece):
    @classmethod
    def gen_mesh(cls) -> None:
        offset = 10 * MESH_DIVS

        sintab = tuple(sin(radians(t)) for t in range(0, 360, 360 // MESH_DIVS))
        points = []
        triangles = set()
        lines = set()

        for (y, m) in [
            (-1, 0.38),      # 0 * MESH_DIVS
            (-0.85, 0.35),    # 1 * MESH_DIVS
            (-0.75, 0.25),    # 2 * MESH_DIVS
            (-0.65, 0.25),    # 3 * MESH_DIVS
            (-0.55, 0.18),    # 4 * MESH_DIVS
            (-0.45, 0.15),   # 5 * MESH_DIVS
            (-0.35, 0.125),  # 6 * MESH_DIVS
            (-0.15, 0.1),    # 7 * MESH_DIVS
            (-0.1, 0.2),    # 8 * MESH_DIVS
            (0.15, 0.2)      # 9 * MESH_DIVS
        ]:
            points += [(
                m * sintab[i],
                y,
                m * sintab[i - (MESH_DIVS>>2)]
            ) for i in range(MESH_DIVS)]

        points += [
            (0, -1, 0),     # offset
            (0, 0.15, 0),    # offset + 1
        ]

        triangles.update({
            ((i + 1) % MESH_DIVS, i, offset) for i in range(MESH_DIVS)
        })

        for j in range(0, offset - MESH_DIVS, MESH_DIVS):
            triangles.update({
                (j + i,
                 j + (i + 1) % MESH_DIVS,
                 j + MESH_DIVS + i) for i in range(MESH_DIVS)
            })

        for j in range(0, offset - MESH_DIVS, MESH_DIVS):
            triangles.update({
                (j + (i + 1) % MESH_DIVS,
                 j + MESH_DIVS + (i + 1) % MESH_DIVS,
                 j + MESH_DIVS + i) for i in range(MESH_DIVS)
            })

        triangles.update({
            (offset - 1 - (i + 1) % MESH_DIVS,
             offset - 1 - i,
             offset + 1) for i in range(MESH_DIVS)
        })

        cls.points = tuple(Point(p) for p in points)
        cls.triangles = triangles
        cls.lines = lines

class Bishop(Piece):
    @classmethod
    def gen_mesh(cls) -> None:
        offset = 11 * MESH_DIVS

        sintab = tuple(sin(radians(t)) for t in range(0, 360, 360 // MESH_DIVS))
        points = []
        triangles = set()
        lines = set()

        for (y, m) in [
            (-1, 0.35),     # 0 * MESH_DIVS
            (-0.85, 0.35),  # 1 * MESH_DIVS
            (-0.75, 0.25),  # 2 * MESH_DIVS
            (-0.65, 0.25),  # 3 * MESH_DIVS
            (-0.55, 0.18),  # 4 * MESH_DIVS
            (-0.45, 0.15),  # 5 * MESH_DIVS
            (-0.35, 0.125), # 6 * MESH_DIVS
            (-0.05, 0.1),   # 7 * MESH_DIVS
            (0, 0.2),       # 8 * MESH_DIVS
            (0.05, 0.1),    # 8 * MESH_DIVS
            (0.22, 0.17)     # 9 * MESH_DIVS
        ]:
            points += [(
                m * sintab[i],
                y,
                m * sintab[i - (MESH_DIVS>>2)]
            ) for i in range(MESH_DIVS)]

        points += [
            (0, -1, 0),     # offset
            (0, 0.5, 0),   # offset + 1
        ]

        triangles.update({
            ((i + 1) % MESH_DIVS, i, offset) for i in range(MESH_DIVS)
        })

        for j in range(0, offset - MESH_DIVS, MESH_DIVS):
            triangles.update({
                (j + i,
                 j + (i + 1) % MESH_DIVS,
                 j + MESH_DIVS + i) for i in range(MESH_DIVS)
            })

        for j in range(0, offset - MESH_DIVS, MESH_DIVS):
            triangles.update({
                (j + (i + 1) % MESH_DIVS,
                 j + MESH_DIVS + (i + 1) % MESH_DIVS,
                 j + MESH_DIVS + i) for i in range(MESH_DIVS)
            })

        triangles.update({
            (offset - 1 - (i + 1) % MESH_DIVS,
             offset - 1 - i,
             offset + 1) for i in range(MESH_DIVS)
        })

        cls.points = tuple(Point(p) for p in points)
        cls.triangles = triangles
        cls.lines = lines

class Knight(Piece):
    @classmethod
    def gen_mesh(cls) -> None:
        offset = 4 * MESH_DIVS

        sintab = tuple(sin(radians(t)) for t in range(0, 360, 360 // MESH_DIVS))
        points = []
        triangles = set()
        lines = set()

        for (y, m) in [
            (-1, 0.35),     # 0 * MESH_DIVS
            (-0.85, 0.35),  # 1 * MESH_DIVS
            (-0.75, 0.25),  # 2 * MESH_DIVS
            (-0.65, 0.25),  # 3 * MESH_DIVS
        ]:
            points += [(
                m * sintab[i],
                y,
                m * sintab[i - (MESH_DIVS>>2)]
            ) for i in range(MESH_DIVS)]

        points += [
            (0, -1, 0),     # offset
            (0, 0.25, 0),   # offset + 1
        ]

        triangles.update({
            ((i + 1) % MESH_DIVS, i, offset) for i in range(MESH_DIVS)
        })

        for j in range(0, offset - MESH_DIVS, MESH_DIVS):
            triangles.update({
                (j + i,
                 j + (i + 1) % MESH_DIVS,
                 j + MESH_DIVS + i) for i in range(MESH_DIVS)
            })

        for j in range(0, offset - MESH_DIVS, MESH_DIVS):
            triangles.update({
                (j + (i + 1) % MESH_DIVS,
                 j + MESH_DIVS + (i + 1) % MESH_DIVS,
                 j + MESH_DIVS + i) for i in range(MESH_DIVS)
            })

        triangles.update({
            (offset - 1 - (i + 1) % MESH_DIVS,
             offset - 1 - i,
             offset + 1) for i in range(MESH_DIVS)
        })

        cls.points = tuple(Point(p) for p in points)
        cls.triangles = triangles
        cls.lines = lines

class Pawn(Piece):
    @classmethod
    def gen_mesh(cls) -> None:
        offset = 11 * MESH_DIVS

        sintab = tuple(sin(radians(t)) for t in range(0, 360, 360 // MESH_DIVS))
        points = []
        triangles = set()
        lines = set()

        for (y, m) in [
            (-1, 0.35),     # 0 * MESH_DIVS
            (-0.87, 0.32),  # 1 * MESH_DIVS
            (-0.8, 0.22),   # 2 * MESH_DIVS
            (-0.67, 0.13),  # 3 * MESH_DIVS
            (-0.37, 0.1),   # 4 * MESH_DIVS
            (-0.32, 0.2),   # 5 * MESH_DIVS
            (-0.27, 0.1),   # 6 * MESH_DIVS
            (-0.2, 0.17),   # 7 * MESH_DIVS
            (-0.1, 0.2),    # 8 * MESH_DIVS
            (0, 0.17),      # 9 * MESH_DIVS
            (0.07, 0.1)     # 10 * MESH_DIVS
        ]:
            points += [(
                m * sintab[i],
                y,
                m * sintab[i - (MESH_DIVS>>2)]
            ) for i in range(MESH_DIVS)]

        points += [
            (0, -1, 0),     # offset
            (0, 0.1, 0),    # offset + 1
        ]

        triangles.update({
            ((i + 1) % MESH_DIVS, i, offset) for i in range(MESH_DIVS)
        })

        for j in range(0, offset - MESH_DIVS, MESH_DIVS):
            triangles.update({
                (j + i,
                 j + (i + 1) % MESH_DIVS,
                 j + MESH_DIVS + i) for i in range(MESH_DIVS)
            })

        for j in range(0, offset - MESH_DIVS, MESH_DIVS):
            triangles.update({
                (j + (i + 1) % MESH_DIVS,
                 j + MESH_DIVS + (i + 1) % MESH_DIVS,
                 j + MESH_DIVS + i) for i in range(MESH_DIVS)
            })

        triangles.update({
            (offset - 1 - (i + 1) % MESH_DIVS,
             offset - 1 - i,
             offset + 1) for i in range(MESH_DIVS)
        })

        cls.points = tuple(Point(p) for p in points)
        cls.triangles = triangles
        cls.lines = lines


# def aproject(obj: AbstractObject, camera: Camera, kw):
#     p = obj.project(camera)

#     return obj, p, kw

def render(objs: Collection[tuple[AbstractObject, dict]], camera: Camera) -> None:
    global screenbuffer, zbuffer
    screenbuffer = "\x1b[2J"
    zbuffer = {}

    # with Pool(5) as p:
    #     proj = p.starmap(aproject, [(o, camera, kw) for o, kw in objs])

    # for o, p, kw in proj:
    #     o.render(*p, **kw)

    for o, kw in objs:
        p = o.project(camera)

        o.render(*p, **kw)

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

    screenbuffer += "\x1b[H3D Terminal Chess\n\nWASD + QE\tmove\n ← ↑ ↓ → \trotate\nspace/tab\tup/down"

    print(end=screenbuffer, flush=True)


King.gen_mesh()
Queen.gen_mesh()
Rook.gen_mesh()
Bishop.gen_mesh()
Knight.gen_mesh()
Pawn.gen_mesh()