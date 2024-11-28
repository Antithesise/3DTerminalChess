from abc import ABC, abstractmethod
from bisect import bisect_left
from functools import lru_cache
from math import ceil, copysign, cos, radians, sin, sqrt
from os import get_terminal_size
# from random import randint
from sys import float_info
from typing import Any, Collection, NamedTuple, Never, NoReturn, Self, overload
from numpy import array, cross, dot, empty, float64, sqrt as npsqrt
from numpy.typing import ArrayLike, NDArray


CACHE_SIZE = 1024
MESH_DIVS = 8 # 4 times a factor of 90

LIGHTNORM = (sqrt(0.5), sqrt(0.5), 0)

VW, VH = get_terminal_size()
HVW, HVH = VW >> 1, VH >> 1
SCALE = 200
CRATIO = 1 / 2.32

FMIN = (5e-324 or float_info.min) # smallest floating point value

EMPTYBUFFER = [""] * (VH * VW)

GRADCHARS = {
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
GRADS = [-4, -2, -1, -0.5, -0.2, 0, 0.2, 0.5, 1, 2, 4]

Vec3 = tuple[float, float, float] | ArrayLike

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
        return Vec2(int(self.x + 0.5), int(self.y + 0.5))

    def __eq__(self, o: "Vec2[T] | tuple[T, T]") -> bool:
        return self.x == o[0] and self.y == o[1]


screenbuffer = EMPTYBUFFER[:]
zbuffer: dict[int, float] = {}

polygons = verts = 0


@lru_cache(CACHE_SIZE)
def gradmap(grad: float):
    pos = bisect_left(GRADS, grad)

    if pos == 0 or pos == 11:
        return "|"

    before = GRADS[pos - 1]
    after = GRADS[pos]

    if after - grad < grad - before:
        return GRADCHARS[after]
    else:
        return GRADCHARS[before]

@lru_cache(CACHE_SIZE)
def normal(p1: tuple[float, float, float],
           p2: tuple[float, float, float],
           p3: tuple[float, float, float]) -> NDArray:
    p = array(p1, float64)
    N = cross(p2 - p, p3 - p)

    return N / npsqrt((N ** 2).sum())

@lru_cache(CACHE_SIZE)
def rotmat(rx: float, ry: float, rz: float) -> NDArray:
    r = radians(rx), radians(ry), radians(rz)
    sx, sy, sz = map(sin, r)
    cx, cy, cz = map(cos, r)

    return array([
        [ 1,  0,  0 ],
        [ 0,  cx, sx],
        [ 0, -sx, cx]
    ]) @ array([
        [ cy, 0, -sy],
        [ 0,  1, 0  ],
        [ sy, 0,  cy]
    ]) @ array([
        [ cz, sz, 0 ],
        [-sz, cz, 0 ],
        [ 0,  0,  1 ]
    ])

@lru_cache(CACHE_SIZE)
def transform(x: float, y: float) -> Vec2[float]:
    return Vec2(HVW + x * SCALE, HVH - y * SCALE * CRATIO)

def xyclip(p1: Vec2, p2: Vec2) -> tuple[Vec2, Vec2] | None:
    """
    liang-barsky for the win
    """

    dx, dy = p2 - p1
    x, y = p1

    p = (-dx, dx, -dy, dy)
    q = (x - 1, VW - x, y - 1, VH - y)
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

def write(x: int | float, y: int | float, z: float, char: str, overdraw: bool=False) -> None:
    x, y = int(x + 0.5), int(y + 0.5)

    if 0 < x < VW and 0 < y < VH:
        i = (y - 1) * VW + x - 1

        if overdraw or zbuffer.setdefault(i, z) >= z:
            screenbuffer[i] = char
            zbuffer[i] = z


class AbstractObject(ABC):
    pos: NDArray
    rot: NDArray

    def __new__(cls, *args, **kwargs) -> Self:
        obj = super().__new__(cls)

        obj.pos = empty((3,), float64)
        obj.rot = empty((3,), float64)

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
    def rotmat(self) -> NDArray:
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
    def __yrotmat(self, ry: float) -> NDArray:
        return array([
            [cos(radians(ry)), 0, -sin(radians(ry))],
            [0, 1, 0 ],
            [sin(radians(ry)), 0, cos(radians(ry))]
        ])

    @property
    def yrotmat(self) -> NDArray:
        return self.__yrotmat(self.ry)

    def __new__(cls, *args, **kwards) -> Self:
        obj = super().__new__(cls)

        return obj

    def __init__(self,
                 pos: Vec3=(0,0,0),
                 rot: Vec3=(0,0,0),
                 foclen: float=0.1) -> None:

        self.pos = array(pos, float64)
        self.rot = array(rot, float64)
        self.foclen = foclen

    @lru_cache(CACHE_SIZE)
    def __project(self, rot: tuple[float, float, float], pos: tuple[float, float, float], fl: float) -> tuple[tuple[Vec2], tuple[float]]:
        dx, dy, dz = (self.rotmat @ (array(pos) + [0, 0, fl]))

        if dz == 0: dz = FMIN

        bx = dx / abs(dz)
        by = dy / abs(dz)

        return (transform(bx, by),), (dz,)

    def project(self, obj: AbstractObject) -> tuple[tuple[Vec2], tuple[float]]:
        res, clip = self.__project(tuple(self.rot), tuple(obj.pos - self.pos), self.foclen)

        return (Vec2(*res[0]),), clip # deepcopy to prevent side effects

    @classmethod
    def render(cls, proj: Never, z: Never) -> NoReturn:
        raise NotImplementedError("No Escherian shenanigans here please.")

class Point(AbstractObject):
    def __init__(self,
                 pos: Vec3=(0,0,0),
                 rot: Vec3=(0,0,0)) -> None:

        self.pos = array(pos, float64)
        self.rot = array(rot, float64)

    def project(self, camera: Camera) -> tuple[tuple[Vec2, ...], tuple[float, ...]]:
        return camera.project(self)

    @classmethod
    def render(cls, proj: tuple[Vec2], z: tuple[float], char: str="#") -> None:
        global verts

        for p, c in zip(proj, z):
            if c <= 0:
                continue

            verts += 1
            write(*p, c, char)

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

        self.p1, self.p2 = p # hope p is correct length
        self.worldloc = worldloc

        self.pos = array(pos, float64)
        self.rot = array(rot, float64)

    def project(self, camera: Camera) -> tuple[tuple[Vec2, ...], tuple[float, ...]]:
        res = []
        z = []

        p1, p2 = self.p1, self.p2

        if self.worldloc:
            res, z = camera.project(p1)
            r, c = camera.project(p2)
            res += r
            z += c

        else:
            res, z = camera.project(Point(
                self.pos + (self.rotmat @ p1.pos.T).T,
                self.rot + p1.rot
            ))
            r, c = camera.project(Point(
                self.pos + (self.rotmat @ p2.pos.T).T,
                self.rot + p2.rot
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

        global verts

        if z[0] <= 0 and z[1] <= 0:
            return

        clipped = xyclip(*proj) # hope proj is correct length
        if clipped is None:
            return

        p1, p2 = (round(p) for p in clipped)

        if  p1.x < 1 > p2.x or p1.x > VW - 1 < p2.x or \
            p1.y < 1 > p2.y or p1.y > VH - 1 < p2.y:
            return

        verts += 2

        c = max(z)

        dx, dy = p2 - p1
        sx = int(copysign(1, dx))
        sy = int(copysign(1, dy))
        x1, y1 = round(p1)
        x2, y2 = round(p2)

        char = char or gradmap(-dy / (dx or FMIN))

        if dx == 0:
            y = max(min(y1, y2), 0)
            dy = min(max(y1, y2), VH) + 1 - y

            for y in range(y, y + dy):
                write(x1, y, c, char, overdraw)

            return

        elif dy == 0:
            x = max(min(x1, x2), 0)
            dx = min(max(x1, x2), VW) + 1 - x

            for x in range(x, x + dx):
                write(x, y1, c, char, overdraw)

            return

        else:
            dx = abs(dx)
            dy = -abs(dy)
            error = dx + dy

            write(x1, y1, c, char, overdraw)

            if sx > 0 and sy > 0:
                x2 = min(VW, x2)
                y2 = min(VH, y2)
                while True:
                    e2 = 2 * error

                    if e2 >= dy:
                        error += dy
                        x1 += sx

                    if e2 <= dx:
                        error += dx
                        y1 += sy

                    if x1 > x2 or y1 > y2:
                        break

                    write(x1, y1, c, char, overdraw)
            elif sx > 0:
                x2 = min(VW, x2)
                y2 = max(1, y2)
                while True:
                    e2 = 2 * error

                    if e2 >= dy:
                        error += dy
                        x1 += sx

                    if e2 <= dx:
                        error += dx
                        y1 += sy

                    if x1 > x2 or y1 < y2:
                        break

                    write(x1, y1, c, char, overdraw)
            elif sy > 0:
                x2 = max(1, x2)
                y2 = min(VH, y2)
                while True:
                    e2 = 2 * error

                    if e2 >= dy:
                        error += dy
                        x1 += sx

                    if e2 <= dx:
                        error += dx
                        y1 += sy

                    if x1 < x2 or y1 > y2:
                        break

                    write(x1, y1, c, char, overdraw)
            else:
                x2 = max(1, x2)
                y2 = max(1, y2)
                while True:
                    e2 = 2 * error

                    if e2 >= dy:
                        error += dy
                        x1 += sx

                    if e2 <= dx:
                        error += dx
                        y1 += sy

                    if x1 < x2 or y1 < y2:
                        break

                    write(x1, y1, c, char, overdraw)


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

        self.pos = array(pos, float64)
        self.rot = array(rot, float64)

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
               char: str="#") -> None:
        """
        Half space rasterisation algorithm based on Nicolas Capens'
        https://web.archive.org/web/20050408192410/http://sw-shader.sourceforge.net/rasterizer.html
        """

        global polygons, verts

        if z[0] <= 0 and z[1] <= 0 and z[2] <= 0:
            return

        (x1, y1), (x2, y2), (x3, y3) = proj # hope proj is correct length

        # backface culling
        if x2*y1 + x3*y2 + x1*y3 < x1*y2 + x2*y3 + x3*y1:
            return

            # (x2, y2), (x1, y1), (x3, y3) = proj # reverse winding

        # viewport culling
        if  (x1 < 1 and x2 < 1 and x3 < 1) or (x1 > VW and x2 > VW and x3 > VW) or\
            (y1 < 1 and y2 < 1 and y3 < 1) or (y1 > VH and y2 > VH and y3 > VH):
                return

        if shading >= 0:
            char = "\x1b[38;5;%dm%c\x1b[0m" % (
                int(255.5 - 23 * sqrt(shading**3)),
                char
            )

        polygons += 1
        verts += 3

        c = max(z)

        dx12 = x1 - x2
        dx23 = x2 - x3
        dx31 = x3 - x1

        dy12 = y1 - y2
        dy23 = y2 - y3
        dy31 = y3 - y1

        minx = int(min(x1, x2, x3, VW))
        maxx = ceil(max(x1, x2, x3, 0))
        miny = int(min(y1, y2, y3, VH))
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
                if cx1 >= 0 and cx2 >= 0 and cx3 >= 0:
                    write(x, y, c, char)

                cx1 -= dy12
                cx2 -= dy23
                cx3 -= dy31

            cy1 += dx12
            cy2 += dx23
            cy3 += dx31

class Mesh(AbstractObject):
    points: tuple[Point, ...]
    triangles: tuple[tuple[int, int, int], ...]
    lines: tuple[tuple[int, ...], ...]
    worldloc: bool

    def __init__(self,
                 points: tuple[Point, ...],
                 triangles: tuple[tuple[int, int, int], ...],
                 lines: tuple[tuple[int, ...], ...],
                 pos: Vec3=(0,0,0),
                 rot: Vec3=(0,0,0),
                 worldloc: bool=True) -> None:

        self.points = points
        self.triangles = triangles
        self.lines = lines
        self.pos = array(pos, float64)
        self.rot = array(rot, float64)
        self.worldloc = worldloc


    @lru_cache(CACHE_SIZE)
    def _shade(self, tri: tuple[int, int, int]) -> float:
        i, j, k = tri

        points = self.points

        return 0.5 + dot(normal(
            tuple(points[i].pos),
            tuple(points[j].pos),
            tuple(points[k].pos)
        ), LIGHTNORM) / 2

    @lru_cache(CACHE_SIZE)
    def _transform(self, p: Point) -> Point:
        return Point(
            self.pos + (self.rotmat @ p.pos),
            self.rot + p.rot
        )

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
                r, c = camera.project(self._transform(p))
                res += r
                clip += c

        shading = tuple(map(self._shade, self.triangles))

        return res, clip, shading

    def render(self,
               proj: tuple[Vec2, ...],
               z: tuple[float, ...],
               shading: tuple[float, ...],
               *,
               char: str="") -> None:

        for i, (j, k, l) in enumerate(self.triangles):
            Triangle.render(
                (proj[j], proj[k], proj[l]),
                (z[j], z[k], z[l]),
                shading[i],
                char=char
            )

        for p in self.lines:
            for i in range(len(p) - 1):
                Line.render(
                    (proj[p[i]], proj[p[i+1]]),
                    (z[p[i]], z[p[i+1]]),
                    char=char
                )

class Piece(Mesh):
    points: tuple[Point, ...]
    triangles: tuple[tuple[int, int, int], ...]
    lines: tuple[tuple[int, int], ...]
    worldloc = False

    def __init__(self,
                 pos: Vec3=(0,0,0),
                 rot: Vec3=(0,0,0)) -> None:

        self.pos = array(pos, float64)
        self.rot = array(rot, float64)

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
        triangles = []
        lines = ()

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

        triangles += [
            ((i + 1) % MESH_DIVS, i, offset) for i in range(MESH_DIVS)
        ]

        for j in range(0, offset - MESH_DIVS, MESH_DIVS):
            triangles += [
                (j + i,
                 j + (i + 1) % MESH_DIVS,
                 j + MESH_DIVS + i) for i in range(MESH_DIVS)
            ] + [
                (j + (i + 1) % MESH_DIVS,
                 j + MESH_DIVS + (i + 1) % MESH_DIVS,
                 j + MESH_DIVS + i) for i in range(MESH_DIVS)
            ]

        triangles += [
            (offset - 1 - (i + 1) % MESH_DIVS,
             offset - 1 - i,
             offset + 1) for i in range(MESH_DIVS)
        ]

        lines = (
            (offset + 1, offset + 2),
            (offset + 3, offset + 4)
        )

        cls.points = tuple(Point(p) for p in points)
        cls.triangles = tuple(triangles)
        cls.lines = lines

class Queen(Piece):
    @classmethod
    def gen_mesh(cls) -> None:
        offset = 11 * MESH_DIVS

        sintab = tuple(sin(radians(t)) for t in range(0, 360, 360 // MESH_DIVS))
        points = []
        triangles = []
        lines = ()

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

        triangles += [
            ((i + 1) % MESH_DIVS, i, offset) for i in range(MESH_DIVS)
        ]

        for j in range(0, offset - MESH_DIVS, MESH_DIVS):
            triangles += [
                (j + i,
                 j + (i + 1) % MESH_DIVS,
                 j + MESH_DIVS + i) for i in range(MESH_DIVS)
            ] + [
                (j + (i + 1) % MESH_DIVS,
                 j + MESH_DIVS + (i + 1) % MESH_DIVS,
                 j + MESH_DIVS + i) for i in range(MESH_DIVS)
            ]

        triangles += [
            (offset - 1 - (i + 1) % MESH_DIVS,
             offset - 1 - i,
             offset + 1) for i in range(MESH_DIVS)
        ]

        cls.points = tuple(Point(p) for p in points)
        cls.triangles = tuple(triangles)
        cls.lines = lines

class Rook(Piece):
    @classmethod
    def gen_mesh(cls) -> None:
        offset = 10 * MESH_DIVS

        sintab = tuple(sin(radians(t)) for t in range(0, 360, 360 // MESH_DIVS))
        points = []
        triangles = []
        lines = ()

        for (y, m) in [
            (-1, 0.35),     # 0 * MESH_DIVS
            (-0.85, 0.35),  # 1 * MESH_DIVS
            (-0.75, 0.25),  # 2 * MESH_DIVS
            (-0.65, 0.25),  # 3 * MESH_DIVS
            (-0.55, 0.18),  # 4 * MESH_DIVS
            (-0.45, 0.15),  # 5 * MESH_DIVS
            (-0.35, 0.125), # 6 * MESH_DIVS
            (-0.15, 0.1),   # 7 * MESH_DIVS
            (-0.1, 0.2),    # 8 * MESH_DIVS
            (0.15, 0.2)     # 9 * MESH_DIVS
        ]:
            points += [(
                m * sintab[i],
                y,
                m * sintab[i - (MESH_DIVS>>2)]
            ) for i in range(MESH_DIVS)]

        points += [
            (0, -1, 0),     # offset
            (0, 0.15, 0),   # offset + 1
        ]

        triangles += [
            ((i + 1) % MESH_DIVS, i, offset) for i in range(MESH_DIVS)
        ]

        for j in range(0, offset - MESH_DIVS, MESH_DIVS):
            triangles += [
                (j + i,
                 j + (i + 1) % MESH_DIVS,
                 j + MESH_DIVS + i) for i in range(MESH_DIVS)
            ] + [
                (j + (i + 1) % MESH_DIVS,
                 j + MESH_DIVS + (i + 1) % MESH_DIVS,
                 j + MESH_DIVS + i) for i in range(MESH_DIVS)
            ]

        triangles += [
            (offset - 1 - (i + 1) % MESH_DIVS,
             offset - 1 - i,
             offset + 1) for i in range(MESH_DIVS)
        ]

        cls.points = tuple(Point(p) for p in points)
        cls.triangles = tuple(triangles)
        cls.lines = lines

class Bishop(Piece):
    @classmethod
    def gen_mesh(cls) -> None:
        offset = 11 * MESH_DIVS

        sintab = tuple(sin(radians(t)) for t in range(0, 360, 360 // MESH_DIVS))
        points = []
        triangles = []
        lines = ()

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
            (0.22, 0.17)    # 9 * MESH_DIVS
        ]:
            points += [(
                m * sintab[i],
                y,
                m * sintab[i - (MESH_DIVS>>2)]
            ) for i in range(MESH_DIVS)]

        points += [
            (0, -1, 0),     # offset
            (0, 0.5, 0),    # offset + 1
        ]

        triangles += [
            ((i + 1) % MESH_DIVS, i, offset) for i in range(MESH_DIVS)
        ]

        for j in range(0, offset - MESH_DIVS, MESH_DIVS):
            triangles += [
                (j + i,
                 j + (i + 1) % MESH_DIVS,
                 j + MESH_DIVS + i) for i in range(MESH_DIVS)
            ] + [
                (j + (i + 1) % MESH_DIVS,
                 j + MESH_DIVS + (i + 1) % MESH_DIVS,
                 j + MESH_DIVS + i) for i in range(MESH_DIVS)
            ]

        triangles += [
            (offset - 1 - (i + 1) % MESH_DIVS,
             offset - 1 - i,
             offset + 1) for i in range(MESH_DIVS)
        ]

        cls.points = tuple(Point(p) for p in points)
        cls.triangles = tuple(triangles)
        cls.lines = lines

class Knight(Piece):
    @classmethod
    def gen_mesh(cls) -> None:
        offset = 4 * MESH_DIVS

        sintab = tuple(sin(radians(t)) for t in range(0, 360, 360 // MESH_DIVS))
        points = []
        triangles = []
        lines = ()

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

        triangles += [
            ((i + 1) % MESH_DIVS, i, offset) for i in range(MESH_DIVS)
        ]

        for j in range(0, offset - MESH_DIVS, MESH_DIVS):
            triangles += [
                (j + i,
                 j + (i + 1) % MESH_DIVS,
                 j + MESH_DIVS + i) for i in range(MESH_DIVS)
            ] + [
                (j + (i + 1) % MESH_DIVS,
                 j + MESH_DIVS + (i + 1) % MESH_DIVS,
                 j + MESH_DIVS + i) for i in range(MESH_DIVS)
            ]

        triangles += [
            (offset - 1 - (i + 1) % MESH_DIVS,
             offset - 1 - i,
             offset + 1) for i in range(MESH_DIVS)
        ]

        cls.points = tuple(Point(p) for p in points)
        cls.triangles = tuple(triangles)
        cls.lines = lines

class Pawn(Piece):
    @classmethod
    def gen_mesh(cls) -> None:
        offset = 11 * MESH_DIVS

        sintab = tuple(sin(radians(t)) for t in range(0, 360, 360 // MESH_DIVS))
        points = []
        triangles = []
        lines = ()

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

        triangles += [
            ((i + 1) % MESH_DIVS, i, offset) for i in range(MESH_DIVS)
        ]

        for j in range(0, offset - MESH_DIVS, MESH_DIVS):
            triangles += [
                (j + i,
                 j + (i + 1) % MESH_DIVS,
                 j + MESH_DIVS + i) for i in range(MESH_DIVS)
            ] + [
                (j + (i + 1) % MESH_DIVS,
                 j + MESH_DIVS + (i + 1) % MESH_DIVS,
                 j + MESH_DIVS + i) for i in range(MESH_DIVS)
            ]

        triangles += [
            (offset - 1 - (i + 1) % MESH_DIVS,
             offset - 1 - i,
             offset + 1) for i in range(MESH_DIVS)
        ]

        cls.points = tuple(Point(p) for p in points)
        cls.triangles = tuple(triangles)
        cls.lines = lines


# def arender(camera: Camera, obj: AbstractObject, kw):
#     return obj.project(camera), obj, kw

def render(objs: Collection[tuple[AbstractObject, dict[str, Any]]], camera: Camera) -> tuple[int, int]:
    global screenbuffer, zbuffer, polygons, verts
    screenbuffer = EMPTYBUFFER[:]
    zbuffer = {}

    polygons = 0
    verts = 0

    # with Pool() as pool:
    #     proj = pool.starmap(partial(arender, camera), objs)

    # for p, o, kw in proj:
    #     o.render(*p, **kw)

    for o, kw in objs:
        p = o.project(camera)

        o.render(*p, **kw)


    res = "\x1b[2J\x1b[H"
    res += "".join((c if screenbuffer[i - 1] else f"\x1b[{i//VW+1};{i%VW+1}H{c}") for i, c in enumerate(screenbuffer) if c)
    res += "\x1b[H3D Terminal Chess\n\nWASD + QE\tmove\n ← ↑ ↓ → \trotate\nspace/tab\tup/down"

    print(end=res, flush=True)
    return polygons, verts


King.gen_mesh()
Queen.gen_mesh()
Rook.gen_mesh()
Bishop.gen_mesh()
Knight.gen_mesh()
Pawn.gen_mesh()