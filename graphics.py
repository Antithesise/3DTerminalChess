from abc import ABC, abstractmethod
from bisect import bisect_left
from collections.abc import Collection
from functools import lru_cache
from math import copysign, cos, hypot, radians, sin, sqrt
from os import get_terminal_size
from sys import float_info
from typing import Any, Never, NoReturn, Self
from numpy import array, cross, dot, empty, float64
from numpy.linalg import det
from numpy.typing import ArrayLike, NDArray

from config import *


FMIN = (5e-324 or float_info.min) # smallest floating point value

VW, VH = get_terminal_size()
HVW, HVH = VW >> 1, VH >> 1

EMPTYBUFFER = [" "] * (VH * VW)

GRADS = [-4, -2, -1, -0.5, -0.2, 0, 0.2, 0.5, 1, 2, 4]
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

Vec3 = tuple[float, float, float] | ArrayLike

screenbuffer = EMPTYBUFFER[:]
zbuffer: dict[int, float] = {}

zbupdate = zbuffer.setdefault

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
def shade(p1: tuple[float, float, float],
          p2: tuple[float, float, float],
          p3: tuple[float, float, float]) -> float:
    p = array(p1, float64)
    N = cross(p2 - p, p3 - p)

    return 0.5 + dot(N / hypot(*N), LIGHTNORM) / 2

@lru_cache(CACHE_SIZE)
def rotmat(rx: float, ry: float, rz: float) -> NDArray:
    r = radians(rx), radians(ry), radians(rz)
    sx, sy, sz = map(sin, r)
    cx, cy, cz = map(cos, r)

    # x = radians(rx)
    # y = radians(ry)
    # z = radians(rz)

    # sx = sin(x)
    # sy = sin(y)
    # sz = sin(z)

    # cx = cos(x)
    # cy = cos(y)
    # cz = cos(z)

    return array((
        ( 1,  0,  0 ),
        ( 0,  cx, sx),
        ( 0, -sx, cx)
    )) @ array((
        ( cy, 0, -sy),
        ( 0,  1,  0 ),
        ( sy, 0,  cy)
    )) @ array((
        ( cz, sz, 0 ),
        (-sz, cz, 0 ),
        ( 0,  0,  1 )
    ))

@lru_cache(CACHE_SIZE)
def yrotmat(ry: float) -> NDArray:
    a = radians(ry)
    s = sin(a)
    c = cos(a)

    return array((
        ( c,  0, -s ),
        ( 0,  1,  0 ),
        ( s,  0,  c )
    ))

def xyclip(x1: float, y1: float, x2: float, y2: float) -> tuple[int, int, int, int] | None:
    """
    liang-barsky for the win
    """

    dx = x2 - x1
    dy = y2 - y1

    p = (-dx, dx, -dy, dy)
    q = (x1 - 1, VW - x1, y1 - 1, VH - y1)
    start = 0
    stop = 1

    for i in range(4):
        if p[i] == 0:
            if q[i] < 0.0:
                return
        else:
            if p[i] < 0.0:
                start = max(q[i] / p[i], start)
            else:
                stop = min(q[i] / p[i], stop)

    if stop < start:
        return

    x2 = x1 + stop * dx
    y2 = y1 + stop * dy
    x1 = x1 + start * dx
    y1 = y1 + start * dy

    return int(x1 + 0.5), int(y1 + 0.5), int(x2 + 0.5), int(y2 + 0.5)

def write(x: int, y: int, depth: float, fill: str, overdraw: bool=False) -> None:
    if 0 < x < VW and 0 < y < VH:
        i = (y - 1) * VW + x - 1

        if overdraw or depth <= zbupdate(i, depth):
            screenbuffer[i] = fill
            zbuffer[i] = depth


class AbstractObject(ABC):
    pos: NDArray
    rot: NDArray

    def __new__(cls, *args, **kwargs) -> Self:
        obj = super().__new__(cls)

        obj.pos = empty((3,), float64)
        obj.rot = empty((3,), float64)

        return obj

    # @property
    # def x(self) -> float:
    #     return self.pos[0]

    # @property
    # def y(self) -> float:
    #     return self.pos[1]

    # @property
    # def z(self) -> float:
    #     return self.pos[2]

    # @property
    # def rx(self) -> float:
    #     return self.rot[0]

    # @property
    # def ry(self) -> float:
    #     return self.rot[1]

    # @property
    # def rz(self) -> float:
    #     return self.rot[2]

    @property
    def rotmat(self) -> NDArray:
        """
        Useful for transforming/aligning children
        """

        return rotmat(self.rot[0], self.rot[1], self.rot[2])

    @property
    def yrotmat(self) -> NDArray:
        return yrotmat(self.rot[1])

    @abstractmethod
    def project(self, obj: "AbstractObject") -> tuple[tuple[float | tuple[float, float], ...], tuple[float, ...]] | tuple[tuple[float | tuple[float, float], ...], tuple[float, ...], tuple[float, ...]]: ...

    @classmethod
    @abstractmethod
    def render(cls, proj: tuple[float | tuple[float, float], ...], depth: tuple[float, ...], *args, **kwargs) -> None: ...


class Camera(AbstractObject):
    def __init__(self,
                 pos: Vec3=(0,0,0),
                 rot: Vec3=(0,0,0)) -> None:

        self.pos = array(pos, float64)
        self.rot = array(rot, float64)

    @lru_cache(CACHE_SIZE)
    def __project2(self, dx: float, dy: float, dz: float) -> tuple[float, float, float]:
        if not dz:
            dz = FMIN

        scaling = SCALE / abs(dz)

        return HVW + dx * scaling, HVH - dy * scaling * CRATIO, dz

    def project(self, pos: Vec3) -> tuple[tuple[float, float], float]:
        x, y, z = self.__project2(
            *(self.rotmat @ (pos - self.pos)) # faster than keying rot in cache
        )

        return (x, y), z

    @classmethod
    def render(cls, proj: Never, depth: Never) -> NoReturn:
        raise NotImplementedError("No Escherian shenanigans here please.")

class Point(AbstractObject):
    def __init__(self,
                 pos: Vec3=(0,0,0),
                 rot: Vec3=(0,0,0)) -> None:

        self.pos = array(pos, float64)
        self.rot = array(rot, float64)

    def project(self, camera: Camera) -> tuple[tuple[float, float], float]:
        return camera.project(self.pos)

    @classmethod
    def render(cls, proj: tuple[float, float], depth: float, fill: str="#") -> None:
        if 0 < depth:
            x, y = proj

            write(int(x + 0.5), int(y + 0.5), depth, fill)

class Line(AbstractObject):
    p1: tuple[float, float, float]
    p2: tuple[float, float, float]
    worldloc: bool

    def __init__(self,
                 p1: tuple[float, float, float],
                 p2: tuple[float, float, float],
                 *,
                 pos: Vec3=(0,0,0),
                 rot: Vec3=(0,0,0),
                 worldloc: bool=True) -> None:

        self.p1, self.p2 = p1, p2
        self.worldloc = worldloc

        self.pos = array(pos, float64)
        self.rot = array(rot, float64)

    def project(self, camera: Camera) -> tuple[tuple[float, float, float, float], tuple[float, float]]:
        if self.worldloc:
            r1, c1 = camera.project(self.p1)
            r2, c2 = camera.project(self.p2)

        else:
            rm = self.rotmat
            pos = self.pos

            r1, c1 = camera.project(pos + (rm @ self.p1))
            r2, c2 = camera.project(pos + (rm @ self.p2))

        return r1 + r2, (c1, c2)

    @classmethod
    def render(cls,
               proj: tuple[float, float, float, float],
               depth: tuple[float, float],
               *,
               fill: str="",
               overdraw: bool=False) -> None:
        """
        Good old Bresenham
        """

        if depth[0] <= 0.0 and depth[1] <= 0.0:
            return

        clipped = xyclip(*proj) # hope proj is correct length
        if clipped is None:
            return

        x1, y1, x2, y2 = clipped

        if  x1 < 1 > x2 or x1 > VW - 1 < x2 or \
            y1 < 1 > y2 or y1 > VH - 1 < y2:
            return

        z = max(depth) # can't be bothered to interpolate for a line

        dx = x2 - x1
        dy = y2 - y1
        sx = int(copysign(1, dx))
        sy = int(copysign(1, dy))

        fill = fill or gradmap(-dy / (dx or FMIN))

        if dx == 0:
            y = max(min(y1, y2), 0)
            dy = min(max(y1, y2), VH) + 1 - y

            for y in range(y, y + dy):
                write(x1, y, z, fill, overdraw)

            return

        elif dy == 0:
            x = max(min(x1, x2), 0)
            dx = min(max(x1, x2), VW) + 1 - x

            for x in range(x, x + dx):
                write(x, y1, z, fill, overdraw)

            return

        else:
            dx = abs(dx)
            dy = -abs(dy)
            error = dx + dy

            write(x1, y1, z, fill, overdraw)

            if 0 < sx and 0 < sy:
                x2 = min(VW, x2)
                y2 = min(VH, y2)
                while True:
                    e2 = 2 * error

                    if dy <= e2:
                        error += dy
                        x1 += sx

                    if e2 <= dx:
                        error += dx
                        y1 += sy

                    if x2 < x1 or y2 < y1:
                        break

                    write(x1, y1, z, fill, overdraw)
            elif 0 < sx:
                x2 = min(VW, x2)
                y2 = max(1, y2)
                while True:
                    e2 = 2 * error

                    if dy <= e2:
                        error += dy
                        x1 += sx

                    if e2 <= dx:
                        error += dx
                        y1 += sy

                    if x2 < x1 or y1 < y2:
                        break

                    write(x1, y1, z, fill, overdraw)
            elif 0 < sy:
                x2 = max(1, x2)
                y2 = min(VH, y2)
                while True:
                    e2 = 2 * error

                    if dy <= e2:
                        error += dy
                        x1 += sx

                    if e2 <= dx:
                        error += dx
                        y1 += sy

                    if x1 < x2 or y2 < y1:
                        break

                    write(x1, y1, z, fill, overdraw)
            else:
                x2 = max(1, x2)
                y2 = max(1, y2)
                while True:
                    e2 = 2 * error

                    if dy <= e2:
                        error += dy
                        x1 += sx

                    if e2 <= dx:
                        error += dx
                        y1 += sy

                    if x1 < x2 or y1 < y2:
                        break

                    write(x1, y1, z, fill, overdraw)


class Triangle(AbstractObject):
    p1: tuple[float, float, float]
    p2: tuple[float, float, float]
    p3: tuple[float, float, float]
    worldloc: bool

    def __init__(self,
                 p1: tuple[float, float, float],
                 p2: tuple[float, float, float],
                 p3: tuple[float, float, float],
                 *,
                 pos: Vec3=(0,0,0),
                 rot: Vec3=(0,0,0),
                 worldloc: bool=True) -> None:
        self.p1, self.p2, self.p3 = p1, p2, p3
        self.worldloc = worldloc

        self.pos = array(pos, float64)
        self.rot = array(rot, float64)

    def project(self, camera: Camera) -> tuple[tuple[float, float, float, float, float, float], tuple[float, float, float]]:
        if self.worldloc:
            r1, c1 = camera.project(self.p1)
            r2, c2 = camera.project(self.p2)
            r3, c3 = camera.project(self.p3)

        else:
            rm = self.rotmat
            pos = self.pos

            r1, c1 = camera.project(pos + (rm @ self.p1))
            r2, c2 = camera.project(pos + (rm @ self.p2))
            r3, c3 = camera.project(pos + (rm @ self.p3))

        return r1 + r2 + r3, (c1, c2, c3)

    @classmethod
    def render(cls,
               proj: tuple[float, float, float, float, float, float],
               depth: tuple[float, float, float],
               shading: float=-1,
               *,
               fill: str="#") -> None:
        """
        Barycentric rasterisation algorithm based on Nicolas Capens'
        https://web.archive.org/web/20050408192410/http://sw-shader.sourceforge.net/rasterizer.html
        """

        x1, y1, x2, y2, x3, y3 = proj # hope proj is correct length

        # backface culling
        if x2*y1 + x3*y2 + x1*y3 < x1*y2 + x2*y3 + x3*y1:
            return

            # (x2, y2), (x1, y1), (x3, y3) = proj # reverse winding

        z1, z2, z3 = depth
        z1, z2, z3 = 1/z1, 1/z2, 1/z3

        # viewport culling
        if (z1 < 0.0 and z2 < 0.0 and z3 < 0.0) or\
            (x1 < 1.0 and x2 < 1.0 and x3 < 1.0) or (VW < x1 and VW < x2 and VW < x3) or\
            (y1 < 1.0 and y2 < 1.0 and y3 < 1.0) or (VH < y1 and VH < y2 and VH < y3):
                return

        elif 0.0 <= shading:
            fill = f"\x1b[38;5;{int(255.5 - 23.0 * sqrt(shading**3))}m{fill}\x1b[0m"

        dx32 = x3 - x2
        dx13 = x1 - x3
        dx21 = x2 - x1

        dy32 = y3 - y2
        dy13 = y1 - y3
        dy21 = y2 - y1

        area = dy13 * dx21 - dx13 * dy21

        minx = int(min(x1, x2, x3, VW))
        maxx = int(max(x1, x2, x3, 0) + 0.999999) # not the most accurate
        miny = int(min(y1, y2, y3, VH))
        maxy = int(max(y1, y2, y3, 0) + 0.999999) # not the most accurate

        wy1 = (y2 - miny - 0.5) * dx32 + (minx + 0.5 - x2) * dy32
        wy2 = (y3 - miny - 0.5) * dx13 + (minx + 0.5 - x3) * dy13
        wy3 = (y1 - miny - 0.5) * dx21 + (minx + 0.5 - x1) * dy21

        for y in range(miny, maxy):
            w1 = wy1
            w2 = wy2
            w3 = wy3

            for x in range(minx, maxx):
                c = (0 < w1 or (w1 == 0 and (0 < dy32 or dy32 == 0 < dx32))) and\
                (0 < w2 or (w2 == 0 and (0 < dy13 or dy13 == 0 < dx13))) and\
                (0 < w3 or (w3 == 0 and (0 < dy21 or dy21 == 0 < dx21)))

                if c:
                    z = area / (w1 * z1 + w2 * z2 + w3 * z3)

                    write(x, y, z, fill)

                w1 += dy32
                w2 += dy13
                w3 += dy21

            wy1 -= dx32
            wy2 -= dx13
            wy3 -= dx21

class Mesh(AbstractObject):
    points: tuple[tuple[float, float, float], ...]
    triangles: tuple[tuple[int, int, int], ...]
    lines: tuple[tuple[int, ...], ...]
    shading: tuple[float, ...]
    fill: str
    worldloc: bool

    def __init__(self,
                 points: tuple[tuple[float, float, float], ...],
                 triangles: tuple[tuple[int, int, int], ...],
                 lines: tuple[tuple[int, ...], ...],
                 pos: Vec3=(0,0,0),
                 rot: Vec3=(0,0,0),
                 fill: str="#",
                 worldloc: bool=True) -> None:

        self.points = points
        self.triangles = triangles
        self.lines = lines
        self.pos = array(pos, float64)
        self.rot = array(rot, float64)
        self.fill = fill
        self.worldloc = worldloc

        self.bake()

    def bake(self) -> None:
        rm = self.rotmat

        points = [tuple(rm @ p) for p in self.points]

        self.shading = tuple(
            shade(points[i], points[j], points[k]) for i, j, k in self.triangles
        )

    def project(self, camera: Camera) -> tuple[tuple[tuple[float, float], ...], tuple[float, ...]]:
        res = []
        depth = []

        if self.worldloc:
            for p in self.points:
                r, z = camera.project(p)
                res.append(r)
                depth.append(z)

        else:
            rm = self.rotmat
            pos = self.pos

            for p in self.points:
                r, z = camera.project(pos + (rm @ p))

                res.append(r)
                depth.append(z)

        return tuple(res), tuple(depth)

    def render(self,
               proj: tuple[tuple[float, float], ...],
               depth: tuple[float, ...],) -> None:
        shading = self.shading
        fill = self.fill

        for i, (j, k, l) in enumerate(self.triangles):
            Triangle.render(
                proj[j] + proj[k] + proj[l],
                (depth[j], depth[k], depth[l]),
                shading[i],
                fill=fill
            )

        for p in self.lines:
            for i in range(len(p) - 1):
                Line.render(
                    proj[p[i]] + proj[p[i+1]],
                    (depth[p[i]], depth[p[i+1]]),
                    fill=fill
                )

class Piece(Mesh):
    side: bool # True = white, False = black
    bakedmat: NDArray # baked rotmat
    bakedpoints: list[tuple[float, float, float]] # baked relative points
    worldloc = False

    def __init__(self,
                 pos: Vec3=(0,0,0),
                 side: Any=True) -> None:

        self.side = side
        self.fill = "#" if self.side else ":"

        self.pos = array(pos, float64)
        self.rot = array((0, 180 * (not side), 0), float64)

        self.bake()

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)

        cls.gen_mesh()

    def bake(self) -> None:
        rm = self.bakedmat = self.rotmat # rotation is never going to change

        bp = self.bakedpoints = [tuple(rm @ p) for p in self.points]

        self.shading = tuple(
            shade(bp[i], bp[j], bp[k]) for i, j, k in self.triangles
        )

    def project(self, camera: Camera) -> tuple[tuple[tuple[float, float], ...], tuple[float, ...]]:
        res = []
        depth = []

        pos = self.pos

        for p in self.bakedpoints:
            r, z = camera.project(pos + p)

            res.append(r)
            depth.append(z)

        return tuple(res), tuple(depth)

    def render(self,
               proj: tuple[tuple[float, float], ...],
               depth: tuple[float, ...]) -> None:
        shading = self.shading
        fill = self.fill

        for i, (j, k, l) in enumerate(self.triangles):
            Triangle.render(
                proj[j] + proj[k] +  proj[l],
                (depth[j], depth[k], depth[l]),
                shading[i],
                fill=fill
            )

        for i, j in self.lines:
            Line.render(
                proj[i] + proj[j],
                (depth[i], depth[j]),
                fill=fill
            )

    def torankfile(self, rank: int, file: int) -> None:
        self.pos = array((file - 3.5, 0, rank - 3.5), float64)

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
            (0.2, 0.1),     # 7 * MESH_DIVS
            (0.25, 0.2),    # 8 * MESH_DIVS
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

        cls.points = tuple(points)
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
            (0.2, 0.1),     # 7 * MESH_DIVS
            (0.25, 0.2),    # 8 * MESH_DIVS
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

        cls.points = tuple(points)
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

        cls.points = tuple(points)
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

        cls.points = tuple(points)
        cls.triangles = tuple(triangles)
        cls.lines = lines

class Knight(Piece):
    @classmethod
    def gen_mesh(cls) -> None:
        offset = 5 * MESH_DIVS

        sintab = tuple(sin(radians(t)) for t in range(0, 360, 360 // MESH_DIVS))
        points = []
        triangles = []
        lines = ()

        for (y, m) in [
            (-1, 0.35),     # 0 * MESH_DIVS
            (-0.9, 0.35),  # 1 * MESH_DIVS
            (-0.8, 0.25),  # 2 * MESH_DIVS
            (-0.7, 0.25),  # 3 * MESH_DIVS
            (-0.675, 0.2),  # 4 * MESH_DIVS
        ]:
            points += [(
                m * sintab[i],
                y,
                m * sintab[i - (MESH_DIVS>>2)]
            ) for i in range(MESH_DIVS)]

        points += [
            (0, -1, 0),     # offset
            (0, -0.675, 0), # offset + 1
            (0, -0.675, 0.2), # offset + 2 front point
            (0, -0.675, -0.2), # offset + 3 back point
            (-0.08, -0.59, 0.36), # offset + 4 front L
            (0.08, -0.59, 0.36), # offset + 5 front R
            (-0.1, -0.61, -0.35), # offset + 6 back L
            (0.1, -0.61, -0.35), # offset + 7 back R
            (-0.22, -0.63, 0.2), # offset + 8 wing L
            (0.22, -0.63, 0.2), # offset + 9 wing R
            (-0.1, -0.2, 0), # offset + 10 neck L
            ( 0.1, -0.2, 0), # offset + 11 neck R
            (-0.15, -0.42, 0.16), # offset + 12 curve L
            (0.15, -0.42, 0.16), # offset + 13 curve R
            (-0.08, -0.1, -0.42), # offset + 14 shoulder L
            (0.08, -0.1, -0.42), # offset + 15 shoulder R
            (-0.12, 0.35, -0.05), # offset + 16 ear L
            ( 0.12, 0.35, -0.05), # offset + 17 ear R
            (0, 0.18, -0.18), # offset + 18 sagittal back
            (0, 0.27, -0.05), # offset + 19 sagittal front
            (-0.116, 0.24, -0.05), # offset + 20 brow L
            ( 0.116, 0.24, -0.05), # offset + 21 brow R
            (-0.116, -0.03, 0.1), # offset + 22 cheek L
            ( 0.116, -0.03, 0.1), # offset + 23 cheek R
            (0, -0.1, 0.35), # offset + 24 jaw
            (0, 0.05, 0.36) # offset + 25 snout
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
            (offset + 2, offset + 3, offset + 8),
            (offset + 2, offset + 9, offset + 3),
            (offset + 2, offset + 4, offset + 5),
            (offset + 3, offset + 7, offset + 6),
            (offset + 2, offset + 8, offset + 4),
            (offset + 2, offset + 5, offset + 9),
            (offset + 3, offset + 6, offset + 8),
            (offset + 3, offset + 9, offset + 7),
            (offset + 4, offset + 10, offset + 11),
            (offset + 4, offset + 11, offset + 5),
            (offset + 4, offset + 8, offset + 12),
            (offset + 5, offset + 13, offset + 9),
            (offset + 4, offset + 12, offset + 10),
            (offset + 5, offset + 11, offset + 13),
            (offset + 8, offset + 6, offset + 14),
            (offset + 7, offset + 9, offset + 15),
            (offset + 12, offset + 8, offset + 14),
            (offset + 9, offset + 13, offset + 15),
            (offset + 10, offset + 12, offset + 14),
            (offset + 13, offset + 11, offset + 15),
            (offset + 6, offset + 15, offset + 14),
            (offset + 6, offset + 7, offset + 15),
            (offset + 10, offset + 14, offset + 16),
            (offset + 11, offset + 17, offset + 15),
            (offset + 18, offset + 19, offset + 16),
            (offset + 18, offset + 17, offset + 19),
            (offset + 18, offset + 16, offset + 14),
            (offset + 18, offset + 15, offset + 17),
            (offset + 19, offset + 20, offset + 16),
            (offset + 19, offset + 17, offset + 21),
            (offset + 10, offset + 20, offset + 22),
            (offset + 21, offset + 11, offset + 23),
            (offset + 10, offset + 22, offset + 24),
            (offset + 11, offset + 24, offset + 23),
            (offset + 25, offset + 24, offset + 22),
            (offset + 25, offset + 23, offset + 24),
            (offset + 22, offset + 20, offset + 25),
            (offset + 23, offset + 25, offset + 21),
            (offset + 19, offset + 25, offset + 20),
            (offset + 19, offset + 21, offset + 25),
            (offset + 10, offset + 24, offset + 11),
            (offset + 14, offset + 15, offset + 18)
        ] + [
            (offset - 1 - (i + 1) % MESH_DIVS,
             offset - 1 - i,
             offset + 1) for i in range(MESH_DIVS)
        ]

        cls.points = tuple(points)
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

        cls.points = tuple(points)
        cls.triangles = tuple(triangles)
        cls.lines = lines


PieceType = King | Queen | Rook | Bishop | Knight | Pawn


def render(objs: Collection[AbstractObject], camera: Camera) -> str:
    screenbuffer[:] = EMPTYBUFFER
    zbuffer.clear()

    for o in objs:
        p = o.project(camera)
        o.render(*p) # type: ignore

    out = "".join(screenbuffer)

    return (
        f"\x1b[H{out}\x1b[H3D Terminal Chess\n\n"
        "WASD + QE\tmove\n ← ↑ ↓ → \trotate\nspace/tab\tup/down\n    X    \thome"
    )
