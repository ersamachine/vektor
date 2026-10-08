"""OpenCV'nin ERSA Vektör'de kullanılan küçük bir alt kümesi — yalnız numpy ve Pillow ile.
Tarayıcıda (Pyodide) OpenCV'yi yüklemek bir dakikayı bulduğu için bu modül onun yerini tutar.
İşlev adları ve imzaları cv2 ile aynıdır; çekirdek `import ncv as cv2` ile kullanır."""
import math

import numpy as np
from PIL import Image

INTER_NEAREST, INTER_CUBIC, INTER_AREA = 0, 2, 3
COLOR_RGB2Lab, COLOR_Lab2RGB, COLOR_RGB2GRAY = 44, 56, 7
MORPH_ELLIPSE = 2
CC_STAT_LEFT, CC_STAT_TOP, CC_STAT_WIDTH, CC_STAT_HEIGHT, CC_STAT_AREA = 0, 1, 2, 3, 4

_WHITE = np.array([0.95047, 1.0, 1.08883], np.float32)
_M = np.array([[0.412453, 0.357580, 0.180423],
               [0.212671, 0.715160, 0.072169],
               [0.019334, 0.119193, 0.950227]], np.float32)
_MI = np.linalg.inv(_M).astype(np.float32)


# ---------------------------------------------------------------- renk

def cvtColor(img, code):
    a = np.asarray(img, np.float32)
    if code == COLOR_RGB2Lab:
        lin = np.where(a > 0.04045, ((a + 0.055) / 1.055) ** 2.4, a / 12.92)
        xyz = lin @ _M.T / _WHITE
        f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
        L = np.where(xyz[..., 1] > 0.008856, 116 * f[..., 1] - 16, 903.3 * xyz[..., 1])
        return np.stack([L, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1).astype(np.float32)
    if code == COLOR_RGB2GRAY:
        return (a[..., 0] * 0.299 + a[..., 1] * 0.587 + a[..., 2] * 0.114).astype(np.float32)
    if code == COLOR_Lab2RGB:
        L, A, B = a[..., 0], a[..., 1], a[..., 2]
        fy = (L + 16) / 116
        f = np.stack([fy + A / 500, fy, fy - B / 200], -1)
        xyz = np.where(f > 0.2069, f ** 3, (f - 16 / 116) / 7.787) * _WHITE
        lin = np.clip(xyz @ _MI.T, 0, 1)
        return np.where(lin > 0.0031308, 1.055 * lin ** (1 / 2.4) - 0.055, 12.92 * lin).astype(np.float32)
    raise ValueError(code)


# ---------------------------------------------------------------- boyutlandırma ve bulanıklaştırma

_FILTERS = {INTER_NEAREST: Image.NEAREST, INTER_CUBIC: Image.BICUBIC, INTER_AREA: Image.BOX}


def resize(img, size, interpolation=INTER_AREA):
    a = np.asarray(img)
    w, h = int(size[0]), int(size[1])
    if a.ndim == 3:
        return np.stack([resize(a[..., c], (w, h), interpolation) for c in range(a.shape[2])], 2)
    if interpolation == INTER_NEAREST and a.dtype == np.uint8:
        return np.asarray(Image.fromarray(a).resize((w, h), Image.NEAREST))
    out = np.asarray(Image.fromarray(np.ascontiguousarray(a, np.float32), "F").resize((w, h), _FILTERS[interpolation]))
    return out.astype(a.dtype) if a.dtype.kind == "f" else out


def _conv1d(a, k, axis):
    r = len(k) // 2
    pad = [(0, 0)] * a.ndim
    pad[axis] = (r, r)
    p = np.pad(a, pad, mode="reflect")
    out = np.zeros_like(a)
    n = a.shape[axis]
    for i, wv in enumerate(k):
        sl = [slice(None)] * a.ndim
        sl[axis] = slice(i, i + n)
        out += wv * p[tuple(sl)]
    return out


def GaussianBlur(img, ksize, sigma):
    a = np.asarray(img, np.float32)
    r = max(1, int(math.ceil(3 * sigma)))
    x = np.arange(-r, r + 1, dtype=np.float32)
    k = np.exp(-x * x / (2 * sigma * sigma))
    k /= k.sum()
    return _conv1d(_conv1d(a, k, 0), k, 1)


def bilateralFilter(img, d, sigmaColor, sigmaSpace):
    a = np.asarray(img, np.float32)
    r = d // 2
    p = np.pad(a, ((r, r), (r, r), (0, 0)), mode="reflect")
    h, w = a.shape[:2]
    num = np.zeros_like(a)
    den = np.zeros(a.shape[:2], np.float32)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dy * dy + dx * dx > r * r:
                continue
            q = p[r + dy:r + dy + h, r + dx:r + dx + w]
            diff = np.abs(q - a).sum(2)
            wgt = np.exp(-(dy * dy + dx * dx) / (2 * sigmaSpace ** 2) - diff * diff / (2 * sigmaColor ** 2))
            num += q * wgt[..., None]
            den += wgt
    return num / den[..., None]


# ---------------------------------------------------------------- biçimsel işlemler (dikdörtgen çekirdek, ayrık)

def getStructuringElement(shape, ksize):
    return np.ones((int(ksize[1]), int(ksize[0])), np.uint8)


def _rank(a, ky, kx, fn, fill):
    out = np.asarray(a)
    for axis, k in ((0, ky), (1, kx)):
        r = k // 2
        if r == 0:
            continue
        pad = [(0, 0)] * out.ndim
        pad[axis] = (r, r)
        p = np.pad(out, pad, mode="constant", constant_values=fill)
        n = out.shape[axis]
        res = None
        for i in range(2 * r + 1):
            sl = [slice(None)] * out.ndim
            sl[axis] = slice(i, i + n)
            v = p[tuple(sl)]
            res = v.copy() if res is None else fn(res, v, out=res)
        out = res
    return out


def _limits(a):
    a = np.asarray(a)
    if a.dtype == bool:
        return False, True
    if a.dtype.kind in "iu":
        info = np.iinfo(a.dtype)
        return info.min, info.max
    return -np.inf, np.inf


def dilate(img, kernel, iterations=1):
    lo, _ = _limits(img)
    out = np.asarray(img)
    for _ in range(iterations):
        out = _rank(out, kernel.shape[0], kernel.shape[1], np.maximum, lo)
    return out


def erode(img, kernel, iterations=1):
    _, hi = _limits(img)
    out = np.asarray(img)
    for _ in range(iterations):
        out = _rank(out, kernel.shape[0], kernel.shape[1], np.minimum, hi)
    return out


# ---------------------------------------------------------------- bağlı bileşenler (satır parçası + birleşim-bul)

def connectedComponentsWithStats(img, connectivity=8):
    m = np.asarray(img) != 0
    h, w = m.shape
    padded = np.zeros((h, w + 2), np.int8)
    padded[:, 1:-1] = m
    d = np.diff(padded, axis=1)
    rows, a = np.nonzero(d == 1)        # parça başı (dahil)
    _, e = np.nonzero(d == -1)          # parça sonu (hariç)
    b = e - 1
    n = len(rows)
    labels = np.zeros((h, w), np.int32)
    if n == 0:
        stats = np.array([[0, 0, w, h, h * w]], np.int32)
        return 1, labels, stats, np.array([[(w - 1) / 2, (h - 1) / 2]])

    W = w + 2
    ext = 1 if connectivity == 8 else 0
    keyA = rows.astype(np.int64) * W + a
    keyB = rows.astype(np.int64) * W + b
    # alt satırdaki her parça j için üst satırda değen parçalar [lo, hi)
    up = rows > 0
    j = np.nonzero(up)[0]
    r_up = rows[j] - 1
    lo = np.searchsorted(keyB, r_up.astype(np.int64) * W + a[j] - ext, side="left")
    hi = np.searchsorted(keyA, r_up.astype(np.int64) * W + b[j] + ext, side="right")
    cnt = np.maximum(hi - lo, 0)
    pj = np.repeat(j, cnt)
    start = np.repeat(np.cumsum(cnt) - cnt, cnt)
    pi = np.repeat(lo, cnt) + (np.arange(pj.size) - start)

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for x, y in zip(pi.tolist(), pj.tolist()):
        rx, ry = find(x), find(y)
        if rx != ry:
            if rx < ry:
                parent[ry] = rx
            else:
                parent[rx] = ry
    root = np.array([find(x) for x in range(n)], np.intp)
    uniq, lab = np.unique(root, return_inverse=True)
    lab = lab.ravel().astype(np.int32) + 1
    N = len(uniq) + 1

    lens = (b - a + 1).astype(np.intp)
    total = int(lens.sum())
    st = (rows.astype(np.intp) * w + a).astype(np.intp)
    pos = np.repeat(st, lens) + (np.arange(total, dtype=np.intp) - np.repeat(np.cumsum(lens) - lens, lens))
    flat = labels.reshape(-1)
    flat[pos] = np.repeat(lab, lens)

    area = np.bincount(lab, weights=lens, minlength=N)
    left = np.full(N, w, np.int64)
    right = np.full(N, -1, np.int64)
    top = np.full(N, h, np.int64)
    bot = np.full(N, -1, np.int64)
    np.minimum.at(left, lab, a)
    np.maximum.at(right, lab, b)
    np.minimum.at(top, lab, rows)
    np.maximum.at(bot, lab, rows)
    cx = np.bincount(lab, weights=lens * (a + b) / 2.0, minlength=N)
    cy = np.bincount(lab, weights=lens * rows.astype(np.float64), minlength=N)
    stats = np.zeros((N, 5), np.int32)
    stats[:, 0] = left
    stats[:, 1] = top
    stats[:, 2] = right - left + 1
    stats[:, 3] = bot - top + 1
    stats[:, 4] = area
    fg = int(area[1:].sum())
    stats[0] = (0, 0, w, h, h * w - fg)
    cent = np.zeros((N, 2))
    cent[1:, 0] = cx[1:] / np.maximum(area[1:], 1)
    cent[1:, 1] = cy[1:] / np.maximum(area[1:], 1)
    return N, labels, stats, cent


# ---------------------------------------------------------------- çokgen yardımcıları

def contourArea(poly):
    p = np.asarray(poly, np.float64).reshape(-1, 2)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def pointPolygonTest(poly, pt, measureDist=False):
    p = np.asarray(poly, np.float64).reshape(-1, 2)
    x, y = float(pt[0]), float(pt[1])
    x0, y0 = p[:, 0], p[:, 1]
    x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
    # kenar üzerinde mi?
    dx, dy = x1 - x0, y1 - y0
    seg = dx * dx + dy * dy
    t = np.clip(((x - x0) * dx + (y - y0) * dy) / np.where(seg > 0, seg, 1), 0, 1)
    if np.min((x0 + t * dx - x) ** 2 + (y0 + t * dy - y) ** 2) < 1e-12:
        return 0.0
    cross = (y0 > y) != (y1 > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        xi = x0 + (y - y0) * dx / np.where(dy != 0, dy, 1)
    inside = np.count_nonzero(cross & (x < xi)) % 2 == 1
    return 1.0 if inside else -1.0
