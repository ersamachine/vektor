"""ERSA Vektör çekirdeği: logo görselini CorelDRAW'da düzenlenebilir EPS'ye çevirir
ve üretilen dosyayı geri okuyup orijinalle karşılaştırarak doğrular."""
import csv
import math
import os
import re
import time
from dataclasses import dataclass, field

import ncv as cv2   # OpenCV yerine numpy+Pillow alt kümesi (tarayıcıda hızlı açılır)
import numpy as np
import potrace
from PIL import Image, ImageDraw, ImageOps

VERSION = "1.0"
MARK = f"%ERSA_Vektor: {VERSION}"
SUPPORTED = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".webp")

PRESETS = {
    "Hassas": dict(alphamax=0.75, opttolerance=0.1, speck=0.5),
    "Normal": dict(alphamax=1.0, opttolerance=0.2, speck=1.0),
    "Yumuşak": dict(alphamax=1.2, opttolerance=0.35, speck=2.5),
}
MAX_COLORS = 8
PROC_MAX = 2400      # işleme çözünürlüğü üst sınırı (uzun kenar, px)
WORK_TARGET = 2600   # izleme çözünürlüğü hedefi (uzun kenar, px)


@dataclass
class Settings:
    colors: object = "auto"   # "auto", "mono" (siyah), "own" (logonun rengi, efektli logolar) ya da 1..8
    preset: str = "Normal"
    size_mm: float = 200.0
    pdf: bool = False
    svg: bool = False
    out_dir: str = ""         # boşsa görselin yanına
    tamper: object = None     # yalnız testler için: kontrolden önce EPS'yi bozan fonksiyon
    debug: bool = False       # yalnız testler için: ara görüntüleri sonuca ekler


@dataclass
class Layer:
    rgb: tuple
    cmyk: tuple
    objects: list = field(default_factory=list)   # nesne = kontur listesi, kontur = [(op, noktalar)]

    @property
    def name(self):
        c, m, y, k = (round(v * 100) for v in self.cmyk)
        return f"C{c} M{m} Y{y} K{k}"


@dataclass
class Result:
    src: str
    status: str = "fail"          # ok | warn | fail
    headline: str = ""
    eps: str = ""
    extra: list = field(default_factory=list)
    layers: list = field(default_factory=list)
    n_objects: int = 0
    n_contours: int = 0
    size_mm: tuple = (0, 0)
    shape_score: float = 0.0
    color_score: float = 0.0
    shape_regions: int = 0
    color_regions: int = 0
    specks: int = 0
    warnings: list = field(default_factory=list)
    checks: list = field(default_factory=list)   # (geçti mi, açıklama)
    seconds: float = 0.0
    preview_orig: object = None
    preview_vec: object = None
    preview_diff: object = None

    def report(self):
        icon = {"ok": "✔", "warn": "⚠", "fail": "✖"}[self.status]
        lines = [f"{icon} {self.headline}", ""]
        if self.eps:
            lines.append(f"Dosya: {os.path.basename(self.eps)}")
            for p in self.extra:
                lines.append(f"       {os.path.basename(p)}")
            lines.append(f"Boyut: {self.size_mm[0]:.0f} × {self.size_mm[1]:.0f} mm")
            lines.append(f"Renk: {len(self.layers)}  ·  Nesne: {self.n_objects}  ·  Kontur: {self.n_contours}")
            for i, ly in enumerate(self.layers, 1):
                lines.append(f"   {i}. {ly.name}  ({len(ly.objects)} nesne)")
            lines.append("")
            lines.append(f"Çizim doğruluğu:  %{self.shape_score:.2f}".replace(".", ","))
            lines.append(f"Orijinale uyum:   %{self.color_score:.2f}".replace(".", ","))
            if self.specks:
                lines.append(f"Temizlenen leke/kir: {self.specks}")
        if self.checks:
            lines.append("")
            lines.append("Kontroller:")
            for ok, text in self.checks:
                lines.append(f"  {'✔' if ok else '✖'} {text}")
        if self.warnings:
            lines.append("")
            lines.append("Uyarılar:")
            for w in self.warnings:
                lines.append(f"  ⚠ {w}")
        lines.append("")
        lines.append(f"Süre: {self.seconds:.1f} sn")
        return "\n".join(lines)


# ---------------------------------------------------------------- renk yardımcıları

def to_lab(rgb):
    a = np.ascontiguousarray(rgb, dtype=np.float32)
    shape = a.shape
    lab = cv2.cvtColor(a.reshape(-1, 1, 3), cv2.COLOR_RGB2Lab)
    return lab.reshape(shape)


def rgb_to_cmyk(rgb):
    r, g, b = (float(v) for v in rgb)
    mx, mn = max(r, g, b), min(r, g, b)
    if mx < 0.2 and mx - mn < 0.08:
        return (0.0, 0.0, 0.0, 1.0)
    if mn > 0.97:
        return (0.0, 0.0, 0.0, 0.0)
    k = 1 - mx
    if mx - mn < 0.04:
        return (0.0, 0.0, 0.0, round(k, 2))
    c, m, y = ((1 - v - k) / (1 - k) for v in (r, g, b))
    return tuple(round(min(1, max(0, v)), 2) for v in (c, m, y, k))


# ---------------------------------------------------------------- 1) görseli yükle

def load_image(path):
    im = Image.open(path)
    try:
        im.seek(0)
    except EOFError:
        pass
    im = ImageOps.exif_transpose(im)
    if im.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
        arr = np.asarray(im).astype(np.float32)
        hi = 65535.0 if arr.max() > 255 else 255.0
        im = Image.fromarray(np.clip(arr / hi * 255, 0, 255).astype(np.uint8), "L")
    is_jpeg = (getattr(im, "format", None) or "").upper() == "JPEG" or path.lower().endswith((".jpg", ".jpeg"))
    has_alpha = im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info)
    display = None
    if has_alpha:
        rgba = np.asarray(im.convert("RGBA")).astype(np.float32) / 255
        a = rgba[..., 3:4]
        if a.min() > 0.99:
            has_alpha = False
            rgb = rgba[..., :3]
        else:
            opaque = rgba[rgba[..., 3] > 0.5][:, :3]
            if len(opaque) > 40000:
                opaque = opaque[np.random.default_rng(0).choice(len(opaque), 40000, replace=False)]
            keys = np.array([[1, 1, 1], [0, 0, 0], [1, 0, 1], [0, 1, 0], [0, 1, 1], [1, 1, 0]], np.float32)
            if len(opaque):
                dist = np.linalg.norm(opaque[:, None, :] - keys[None], axis=2).min(0)
                key = keys[int(np.argmax(dist))]
            else:
                key = keys[0]
            rgb = rgba[..., :3] * a + key * (1 - a)
            display = rgba[..., :3] * a + (1 - a)
    else:
        rgb = np.asarray(im.convert("RGB")).astype(np.float32) / 255
    if display is None:
        display = rgb
    return np.ascontiguousarray(rgb), np.ascontiguousarray(display), has_alpha, is_jpeg


# ---------------------------------------------------------------- 2) arka plan ve renk paleti

def detect_background(lab, rgb):
    h, w = lab.shape[:2]
    b = max(2, int(0.01 * min(h, w)))
    ring_lab = np.concatenate([lab[:b].reshape(-1, 3), lab[-b:].reshape(-1, 3),
                               lab[:, :b].reshape(-1, 3), lab[:, -b:].reshape(-1, 3)])
    ring_rgb = np.concatenate([rgb[:b].reshape(-1, 3), rgb[-b:].reshape(-1, 3),
                               rgb[:, :b].reshape(-1, 3), rgb[:, -b:].reshape(-1, 3)])
    q = np.floor(ring_lab / 6).astype(np.int32)
    keys, inv, counts = np.unique(q, axis=0, return_inverse=True, return_counts=True)
    top = int(np.argmax(counts))
    seed = ring_lab[inv.ravel() == top].mean(0)
    near = np.linalg.norm(ring_lab - seed, axis=1) < 12
    frac = float(near.mean())
    return ring_rgb[near].mean(0), ring_lab[near].mean(0), frac


def solid_mask(lab, tol):
    d = np.zeros(lab.shape[:2], np.float32)
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        d = np.maximum(d, np.linalg.norm(lab - np.roll(lab, (dy, dx), axis=(0, 1)), axis=2))
    return d < tol


def kmeans(pts, centers, iters=8):
    centers = centers.copy()
    for _ in range(iters):
        idx = np.argmin(((pts[:, None, :] - centers[None]) ** 2).sum(2), 1)
        for j in range(len(centers)):
            sel = idx == j
            if sel.any():
                centers[j] = pts[sel].mean(0)
    idx = np.argmin(((pts[:, None, :] - centers[None]) ** 2).sum(2), 1)
    return centers, idx


def find_palette(rgb, lab, bg_lab, want):
    """Logodaki düz renkleri bulur. Dönüş: (rgb merkezleri, degrade oranı)."""
    h, w = lab.shape[:2]
    f = min(1.0, 900 / max(h, w))
    if f < 1:
        rgb_s = cv2.resize(rgb, (max(1, int(w * f)), max(1, int(h * f))), interpolation=cv2.INTER_AREA)
        lab_s = to_lab(rgb_s)
    else:
        rgb_s, lab_s = rgb, lab
    fg = np.linalg.norm(lab_s - bg_lab, axis=2) > 15
    if not fg.any():
        return np.zeros((0, 3), np.float32), 0.0, 0
    solid = solid_mask(lab_s, 5) & fg
    use = solid if solid.sum() >= 30 else fg
    pts, prgb = lab_s[use], rgb_s[use]
    if len(pts) > 250000:
        sel = np.random.default_rng(1).choice(len(pts), 250000, replace=False)
        pts, prgb = pts[sel], prgb[sel]

    q = np.floor(pts / 4).astype(np.int32)
    _, inv, counts = np.unique(q, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    means = np.stack([np.bincount(inv, weights=pts[:, c]) / counts for c in range(3)], 1)
    minc = max(8, 0.003 * len(pts))
    centers, weights = [], []
    for i in np.argsort(-counts):
        m = means[i]
        if centers:
            d = np.linalg.norm(np.array(centers) - m, axis=1)
            j = int(np.argmin(d))
            if d[j] < 12:
                wsum = weights[j] + counts[i]
                centers[j] = (centers[j] * weights[j] + m * counts[i]) / wsum
                weights[j] = wsum
                continue
        if counts[i] >= minc:
            centers.append(m.astype(np.float64))
            weights.append(float(counts[i]))
    if not centers:
        centers, weights = [pts.mean(0)], [len(pts)]
    C = np.array(centers, np.float32)
    C, idx = kmeans(pts, C)
    share = np.bincount(idx, minlength=len(C)) / len(pts)
    keep = share >= 0.004
    if keep.sum() == 0:
        keep[np.argmax(share)] = True
    C = C[keep]
    # birbirine çok yakın renkleri birleştir
    merged = True
    while merged and len(C) > 1:
        merged = False
        d = np.linalg.norm(C[:, None] - C[None], axis=2) + np.eye(len(C)) * 1e9
        i, j = np.unravel_index(np.argmin(d), d.shape)
        if d[i, j] < 10:
            C = np.delete(C, j, 0)
            C, idx = kmeans(pts, C, 3)
            merged = True
    C, idx = kmeans(pts, C)
    share = np.bincount(idx, minlength=len(C)) / len(pts)
    order = np.argsort(-share)
    C = C[order]

    found = n = len(C)
    if isinstance(want, int):
        n = want
    elif n > MAX_COLORS:
        n = MAX_COLORS
    if n != len(C):
        if n < len(C):
            init = C[:n]
        else:
            extra = pts[np.random.default_rng(2).choice(len(pts), n - len(C), replace=False)]
            init = np.concatenate([C, extra])
        C, idx = kmeans(pts, init.astype(np.float32), 10)
    else:
        idx = np.argmin(((pts[:, None, :] - C[None]) ** 2).sum(2), 1)

    # degrade/gölge ölçüsü: düz görünen logo piksellerinin seçilen renge uzaklığı
    dmin = np.sqrt(((pts[:, None, :] - C[None]) ** 2).sum(2).min(1))
    grad = float((dmin > 6).mean())

    out = []
    for j in range(len(C)):
        sel = idx == j
        out.append(prgb[sel].mean(0) if sel.any() else cv2.cvtColor(C[j][None, None].astype(np.float32), cv2.COLOR_Lab2RGB)[0, 0])
    return np.clip(np.array(out, np.float32), 0, 1), grad, found


# ---------------------------------------------------------------- 3) piksel → renk payı

def blend_assign(rgb, centers):
    """Her piksel için iki renk ve aralarındaki karışım oranı. Düz alanlar en yakın renge,
    kenar pikselleri ise üzerinde durdukları renk çiftine atanır (ör. gri yazı kenarı
    gri↔beyaz karışımıdır, en yakın renk mavi olsa bile)."""
    h, w = rgb.shape[:2]
    P = rgb.reshape(-1, 3)
    n = len(P)
    C = centers.astype(np.float32)
    a = np.empty(n, np.uint8)
    r = np.empty(n, np.float32)
    for s in range(0, n, 1_000_000):
        p = P[s:s + 1_000_000]
        d2 = ((p[:, None, :] - C[None]) ** 2).sum(2)
        a[s:s + len(p)] = np.argmin(d2, 1)
        r[s:s + len(p)] = np.sqrt(d2.min(1))
    b = a.copy()
    t = np.zeros(n, np.float32)
    if len(C) < 2:
        return a.reshape(h, w), b.reshape(h, w), t.reshape(h, w), r.reshape(h, w)
    d = np.zeros((h, w), np.float32)
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)):
        d = np.maximum(d, np.abs(rgb - np.roll(rgb, (dy, dx), axis=(0, 1))).max(2))
    edge = np.nonzero(d.ravel() > 0.03)[0]
    pairs = [(i, j) for i in range(len(C)) for j in range(i + 1, len(C))]
    I = np.array([p[0] for p in pairs])
    J = np.array([p[1] for p in pairs])
    V = C[J] - C[I]
    VV = np.maximum((V * V).sum(1), 1e-6)
    for s in range(0, len(edge), 200_000):
        ix = edge[s:s + 200_000]
        p = P[ix]
        rel = p[:, None, :] - C[I][None]
        tt = np.clip((rel * V[None]).sum(2) / VV[None], 0, 1)
        res = ((rel - tt[..., None] * V[None]) ** 2).sum(2)
        best = np.argmin(res, 1)
        tb = tt[np.arange(len(ix)), best]
        a[ix] = I[best]
        b[ix] = J[best]
        t[ix] = tb
        r[ix] = np.sqrt(res[np.arange(len(ix)), best])
    return a.reshape(h, w), b.reshape(h, w), t.reshape(h, w), r.reshape(h, w)


def banding(lbl, palette):
    """Birbirine çok benzeyen iki rengin uzun sınır paylaşması = degrade bantlaması."""
    K = len(palette) + 1
    lab = to_lab(palette[None])[0]
    pairs = []
    for x, y in ((lbl[:, 1:], lbl[:, :-1]), (lbl[1:], lbl[:-1])):
        m = (x != y) & (x > 0) & (y > 0)
        pairs.append(np.minimum(x[m], y[m]).astype(np.intp) * K + np.maximum(x[m], y[m]))
    if not sum(len(p) for p in pairs):
        return False
    cnt = np.bincount(np.concatenate(pairs), minlength=K * K)
    fg = int((lbl > 0).sum())
    for code in np.nonzero(cnt > max(30, 0.002 * fg))[0]:
        i, j = divmod(int(code), K)
        if np.linalg.norm(lab[i - 1] - lab[j - 1]) < 20:
            return True
    return False


EFFECT_BLUR, EFFECT_CLOSE, EFFECT_SOFT = 0.8, 0, 0.5   # kenar yumuşatma; kapama 0 = ince ayırıcı çizgiler korunur


def otsu(values, bins=256):
    v = values[np.isfinite(values)]
    hi = float(np.percentile(v, 99.5)) if len(v) else 1.0
    hist, edges = np.histogram(np.clip(v, 0, hi), bins=bins, range=(0, max(hi, 1e-6)))
    p = hist.astype(np.float64) / max(1, hist.sum())
    c = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(p)
    m0 = np.cumsum(p * c)
    mt = m0[-1]
    between = (mt * w0 - m0) ** 2 / np.maximum(w0 * (1 - w0), 1e-12)
    return float(c[int(np.argmax(between))])


def effect_segment(rgb, lab, bg_lab):
    """Efektli logo (3B, metalik, gölgeli, değişken zemin): logonun zeminden ayrılma payı (0..1)
    ve ana rengi. Zemin yerel olarak tahmin edilir; ayrım renk doygunluğu ağırlıklıdır."""
    h, w = lab.shape[:2]
    f = min(1.0, 320 / max(h, w))
    sw, sh = max(8, int(w * f)), max(8, int(h * f))
    lab_s = cv2.resize(lab, (sw, sh), interpolation=cv2.INTER_AREA)
    bgm = (np.linalg.norm(lab_s - bg_lab, axis=2) < 12).astype(np.float32)
    sig = max(3.0, 0.05 * max(sw, sh))
    score = None
    for _ in range(3):
        den = cv2.GaussianBlur(bgm, (0, 0), sig)
        num = cv2.GaussianBlur(lab_s * bgm[..., None], (0, 0), sig)
        local_s = np.where(den[..., None] > 1e-3, num / np.maximum(den[..., None], 1e-6), bg_lab)
        local = cv2.resize(local_s.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
        d = lab - local
        score = np.sqrt((0.6 * d[..., 0]) ** 2 + d[..., 1] ** 2 + d[..., 2] ** 2)
        thr = otsu(score)
        fg_s = cv2.resize(score.astype(np.float32), (sw, sh), interpolation=cv2.INTER_AREA) > thr
        k = np.ones((5, 5), np.uint8)
        bgm = (~cv2.dilate(fg_s.astype(np.uint8), k).astype(bool)).astype(np.float32)
    # kenar piksel altı hassasiyetle: puan hafifçe yumuşatılır, geçiş geniş tutulur
    # (keskin "ya logo ya zemin" kararı kenarda piksel basamakları bırakır)
    score_b = cv2.GaussianBlur(score.astype(np.float32), (0, 0), EFFECT_BLUR)
    cov = np.clip((score_b - thr) / (0.5 * thr) + 0.5, 0, 1).astype(np.float32)
    m = cov >= 0.5
    # gölge dikişlerini kapat (2r pikselden dar koyu çizgiler) ve küçük karanlık delikleri doldur
    r = EFFECT_CLOSE
    kk = np.ones((2 * r + 1, 2 * r + 1), np.uint8)
    closed = cv2.erode(cv2.dilate(m.astype(np.uint8), kk), kk).astype(bool) if r else m.copy()
    n, cc, stats, _ = cv2.connectedComponentsWithStats((~closed).astype(np.uint8), connectivity=4)
    border = np.unique(np.concatenate([cc[0], cc[-1], cc[:, 0], cc[:, -1]]))
    small = np.zeros(n, bool)
    small[1:] = stats[1:, cv2.CC_STAT_AREA] < 0.0004 * h * w
    small[border] = False
    filled = closed | small[cc]
    cov = np.where(filled & ~m, 1.0, cov).astype(np.float32)
    cov = cv2.GaussianBlur(cov, (0, 0), EFFECT_SOFT) if EFFECT_SOFT else cov
    core = cov > 0.9
    color = np.median(rgb[core], 0).astype(np.float32) if core.any() else np.array([0, 0, 0], np.float32)
    return cov, color, int(small[1:].sum())


UNEXPLAINED = 0.15   # hiçbir renk çiftinin karışımıyla açıklanamayan piksel eşiği (RGB)


def augment_palette(rgb, bg_rgb, palette):
    """Paletteki renklerle açıklanamayan piksel kümesi varsa (ör. ince yazının rengi)
    o rengi palete ekler."""
    h, w = rgb.shape[:2]
    f = min(1.0, 1500 / max(h, w))
    small = cv2.resize(rgb, (max(1, int(w * f)), max(1, int(h * f))), interpolation=cv2.INTER_AREA) if f < 1 else rgb
    added = 0
    while len(palette) < MAX_COLORS:
        centers = np.concatenate([bg_rgb[None].astype(np.float32), palette])
        _, _, _, res = blend_assign(small, centers)
        # yalnız en az 3 piksellik kümeler sayılır (tek tük gürültü değil, küçük nokta bile olsa)
        n, cc, stt, _ = cv2.connectedComponentsWithStats((res > UNEXPLAINED).astype(np.uint8), connectivity=8)
        okc = np.zeros(n, bool)
        okc[1:] = stt[1:, cv2.CC_STAT_AREA] >= 3
        bad = small[okc[cc]]
        if len(bad) < 12:
            break
        q = np.floor(bad / 0.08).astype(np.int32)
        _, inv, counts = np.unique(q, axis=0, return_inverse=True, return_counts=True)
        seed = bad[inv.ravel() == int(np.argmax(counts))].mean(0)
        members = bad[np.abs(bad - seed).max(1) < 0.12]
        if len(members) < 8:
            break
        dist = np.linalg.norm(members - bg_rgb, axis=1)
        core = members[dist >= np.percentile(dist, 70)]
        new = np.median(core, 0).astype(np.float32)
        if np.abs(centers - new).max(1).min() < 0.08:
            break
        # JPEG halesi: mevcut bir rengin zeminden uzağa taşmış (aşırı doymuş) kopyası yeni renk değildir
        halo = False
        for c in palette:
            v = c - bg_rgb
            vv = float((v * v).sum())
            if vv < 1e-6:
                continue
            tt = float(((new - bg_rgb) * v).sum() / vv)
            if tt > 0.85 and np.linalg.norm(new - (bg_rgb + tt * v)) < 0.2 and np.abs(new - c).max() < 0.25:
                halo = True
                break
        if halo:
            break
        palette = np.concatenate([palette, new[None]])
        added += 1
    return palette, added


def boost_ridges(cov, lo=0.2):
    """Yarım pikselden ince, soluk çizgileri (incelerek biten uçlar, kıl çizgiler) korur:
    bir sırt boyunca uzanan pikseller eşiğin hemen üstüne çekilir, böylece silinmez."""
    cand = (cov >= lo) & (cov < 0.75)
    if not cand.any():
        return cov
    h, w = cov.shape
    p = np.pad(cov, 1, mode="edge")

    def sh(dy, dx):
        return p[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
    ridge = np.zeros_like(cand)
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        n1, n2 = sh(dy, dx), sh(-dy, -dx)
        ridge |= (cov >= n1) & (cov >= n2) & (cov > np.minimum(n1, n2) + 0.08)
    ridge &= cand
    n, cc, stats, _ = cv2.connectedComponentsWithStats(ridge.astype(np.uint8), connectivity=8)
    if n <= 1:
        return cov
    big = np.zeros(n, bool)
    big[1:] = stats[1:, cv2.CC_STAT_AREA] >= 4
    ridge = big[cc]
    return np.where(ridge, np.maximum(cov, 0.75), cov).astype(np.float32)


def clean_slivers(label, K, S):
    """Renk sınırlarında kalan, 1 pikselden ince kıymıkları komşu renge katar."""
    H, W = label.shape
    thr = max(1.0, 0.5 * S)
    k3 = np.ones((3, 3), np.uint8)
    fixed = 0
    for k in range(1, K + 1):
        m = (label == k).astype(np.uint8)
        if not m.any():
            continue
        r = max(0, math.ceil(thr) - 1)
        core = cv2.erode(m, np.ones((2 * r + 1, 2 * r + 1), np.uint8)) if r else m
        n, cc, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        thick = np.zeros(n, bool)
        thick[np.unique(cc[core > 0])] = True
        thick[0] = True
        small = stats[:, cv2.CC_STAT_AREA] < 9 * S * S   # ~3×3 orijinal pikselden küçük
        for i in range(1, n):
            if thick[i] and not small[i]:
                continue
            x, y, w, h = stats[i][:4]
            x0, y0, x1, y1 = max(x - 2, 0), max(y - 2, 0), min(x + w + 2, W), min(y + h + 2, H)
            comp = cc[y0:y1, x0:x1] == i
            ring = cv2.dilate(comp.astype(np.uint8), k3).astype(bool) & ~comp
            sub = label[y0:y1, x0:x1]
            nb = sub[ring]
            nb = nb[nb != k]
            others = nb[nb != 0]
            if not thick[i]:
                # ince kıymık: başka bir renge değiyorsa o renge kat
                if len(others):
                    sub[comp] = np.bincount(others).argmax()
                    fixed += 1
            elif len(np.unique(nb)) >= 2 and len(others):
                # iki rengin sınırına sıkışmış küçük leke (JPEG karışım kiri)
                sub[comp] = np.bincount(nb).argmax()
                fixed += 1
    return fixed


# ---------------------------------------------------------------- 4) izleme

def trace_mask(mask, prm, turd):
    pad = np.pad(mask, 1)
    bm = potrace.Bitmap(np.zeros((1, 1), bool))
    bm.data = pad
    curves = bm.trace(turdsize=turd, turnpolicy=potrace.POTRACE_TURNPOLICY_MINORITY,
                      alphamax=prm["alphamax"], opticurve=True, opttolerance=prm["opttolerance"])
    out = []
    for c in curves:
        ops = [("M", [(c.start_point.x - 1, c.start_point.y - 1)])]
        for s in c.segments:
            if s.is_corner:
                ops.append(("L", [(s.c.x - 1, s.c.y - 1)]))
                ops.append(("L", [(s.end_point.x - 1, s.end_point.y - 1)]))
            else:
                ops.append(("C", [(s.c1.x - 1, s.c1.y - 1), (s.c2.x - 1, s.c2.y - 1),
                                  (s.end_point.x - 1, s.end_point.y - 1)]))
        poly = np.array([(p.x - 1, p.y - 1) for p in c._path.pt], np.float32)
        out.append(dict(ops=ops, poly=poly, outer=bool(c._path.sign),
                        area=abs(float(cv2.contourArea(poly))),
                        box=(poly[:, 0].min(), poly[:, 1].min(), poly[:, 0].max(), poly[:, 1].max())))
    return out


def group_objects(contours):
    """Dış konturları ve içlerindeki boşlukları nesnelere ayırır (harf başına bir nesne)."""
    outers = [c for c in contours if c["outer"]]
    holes = [c for c in contours if not c["outer"]]
    groups = {id(o): [o] for o in outers}
    orphans = 0
    for h in holes:
        hx0, hy0, hx1, hy1 = h["box"]
        cands = sorted((o for o in outers if o["box"][0] <= hx0 and o["box"][1] <= hy0
                        and o["box"][2] >= hx1 and o["box"][3] >= hy1), key=lambda o: o["area"])
        step = max(1, len(h["poly"]) // 7)
        probe = h["poly"][::step][:7]
        parent = None
        for o in cands:
            res = [cv2.pointPolygonTest(o["poly"], (float(x), float(y)), False) for x, y in probe]
            if sum(r > 0 for r in res) > sum(r < 0 for r in res):
                parent = o
                break
        if parent is None and cands:
            parent = cands[0]
        if parent is None:
            orphans += 1
            continue
        groups[id(parent)].append(h)

    def key(o):
        x0, y0, x1, y1 = o["box"]
        return (round((y0 + y1) / 2 / 40), x0)
    objs = [groups[id(o)] for o in sorted(outers, key=key)]
    return objs, orphans


# ---------------------------------------------------------------- 5) dosya yazıcılar

def fnum(v):
    s = f"{v:.3f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def write_eps(path, layers, Wpt, Hpt, title):
    safe = re.sub(r"[^A-Za-z0-9 ._-]", "_", title)[:60]
    out = ["%!PS-Adobe-3.0 EPSF-3.0",
           "%%Creator: Adobe Illustrator(R) 8.0",
           f"%%Title: ({safe})",
           MARK,
           f"%%BoundingBox: 0 0 {math.ceil(Wpt)} {math.ceil(Hpt)}",
           f"%%HiResBoundingBox: 0 0 {Wpt:.3f} {Hpt:.3f}",
           "%%DocumentProcessColors: Cyan Magenta Yellow Black",
           "%AI5_FileFormat 4.0",
           "%AI3_ColorUsage: Color",
           f"%AI3_TemplateBox: {Wpt / 2:.1f} {Hpt / 2:.1f} {Wpt / 2:.1f} {Hpt / 2:.1f}",
           f"%AI3_TileBox: 0 0 {math.ceil(Wpt)} {math.ceil(Hpt)}",
           "%AI3_DocumentPreview: None",
           "%%EndComments",
           "%%BeginProlog",
           "/m {moveto} bind def /l {lineto} bind def /L {lineto} bind def /c {curveto} bind def",
           "/k {setcmykcolor} bind def /A {pop} bind def /XR {pop} bind def",
           "/f {closepath} bind def /*u {newpath} bind def /*U {eofill} bind def",
           "/u {} def /U {} def /Lb {10 {pop} repeat} bind def /Ln {pop} bind def /LB {} def",
           "%%EndProlog",
           "%%BeginSetup",
           "%%EndSetup"]
    for i, ly in enumerate(layers, 1):
        col = " ".join(fnum(v) for v in ly.cmyk)
        out += ["%AI5_BeginLayer", "1 1 1 1 0 0 0 79 128 255 Lb", f"(Renk {i} - {ly.name}) Ln", "0 A", "1 XR"]
        for obj in ly.objects:
            out += ["*u", f"{col} k"]
            for ops in obj:
                for op, pts in ops:
                    xy = " ".join(f"{fnum(x)} {fnum(y)}" for x, y in pts)
                    out.append(f"{xy} {'m' if op == 'M' else 'L' if op == 'L' else 'c'}")
                out.append("f")
            out.append("*U")
        out += ["LB", "%AI5_EndLayer--"]
    out += ["%%PageTrailer", "showpage", "%%Trailer", "%%EOF", ""]
    with open(path, "w", encoding="ascii", newline="\r\n") as fh:
        fh.write("\n".join(out))


def write_pdf(path, layers, Wpt, Hpt):
    parts = []
    for ly in layers:
        parts.append(" ".join(fnum(v) for v in ly.cmyk) + " k")
        for obj in ly.objects:
            for ops in obj:
                for op, pts in ops:
                    xy = " ".join(f"{fnum(x)} {fnum(y)}" for x, y in pts)
                    parts.append(f"{xy} {'m' if op == 'M' else 'l' if op == 'L' else 'c'}")
                parts.append("h")
            parts.append("f*")
    content = ("\n".join(parts) + "\n").encode("ascii")
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {Wpt:.3f} {Hpt:.3f}] /Contents 4 0 R >>".encode(),
            b"<< /Length %d >>\nstream\n" % len(content) + content + b"endstream"]
    data = b"%PDF-1.4\n"
    offs = []
    for i, o in enumerate(objs, 1):
        offs.append(len(data))
        data += f"{i} 0 obj\n".encode() + o + b"\nendobj\n"
    xref = len(data)
    data += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    data += b"".join(f"{o:010d} 00000 n \n".encode() for o in offs)
    data += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    with open(path, "wb") as fh:
        fh.write(data)


def write_svg(path, layers, Wpt, Hpt, title):
    def hexc(rgb):
        return "#" + "".join(f"{int(round(v * 255)):02X}" for v in rgb)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {Wpt:.3f} {Hpt:.3f}" '
           f'width="{Wpt * 25.4 / 72:.2f}mm" height="{Hpt * 25.4 / 72:.2f}mm">',
           f"<title>{re.sub(r'[<>&]', '', title)}</title>"]
    for i, ly in enumerate(layers, 1):
        out.append(f'<g id="renk-{i}" fill="{hexc(ly.rgb)}" fill-rule="evenodd">')
        for obj in ly.objects:
            d = []
            for ops in obj:
                for op, pts in ops:
                    d.append(op + " ".join(f"{fnum(x)} {fnum(Hpt - y)}" for x, y in pts))
                d.append("Z")
            out.append(f'<path d="{"".join(d)}"/>')
        out.append("</g>")
    out.append("</svg>\n")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out))


# ---------------------------------------------------------------- 6) kontrol: EPS'yi geri oku ve çiz

def parse_eps(path):
    """Yazılan EPS'yi baştan okur. Dönüş: (nesneler, bbox, yapı hataları)."""
    errors = []
    with open(path, "r", encoding="ascii", errors="replace") as fh:
        text = fh.read()
    lines = text.splitlines()
    if not lines or not lines[0].startswith("%!PS-Adobe-3.0 EPSF-3.0"):
        errors.append("EPS başlığı hatalı")
    m = re.search(r"^%%BoundingBox: 0 0 (\d+) (\d+)$", text, re.M)
    bbox = (int(m.group(1)), int(m.group(2))) if m else None
    if not bbox:
        errors.append("BoundingBox yok")
    if "%%EOF" not in text[-20:]:
        errors.append("Dosya sonu (EOF) eksik")
    objects, cur, contour, color = [], None, None, None
    in_layer = False
    for no, line in enumerate(lines, 1):
        if line.startswith("%AI5_BeginLayer"):
            in_layer = True
            continue
        if line.startswith("%AI5_EndLayer"):
            in_layer = False
            continue
        if not in_layer or not line or line.startswith("%"):
            continue
        tok = line.split()
        op, args = tok[-1], tok[:-1]
        if op in ("Lb", "Ln", "A", "XR", "LB"):
            continue
        try:
            nums = [float(v) for v in args]
        except ValueError:
            errors.append(f"Satır {no}: sayı okunamadı")
            continue
        if any(not math.isfinite(v) for v in nums):
            errors.append(f"Satır {no}: geçersiz sayı")
        if op == "*u":
            if cur is not None:
                errors.append(f"Satır {no}: kapanmamış nesne")
            cur = dict(color=color, contours=[])
        elif op == "*U":
            if cur is None:
                errors.append(f"Satır {no}: fazladan nesne sonu")
            elif contour is not None:
                errors.append(f"Satır {no}: kapanmamış kontur")
            else:
                objects.append(cur)
            cur, contour = None, None
        elif op == "k":
            if len(nums) != 4 or any(v < 0 or v > 1 for v in nums):
                errors.append(f"Satır {no}: renk değeri hatalı")
            color = tuple(nums)
            if cur is not None:
                cur["color"] = color
        elif op == "m":
            if cur is None or contour is not None or len(nums) != 2:
                errors.append(f"Satır {no}: kontur başlangıcı hatalı")
            contour = [("M", [tuple(nums)])]
        elif op in ("L", "l", "c"):
            need = 6 if op == "c" else 2
            if contour is None or len(nums) != need:
                errors.append(f"Satır {no}: çizgi/eğri hatalı")
                continue
            pts = [(nums[i], nums[i + 1]) for i in range(0, need, 2)]
            contour.append(("C" if op == "c" else "L", pts))
        elif op == "f":
            if contour is None or cur is None:
                errors.append(f"Satır {no}: boş kontur kapatma")
            else:
                cur["contours"].append(contour)
            contour = None
        else:
            errors.append(f"Satır {no}: bilinmeyen komut '{op}'")
    if cur is not None:
        errors.append("Dosya sonunda kapanmamış nesne")
    if bbox:
        for o in objects:
            for c in o["contours"]:
                for _, pts in c:
                    for x, y in pts:
                        if x < -2 or y < -2 or x > bbox[0] + 2 or y > bbox[1] + 2:
                            errors.append("Çizim, sayfa sınırının dışına taşıyor")
                            break
                    else:
                        continue
                    break
    return objects, bbox, sorted(set(errors), key=errors.index)


def flatten(contour, tf):
    pts = []
    last = None
    for op, p in contour:
        p = [tf(x, y) for x, y in p]
        if op in ("M", "L"):
            pts.append(p[0])
            last = p[0]
        else:
            p0 = np.array(last)
            c1, c2, p3 = (np.array(v) for v in p)
            est = np.linalg.norm(c1 - p0) + np.linalg.norm(c2 - c1) + np.linalg.norm(p3 - c2)
            n = int(np.clip(math.ceil(est / 1.5), 2, 120))
            t = np.linspace(1 / n, 1, n)[:, None]
            seg = ((1 - t) ** 3) * p0 + 3 * ((1 - t) ** 2) * t * c1 + 3 * (1 - t) * t * t * c2 + (t ** 3) * p3
            pts.extend(map(tuple, seg))
            last = p[2]
    return np.array(pts, np.float64)


def raster_evenodd(polys, h, w):
    """Çokgenleri piksel merkezinden örnekleyerek, çift-tek kuralıyla tarar (kenar şişmesi yok)."""
    P = [np.asarray(p, np.float64) for p in polys if len(p) >= 3]
    if not P:
        return np.zeros((h, w), bool)
    x0 = np.concatenate([p[:, 0] for p in P])
    y0 = np.concatenate([p[:, 1] for p in P])
    x1 = np.concatenate([np.roll(p[:, 0], -1) for p in P])
    y1 = np.concatenate([np.roll(p[:, 1], -1) for p in P])
    lo, hi = np.minimum(y0, y1), np.maximum(y0, y1)
    r0 = np.clip(np.ceil(lo - 0.5), 0, h).astype(np.intp)
    r1 = np.clip(np.ceil(hi - 0.5), 0, h).astype(np.intp)
    n = r1 - r0
    keep = n > 0
    if not keep.any():
        return np.zeros((h, w), bool)
    x0, y0, x1, y1, r0, n = x0[keep], y0[keep], x1[keep], y1[keep], r0[keep], n[keep]
    rep = np.repeat(np.arange(len(n)), n)
    start = np.repeat(np.cumsum(n) - n, n)
    rows = r0[rep] + (np.arange(rep.size) - start)
    yc = rows + 0.5
    xe = x0[rep] + (yc - y0[rep]) * (x1[rep] - x0[rep]) / (y1[rep] - y0[rep])
    o = np.lexsort((xe, rows))
    rows, xe = rows[o], xe[o]
    rr = rows[0::2]
    c0 = np.clip(np.ceil(xe[0::2] - 0.5), 0, w).astype(np.intp)
    c1 = np.clip(np.ceil(xe[1::2] - 0.5), 0, w).astype(np.intp)
    size = h * (w + 1)
    diff = np.bincount(rr * (w + 1) + c0, minlength=size) - np.bincount(rr * (w + 1) + c1, minlength=size)
    return np.cumsum(diff.reshape(h, w + 1)[:, :w], axis=1) > 0


def render_objects(objects, color_index, shape, tf):
    """EPS nesnelerini etiket haritasına çizer (0 = boş, i = renk sırası)."""
    h, w = shape
    lab = np.zeros((h, w), np.uint8)
    for o in objects:
        polys = [flatten(c, tf) for c in o["contours"]]
        polys = [p for p in polys if len(p) >= 3]
        if not polys:
            continue
        allp = np.concatenate(polys)
        x0, y0 = np.maximum(np.floor(allp.min(0)).astype(int) - 1, 0)
        x1, y1 = np.minimum(np.ceil(allp.max(0)).astype(int) + 2, (w, h))
        if x1 <= x0 or y1 <= y0:
            continue
        m = raster_evenodd([p - (x0, y0) for p in polys], y1 - y0, x1 - x0)
        lab[y0:y1, x0:x1][m] = color_index[o["color"]]
    return lab


def render_supersampled(objects, color_index, h, w, tf, k, pal):
    """Kenar yumuşatmalı çizim: k× çözünürlükte şeritler halinde çizip piksel başına ortalar."""
    C = pal.shape[1]
    out = np.zeros((h, w, C), np.float32)
    flat = []
    for o in objects:
        polys = [flatten(c, lambda X, Y: tuple(v * k for v in tf(X, Y))) for c in o["contours"]]
        polys = [p for p in polys if len(p) >= 3]
        if polys:
            allp = np.concatenate(polys)
            flat.append((color_index[o["color"]], polys, allp[:, 0].min(), allp[:, 1].min(),
                         allp[:, 0].max(), allp[:, 1].max()))
    step = max(1, 3_000_000 // max(1, w * k * k))
    for r0 in range(0, h, step):
        r1 = min(h, r0 + step)
        Y0, Y1 = r0 * k, r1 * k
        lab = np.zeros((Y1 - Y0, w * k), np.uint8)
        for ci, polys, xmin, ymin, xmax, ymax in flat:
            if ymax < Y0 or ymin > Y1:
                continue
            x0 = max(int(np.floor(xmin)) - 1, 0)
            x1 = min(int(np.ceil(xmax)) + 2, w * k)
            if x1 <= x0:
                continue
            m = raster_evenodd([p - (x0, Y0) for p in polys], Y1 - Y0, x1 - x0)
            lab[:, x0:x1][m] = ci
        img = cv2.resize(pal[lab], (w, r1 - r0), interpolation=cv2.INTER_AREA)
        out[r0:r1] = img.reshape(r1 - r0, w, C)
    return out


def boundary(label, r):
    k = np.ones((3, 3), np.uint8)
    edge = (cv2.dilate(label, k) != cv2.erode(label, k)).astype(np.uint8)
    if r > 1:
        edge = cv2.dilate(edge, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r - 1, 2 * r - 1)))
    return edge > 0


def blobs(mask, min_area):
    n, _, stats, cent = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    return [(stats[i], cent[i]) for i in keep]


# ---------------------------------------------------------------- ana işlem

def output_path(src, out_dir, ext):
    folder = out_dir or os.path.dirname(src)
    stem = os.path.splitext(os.path.basename(src))[0]
    p = os.path.join(folder, stem + ext)
    if os.path.exists(p) and not is_ours(p):
        p = os.path.join(folder, stem + "_vektor" + ext)
    return p


def is_ours(p):
    try:
        with open(p, "rb") as fh:
            head = fh.read(4096)
    except OSError:
        return False
    if p.lower().endswith(".eps"):
        return MARK.encode() in head
    return True if os.path.exists(os.path.splitext(p)[0] + ".eps") and is_ours(os.path.splitext(p)[0] + ".eps") else False


def checker(h, w, s=12):
    yy, xx = np.mgrid[0:h, 0:w]
    c = (((yy // s) + (xx // s)) % 2).astype(np.float32)
    return (0.93 + 0.05 * c)[..., None].repeat(3, 2)


def convert(src, st: Settings, progress=lambda msg: None):
    t0 = time.time()
    res = Result(src=src)
    prm = PRESETS.get(st.preset, PRESETS["Normal"])
    try:
        progress("Görsel okunuyor")
        rgb, display, has_alpha, is_jpeg = load_image(src)
        h0, w0 = rgb.shape[:2]
        if max(h0, w0) > PROC_MAX:
            f = PROC_MAX / max(h0, w0)
            size = (max(1, round(w0 * f)), max(1, round(h0 * f)))
            rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
            display = cv2.resize(display, size, interpolation=cv2.INTER_AREA)
        h, w = rgb.shape[:2]
        orig_rgb = rgb.copy()
        if is_jpeg:
            rgb = cv2.bilateralFilter(rgb, 5, 0.08, 3)
        lab = to_lab(rgb)
        orig_lab = to_lab(orig_rgb)

        progress("Arka plan ve renkler bulunuyor")
        bg_rgb, bg_lab, bg_frac = detect_background(lab, rgb)
        if bg_frac < 0.45:
            res.warnings.append("Arka plan net değil (logo kenara dayanıyor olabilir); kenar rengi arka plan sayıldı.")
        want = st.colors if isinstance(st.colors, int) else None
        palette, grad, found = find_palette(rgb, lab, bg_lab, want)
        if len(palette) == 0:
            res.headline = "Görselde logo bulunamadı"
            res.checks.append((False, "Arka plandan ayrışan çizim yok"))
            return res
        if found > MAX_COLORS and st.colors == "multi":
            res.warnings.append(f"Görselde çok fazla renk var ({found}+); en baskın {MAX_COLORS} renge indirildi.")
        if bg_rgb.min() < 0.9:
            res.checks.append((True, "Zemin rengi çizilmedi (yalnız logo alındı)"))
        mono = st.colors == "mono"
        effect = st.colors == "own" or (st.colors == "auto" and (grad > 0.12 or found > MAX_COLORS))

        if effect:
            # 3B/metalik/gölgeli logo: tek düz renk (logonun kendi rengi), zemin yerel tahminli
            progress("Efektli logo ayrıştırılıyor")
            cov_e, fg_col, n_holes = effect_segment(rgb, lab, bg_lab)
            palette = fg_col[None]
            centers = np.concatenate([bg_rgb[None].astype(np.float32), palette])
            a = np.zeros((h, w), np.uint8)
            b = np.ones((h, w), np.uint8)
            t = cov_e
            mono = True
            K = 1
            layer_rgb = [fg_col]
            prm = dict(prm, alphamax=max(prm["alphamax"], 1.2), opttolerance=max(prm["opttolerance"], 0.35))
            if st.colors == "auto":
                res.warnings.append("Efektli logo (gölge, parlaklık ya da 3B): tek düz renk olarak, logonun kendi "
                                    "rengiyle çıkarıldı. Koyu gölgeli kenarlara bir göz atın.")
            if n_holes:
                res.checks.append((True, f"Gölgeden oluşan {n_holes} küçük delik dolduruldu"))
        else:
            if not isinstance(st.colors, int) and not mono:
                palette, added = augment_palette(rgb, bg_rgb.astype(np.float32), palette)
                if added:
                    res.checks.append((True, f"İnce ayrıntılarda {added} ek renk bulundu"))
            centers = np.concatenate([bg_rgb[None].astype(np.float32), palette])
            a, b, t, resid = blend_assign(rgb, centers)
            if not mono and (grad > 0.12 or (len(palette) > 1 and banding(np.where(t < 0.5, a, b), palette))):
                grad = max(grad, 0.13)
                res.warnings.append("Görselde degrade/gölge var; düz renk bantlarına indirgendi. "
                                    "Degrade gerekiyorsa Corel'de elle verin.")
            K = len(palette)
            if mono:
                remap = np.array([0] + [1] * K, np.uint8)
                a, b = remap[a], remap[b]
                K = 1
                layer_rgb = [np.array([0, 0, 0], np.float32)]
            else:
                layer_rgb = list(palette)

        # izleme çözünürlüğü
        long = max(h, w)
        S = max(2.0, min(12.0, WORK_TARGET / long))
        W, H = max(1, round(w * S)), max(1, round(h * S))
        sig = 0.0
        if long < 400:
            res.warnings.append(f"Görsel küçük ({w0}×{h0} px); ince ayrıntılar tahmin edilerek yumuşatıldı.")
        if is_jpeg:
            res.checks.append((True, "JPEG sıkıştırma kirleri temizlendi"))

        progress("Kenarlar hesaplanıyor")
        best = None
        label = np.zeros((H, W), np.uint8)
        for k in range(K + 1):
            cov = ((a == k) * (1 - t) + (b == k) * t).astype(np.float32)
            if k:
                cov = boost_ridges(cov)
            up = cv2.resize(cov, (W, H), interpolation=cv2.INTER_CUBIC) if S != 1 else cov
            if sig > 0.3:
                up = cv2.GaussianBlur(up, (0, 0), sig)
            if best is None:
                best = up
            else:
                m = up > best
                best[m] = up[m]
                label[m] = k
        del best

        if K > 1:
            n_sliver = clean_slivers(label, K, S)
            if n_sliver:
                res.checks.append((True, f"Renk sınırındaki {n_sliver} ince kıymık temizlendi"))
        areas = np.bincount(label.ravel(), minlength=K + 1)
        order = [k for k in sorted(range(1, K + 1), key=lambda k: -areas[k]) if areas[k] > 0]
        if not order:
            res.headline = "Görselde logo bulunamadı"
            return res

        ys, xs = np.nonzero(label)
        cx0, cy0, cx1, cy1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        long_pt = st.size_mm * 72 / 25.4
        kpt = long_pt / max(cx1 - cx0, cy1 - cy0)
        margin = 4.0
        Wpt = (cx1 - cx0) * kpt + 2 * margin
        Hpt = (cy1 - cy0) * kpt + 2 * margin

        def to_pt(x, y):
            return ((x - cx0) * kpt + margin, Hpt - ((y - cy0) * kpt + margin))

        turd = max(2, int(prm["speck"] * S * S * 1.5))
        r_trap = max(1, round(0.6 * S))
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_trap + 1, 2 * r_trap + 1))
        layers, orphans = [], 0
        for pos, k in enumerate(order):
            progress(f"Çiziliyor: renk {pos + 1}/{len(order)}")
            own = label == k
            above = np.isin(label, order[pos + 1:]) if pos + 1 < len(order) else None
            mask = own
            if above is not None and above.any():
                mask = own | (cv2.dilate(own.astype(np.uint8), kern).astype(bool) & above)
            contours = trace_mask(mask, prm, turd)
            objs, orph = group_objects(contours)
            orphans += orph
            rgbk = layer_rgb[k - 1]
            ly = Layer(rgb=tuple(float(v) for v in rgbk), cmyk=rgb_to_cmyk(rgbk))
            for obj in objs:
                ly.objects.append([[(op, [to_pt(x, y) for x, y in pts]) for op, pts in c["ops"]] for c in obj])
            if ly.objects:
                layers.append(ly)
        if not layers:
            res.headline = "Görselde logo bulunamadı"
            return res
        if orphans:
            res.warnings.append(f"{orphans} iç boşluk sahibine bağlanamadı ve atlandı.")

        progress("Dosya yazılıyor")
        title = os.path.splitext(os.path.basename(src))[0]
        eps = output_path(src, st.out_dir, ".eps")
        write_eps(eps, layers, Wpt, Hpt, title)
        res.eps = eps
        if st.pdf:
            p = output_path(src, st.out_dir, ".pdf")
            write_pdf(p, layers, Wpt, Hpt)
            res.extra.append(p)
        if st.svg:
            p = output_path(src, st.out_dir, ".svg")
            write_svg(p, layers, Wpt, Hpt, title)
            res.extra.append(p)
        res.layers = layers
        res.n_objects = sum(len(l.objects) for l in layers)
        res.n_contours = sum(len(o) for l in layers for o in l.objects)
        res.size_mm = (Wpt * 25.4 / 72, Hpt * 25.4 / 72)

        # ------------------------------------------------ KONTROL
        progress("Kontrol ediliyor")
        if st.tamper:
            st.tamper(eps)
        objects, bbox, errors = parse_eps(eps)
        res.checks.append((not errors, "EPS yapısı geçerli (Illustrator 8, Corel uyumlu)" if not errors
                           else "EPS yapısı: " + "; ".join(errors[:4])))
        cmyk_index = {}
        for i, ly in enumerate(layers, 1):
            cmyk_index.setdefault(tuple(float(fnum(v)) for v in ly.cmyk), i)
        color_index = {}
        for o in objects:
            col = o["color"]
            if col not in color_index:
                color_index[col] = cmyk_index.get(col, 0)
        n_obj_ok = len(objects) == res.n_objects
        res.checks.append((n_obj_ok, f"Nesne sayısı tutarlı ({len(objects)})" if n_obj_ok
                           else f"Nesne sayısı uyuşmuyor: dosyada {len(objects)}, olması gereken {res.n_objects}"))
        unknown = sum(1 for o in objects if color_index.get(o["color"], 0) == 0)
        if unknown:
            res.checks.append((False, f"{unknown} nesnenin rengi beklenenden farklı"))

        def tf_work(X, Y):
            return ((X - margin) / kpt + cx0, (Hpt - Y - margin) / kpt + cy0)

        # a) şekil: çalışma çözünürlüğünde hedef etiketle karşılaştır
        layer_of = {k: i for i, k in enumerate(order, 1)}
        target = np.zeros_like(label)
        for k, i in layer_of.items():
            target[label == k] = i
        if mono:
            target = (target > 0).astype(np.uint8)
        rend = render_objects(objects, color_index, (H, W), tf_work)
        band = boundary(target, max(2, round(0.75 * S))) | boundary(rend, max(2, round(0.75 * S)))
        mism = rend != target
        fg_area = max(1, int((target > 0).sum()))
        res.shape_score = 100 * (1 - (mism & ~band).sum() / fg_area)
        shape_err = blobs(mism & ~band, turd + 1)
        specks = blobs(mism & ~band, 1)
        res.specks = max(0, len(specks) - len(shape_err))
        res.shape_regions = len(shape_err)
        shape_err_area = sum(s[0][cv2.CC_STAT_AREA] for s in shape_err) / fg_area

        # b) renk: orijinal çözünürlükte, orijinal piksellerle karşılaştır
        def tf_proc(X, Y):
            x, y = tf_work(X, Y)
            return (x / S, y / S)
        rend_p = render_objects(objects, color_index, (h, w), tf_proc)
        pal = np.zeros((len(layers) + 1, 3), np.float32)
        pal[0] = bg_rgb
        for i, ly in enumerate(layers, 1):
            pal[i] = ly.rgb
        # yeniden çizim karşılaştırması: vektör orijinal çözünürlükte kenar yumuşatmalı
        # çizilir ve her piksel orijinalle kıyaslanır. Hata = pikselin yarısından fazlası yanlış
        # (kalınlaşma/incelme, eksik ya da fazla parça, yanlış renk hepsi burada yakalanır).
        def shrink(plane):
            return cv2.resize(plane, (w, h), interpolation=cv2.INTER_AREA) if (W, H) != (w, h) else plane
        # karşılaştırma 1 piksellik yumuşatmayla yapılır: basamaklı kenar ile düzgün eğri
        # arasındaki yarım piksellik fark sıfırlanır, ~0,6 pikselden büyük kayma yakalanır
        def soft(x):
            return cv2.GaussianBlur(x, (0, 0), 1.0)
        if S >= 3:
            ss_plane = None
        else:
            # izleme çözünürlüğü düşükse kontrol çizimi ayrıca 3× örneklenir
            ss_pal = np.array([[0.0], [1.0]] + [[1.0]] * (len(pal) - 2), np.float32) if mono else pal
            ss_plane = render_supersampled(objects, color_index, h, w, tf_proc, 3, ss_pal)
        if mono:
            syn = shrink((rend > 0).astype(np.float32)) if ss_plane is None else ss_plane[..., 0]
            cov_fg = ((a != 0) * (1 - t) + (b != 0) * t).astype(np.float32)
            err = np.abs(soft(syn) - soft(cov_fg))
        else:
            syn = np.stack([shrink(pal[:, ch][rend]) for ch in range(3)], 2) if ss_plane is None else ss_plane
            contrast = np.linalg.norm(centers[a] - centers[b], axis=2)
            same = a == b
            contrast[same] = np.linalg.norm(centers[a[same]] - bg_rgb[None], axis=1)
            cmap = cv2.dilate(np.maximum(contrast, 0.25).astype(np.float32), np.ones((5, 5), np.uint8))
            err = np.linalg.norm(soft(rgb) - soft(syn), axis=2) / cmap
        cerr = err > 0.25
        if st.debug:
            res.debug = dict(err=err, syn=syn, src=rgb, target=target, rend=rend, pal=pal, S=S, centers=centers, a=a, b=b, t=t)
        fg_p = max(1, int(((rend_p > 0) | (a != 0) | (b != 0)).sum()))
        res.color_score = 100 * (1 - cerr.sum() / fg_p)
        min_c = 3
        color_err = blobs(cerr, min_c)
        res.color_regions = len(color_err)
        color_err_area = sum(s[0][cv2.CC_STAT_AREA] for s in color_err) / fg_p

        res.checks.append((res.shape_regions == 0,
                           "Çizim, ayrıştırılan şekille birebir" if res.shape_regions == 0
                           else f"Çizim: {res.shape_regions} bölgede şekilden sapma (kırmızı işaretli)"))
        res.checks.append((res.color_regions == 0,
                           "Orijinalle piksel piksel karşılaştırma: fark yok" if res.color_regions == 0
                           else f"Orijinalle karşılaştırma: {res.color_regions} bölgede fark (kırmızı işaretli)"))

        # durum
        if errors or not n_obj_ok or unknown or res.shape_score < 97 or shape_err_area > 0.01 or color_err_area > 0.03:
            res.status = "fail"
            res.headline = "SORUNLU — elle kontrol edin"
        elif res.shape_regions or res.color_regions or res.warnings or res.shape_score < 99.5:
            res.status = "warn"
            res.headline = "HAZIR — işaretli yerlere göz atın"
        else:
            res.status = "ok"
            res.headline = "KUSURSUZ — orijinalle birebir"

        # önizlemeler
        progress("Önizleme hazırlanıyor")
        res.preview_orig = Image.fromarray((np.clip(display, 0, 1) * 255).astype(np.uint8))
        vec = checker(h, w)
        has = rend_p > 0
        vec[has] = pal[rend_p][has]
        res.preview_vec = Image.fromarray((vec * 255).astype(np.uint8))
        gray = cv2.cvtColor(np.clip(display, 0, 1).astype(np.float32), cv2.COLOR_RGB2GRAY)
        faded = (0.55 + 0.45 * gray)[..., None].repeat(3, 2)
        faded[has] = faded[has] * np.array([0.75, 0.9, 0.75])  # vektörün kapladığı yer hafif yeşil
        diffm = cerr.copy()
        if shape_err:
            sm = np.zeros((H, W), np.uint8)
            for st_, _ in shape_err:
                x, y, ww, hh = st_[:4]
                sm[y:y + hh, x:x + ww] = (mism & ~band)[y:y + hh, x:x + ww]
            diffm |= cv2.resize(sm, (w, h), interpolation=cv2.INTER_NEAREST) > 0
        faded[diffm] = (0.9, 0.1, 0.1)
        img = Image.fromarray((faded * 255).astype(np.uint8))
        dr = ImageDraw.Draw(img)
        rad = max(8, int(0.02 * max(w, h)))
        for st_, c in shape_err:
            dr.ellipse([c[0] / S - rad, c[1] / S - rad, c[0] / S + rad, c[1] / S + rad], outline=(220, 0, 0), width=max(2, rad // 5))
        for st_, c in color_err:
            dr.ellipse([c[0] - rad, c[1] - rad, c[0] + rad, c[1] + rad], outline=(220, 0, 0), width=max(2, rad // 5))
        res.preview_diff = img
    except Exception as e:  # noqa: BLE001 — her hatayı kullanıcıya rapor olarak göster
        res.status = "fail"
        res.headline = f"Çevrilemedi: {e}"
        import traceback
        res.warnings.append(traceback.format_exc(limit=3))
    finally:
        res.seconds = time.time() - t0
    return res


def log_result(res: Result, log_path):
    new = not os.path.exists(log_path)
    try:
        with open(log_path, "a", newline="", encoding="utf-8-sig") as fh:
            w = csv.writer(fh, delimiter=";")
            if new:
                w.writerow(["Tarih", "Kaynak", "EPS", "Durum", "Şekil %", "Renk %", "Renk", "Nesne", "Uyarılar"])
            w.writerow([time.strftime("%Y-%m-%d %H:%M"), res.src, res.eps,
                        {"ok": "Kusursuz", "warn": "Kontrol et", "fail": "Sorunlu"}[res.status],
                        f"{res.shape_score:.2f}", f"{res.color_score:.2f}", len(res.layers), res.n_objects,
                        " | ".join(w.splitlines()[0] for w in res.warnings)])
    except OSError:
        pass
