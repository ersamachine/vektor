// ERSA Vektör — çeviri motoru (tarayıcı içinde Python). Dosyalar bilgisayardan dışarı çıkmaz.
const PYODIDE = "https://cdn.jsdelivr.net/pyodide/v0.28.3/full/";
importScripts(PYODIDE + "pyodide.js");

let py = null;
let run = null;

const PY_HELPER = `
import io, os, sys
import js
from pyodide.ffi import to_js
sys.path.insert(0, "/home/pyodide")
import vektor_core as vc
vc.PROC_MAX = 2000   # tarayıcı belleği için işleme sınırı (uzun kenar, px)

def _png(im, limit=1400):
    if im is None:
        return None
    im = im.copy()
    im.thumbnail((limit, limit))
    b = io.BytesIO()
    im.save(b, "PNG", optimize=False)
    return b.getvalue()

def run(job_id, name, data, colors, preset, size_mm, pdf, svg):
    for d in ("/tmp/in", "/tmp/out"):
        os.makedirs(d, exist_ok=True)
        for f in os.listdir(d):
            os.remove(os.path.join(d, f))
    safe = name.replace("/", "_").replace("\\\\", "_") or "logo.png"
    src = "/tmp/in/" + safe
    with open(src, "wb") as fh:
        fh.write(data.to_py())
    if colors in ("auto", "mono"):
        col = colors
    else:
        col = int(colors)
    st = vc.Settings(colors=col, preset=preset, size_mm=float(size_mm), pdf=bool(pdf), svg=bool(svg), out_dir="/tmp/out")

    def progress(msg):
        js.postMessage(to_js({"type": "progress", "id": job_id, "msg": msg}, dict_converter=js.Object.fromEntries))

    r = vc.convert(src, st, progress=progress)
    files = []
    for p in [r.eps] + list(r.extra):
        if p and os.path.exists(p):
            with open(p, "rb") as fh:
                files.append({"name": os.path.basename(p), "data": fh.read()})
    return to_js({
        "status": r.status, "headline": r.headline, "report": r.report(),
        "shape": r.shape_score, "color": r.color_score,
        "objects": r.n_objects, "colors": len(r.layers),
        "size": [r.size_mm[0], r.size_mm[1]],
        "files": files,
        "orig": _png(r.preview_orig), "vec": _png(r.preview_vec), "diff": _png(r.preview_diff),
    }, dict_converter=js.Object.fromEntries)
`;

async function boot() {
  try {
    post({ type: "boot", msg: "Python hazırlanıyor" });
    const t0 = performance.now();
    py = await loadPyodide({ indexURL: PYODIDE });
    console.log("[motor] python", Math.round(performance.now() - t0), "ms");
    post({ type: "boot", msg: "Görüntü kütüphaneleri yükleniyor" });
    const t = performance.now();
    await py.loadPackage(["numpy", "pillow"], {
      messageCallback: (m) => console.log("[motor]", Math.round(performance.now() - t), "ms", m),
      errorCallback: (m) => console.warn("[motor]", m),
    });
    await py.loadPackage(new URL("py/potracer-0.0.4-py2.py3-none-any.whl", self.location).href);
    post({ type: "boot", msg: "Çeviri motoru yükleniyor" });
    py.FS.mkdirTree("/home/pyodide");
    for (const f of ["ncv.py", "vektor_core.py"]) {
      const code = await (await fetch(new URL(f, self.location), { cache: "no-cache" })).text();
      py.FS.writeFile("/home/pyodide/" + f, code);
    }
    await py.runPythonAsync(PY_HELPER);
    run = py.globals.get("run");
    const ver = py.runPython("vc.VERSION");
    post({ type: "ready", version: ver });
  } catch (e) {
    post({ type: "fatal", msg: String(e && e.message || e) });
  }
}

function post(m, transfer) {
  self.postMessage(m, transfer || []);
}

const queue = [];
let busy = false;

self.onmessage = (ev) => {
  queue.push(ev.data);
  pump();
};

async function pump() {
  if (busy || !run) return;
  const job = queue.shift();
  if (!job) return;
  busy = true;
  post({ type: "start", id: job.id });
  try {
    const o = job.opts;
    const res = run(job.id, job.name, job.data, String(o.colors), o.preset, o.size, !!o.pdf, !!o.svg);
    const transfer = [];
    for (const f of res.files) transfer.push(f.data.buffer);
    for (const k of ["orig", "vec", "diff"]) if (res[k]) transfer.push(res[k].buffer);
    post({ type: "done", id: job.id, res }, transfer);
  } catch (e) {
    post({ type: "done", id: job.id, res: { status: "fail", headline: "Çevrilemedi", report: String(e && e.message || e), files: [] } });
  }
  busy = false;
  pump();
}

boot().then(pump);
