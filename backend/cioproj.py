# -*- coding: utf-8 -*-
"""CIO 投影模块：原始场 → projected CIO 序列（完整链路的第一步）

口径 = 2026-09-15 在服务器上逐点验证锁死的那条（交接文档 §9）：
  1. 读原始逐日距平场（每月只含 2–29 日，28 天/月）
  2. 裁剪到 -20..20N / 40..120E，双线性插值到 1°（41×81）
  3. 减"按日序 1–28 跨年"的平均场（日序 >28 无平均，按 0）—— 跨月合并
  4. 减"沿纬度"的高通项 U850_high                       ← 见下面的警告
  5. (x - Umean) / Ustd
  6. 纬度快变（F 序）展平成 3321，点乘 e 的后 3321 列

⚠⚠ 第 4 步是有意照抄的历史 bug，不要"修好" ⚠⚠
  rain_uwnd_filter.py 把 (lat,lon,time) 当成 (time,lat,lon)，sosfiltfilt(axis=0)
  实际沿**纬度**滤波。Proj_CIOmode_new.m 第 25 行确实减掉了这一项，现有 327 万
  个 checkpoint 就是在这份输入上训出来的。改成"沿时间"或删掉这一步，输入分布
  都会与权重不匹配（沿时间那版与现口径相关只有 0.70，必须全部重训）。
  依据：pre1 用本口径复现 corr=1.00000 / 斜率=1.0000 / 残差=0.0000。

时间窗：lead=k 的窗口 = (5月30−k 日) … (9月29−k 日)，落在每月 2–29 日上的天数
  恰为 112（k=20 时窗含 5 月，缺 5 月数据会少几天——如实反映，不补齐）。
"""

from __future__ import annotations

import glob
import os
import re

import numpy as np

NLAT, NLON = 41, 81                 # -20..20 / 40..120 @1°
LAT_TGT = np.arange(-20, 21, 1.0)
LON_TGT = np.arange(40, 121, 1.0)
MONTHS = (5, 6, 7, 8, 9)
N_SST = 2707                        # e = [SST 2707 | U850 3321]
N_U850 = 3321

DEFAULT_MAT = "CIOmode_1982_2017.mat"


# ----------------------------------------------------------------- 工具

def read_nc(path: str) -> dict:
    """读 nc，返回 {变量名: 数组}；依次尝试 netCDF4 / h5py / xarray / scipy。"""
    errs = []
    for lib in ("netCDF4", "h5py", "xarray", "scipy"):
        try:
            if lib == "netCDF4":
                import netCDF4
                with netCDF4.Dataset(path) as ds:
                    return {v: np.array(ds.variables[v][:]) for v in ds.variables}
            if lib == "h5py":
                import h5py
                with h5py.File(path, "r") as f:
                    return {k: np.array(f[k]) for k in f.keys()}
            if lib == "xarray":
                import xarray as xr
                with xr.open_dataset(path) as ds:
                    return {v: np.asarray(ds[v].values) for v in ds.variables}
            import scipy.io as sio
            ds = sio.netcdf_file(path, "r", mmap=False)
            out = {k: np.array(v[:]) for k, v in ds.variables.items()}
            ds.close()
            return out
        except Exception as exc:                      # noqa: BLE001
            errs.append("%s: %r" % (lib, exc))
    raise RuntimeError("没有库能读 %s → %s" % (path, " | ".join(errs)))


def _pick_field(d: dict, hints=("u850", "uwnd")) -> np.ndarray:
    best = None
    for k, v in d.items():
        if getattr(v, "ndim", 0) < 3 or k in ("lat", "lon", "time"):
            continue
        if any(h in k.lower() for h in hints):
            return np.asarray(v, dtype=np.float64)
        best = v if best is None else best
    if best is None:
        raise ValueError("nc 里找不到三维场：%s" % list(d))
    return np.asarray(best, dtype=np.float64)


def _orient(arr, lat, lon):
    """保证纬度递增、经度递增。"""
    if lat[0] > lat[-1]:
        lat, arr = lat[::-1], arr[:, ::-1, :]
    if lon[0] > lon[-1]:
        lon, arr = lon[::-1], arr[:, :, ::-1]
    return arr, lat, lon


# ----------------------------------------------------------------- 时间窗

def window(lead: int):
    """lead=k 的窗口：(5月, 30−k) … (9月, 29−k) 闭区间。"""
    if not 1 <= lead <= 20:
        raise ValueError("lead 必须在 1..20")
    return (5, 30 - lead), (9, 29 - lead)


def in_window(month: int, day: int, lead: int) -> bool:
    return window(lead)[0] <= (month, day) <= window(lead)[1]


# ----------------------------------------------------------------- 读场

def find_nc_files(nc_dir: str, year: int, month: int):
    """按月找文件：优先按官方命名，找不到再放宽正则。"""
    pat = "*%04d%02d02-28days*" % (year, month)
    hit = sorted(glob.glob(os.path.join(nc_dir, pat)))
    if hit:
        return hit[0]
    rx = re.compile(r"\.%04d%02d\d{2}-28days" % (year, month))
    for f in sorted(os.listdir(nc_dir)):
        if rx.search(f) and os.path.isfile(os.path.join(nc_dir, f)):
            return os.path.join(nc_dir, f)
    return None


def load_field(nc_dir: str, years, lead: int):
    """读窗口内的原始场并插值到 1°。

    返回 (U3 (41,81,T), dom (T,), rows, missing)：
      dom  —— 每一天的"日序"（2..29），第 3 步按它取平均
      rows —— [(年, 月, 日), ...]，对齐用
    """
    from scipy.interpolate import RegularGridInterpolator

    lon_g, lat_g = np.meshgrid(LON_TGT, LAT_TGT)
    pts = np.array([lat_g.ravel(), lon_g.ravel()]).T
    seq, dom, rows, missing = [], [], [], []

    for y in years:
        for mo in MONTHS:
            fp = find_nc_files(nc_dir, y, mo)
            if fp is None:
                if in_window(mo, 28, lead) or in_window(mo, 2, lead):
                    missing.append((y, mo))
                continue
            d = read_nc(fp)
            a = np.squeeze(_pick_field(d))
            lat = np.ravel(d["lat"]) if "lat" in d else np.ravel(d["latitude"])
            lon = np.ravel(d["lon"]) if "lon" in d else np.ravel(d["longitude"])
            a, lat, lon = _orient(a, lat, lon)

            lat_m = (lat >= -20 - 1e-6) & (lat <= 20 + 1e-6)
            lon_m = (lon >= 40 - 1e-6) & (lon <= 120 + 1e-6)
            sub, slat, slon = a[:, lat_m, :][:, :, lon_m], lat[lat_m], lon[lon_m]

            for k in range(sub.shape[0]):
                day = 2 + k                    # 文件里只存 2–29 日
                if not in_window(mo, day, lead):
                    continue
                f = RegularGridInterpolator((slat, slon), sub[k], bounds_error=False,
                                            fill_value=np.nan)
                seq.append(f(pts).reshape(NLAT, NLON))
                dom.append(day)
                rows.append((y, mo, day))

    if not seq:
        raise RuntimeError("窗口内一天数据都没读到：nc_dir=%s lead=%d" % (nc_dir, lead))
    U3 = np.stack(seq, axis=2)                 # (41,81,T)
    if not np.isfinite(U3).all():
        raise RuntimeError("插值后出现 NaN/Inf —— 源网格没覆盖 -20..20/40..120？")
    return U3, np.array(dom, dtype=int), rows, missing


# ----------------------------------------------------------------- 投影

def load_consts(mat_path: str) -> dict:
    """读 CIOmode .mat 的模态向量与归一化常数。"""
    import scipy.io as sio
    m = sio.loadmat(mat_path)
    e = np.asarray(m["e"], dtype=np.float64)
    if e.ndim != 2 or e.shape[1] != N_SST + N_U850:
        raise ValueError("e 形状异常：%s（期望 %d 列）" % (e.shape, N_SST + N_U850))
    get = lambda k: (float(np.ravel(m[k])[0]) if k in m else None)
    return {"e_u850": e[0, N_SST:],            # 后 3321 列
            "e_sst": e[0, :N_SST],
            "Umean": get("Umean"), "Ustd": get("Ustd"),
            "SSTmean": get("SSTmean"), "SSTstd": get("SSTstd")}


def dom_average(U3: np.ndarray, dom: np.ndarray) -> np.ndarray:
    """按日序（跨年、跨月合并）求平均场；日序 >28 的槽位没有平均，取 0。"""
    avg28 = np.zeros((NLAT, NLON, 28))
    for i in range(28):
        sel = np.where(dom == i + 1)[0]
        if sel.size:
            avg28[:, :, i] = U3[:, :, sel].mean(axis=2)
    out = np.zeros_like(U3)
    for t in range(U3.shape[2]):
        if 1 <= dom[t] <= 28:
            out[:, :, t] = avg28[:, :, dom[t] - 1]
    return out


def lat_highpass(U3: np.ndarray) -> np.ndarray:
    """⚠ 坏高通：沿纬度（轴 0）滤。照抄历史口径，见文件头警告。"""
    from scipy.signal import butter, sosfiltfilt
    sos = butter(5, (1 / 20.) / 0.5, btype="highpass", output="sos")
    return sosfiltfilt(sos, U3, axis=0)


def project_u850(U3: np.ndarray, dom: np.ndarray, consts: dict) -> np.ndarray:
    """(41,81,T) 原始场 → 一维投影序列 (T,)。"""
    remain = U3 - dom_average(U3, dom) - lat_highpass(U3)
    X = (remain - consts["Umean"]) / consts["Ustd"]
    M = X.reshape(NLAT * NLON, X.shape[2], order="F")                      # 纬度快变
    return consts["e_u850"] @ M


# ----------------------------------------------------------------- 分派入口

def detect_kind(name: str, array=None) -> str:
    """按交接文档 §9.2 判类：只认文件名/变量名，判不出来就报错（不默认成 U850）。"""
    s = (name or "").lower()
    has_sst = any(k in s for k in ("sst", "sea_surface", "sea-surface"))
    has_u = any(k in s for k in ("u850", "uwnd", "u-wind", "u_wind", "u_anom"))
    if has_sst and has_u:
        return "both"
    if has_sst:
        return "sst"
    if has_u:
        return "u850"
    if array is not None and np.asarray(array).ndim >= 2:
        pass                                   # 预留：形状/掩膜判据（SST 有 614 个空点）
    raise ValueError("判不出输入类别（文件名 %r）；请显式指定 which=" % name)


def compute_cioproj(which: str = "u850", nc_dir: str = None, years=(),
                    lead: int = 1, mat_path: str = None, name: str = ""):
    """原始场 → projected CIO 序列。

    which: "u850" | "sst" | "both" | "auto"
    返回 (序列 (T,), meta)
    """
    if which == "auto":
        which = detect_kind(name)
    if which == "sst":
        raise NotImplementedError(
            "SST 分支尚未实现：缺 SST 原始场及其预处理源码（哪套资料、如何插值、"
            "是否也做日序平均、是否也过沿纬度滤波）。见 交接文档 §9.3。")
    if which == "both":
        compute_cioproj("sst", name=name)      # 直接抛错，不做近似替代
        raise AssertionError("unreachable")
    if which != "u850":
        raise ValueError("which 只能是 u850 / sst / both / auto")
    if not mat_path:
        raise ValueError("缺少 CIOmode .mat 路径（mat_path）")

    years = list(years)
    U3, dom, rows, missing = load_field(nc_dir, years, lead)
    consts = load_consts(mat_path)
    series = project_u850(U3, dom, consts)
    per_year = {}
    for (y, _mo, _d) in rows:
        per_year[y] = per_year.get(y, 0) + 1
    dates = ["%04d-%02d-%02d" % row for row in rows]
    meta = {"which": "u850", "lead": lead, "years": years,
            "length": int(series.size), "days_per_year": per_year,
            "missing_months": missing, "window": window(lead), "dates": dates}
    if missing:
        meta["warning"] = ("缺这些月份的源文件 %s —— 窗口天数不足 112/年，"
                           "与官方序列会有偏差" % missing)
    return series, meta
