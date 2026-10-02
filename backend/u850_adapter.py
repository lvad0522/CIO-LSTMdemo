"""Metadata-driven daily U850 anomaly adapter; preserves the trained 112-point calendar.

No climatology is invented for absolute winds, and missing days are never filled.
Only the explicitly named historical 28-day format may infer missing metadata.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
import re

import numpy as np
import xarray as xr

LEGACY = re.compile(r"\.(\d{4})(\d{2})02-28days\.nc$", re.I)
MONTH_LENGTHS = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)


def nc_files(root):
    return sorted(p for p in Path(root).rglob('*')
                  if p.is_file() and p.suffix.lower() in ('.nc', '.nc4'))


def open_dataset(path):
    try:
        return xr.open_dataset(path, decode_times=False, mask_and_scale=False)
    except Exception as exc:
        raise ValueError(f"无法读取 NetCDF {Path(path).name}（{type(exc).__name__}）；"
                         "请使用标准 NetCDF3/4 文件，NetCDF4 需安装 h5netcdf 或 netCDF4。") from exc


def coordinate(ds, role):
    names = {'lat': ('lat', 'latitude'), 'lon': ('lon', 'longitude'),
             'time': ('time', 'valid_time', 'date'),
             'level': ('level', 'lev', 'plev', 'pressure_level', 'isobaricinhpa')}
    standard = {'lat': 'latitude', 'lon': 'longitude', 'time': 'time',
                'level': 'air_pressure'}
    axis = {'lat': 'Y', 'lon': 'X', 'time': 'T', 'level': 'Z'}
    found = [k for k, v in ds.variables.items()
             if k.lower() in names[role]
             or v.attrs.get('standard_name') == standard[role]
             or str(v.attrs.get('axis', '')).upper() == axis[role]]
    if len(found) != 1:
        if role == 'level' and not found:
            return None
        raise ValueError(f"{role} 坐标无法唯一识别：{found}；需要明确的坐标名称或 CF 属性")
    return found[0]


def numeric_values(var):
    """Decode CF packing explicitly, including coordinate variables."""
    raw = np.asarray(var.values, dtype=np.float64)
    if str(var.attrs.get('_Unsigned', '')).lower() == 'true':
        raise ValueError('暂不支持 _Unsigned 编码；请导出解码后的有符号数值或浮点 NetCDF')
    for attr in ('_FillValue', 'missing_value'):
        for fill in np.ravel(var.attrs.get(attr, [])):
            raw = np.where(raw == fill, np.nan, raw)
    return raw * float(var.attrs.get('scale_factor', 1)) + float(var.attrs.get('add_offset', 0))


def decode_dates(var, filename):
    values = np.asarray(var.values).ravel()
    if values.dtype.kind in 'iuf':
        values = numeric_values(var).ravel()
    if not values.size:
        raise ValueError('时间坐标为空')
    calendar = str(var.attrs.get('calendar', 'standard')).lower()
    if calendar not in ('standard', 'gregorian', 'proleptic_gregorian', 'noleap', '365_day'):
        raise ValueError(f"不支持日历 {calendar}；不能无损映射到当前模型日期")
    units = str(var.attrs.get('units', ''))
    match = re.fullmatch(r'(days?|hours?|minutes?|seconds?) since (.+)', units.strip(), re.I)
    legacy = LEGACY.search(filename)
    if not match:
        if values.dtype.kind in 'USM':
            try:
                stamps = [datetime.fromisoformat(str(v).replace('Z', '+00:00')) for v in values]
            except ValueError as exc:
                raise ValueError('时间坐标不是可解析的 ISO 日期') from exc
            source = 'ISO time coordinate'
        elif legacy and len(values) == 28 and not units:
            stamps = [datetime(int(legacy[1]), int(legacy[2]), d) for d in range(2, 30)]
            source = 'legacy filename (28 days)'
        else:
            raise ValueError('时间坐标缺少有效的 CF units（如 days since 2000-01-01）；不按样本位置猜日期')
    else:
        try:
            base = datetime.fromisoformat(match[2].strip().replace('Z', '+00:00'))
        except ValueError as exc:
            raise ValueError('无法解析 CF 时间起点') from exc
        if base.tzinfo is not None and base.utcoffset() != timedelta(0):
            raise ValueError('时间起点必须使用 UTC 或无时区日期')
        seconds = {'day': 86400, 'hour': 3600, 'minute': 60, 'second': 1}[match[1].lower().rstrip('s')]
        if not np.isfinite(values.astype(float)).all():
            raise ValueError('时间坐标含 NaN/Inf')
        if calendar in ('noleap', '365_day'):
            if base.month == 2 and base.day == 29:
                raise ValueError('noleap 日历起点不能是 2 月 29 日')
            origin = (base.year - 1) * 365 + sum(MONTH_LENGTHS[:base.month-1]) + base.day - 1
            stamps = []
            for value in values:
                offset = base.hour*3600 + base.minute*60 + base.second + float(value)*seconds
                day_offset, remainder = divmod(offset, 86400)
                year_index, doy = divmod(origin + int(day_offset), 365)
                month = 1
                while doy >= MONTH_LENGTHS[month-1]:
                    doy -= MONTH_LENGTHS[month-1]
                    month += 1
                stamps.append(datetime(year_index+1, month, doy+1) + timedelta(seconds=remainder))
        else:
            stamps = [base + timedelta(seconds=float(v)*seconds) for v in values]
        source = 'CF time coordinate'
    dates = [s.date() for s in stamps]
    if any(s.utcoffset() not in (None, timedelta(0)) for s in stamps):
        raise ValueError('时间坐标必须使用 UTC 或无时区日期')
    if len(set(dates)) != len(dates):
        raise ValueError('存在重复日期或日内多时次；当前只接受逐日距平，不自动平均小时数据')
    return dates, source, calendar


def describe(ds, filename):
    lat, lon, time = (coordinate(ds, role) for role in ('lat', 'lon', 'time'))
    if any(ds[k].ndim != 1 for k in (lat, lon, time)):
        raise ValueError('只支持一维时间和规则经纬度坐标；二维曲线网格需要专用重网格适配')
    dims = tuple(ds[k].dims[0] for k in (time, lat, lon))
    if len(set(dims)) != 3:
        raise ValueError('时间、纬度、经度必须对应三个独立维度')
    candidates = []
    for name, v in ds.data_vars.items():
        if not set(dims).issubset(v.dims):
            continue
        hint = ' '.join((name, str(v.attrs.get('standard_name', '')), str(v.attrs.get('long_name', '')))).lower()
        if name.lower() in ('u', 'ua') or any(s in hint for s in ('u850', 'uwnd', 'u_anom', 'u-wind', 'u_wind', 'eastward_wind', 'zonal wind', 'zonal_wind')):
            candidates.append(name)
    if len(candidates) != 1:
        raise ValueError(f'U850 变量无法唯一识别：{candidates}；需要 u850/uwnd/u/ua 或 eastward_wind 元数据')
    name = candidates[0]
    field = ds[name]
    hint = ' '.join((name, filename, str(field.attrs), str(ds.attrs))).lower()
    if not any(s in hint for s in ('anom', '距平')):
        raise ValueError('未确认输入为距平场：变量、long_name 或文件名需标明 anomaly/anom；绝对风场需先按训练基准计算距平')
    level = coordinate(ds, 'level')
    selected = {}
    if level:
        lv = ds[level]
        if lv.ndim > 1:
            raise ValueError('气压坐标必须是一维或标量')
        pressure = numeric_values(lv).ravel()
        unit = str(lv.attrs.get('units', '')).lower().strip()
        if unit in ('pa', 'pascal', 'pascals'):
            pressure = pressure / 100
        elif unit not in ('hpa', 'mbar', 'millibar', 'millibars'):
            raise ValueError('气压层必须明确标注 Pa 或 hPa 单位')
        indices = np.flatnonzero(np.isclose(pressure, 850, rtol=0, atol=0.01))
        if len(indices) != 1:
            raise ValueError('数据中没有唯一的 850 hPa 层，不用邻近气压层替代')
        if lv.ndim == 1:
            if lv.dims[0] not in field.dims:
                raise ValueError('气压坐标与 U 风场维度不对应')
            selected[lv.dims[0]] = int(indices[0])
    elif not re.search(r'(?:u[ _-]?850|850\s*(?:hpa|mb))', hint):
        raise ValueError('未确认 850 hPa：需要气压坐标或明确的 U850 变量/文件元数据')
    for dim in field.dims:
        if dim not in dims and dim not in selected:
            if field.sizes[dim] != 1:
                raise ValueError(f'不支持非单例附加维度 {dim}；请先选择成员或实验版本')
            selected[dim] = 0
    unit = str(field.attrs.get('units', '')).lower().replace(' ', '').replace('**', '^')
    factors = {'m/s': 1., 'ms-1': 1., 'ms^-1': 1., 'm.s-1': 1.,
               'cm/s': .01, 'cms-1': .01, 'cms^-1': .01,
               'km/h': 1/3.6, 'kmh-1': 1/3.6, 'kmh^-1': 1/3.6}
    if unit not in factors:
        raise ValueError(f'不支持或缺少风速单位 {unit!r}；支持 m/s、cm/s、km/h')
    dates, date_source, calendar = decode_dates(ds[time], filename)
    latv = numeric_values(ds[lat])
    lonv = numeric_values(ds[lon])
    for key, values in ((lat, latv), (lon, lonv)):
        coord_unit = str(ds[key].attrs.get('units', '')).lower()
        if coord_unit and 'degree' not in coord_unit:
            raise ValueError(f'{key} 坐标必须使用角度单位')
        if len(values) < 2 or not np.isfinite(values).all():
            raise ValueError(f'{key} 坐标无效')
    lonv = (lonv + 180) % 360 - 180
    if len(np.unique(latv)) != len(latv) or len(np.unique(lonv)) != len(lonv):
        raise ValueError('经纬度存在重复格点（含 0°/360° 重复端点），请先去重')
    if latv.min() > -20 or latv.max() < 20 or lonv.min() > 40 or lonv.max() < 120:
        raise ValueError('源网格未覆盖 20°S–20°N、40°E–120°E，不进行外推')
    return dict(variable=name, dims=dims, selected=selected, dates=dates,
                lat=latv, lon=lonv, factor=factors[unit],
                report=dict(file=filename, variable=name, sourceDimensions=list(field.dims),
                            sourceShape=list(field.shape), sourceUnit=field.attrs.get('units'),
                            targetUnit='m/s', unitFactor=factors[unit], pressureHpa=850,
                            dateSource=date_source, calendar=calendar,
                            sourceSamples=len(dates), dateStart=min(dates).isoformat(),
                            dateEnd=max(dates).isoformat()))


def inventory(root):
    files = nc_files(root)
    if not files:
        raise ValueError('数据包中没有 NetCDF 文件')
    entries, seen = [], set()
    for path in files:
        with open_dataset(path) as ds:
            spec = describe(ds, path.name)
        for d in spec['dates']:
            if d in seen:
                raise ValueError(f'多个文件包含重复日期 {d.isoformat()}，不自动覆盖或合并')
            seen.add(d)
        entries.append((path, spec))
    return entries


def expected_dates(year, lead):
    start, end = date(year, 5, 30-lead), date(year, 9, 29-lead)
    return [start + timedelta(days=i) for i in range((end-start).days+1)
            if 2 <= (start + timedelta(days=i)).day <= 29]


def require_window(entries, year, lead):
    all_dates = {d for _, spec in entries for d in spec['dates']}
    if year not in {d.year for d in all_dates}:
        raise ValueError(f'zip 里没有 {year} 年的 nc 时间数据')
    missing = sorted(set(expected_dates(year, lead)) - all_dates)
    if missing:
        example = ', '.join(d.isoformat() for d in missing[:8])
        raise ValueError(f'目标年 {year} 缺少 {len(missing)} 个模型窗口日期：{example}；不补值、不按位置错移')


def load_fields(root, years, lead, *, target_year=None):
    from scipy.interpolate import RegularGridInterpolator
    if not 1 <= lead <= 20:
        raise ValueError('lead 必须在 1..20')
    entries = inventory(root)
    if target_year is not None:
        require_window(entries, target_year, lead)
    available_years = sorted({d.year for _, s in entries for d in s['dates']})
    years = sorted(set(years)) if years else available_years
    wanted = {d for year in years for d in expected_dates(year, lead)}
    lon_grid, lat_grid = np.meshgrid(np.arange(40, 121.), np.arange(-20, 21.))
    points = np.column_stack((lat_grid.ravel(), lon_grid.ravel()))
    frames, reports = {}, []
    for path, spec in entries:
        chosen = [(i, d) for i, d in enumerate(spec['dates']) if d in wanted]
        report = dict(spec['report'], selectedSamples=len(chosen))
        reports.append(report)
        if not chosen:
            continue
        lati, loni = np.argsort(spec['lat']), np.argsort(spec['lon'])
        slat, slon = spec['lat'][lati], spec['lon'][loni]
        # Include bracketing cells outside the target domain, so shifted/coarse
        # source grids remain interpolable at the domain boundaries.
        la = slice(max(0, np.searchsorted(slat, -20)-1), min(len(slat), np.searchsorted(slat, 20, side='right')+1))
        lo = slice(max(0, np.searchsorted(slon, 40)-1), min(len(slon), np.searchsorted(slon, 120, side='right')+1))
        with open_dataset(path) as ds:
            field = ds[spec['variable']]
            if str(field.attrs.get('_Unsigned', '')).lower() == 'true':
                raise ValueError('暂不支持 _Unsigned 风场编码；请导出浮点 NetCDF')
            remaining = [d for d in field.dims if d not in spec['selected'] and d != spec['dims'][0]]
            for index, day in chosen:
                select = {**spec['selected'], spec['dims'][0]: index}
                key = tuple(select.get(d, slice(None)) for d in field.dims)
                raw = np.asarray(field.variable[key].values, dtype=np.float64)
                raw = np.transpose(raw, [remaining.index(d) for d in spec['dims'][1:]])
                for attr in ('_FillValue', 'missing_value'):
                    for fill in np.ravel(field.attrs.get(attr, [])):
                        raw = np.where(raw == fill, np.nan, raw)
                raw = (raw * float(field.attrs.get('scale_factor', 1)) + float(field.attrs.get('add_offset', 0))) * spec['factor']
                raw = raw[np.ix_(lati[la], loni[lo])]
                result = RegularGridInterpolator((slat[la], slon[lo]), raw, bounds_error=True)(points).reshape(41, 81)
                if not np.isfinite(result).all():
                    raise ValueError(f'{day.isoformat()} 目标区域含缺测值或 NaN/Inf；不自动填补')
                frames[day] = result
    if not frames:
        raise ValueError('源数据在所选年份的模型窗口内没有有效日期')
    dates = sorted(frames)
    missing_dates = sorted(wanted - frames.keys())
    report = dict(adapter='cf-daily-u850-anomaly-v1', files=reports,
                  sourceSamples=sum(s['sourceSamples'] for s in reports), selectedSamples=len(dates),
                  ignoredSamples=sum(s['sourceSamples'] for s in reports)-len(dates),
                  targetGrid=dict(lat=[-20, 20], lon=[40, 120], shape=[41, 81], resolutionDegrees=1),
                  sampling='训练窗口内每月 2–29 日；其余日期不送入当前模型',
                  missingDates=[d.isoformat() for d in missing_dates],
                  preprocessing='保留训练时的日序平均、沿纬度滤波和固定 EOF 标准化；适配不等于跨数据集精度保证')
    rows = [(d.year, d.month, d.day) for d in dates]
    missing = sorted({(d.year, d.month) for d in missing_dates})
    return np.stack([frames[d] for d in dates], axis=2), np.array([d.day for d in dates]), rows, missing, report
