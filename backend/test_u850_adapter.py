"""Analytic NetCDF fixtures test coordinate/date semantics independently of CIO."""
import io
from datetime import date
from pathlib import Path
import zipfile

import numpy as np
import pytest
from scipy.io import netcdf_file

import u850_adapter as adapter


def write_nc(path, *, times=None, lat=None, lon=None, dims=('time', 'lat', 'lon'),
             unit='m/s', pressure=None, anomaly=True, fill=False, calendar='standard',
             time_units='days since 2000-01-01', variable='U850_anom', packed=False):
    times = np.arange(366) if times is None else np.asarray(times)
    lat = np.arange(-22.5, 23, 5.) if lat is None else np.asarray(lat)
    lon = np.arange(37.5, 124, 5.) if lon is None else np.asarray(lon)
    coords = {'time': times, 'lat': lat, 'lon': lon}
    if pressure is not None:
        coords['level'] = np.asarray(pressure)
    with netcdf_file(str(path), 'w') as ds:
        for key, vals in coords.items():
            ds.createDimension(key, len(vals))
            v = ds.createVariable(key, 'd', (key,))
            v[:] = vals
            if key == 'time':
                if time_units is not None:
                    v.units = time_units
                v.calendar = calendar
            elif key == 'level':
                v.units = 'Pa'
            else:
                v.units = 'degrees_north' if key == 'lat' else 'degrees_east'
        field = ds.createVariable(variable, 'h' if packed else 'd', dims)
        field.units = unit
        field.long_name = '850 hPa eastward wind anomaly' if anomaly else '850 hPa eastward wind'
        # An affine spatial field has an exact bilinear-interpolation answer.
        vals = 2 * lat[None, :, None] + ((lon+180)%360-180)[None, None, :] + times[:, None, None] * .01
        original = ['time', 'lat', 'lon']
        if pressure is not None:
            vals = vals[:, None, :, :] + (np.asarray(pressure)[None, :, None, None]-85000)*.01
            original.insert(1, 'level')
        vals = np.transpose(vals, [original.index(d) for d in dims])
        if unit == 'cm/s':
            vals *= 100
        if packed:
            field.scale_factor = .1
            field.add_offset = 2.
            vals = np.rint((vals-2)/.1)
        if fill:
            field._FillValue = -9999.
            vals[...] = -9999
        field[:] = vals


@pytest.mark.parametrize('dims,lat,lon,unit,pressure', [
    (('time', 'lat', 'lon'), None, None, 'm/s', None),
    (('lon', 'time', 'lat'), np.arange(22.5,-23,-5), np.arange(122.5,37,-5)-360, 'cm/s', None),
    (('lon', 'level', 'lat', 'time'), None, None, 'm/s', [100000,85000,50000]),
])
def test_affine_field_and_calendar(tmp_path, dims, lat, lon, unit, pressure):
    write_nc(tmp_path/'annual.nc', dims=dims, lat=lat, lon=lon, unit=unit, pressure=pressure)
    data, dom, rows, missing, report = adapter.load_fields(tmp_path, [], 1, target_year=2000)
    assert data.shape == (41,81,112)
    assert rows[0] == (2000,5,29) and rows[-1] == (2000,9,28)
    assert missing == [] and len(dom) == 112
    assert report['sourceSamples'] == 366 and report['ignoredSamples'] == 254
    la, lo = np.meshgrid(np.arange(-20,21), np.arange(40,121), indexing='ij')
    for index in (0,50,111):
        offset = (date(*rows[index])-date(2000,1,1)).days
        np.testing.assert_allclose(data[:,:,index], 2*la+lo+offset*.01, atol=1e-10)


def test_split_files_shuffled_time_and_variable_length(tmp_path):
    write_nc(tmp_path/'part-z.nc', times=np.arange(0,183)[::-1])
    write_nc(tmp_path/'part-a.nc', times=np.arange(183,731)[::-1])
    data, _, rows, _, report = adapter.load_fields(tmp_path, [], 20, target_year=2001)
    assert data.shape[-1] == 224 and rows == sorted(rows)
    assert report['selectedSamples'] == 224
    assert rows[0] == (2000,5,10) and rows[-1] == (2001,9,9)


@pytest.mark.parametrize('kwargs,match', [
    ({'times': np.delete(np.arange(366),149)}, '缺少 1 个'),
    ({'times': [149,149.5]}, '重复日期或日内'),
    ({'unit': 'knots'}, '风速单位'),
    ({'unit': ''}, '风速单位'),
    ({'pressure': [50000], 'dims': ('time','level','lat','lon')}, '850 hPa'),
    ({'anomaly': False, 'variable': 'u850'}, '未确认输入为距平'),
    ({'calendar': '360_day'}, '不支持日历'),
    ({'lon': np.arange(50,121,5)}, '未覆盖'),
    ({'lat': [-20,0,0,20]}, '重复格点'),
    ({'fill': True}, '缺测值'),
    ({'time_units': None}, 'CF units'),
])
def test_reject_incompatible_inputs(tmp_path, kwargs, match):
    write_nc(tmp_path/'annual.nc', **kwargs)
    with pytest.raises(ValueError, match=match):
        adapter.load_fields(tmp_path, [], 1, target_year=2000)


def test_duplicate_dates_across_files(tmp_path):
    write_nc(tmp_path/'a.nc')
    write_nc(tmp_path/'b.nc')
    with pytest.raises(ValueError, match='多个文件包含重复日期'):
        adapter.inventory(tmp_path)


def test_noleap_calendar_and_legacy_fallback(tmp_path):
    p = tmp_path/'cesm.u850.anom.daily.20000502-28days.nc'
    write_nc(p, times=np.arange(121,149), calendar='noleap')
    assert adapter.inventory(tmp_path)[0][1]['dates'][0] == date(2000,5,2)
    write_nc(p, times=np.arange(28), time_units=None)
    spec = adapter.inventory(tmp_path)[0][1]
    assert spec['dates'][0] == date(2000,5,2)
    assert spec['report']['dateSource'].startswith('legacy filename')


def test_packed_data(tmp_path):
    write_nc(tmp_path/'packed.nc', packed=True)
    data, _, rows, _, _ = adapter.load_fields(tmp_path, [], 1, target_year=2000)
    offset = (date(*rows[0])-date(2000,1,1)).days
    assert abs(data[20,0,0] - (40+offset*.01)) <= .05


def test_packed_coordinates(tmp_path):
    path = tmp_path/'packed-coordinates.nc'
    write_nc(path)
    with netcdf_file(str(path), 'a') as ds:
        for name in ('lat', 'lon', 'time'):
            ds.variables[name][:] *= 10
            ds.variables[name].scale_factor = .1
    data, _, rows, _, _ = adapter.load_fields(tmp_path, [], 1, target_year=2000)
    assert rows[0] == (2000,5,29)
    np.testing.assert_allclose(data[20,0,0], 41.49, atol=1e-6)


def test_missing_other_year_is_reported_without_filling(tmp_path):
    times = np.delete(np.arange(731),149)
    write_nc(tmp_path/'two-years.nc', times=times)
    data, _, rows, missing, report = adapter.load_fields(tmp_path, [], 1, target_year=2001)
    assert data.shape[-1] == 223
    assert sum(y == 2001 for y,_,_ in rows) == 112
    assert missing == [(2000,5)] and report['missingDates'] == ['2000-05-29']


def test_netcdf4_with_cf_coordinate_names(tmp_path):
    import h5netcdf
    with h5netcdf.File(tmp_path/'climate.nc4', 'w') as ds:
        ds.dimensions = dict(valid_time=366, latitude=9, longitude=17)
        for name, vals, attrs in (
            ('valid_time', np.arange(366), {'units':'days since 2000-01-01', 'standard_name':'time'}),
            ('latitude', np.arange(-20,21,5), {'units':'degrees_north'}),
            ('longitude', np.arange(40,121,5), {'units':'degrees_east'}),
        ):
            v = ds.create_variable(name, (name,), dtype='f8', data=vals)
            v.attrs.update(attrs)
        field = ds.create_variable('u850', ('longitude','latitude','valid_time'), dtype='f8')
        field.attrs.update(units='m s**-1', long_name='eastward wind anomaly')
        field[:] = np.arange(366)[None,None,:] + np.zeros((17,9,1))
    data, _, rows, _, _ = adapter.load_fields(tmp_path, [], 1, target_year=2000)
    assert data.shape == (41,81,112)
    assert data[0,0,0] == (date(*rows[0])-date(2000,1,1)).days


def test_chain_accepts_anonymous_annual_archive_and_exports_report(tmp_path, monkeypatch):
    import chain
    source = tmp_path/'source'
    source.mkdir()
    write_nc(source/'anonymous.nc')
    content = io.BytesIO()
    with zipfile.ZipFile(content, 'w') as z:
        z.write(source/'anonymous.nc', 'nested/annual.nc')
    job = tmp_path/'jobs'/'adapter-test'
    job.mkdir(parents=True)
    monkeypatch.setattr(chain, '_mat_identity', lambda required: {})
    metadata = chain._intake_zip(content.getvalue(), job, 'auto', 'dataset.zip', 2000, 1)
    assert metadata['years'] == [2000] and metadata['kindSource'] == 'netcdf-metadata'
    assert metadata['metadataValidated']
    monkeypatch.setattr(chain, 'JOB_ROOT', job.parent)
    monkeypatch.setattr(chain.prediction, '_mat_path', lambda: 'unused')
    monkeypatch.setattr(chain.cioproj, 'load_consts', lambda _: dict(
        e_u850=np.linspace(-.01,.02,3321), Umean=.2, Ustd=3.))
    with chain._jobs_lock:
        chain._jobs['adapter-test'] = {'jobId':'adapter-test', 'projection':metadata}
    try:
        series = chain._run_zip_stage1('adapter-test', job, metadata, 1, 'dataset.zip')
        assert series.size == 112
        stored = chain._jobs['adapter-test']['projection']['adaptation']
        assert stored['selectedSamples'] == 112
        assert stored['files'][0]['dateSource'] == 'CF time coordinate'
        assert (job/'projection_preview.npz').is_file()
        assert (job/'adaptation.json').is_file()
    finally:
        with chain._jobs_lock:
            chain._jobs.pop('adapter-test', None)
