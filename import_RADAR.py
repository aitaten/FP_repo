"""
Methods for importing files containing two-dimestional radar mosaics.

There are:

    1.- Filename importers (for both archived [.gz.] and h5 files [.hdf])
    2.- Filename generators based on datetimes and internal ZAMG structure

Functions for creating the projection or coordinates for the different
files is also provided in different functions.
"""
import gzip
from contextlib import contextmanager
import numpy as np
from datetime import datetime, timedelta
from os.path import join, isfile
import logging

import pdb

logger = logging.getLogger(__name__)

try:
    import h5py

    H5PY_IMPORTED = True
except ImportError:
    H5PY_IMPORTED = False

try:
    import pyproj

    PYPROJ_IMPORTED = True
except ImportError:
    PYPROJ_IMPORTED = False

@contextmanager
def open_compressed(filename, mode='r'):
    """Open a file directly or through gzip library depending on extension.

    Args:
        mode (str): file access mode
    """

    if not H5PY_IMPORTED:
        raise MissingOptionalDependency(
            "h5py package is required to import hdf radar files but it is not installed"
        )

    if filename.endswith(".gz"):
        f = gzip.GzipFile(filename, mode)
        h5file=h5py.File(f,mode)
    else:
        h5file=h5py.File(filename,"r")
    yield h5file
    h5file.close()
    # del f

def read_from_hdf(filename):
    """Read hdf radar files

    Args:
        filename (str): file name
    """
    with open_compressed(filename,"r") as h5file:

        dataset=h5file['dataset1']
        datagroup=dataset['data1']
        data=np.array(datagroup['data'],dtype=float)

        whatgroup=dataset['what']
        gain=whatgroup.attrs.get('gain')
        offset=whatgroup.attrs.get('offset')
        nodata=whatgroup.attrs.get('nodata')
        undetect=whatgroup.attrs.get('undetect')

        data[data==undetect]=0
        data[data==nodata]=np.nan
        data=data*gain+offset

        # return np.flipud(data).T
        return data

def import_ATNT_composite(RadarDate,accum_r = "ACC5",
    RADAR_ARC = '/radar_arch/COMPOSITES/ATNTCOMPO/'):
    """Read hdf radar files from date creating the
        proper filename structure and path

    Args:
        RadarDate (datetime): date for the radar file
        [accum_r] (str): acronym of the accumulation composite
        [RADAR_ARC] (path): Path of the radar files archive
    """

    from os.path import join, isfile
    
    radarfile_old = join(RADAR_ARC, f"{RadarDate:%Y}", f"{RadarDate:%m}", f"{RadarDate:%d}", accum_r, "HDF", "COMPO_ATNTCOMPO_" + accum_r + "_MAXRRSUM_" + f"{RadarDate:%Y%m%d%H%M}" + ".hdf")

    if not isfile(radarfile_old):
        radarfile_old +='.gz'

    return read_from_hdf(radarfile_old)

def meta_ATNT_composite(qty="DBZH", accutime=5.0, pixelsize=1000.0, 
    x1=0,x2=999000,y1=-600000.0,y2=-1000.0,**kwargs):

    if not PYPROJ_IMPORTED:
        raise MissingOptionalDependency(
            "pyproj package is required to create radar metadata but it is not installed"
        )

    metadata = {}

    if qty == "ACRR":
        unit = "mm"
        transform = None
    elif qty == "DBZH":
        unit = "dBZ"
        transform = "dB"

    proj4str = "+proj=lcc +lat_1=49 +lat_2=46 +lat_0=47.5 +lon_0=13.333333 +x_0=500000 +y_0=-320000 +ellps=bessel +towgs84=577.326,90.129,463.919,5.137,1.474,5.297,2.4232 +units=m +no_defs"
    pr = pyproj.Proj(proj4str)
    metadata["projection"] = proj4str

    # Fill in the metadata
    metadata["x1"] = x1
    metadata["y1"] = y1
    metadata["x2"] = x2
    metadata["y2"] = y2

    metadata["xpixelsize"] = pixelsize
    metadata["ypixelsize"] = pixelsize
    metadata["cartesian_unit"] = "m"
    metadata["yorigin"] = "upper"

    metadata["institution"] = "ZAMG - Zentralanstalt für Meteorologie und Geodynamik"
    metadata["accutime"] = accutime
    metadata["unit"] = unit
    metadata["transform"] = transform
    metadata["zerovalue"] = 0.0
    metadata["threshold"] = 8.0
    metadata["zr_a"] = 200.0
    metadata["zr_b"] = 1.6

    return metadata

def meta_INCALBINA(qty="DBZH", accutime=5.0, pixelsize=1000.0, 
    x1=20000.,x2=720000.,y1=190000.,y2=620000.,**kwargs):

    if not PYPROJ_IMPORTED:
        raise MissingOptionalDependency(
            "pyproj package is required to create radar metadata but it is not installed"
        )

    metadata = {}

    if qty == "ACRR":
        unit = "mm"
        transform = None
    elif qty == "DBZH":
        unit = "dBZ"
        transform = "dB"

    proj4str = "+proj=lcc +lat_1=46 +lat_2=49 +lat_0=47.5 +lon_0=13.33333333333333 +x_0=400000 +y_0=400000 +ellps=bessel +towgs84=577.326,90.129,463.919,5.137,1.474,5.297,2.4232 +units=m +no_defs"
    pr = pyproj.Proj(proj4str)
    metadata["projection"] = proj4str

    # Fill in the metadata
    metadata["x1"] = x1
    metadata["y1"] = y1
    metadata["x2"] = x2
    metadata["y2"] = y2

    metadata["xpixelsize"] = pixelsize
    metadata["ypixelsize"] = pixelsize
    metadata["cartesian_unit"] = "m"
    metadata["yorigin"] = "upper"

    metadata["institution"] = "ZAMG - Zentralanstalt für Meteorologie und Geodynamik"
    metadata["accutime"] = accutime
    metadata["unit"] = unit
    metadata["transform"] = transform
    metadata["zerovalue"] = 0.0
    metadata["threshold"] = 8.0
    metadata["zr_a"] = 200.0
    metadata["zr_b"] = 1.6

    return metadata


def from_meta_to_latlon(meta, return_grid = "quadmesh"):

    proj = pyproj.Proj(meta['projection'])

    NX = int((meta['x2']-meta['x1'])/meta['xpixelsize'])
    NY = int((meta['y2']-meta['y1'])/meta['ypixelsize'])

    if return_grid == "coords":
        y_coord = np.linspace(meta['y1'], meta['y2'], NY) + meta["ypixelsize"] / 2.0
        x_coord = np.linspace(meta['x1'], meta['x2'], NX) + meta["xpixelsize"] / 2.0
    elif return_grid == "quadmesh":
        y_coord = np.linspace(meta['y1'], meta['y2'], NY + 1) + meta["ypixelsize"] / 2.0
        x_coord = np.linspace(meta['x1'], meta['x2'], NX + 1) + meta["ypixelsize"] / 2.0

    x_grid, y_grid = np.meshgrid(x_coord, y_coord)

    if meta["yorigin"] == "upper":
        y_grid = np.flipud(y_grid)

    return proj(x_grid, y_grid , inverse=True)

def read_hbin(file,hbin_NX=600,hbin_NY=1000):
    """
    Read hbin format

    Args:
        file (string): the radar file

    Returns:
        RX (N,M) ndarray: The radar field from the file

    """
    import struct

    with file:
        typeFormat = 'l'
        typeSize = struct.calcsize(typeFormat)
        value = file.read(typeSize)
        radar_available = struct.unpack(typeFormat, value)[0]

        typeFormat = 'l'
        typeSize = struct.calcsize(typeFormat)
        value = file.read(typeSize)
        radar_total = struct.unpack(typeFormat, value)[0]

        typeFormat = '500c'
        typeSize = struct.calcsize(typeFormat)
        radar_info = file.read(typeSize)

        count=hbin_NX*hbin_NY
        typeFormat = str(count)+'f'
        typeSize = struct.calcsize(typeFormat)
        value = file.read(typeSize)
        radar = struct.unpack(typeFormat, value)
        radar = np.asarray(radar)

        field = radar.reshape(hbin_NY, hbin_NX)

    return field

def read_radar_hbin_single(AnaDate, AccumType = "5min", RXout = 20., hbin_prefix="INCALBINA",
    RADAR_LIST = ["A_PAK", "A_FEL", "A_ZIR", "A_RAU", "I_BOL", "SLO_RAPS_COMPOSIT", "CH_COMPOSIT", "CZ_BRD", "CZ_SKA", "SK_COMPOSIT", "D_RAPS_COMPOSIT"]):
    """
    Read and process radar data from single radars in hbin format

    Args:
        AnaDate (datetime): the date for the radar fields
        AccumType (string): the accumulation for the different radar fields
        RADAR_LIST (list): the list of radars
        RXout (float): Allowed percentage of missing radar files in accumulation
        hbin_prefix (string): Prefix of the Radar files

    Returns:
        RXsingle (N,M,n) ndarray: single radar fields on the INCA grid
        RXn (list of tuples): available and maximum number of time slices of each radar

    """

    import gzip
    import struct

    # Directories for radar data
    RADAR_OP = '/radar_arch/tmp/INCA'
    RADAR_ARC = '/mapp_arch/mgruppe/arc/radar'

    #This information is the INCA ALBINA meta data
    meta = meta_INCALBINA()

    # hbin_NX = int((meta['x2']-meta['x1'])/meta['xpixelsize'])
    # hbin_NY = int((meta['y2']-meta['y1'])/meta['ypixelsize'])
    hbin_NX = int((meta['x2']-meta['x1'])/meta['xpixelsize'])+1
    hbin_NY = int((meta['y2']-meta['y1'])/meta['ypixelsize'])+1

    RX_hbin_N = len(RADAR_LIST)
    RXSingle = np.full((hbin_NY, hbin_NX, RX_hbin_N), -999.)

    if AccumType == "24h":
        TypeRX = "racc24h"
        minutes_back=[0]
    elif AccumType == "1h":
        TypeRX = "racc1h"
        minutes_back=[0,5,10,15]
    elif AccumType == "15min":
        TypeRX = "racc15m"
        minutes_back=[0,5]
    elif AccumType == "5min":
        TypeRX = "racc5m"
        minutes_back=[0,1]
        #The original has
        #minutes_back=[5,6]


    RXn = []
    ct = 0
    for i, rad in enumerate(RADAR_LIST):
        success = False
        for back in minutes_back:
            RadarDate = AnaDate - timedelta(minutes=back)

            radarfile_op = join(RADAR_OP, rad, f"{RadarDate:%Y}", f"{RadarDate:%m}", f"{RadarDate:%d}", TypeRX, hbin_prefix + "_" + rad + "_" + TypeRX + "_" + f"{RadarDate:%Y%m%d%H%M}" + ".hbin")
            radarfile_arc = join(RADAR_ARC, rad, f"{RadarDate:%Y}", f"{RadarDate:%m}", f"{RadarDate:%d}", TypeRX, hbin_prefix + "_" + rad + "_" + TypeRX + "_" + f"{RadarDate:%Y%m%d%H%M}" + ".hbin.gz")


            if isfile(radarfile_op):
                f = open(radarfile_op, 'rb')
                success = True
            elif isfile(radarfile_arc):
                f = gzip.GzipFile(radarfile_arc, 'rb')
                success = True
            else:
                logger.warning(f"Could not open {radarfile_arc}")
                

            # try:
            #     f = open(radarfile_op, 'rb')
            #     logger.info(f"Reading {radarfile_op}")
            #     success = True
            # except IOError:
            #     logger.warning(f"Could not open {radarfile_op}. Looking in archive instead")
            #     try:
            #         f = gzip.GzipFile(radarfile_arc, 'rb')
            #         logger.info(f"Reading {radarfile_arc}")
            #         success = True
            #     except IOError:
            #         logger.warning(f"Could not open {radarfile_arc}")

            if success:
                break

        if not success:
            RXn.append((0, 0))
            continue

        with f:
            typeFormat = 'l'
            typeSize = struct.calcsize(typeFormat)
            value = f.read(typeSize)
            radar_available = struct.unpack(typeFormat, value)[0]

            typeFormat = 'l'
            typeSize = struct.calcsize(typeFormat)
            value = f.read(typeSize)
            radar_total = struct.unpack(typeFormat, value)[0]

            typeFormat = '500c'
            typeSize = struct.calcsize(typeFormat)
            radar_info = f.read(typeSize)

            count=hbin_NX*hbin_NY
            typeFormat = str(count)+'f'
            typeSize = struct.calcsize(typeFormat)
            value = f.read(typeSize)
            radar = struct.unpack(typeFormat, value)
            radar = np.asarray(radar)

            field = radar.reshape(hbin_NY, hbin_NX)

            if np.isnan(field).any():
                logger.warning(f"NaN found for radar {rad}. Setting NaN values to -999.")
                field = np.nan_to_num(field, nan=-999.)

            logger.info("%s: %d of %d radar observations found", rad, radar_available, radar_total)

            if radar_total < 1:
                logger.warning(f"There might be something wrong with radar {rad}: Expected number of time slices is {radar_total} and doesn't make sense")

            if radar_available < radar_total:
                logger.warning(f"A part of the radar information is missing for radar {rad}")
            logger.debug(radar_info.decode('utf-8').rstrip(" \0").rstrip())

            RXn.append((radar_available, radar_total))

            if radar_available < radar_total * (100. - RXout) / 100.:
                logger.warning("Rejecting radar %s because more than %4.1f %% of the files are missing in the accumulation (only %d of %d available)" % (rad, RXout, radar_available, radar_total))
                continue

            if radar_total - radar_available > 0:
                factor = radar_total / radar_available
                logger.info("Upscaling radar %s by factor %6.3f (only %d of %d available)" % (rad, factor, radar_available, radar_total))
                field = field * factor

            RXSingle[..., i] = field

            ct += 1

    if ct > 0:
        return RXSingle, RXn
    else:
        return None, None


if __name__ == "__main__":

    #This tool is useful for debugging (you can stop the code using pdb.set_trace() like after 1st figure)
    import pdb

    ## Example of reading radar (two ways, individual and composite):
    import sys  
    try:
        import sys
        RadarDate = datetime.strptime(sys.argv[1],'%Y%m%d%H%M')
    except:
        print('Running with date from example case')
        RadarDate = datetime.strptime('202308231100','%Y%m%d%H%M')

    RADAR_COMP = import_ATNT_composite(RadarDate)

    RADARs_SIN, RXn = read_radar_hbin_single(RadarDate)

    #To understand the type of array (it is multiple radars in a different domain as composite)
    print(np.array(RADARs_SIN).shape)

    from matplotlib import use, colors
    use('TkAgg')

    from matplotlib import pyplot as plt

    plt.subplot(2,1,1)
    plt.pcolormesh(RADAR_COMP,norm=colors.LogNorm())
    plt.subplot(2,1,2)
    plt.pcolormesh(np.nanmax(RADARs_SIN,axis=2),norm=colors.LogNorm())
    plt.show()

    pdb.set_trace()

    ### import function from another program in this folder
    from f_clean_radar import clean_radar,clean_with_ndimage

    RADAR_CLEAN = clean_radar(np.nanmax(RADARs_SIN,axis=2),Thr=0.1,S_area=100)
    # RADAR_CLEAN = clean_with_ndimage(np.nanmax(RADARs_SIN,axis=2),Thr=0.1,S_area=100)

    plt.subplot(2,1,1)
    plt.pcolormesh(RADAR_CLEAN,norm=colors.LogNorm())
    plt.subplot(2,1,2)
    plt.pcolormesh(np.nanmax(RADARs_SIN,axis=2),norm=colors.LogNorm())
    plt.show()