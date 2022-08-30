import logging
import os
import pickle

from pcse.util import get_user_home

PCSE_USER_HOME = os.path.join(get_user_home(), ".pcse")
METEO_CACHE_DIR = os.path.join(PCSE_USER_HOME, "meteo_cache")


def _get_cache_filename(latitude, longitude):
    """
    Constructs the filename used for cache files given latitude and longitude.
    The latitude and longitude is coded into the filename by truncating on
    0.1 degree. So the cache filename for a point with lat/lon 52.56/-124.78 will be:
    NASAPowerWeatherDataProvider_LAT00525_LON-1247.cache

    Args:
        latitude (float): Latitude
        longitude (float): Longitude

    Returns:
        str: cache_filename
    """

    cache_filename = "%s_LAT%05i_LON%05i.cache" % (
        "NASAPowerWeatherDataProvider",
        int(latitude * 10),
        int(longitude * 10),
    )

    return cache_filename


def _loadndump(cache_fname):
    """ Load cached file and save it to METEO_CACHE_DIR

    Args:
        cache_fname (str): cache filename

    Returns:
        bool: True if load and save successfully else False
    """

    file_load = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")
    file_dump = METEO_CACHE_DIR
    os.makedirs(file_dump, exist_ok=True)

    try:
        with open(os.path.join(file_load, cache_fname), "rb") as fp:
            (store, elevation, longitude, latitude, description, ETModel) = pickle.load(
                fp
            )
        msg = f"Load from {os.path.join(file_load, cache_fname)}"
        logger.debug(msg)
    except Exception as e:
        msg = f"Load fail due to {e}"
        logger.error(msg)
        return False

    try:
        with open(os.path.join(file_dump, cache_fname), "wb") as fp:
            dmp = (store, elevation, longitude, latitude, description, ETModel)
            pickle.dump(dmp, fp, pickle.HIGHEST_PROTOCOL)
        msg = f"Save to {os.path.join(file_dump, cache_fname)}"
        logger.debug(msg)
    except Exception as e:
        msg = f"Save fail due to {e}"
        logger.error(msg)
        return False

    return True


def _write_cache_file():
    """
    Writes the meteo data from NASA Power to a cache file.
    """
    cache_filename = _get_cache_filename(35, 128)

    try:
        _loadndump(cache_filename)

    except (IOError, EnvironmentError) as e:
        msg = "Failed to write cache to file '%s' due to: %s" % (cache_filename, e)
        logger.warning(msg)


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    logger.addHandler(console)
    _write_cache_file()
