import hashlib
import re
from datetime import datetime

def md5(fname):
    """
    Creates md5 checksum from file
    """
    hash_md5 = hashlib.md5()

    with open(fname, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def get_date(regex, fname):
    """
    Extracts date from file name using regex
    """
    ex = re.compile(regex)
    match = re.search(ex, fname)
    date = match.group()
    return date


def get_hemi(fname):
    """
    Extracts hemisphere from file name
    """
    if '_nh_' in fname:
        return 'nh'
    if '_sh_' in fname:
        return 'sh'
    return ''


def valid_date(filename, config):
    file_date = get_date(config['regex'], filename)

    start = config['start'][:8]
    end = str(datetime.now)[
        :8] if config['end'] == 'NOW' else config['end'][:8]

    if file_date >= start and file_date <= end:
        return True
    return False