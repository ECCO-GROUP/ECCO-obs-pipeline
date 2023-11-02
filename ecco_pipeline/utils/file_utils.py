import hashlib
import re
from datetime import datetime
import logging


def md5(fname: str) -> str:
    """
    Creates md5 checksum from file
    """
    hash_md5 = hashlib.md5()

    with open(fname, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def get_date(regex: str, fname: str) -> str:
    """
    Extracts date from file name using regex
    """
    ex = re.compile(regex)
    match = re.search(ex, fname)
    date = match.group()
    return date


def get_hemi(fname: str) -> str:
    """
    Extracts hemisphere from file name
    """
    if '_nh_' in fname:
        return 'nh'
    if '_sh_' in fname:
        return 'sh'
    return ''


def valid_date(filename: str, config: dict) -> bool:
    """
    Determines if date in filename falls within start/end time bounds
    """
    file_date = get_date(config['filename_date_regex'], filename)

    start = config['start'][:8]
    end = str(datetime.now)[:8] if config['end'] == 'NOW' else config['end'][:8]

    if file_date >= start and file_date <= end:
        return True
    return False
