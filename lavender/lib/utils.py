#/usr/bin/env python3
"""Commonly used utilities

    Function    
    ---------------
    throw_error(msg):
        throw error and exit
    
    write_log(msg, lvl=20):
        write logging log to log file
    
    parse_tswildcard(tgt_time, wildcard):
        parse string with timestamp wildcard 
        to datetime object

"""
import numpy as np
import logging


def throw_error(msg):
    '''
    throw error and exit
    '''
    logging.error(msg)
    exit()

def write_log(msg, lvl=20):
    '''
    write logging log to log file
    level code:
        CRITICAL    50
        ERROR   40
        WARNING 30
        INFO    20
        DEBUG   10
        NOTSET  0
    '''

    logging.log(lvl, msg)

def parse_bitunits(str):
    '''
    parse bit units to bytes
    '''
    str=str.upper()
    if str.endswith('K'):
        return int(str[:-1])*1024
    elif str.endswith('M'):
        return int(str[:-1])*1024*1024
    elif str.endswith('B'):
        return int(str[:-1])*1024*1024*1024
    else:
        return int(str)

def parse_tswildcard(tgt_time, wildcard):
    '''
    parse string with timestamp wildcard to datetime object
    '''
    seg_str=wildcard.split('@')
    parsed_str=''
    for seg in seg_str:
        if seg.startswith('%'):
            parsed_str+=tgt_time.strftime(seg)
        else:
            parsed_str+=seg
    return parsed_str
