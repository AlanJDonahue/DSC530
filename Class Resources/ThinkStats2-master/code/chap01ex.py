"""This file contains code for use with "Think Stats",
by Allen B. Downey, available from greenteapress.com

Copyright 2014 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

import numpy as np
import sys

import nsfg
import thinkstats2

def readFemResp(dct_file='2002FemResp.dct', dat_file='2002FemResp.dat.gz'):
    """
    Reads the files

    Parameters
    ----------
    dct_file : file name
        DESCRIPTION. The default is '2002FemResp.dct'.
    dat_file : file name
        DESCRIPTION. The default is '2002FemResp.dat.gz'.

    Returns a dataframe
    ------
    """
    dct = thinkstats2.ReadStataDct(dct_file)
    df = dct.ReadFixedWidth(dat_file, compression = 'gzip')
    #CleanFemResp(df)
    return df

def ValidatePregnum(resp):
    """Validate pregnum in the respondent file.

    resp: respondent DataFrame
    """
    preg = nsfg.ReadFemPreg()
    
    # make the map from caseid to list of pregnancy indices
    preg_map = nsfg.MakePregMap(preg)
    
    # iterate through the respondent pregnum series
    for index, pregnum in resp.pregnum.iteritems():
        caseid = resp.caseid[index]
        indices = preg_map[caseid]

        # check that pregnum from the respondent file equals
        # the number of records in the pregnancy file
        if len(indices) != pregnum:
            print(caseid, len(indices), pregnum)
            return False

    return True




def main(script):
    
    respond = readFemResp()
    
    if len(respond) != 7643:
        print("The length of respond doesn't match")
    elif respond.pregnum.value_counts()[1] != 1267:
        print("The number of 2nd pregnancies don't match")
    elif ValidatePregnum(respond) != True:
        print("Pregnum is not valid.")
    else:
        print('%s: All tests passed.' % script)


if __name__ == '__main__':
    main(*sys.argv)
