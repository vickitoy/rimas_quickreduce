import numpy as np
import pyfits as pf
import glob
import re
import time
import math
import matplotlib.pyplot as plt

def quickreduce(folder, refpixcorr=None, sampling='utr', writefile=False):
    '''
    NAME:
        quickreduce
    PURPOSE:
        Quick reduction of raw frames that can do reference pixel correction, crops the 
        reference pixels, can do Fowler and up-the-ramp sampling result frames, and can 
        save the result frame
    INPUTS:
        folder     - directory with fits files
        refpixcorr - how to correct reference pixels 
                     [None, 'col_allslope', 'col_medslope', 'col_med', 'row_med']
        sampling   - type of CMOS sampling ['utr', 'fowler']
        writefile  - boolean to write fits file after corrections
    OUTPUT:
        final_data - reference pixel corrected, reference pixel cropped, result data
    '''

    # Check in keywords are set appropriately
    if sampling not in ['utr','fowler']:
        sys.exit('Choose either utr or fowler sampling')

    if refpixcorr not in [None, 'col_allslope', 'col_medslope', 'col_med', 'row_med']:
        sys.exit("Reference pixel correction must be in [None, 'col_allslope', 'col_medslope', 'col_med', 'row_med']")

    # Finds fits files within given folder and human sorts them
    files = glob.glob(folder+'/*.fits')
    sort_nicely(files)

    # Dictionary of reference pixel correction functions
    options = {None: refpixnocorr,
           'col_allslope' : refpixcolallslope,
           'col_medslope' : refpixcolmedslope,
           'col_med'      : refpixcolmed,
           'row_med'      : refpixrowmed}
    
    # Read in each file and performs reference pixel correction, crops out reference pixels,
    # calculates the integration time of the frame, and stores the data and integration time
    keeptime = []
    store = np.zeros((2040,2040,len(files)))    
    for i,file in enumerate(files):   
        hdu = pf.open(file)
        header = hdu[0].header        
        data = hdu[0].data
        hdu.close()
    
        if i == len(files)-1:
            keepheader = header
            
        if 0 in data:
            print 'Caution: saturated pixels in %s' %(file)

        if 65535 in data:
            print 'Caution: some pixels out of range in %s' %(file)
        data = options[refpixcorr](data) 
          
        # Crop reference pixels
        data = data[4:2044,4:2044]
        
        tdiff = time.mktime(time.strptime(header['FRMDATE'],"%Y%m%d-%H%M%S"))-\
                time.mktime(time.strptime(header['DATE'],"%Y%m%d-%H%M%S"))
        
        keeptime.append(tdiff)        
        store[:,:,i] = data
    
    # UTR sampling
    keeptime = np.array(keeptime)
    if sampling == 'utr':
    
        # Make into 2-D array with all pixels in 1-d array and 2nd dimension is time
        store2 = store.T.reshape(len(keeptime),-1)
        
        # Take difference between each frame for each pixel and look at median of the 
        # entire array 
        diff = store2[1:,:] - store2[0:-1,:] 
        diffmed = np.median(diff, axis=1)
        
        # Find median and standard deviation of the median of each subtracted frame
        medall = np.median(diffmed)
        stdall = np.std(diffmed)
        
        # Only keep the frames that are within 1-sigma (to remove ramping frames)
        goodframe = np.where(abs(diffmed-medall) < stdall)[0]
        goodframe = np.append(goodframe,goodframe[-1]+1)
        print 'Good frames:', goodframe
        
        store2 = store2[goodframe]
        store = store[:,:,goodframe]
        keeptime = keeptime[goodframe]
        
        # Perform least squares regression on all pixels simultaneously
        A = np.vstack([keeptime, np.ones(len(keeptime))]).T
        regressions = np.linalg.lstsq(A, store2)[0]

        # Reshape regression output to match our data
        slope = regressions[0].reshape((2040,2040))
        slope = slope.transpose()
        
        intercept = regressions[1].reshape((2040,2040))
        intercept = intercept.transpose()

        #plt.plot(keeptime, store[500,500,:], 'o')
        #plt.plot(keeptime, np.array(keeptime)*slope[500,500] + intercept[500,500])
        #plt.show()
        
        # The accumulated counts are the slope of the fit multiplied by the total exposure time
        final_data = slope*(keeptime[-1]-keeptime[0])
        
        keepheader['EXPTIME'] = keeptime[-1]
        keepheader['MODE'] = 'UTR'        

    # Fowler sampling
    if sampling == 'fowler':
    
        # Subtract the median signal frame for each pixel from the median pedestal frame
        # for each pixel
        midpt =  int(math.floor(len(files)/2.))
    
        pedestal = np.median(store[:,:,:midpt],axis=2)
        signal   = np.median(store[:,:,midpt:],axis=2)
        final_data = signal-pedestal
    
        keepheader['EXPTIME'] = np.mean(keeptime[midpt:]) - np.mean(keeptime[:midpt])
        keepheader['MODE'] = 'Fowler'

    # Write fits file with same name except with RESULT instead of frame number
    if writefile:
        splitname = re.split('-',files[0])
        newfilename = splitname[0]+'_'+splitname[1]+'_RESULT.fits'
        pf.writeto(newfilename, final_data, keepheader, clobber=True)


    return final_data

###### Reference pixel corrections ######
def refpixnocorr(data):
    return data

def refpixcolmed(data):
    """
    NAME:
        refpixcolmed
    PURPOSE:
        Finds the median of the 8 reference pixels in the column and subtracts from data
    INPUT:
        data - data to use
    OURPUT:
        data - data with subtracted median column reference pixel 
    EXAMPLE:
        data = refpixcolmed(data)
    """

    # Takes top and bottom reference pixels for columns
    bottom = data[0:4,:]
    top    = data[2044:,:]
    
    allrefpix = np.vstack((bottom,top))    
    medrefpix = np.median(allrefpix, axis=0)
    
    medrefpix[0:4] = 0
    medrefpix[-4:] = 0
    
    data[4:-4,:] -= medrefpix
    
    return data

def refpixrowmed(data):
    """
    NAME:
        refpixrowmed
    PURPOSE:
        Finds the median of the 8 reference pixels in the row and subtracts from data
    INPUT:
        data - data to use
    OURPUT:
        data - data with subtracted median row reference pixel 
    EXAMPLE:
        data = refpixcolmed(data)
    NOTE: 
        Does not effectively remove ADC boundaries
    """

    # Takes top and bottom reference pixels for columns
    left  = data[:,0:4]
    right = data[:,2044:]

    allrefpix = np.hstack((left,right))  
    medrefpix = np.median(allrefpix, axis=1)
    
    medrefpix[0:4] = 0
    medrefpix[-4:] = 0
    
    data[:,4:-4] = (data[:,4:-4].transpose() - medrefpix).transpose()
        
    return data

def refpixcolallslope(data):
    """
    NAME:
        refpixcol
    PURPOSE:
        Fits linear fit to all reference pixels in column and subtracts from data
    INPUT:
        data - data to use
    OUTPUT:
        data - data with subtracted reference pixel columns removed
    EXAMPLE:
        test = refpixcolsub(data)
    """
    
    # Takes top and bottom reference pixels for columns
    bottom = data[0:4,4:2044]
    top    = data[2044:,4:2044]
    
    allrefpix = np.vstack((bottom,top))   

    x = [0,1,2,3,2044,2045,2046,2047]
    slope,intercept = np.polyfit(x, allrefpix, 1) 
    
    # Make array for all columns except first 4 and last 4 populated with slope
    # and intercept values
    col = np.arange(2048)
    dummy = np.transpose(np.tile(col,(2040,1)))
    subdata = dummy*slope + intercept

    # Subtract from data   
    data[:,4:2044] = data[:,4:2044] - subdata 
    
    pf.writeto('testslope.fits', data)
    
    return data
    
def refpixcolmedslope(data):
    """
    NAME:
        refpixcolmedslope
    PURPOSE:
        Fits linear fit to median of top and bottom reference pixels in column and subtracts from data
    INPUT:
        data - data to use
    OUTPUT:
        data - data with subtracted reference pixel columns removed
    EXAMPLE:
        test = refpixcolmedslope(data)
    """
    
    # Takes top and bottom reference pixels for columns
    bottom = data[0:4,4:2044]
    top    = data[2044:,4:2044]
    
    # Median along column (i.e. of 4 reference pixels)
    botmed = np.median(bottom, axis=0)
    topmed = np.median(top, axis=0)
    
    # Find slope and intercept using 1 value for top and 1 value for bottom
    slope = (topmed - botmed)/2044.0
    intercept = botmed
    
    # Make array for all columns except first 4 and last 4 populated with slope
    # and intercept values
    col = np.arange(2048)
    dummy = np.transpose(np.tile(col,(2040,1)))
    subdata = dummy*slope + intercept

    # Subtract from data   
    data[:,4:2044] = data[:,4:2044] - subdata 
    
    return data
    
############################

def tryint(s):
    """
    NAME:
        tryint
    PURPOSE:
        Turn string into int
    OUTPUT:
        If not possible to turn into an int, keep string
    """
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ 
    NAME:
        alphanum_key
    PURPOSE:
        Turn a string into a list of string and number chunks.
    INPUT:  
        s - string
    OUTPUT:
        list of string and number chunks
    EXAMPLE:
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    """ 
    NAME:
        sort_nicely
    PURPOSE:
        Sort the given list in the way that humans expect.
    INPUTS:
        l - list of files
    """
    l.sort(key=alphanum_key)
