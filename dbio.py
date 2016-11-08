# Functions for database input/output 
def sanitize(s):
    return s.strip('% \n')

def parseheader(fileobj):
    '''Parse the information in the header of a neural recording file. 
    
    Keyword arguments:
    fileobj -- stream object returned by the open() function
    
    Returns: 
    hdrdict -- a dictionary of parsed header information 
    '''
    # define constant relevant to the format of the header
    stimpos, attpos, freqpos = (1,3,7)
    
    # read the first line, as a confirmation of the encoding format of the file
    firstline = fileobj.readline() 
    if firstline != '%  START_HEADER\n':
        raise ValueError('Unexpected content on the first line:{}'.format(firstline))
    
    # process the rest of the header
    hdrdict = {}
    stimv = []
    for aline in fileobj:
        # determine whether the end of the header is reached
        if aline == '%  END_HEADER\n':
            hdrdict['stimparams'] = stimv
            return hdrdict
        # parse the content of the header
        if '=' in aline:
            key, value = (sanitize(x) for x in aline.split('=', 1))
            hdrdict[key] = value
        elif ':' in aline:
            key, value = (sanitize(x) for x in aline.split(':', 1))
        elif aline.startswith('%  Stimulus'):
            stimps = sanitize(aline).split()
            stimv.append((stimps[stimpos], stimps[attpos], stimps[freqpos]))

def parsedata(fileobj):
    '''Parse the (rest of the) data in a neural recording file (excluding header). 
    
    Keyword arguments:
    fileobj -- stream object returned by the open() function. 
    data_parser(fileobj) does not reset reader position in fileobj to top when processing.  
    
    Returns: 
    hdrdict -- a dictionary of parsed data information 
    '''
    datadict = {}
    for aline in fileobj:
        # determin whether it reached the end
        if aline == '% trial complete\n':
            return datadict
        item = [int(x) for x in sanitize(aline).split()]
        key = tuple(item[:3])
        if key in datadict:
            datadict[key].append(item[-1])
        else:
            datadict[key] = [item[-1]]
    return datadict

# prepare for data storage
def grabflist(matfile):
    ''' Get information for a list of neural recording files to store from a .mat file.
    
    Keyword arguments:
    matfile -- name of the .mat file that contains the file list information
    
    Return:
    flist -- dictionary containing file list information.
    '''    
    import scipy.io
    flist = {} 
    scipy.io.loadmat(matfile, flist) # read in data from the mat file
    try: # remove unnecessary field
        del flist['__header__']
        del flist['__globals__']
        del flist['__version__']
    except: pass
    return flist

def wrap_toedata(x):
    ''' Wrap time of event (toe) data in string format for storage in a database
    
    Keyword arguments:
    x -- time of event data as a list, each element is a number
    
    Returns:
    s -- a string of wrapped data in comma separated format
    '''
    s = ','.join(str(number) for number in x)
    return s
    
def prepdata(filename):
    ''' Prepare data of a neural recording file to be stored into a database.
    
    Keyword arguments:
    filename -- the name of file containing neural recording data
    
    Returns:
    dbdata -- a dictionary containing the prepared data to be stored
    '''
    # parse the file
    try: 
        with open(filename) as f:
            # parse the header
            hdr = parseheader(f)
            # parse the neural recording data (time of event data with labels)
            data = parsedata(f)
    except OSError:
        print('The file {} was not found.'.format(filename))
        return {}
    dbdata = {};
    # get relevant information from the header 
    dbdata['unitnumber'] = int(hdr.get('Unit Number'))
    dbdata['monkey'] = hdr.get('Experiment')
    dbdata['stimparams'] = hdr.get('stimparams')
    # get relevant information from the neural recording data 
    dbdata['chandata'] = {}
    posdict = {}
    posdict[hdr['D1']] = 0 # get the label positions of the parsed neural recording data
    posdict[hdr['D2']] = 1
    posdict[hdr['D3']] = 2
    dbdata['chandataformat'] = ['stimulusid', 'repetition', 'timeofevents']
    for (dkey, dval) in data.items():
        stimnumber = dkey[posdict['Stimulus Number']]
        repnumber = dkey[posdict['Stimulus Repetition']]
        chan = dkey[posdict['ET1 Channel Number']]
        if chan in dbdata['chandata']: # check whether channel exists in key
            dbdata['chandata'][chan].append((stimnumber, repnumber, wrap_toedata(dval))) 
        else:
            dbdata['chandata'][chan]=[(stimnumber, repnumber, wrap_toedata(dval))]
    return dbdata

# store data into database
def mapstimid(stimparams, stiminfodb):
    ''' map the stimulus number to stimulusid in the database 'auditorycortex'
    Keyword arguments:
    stimparams -- a list of tuples containing the stimulus information (stimnum, atten, freq) from file
    stiminfodb -- a list of tuples containing the stimulus information from the database
    
    Returns:
    stimid_map -- a dictionary with stimulus number in the file as key, stimid in the databse as value
    '''   
    # format the information into a dictionary
    stim_db_dict = {}
    for x in stiminfodb:
        stim_db_dict[(float(x[1]), float(x[2]))] = int(x[0])

    # format the stimparams information from the file
    stim_file_dict = {}
    for x in stimparams:
        stim_file_dict[(float(x[2]), float(x[1]))] = int(x[0])
        
    # form the mapping between stimulus number and stimulus id
    stimid_map = {}
    for k,v in stim_file_dict.items():
        stimid_map[v] = stim_db_dict[k]
    return stimid_map

# read data from database
def unwrap_toedata(toestr):
    ''' upwrap time of event (toe) data from the toe string read from the database
    
    Keyword arguments:
    toestr -- a string of toe data in the format of comma separated numbers
    Returns:
    toelist -- toe data in the format of a numpy array
    '''
    toelist = [int(x) for x in toestr.split(',')]
    return toelist

def readunit(monkey = None, filenum = None, chan = None, stimulusid = None, reconly = True):
    ''' Read data of the unit with given monkey, file number and channel from the database.
    Keyword arguments:
    monkey -- a string defining monkey name
    filenum -- a string or an integer defining file number
    chan -- a string or an integer defining channel number 
    stimulusid -- a list of stimulusids, if None, get all stimulusid available
    reconly -- a boolean type variable defining whether or not to only return recommended record
    
    Returns:
    unitinfo -- a pandas Series containing unit information
    data -- a pandas DataFrame containing data for the unit
    '''
    import mysql.connector
    import pandas as pd
    
    # process input parameters
    if monkey is None or filenum is None or chan is None:
        return None, None
    filenum = '{:04}'.format(int(filenum))
    chan = str(chan)
    
    # establish database connection
    cnx = mysql.connector.connect(user = 'root', database = 'auditorycortex')
    cursor = cnx.cursor()
    # get unit information
    getunitinfo = ("SELECT DISTINCT unitid, monkey, file, chan FROM unitinfo " 
                   "WHERE monkey =%s AND file=%s AND chan=%s")
    cursor.execute(getunitinfo, (monkey, filenum, chan))
    unitinfo = pd.Series(cursor.fetchall()[0], index = ['unitid','monkey','file','chan'])
    # get stimulus_id information
    if not stimulusid:
        cursor.execute("SELECT DISTINCT stimulusid FROM actionpotential")
        stimulusid = [x[0] for x in cursor.fetchall()]
    # get data    
    if reconly:
        getdata = ("SELECT unitid, stimulusid, repetition, timeofevents FROM actionpotential "
                  "WHERE unitid = %s AND recommended = %s AND stimulusid IN ({})"
                  .format(','.join(['%s']*len(stimulusid))))
        params = stimulusid[:]
        params.insert(0, True)
        params.insert(0, unitinfo['unitid'])
        cursor.execute(getdata, params)
    else:
        getdata = ("SELECT unitid, stimulusid, repetition, timeofevents FROM actionpotential "
                  "WHERE unitid = %s AND stimulusid IN ({})"
                   .format(','.join(['%s']*len(stimulusid))))
        params = stimulusid[:]
        params.insert(0, unitinfo['unitid'])
        cursor.execute(getdata, params)
    data = pd.DataFrame(cursor.fetchall(), columns=['unitid','stimulusid','repetition','timeofevents'])
    data['timeofevents'] = data['timeofevents'].map(unwrap_toedata)
    # close connection to the databse
    cursor.close()
    cnx.close()
    # return results
    return unitinfo, data

def readsetonsust():
    ''' Read in the dataset for the onset/sustained project from the database.
    
    Returns: 
    unitinfo -- a pandas DataFrame containing unit information
    data -- a pandas DataFrame 
    '''
    import mysql.connector
    import pandas as pd
    # establish database connection
    cnx = mysql.connector.connect(user = 'root', database = 'auditorycortex')
    cursor = cnx.cursor()
    # get unit_id for the dataset
    getunitid = ("SELECT DISTINCT unitid FROM actionpotential " 
                 "WHERE recommended = True")
    cursor.execute(getunitid)
    unitids = [x[0] for x in cursor.fetchall()]
    # get unit information
    getunitinfo = ("SELECT unitid, monkey, file, chan FROM unitinfo " 
                   "WHERE unitid in ({})".format(','.join(['%s']*len(unitids))))
    cursor.execute(getunitinfo, unitids)
    unitinfo = pd.DataFrame(cursor.fetchall(), columns=['unitid','monkey','file','chan'])
    # get data
    stimids = [1,5,9,13,17,21]
    getdata = ("SELECT unitid, stimulusid, repetition, timeofevents FROM actionpotential "
                  "WHERE recommended = True AND stimulusid IN ({})"\
               .format(','.join(['%s']*len(stimids))))
    cursor.execute(getdata, stimids)
    data = pd.DataFrame(cursor.fetchall(), columns=['unitid','stimulusid', 'repetition', 'timeofevents'])
    data['timeofevents'] = data['timeofevents'].map(unwrap_toedata)
    # close connection to the databse
    cursor.close()
    cnx.close()
    # return results
    return unitinfo, data

def storesetonsust():
    ''' This function stores data relevant to the project of onset/sustained responses to the database
    '''
    import os, mysql.connector
    # get the information of the files to process
    filelist = '/Users/wensheng/Dropbox/LabProjects/Proj_auditorycortex/datainfo_20150209_su.mat'
    flist = grabflist(filelist)
    nfiles = len(flist['chans']) # get the number of files to process
    # get the information of stimulus id from the database
    cnx = mysql.connector.connect(user = 'root', database = 'auditorycortex')
    cursor = cnx.cursor()
    cursor.execute("SELECT * FROM stimulusinfo")
    stiminfodb = cursor.fetchall()
    # define data insertion command
    addunit = ("INSERT INTO unitinfo"
               "(unitnumber, monkey, file, chan)"
               "VALUES (%s, %s, %s, %s)")
    addrecord = ("INSERT INTO actionpotential"
               "(unitid, stimulusid, repetition, timeofevents, recommended)"
               "VALUES (%s, %s, %s, %s, %s)")
    # process each file
    filepath = '/Users/wensheng/Dropbox/LabProjects/Proj_IntensFreq/ExistingData/'
    for i in range(nfiles):
        # grab basic file information
        monkey = str(flist['monks'][i])
        chan = int(flist['chans'][i])
        filenum = '{:04}'.format(int(flist['files'][i]))
        rep_recommend = flist['reps'][i]    
        # commpose file name
        foldername = 'Spikes_{}'.format(monkey)
        filename = monkey + filenum + '.dat'
        filename = os.path.join(filepath, foldername, filename)    
        # get the data from the file
        dbdata = prepdata(filename)  
        ind_stim = dbdata['chandataformat'].index('stimulusid') # get the index of each label
        ind_rep =  dbdata['chandataformat'].index('repetition')
        ind_toe =  dbdata['chandataformat'].index('timeofevents')
        # switch stimulus number to stimulus id in the database
        stimid_map = mapstimid(dbdata['stimparams'], stiminfodb)
        for channow, datanow in dbdata['chandata'].items():
            dataunit = (dbdata['unitnumber'], monkey, filenum, channow)
            cursor.execute(addunit, dataunit)
            cursor.execute("SELECT LAST_INSERT_ID()")
            unitid = cursor.fetchall()[0][0]
            for x in datanow:
                stimulusid = stimid_map[x[ind_stim]]
                repetition = x[ind_rep]
                # determine value for the field 'recommended'
                recommended = False
                if channow != chan:
                    recommended = False
                elif repetition in rep_recommend[0]:
                    recommended = True
                # store data record in table
                datarecord = (unitid, stimulusid, repetition, x[ind_toe], recommended)
                cursor.execute(addrecord, datarecord)   
    cnx.commit()
    cursor.close()
    cnx.close()
    return       