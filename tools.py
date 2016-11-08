import pandas as pd
import numpy as np
def calrate(toe = None, binbeg = None, binend = None, dout = 'rate'):
    ''' Calculate firing rates and/or spike counts from action potential time of events (toe) data. 
    Calculation is done in time intervals defined by corresponding elements in [binbeg, binend)
    Keyword arguments:
    toe -- time of events in microsecond in a list
    binbeg -- numpy array of bin begin time, unit is millisecond
    binend -- numpy array of bin end time, same length as bin beg, unit in millisecond
    dout -- data returned,'rate'=firing rate, 'spkcnt'=spike count
    Returns:
    spkdata -- pandas Series of firing rate or spike count data 
    ''' 
    import bisect
    # process input parameters
    if toe is None or binbeg is None or binend is None:
        return None
    # check that length of binbeg and binend are the same
    if len(binbeg) != len(binend):
        print('binbeg and binend needs to be the same length')
        return None
    # cast the type of toe from list to numpy array
    toe = np.array(toe)
    # store time bin information    
    binbeg_ms = binbeg
    binend_ms = binend
    # change time bin unit to microsecond
    binbeg = binbeg*1000
    binend = binend*1000
    # obtain spike count and/or firing rate information
    if dout == 'spkcnt':
        spkdata = pd.Series([bisect.bisect_left(toe, x2)- bisect.bisect_left(toe,x1) \
                                for (x1, x2) in zip(binbeg, binend)], index = [binbeg_ms, binend_ms])
    if dout == 'rate':
        spkdata = pd.Series([(bisect.bisect_left(toe,x2)- bisect.bisect_left(toe,x1))\
                                    *10**6/(x2-x1) for (x1, x2) in zip(binbeg, binend)],\
                           index = [binbeg_ms, binend_ms])  
    spkdata.index.name = ['binbeg_ms', 'binend_ms']
    return spkdata

def calratetabel(data, binbeg, binend, dout='rate'):
    ''' Calculate firing rates for all rows of spike timing data in a DataFrame 
    Keyword arguments:
    data -- pandas dataframe, spike timing data
    binbeg -- begining time of bins to calculate firing rates
    binend -- end time of bins to calculate firing rates
    dout -- type of data to return, spkcnt = spike count, rate = firing rate
    Returns:
    spkrate -- dataframe of firing rate data
    '''    
    spkrate = []
    index = data.set_index(['unitid', 'stimulusid', 'repetition']).index
    for (ind,v) in data.iterrows():    
        spkrate.append(calrate(v['timeofevents'], binbeg,binend,dout=dout))
    spkrate = pd.DataFrame(spkrate, index = index)
    spkrate.columns.names=['binbeg', 'binend']
    return spkrate

def genbins(t_range = None, binwid = None, stepsize = None):
    ''' Generate time bins based on given parameters.
    Keyword arguments:
    t_range -- a list defining the time range to generate the time bins
    binwid -- width of each bin
    stepsize -- displacement between adjacent bins; if None, will be set to the same value as binwid.
    
    Returns:
    binbeg -- an array containing begin time of all time bins
    binend -- an array containing begin time of all time bins
    '''
    if t_range and binwid:
        if not stepsize:
            stepsize = binwid
    tbeg, tend = min(t_range), max(t_range)
    dtail = 1
    binend = np.arange(tbeg+binwid, tend+dtail, stepsize)
    binbeg = binend - binwid
    return binbeg, binend

def rasterplot(data, t_range = [0,1500], figsize=(7, 7), s_range=None,
               xlabel='Time (ms)', ylabel='Trial #', ax=None):
    ''' Make the raster plot of neural responses.
    Keyword arguments:
    data -- a dataframe of neural response data
    t_range -- time range to plot (ms)
    figsize -- figure size
    s_range -- time range for sound stimulus, if given, the time region will be covered by a colored mask
    xlabel -- xlabel of the plot
    ylabel -- ylabel of the plot
    ax -- axes to make the plot
    '''
    import matplotlib.pyplot as plt
    if ax is None:
        # make a new figure
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    stimidlast = None
    ylast = None
    tmin, tmax = min(t_range), max(t_range)
    ax.set_xlim([tmin,tmax])
    ax.set_ylim([data.index.values.min(), data.index.values.max()])
    if s_range:
        ax.axvspan(min(s_range), max(s_range), facecolor='r', alpha=0.3)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=14)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=14)
    for y, s in data.iterrows():
        # get information of current row
        stimidnow = s['stimulusid']
        ynow = y
        tspks = np.array(s['timeofevents'])
        # plot the current row
        ax.scatter(tspks/1000, [y]*len(tspks), s=8, c='k', marker='o', edgecolor='none')
        # determine whether to plot stimulus line separator
        if stimidlast is not None:
            if stimidnow != stimidlast: # plot when change in stimulus id happens
                yline = (ynow+ylast)/2
                ax.plot([tmin, tmax], [yline,yline], 'k', linewidth=1)
        stimidlast = stimidnow
        ylast = ynow    
    return
def calmeanrate(spkrate):
    ''' Calculate mean firing rates from single-repetition firing rate datas.
    Keyword arguments:
    spkrate -- single-repetition firing rate 
    Returns:
    meanrate -- a dataframe containing mean rates to each stimulus for each unit
    '''
    # drop repetition label
    meanratemid = spkrate.reset_index(level = 'repetition', drop = 'True')
    # get the mean firing rate for each stimulus and unit pair
    meanrate = meanratemid.groupby(meanratemid.index).mean()
    # split out stimulus and unit pair from tuple to individual column
    meanrate.reset_index(inplace = True)
    inddf = pd.DataFrame([[x[0],x[1]] for x in meanrate['index']])
    meanrate.insert(0, 'stimulusid', inddf[1])
    meanrate.insert(0, 'unitid', inddf[0])
    meanrate.drop('index', level = 0,axis = 1, inplace = True)
    return meanrate

def plotmeanrate(mrdata, figsize=(7, 7), s_range=None,
               xlabel='Time (ms)', ylabel='Firing rate (spikes/s)', ax=None):
    ''' Plot mean rate of a unit. 
    Keyword arguments:
    mrdata -- dataframe containing mean rate of a unit
    figsize -- figure size
    s_range -- time range for sound stimulus, if given, the time region will be covered by a colored mask
    xlabel -- xlabel of the plot
    ylabel -- ylabel of the plot
    ax -- axes to make the plot
    '''
    import matplotlib.pyplot as plt   
    # prepare data
    data = mrdata.drop('unitid', axis=1, level=0)
    data.set_index('stimulusid', inplace = True)
    t = [(int(x[0])+ int(x[1]))/2 for x in data.columns.values]
    nstims = data.shape[0]
    ylim = [data.values.min(), data.values.max()]
    # determine axes
    if ax is None:
        ax = []
        # make a new figure
        fig = plt.figure(figsize=figsize)
        for i in range(nstims):
            ax.append(fig.add_subplot(nstims, 1, i+1))
    # make the plots
    i=nstims-1
    for ind, v in data.iterrows():        
        if type(ax) is list:
            axnow = ax[i]           
        else:
            axnow = ax
        axnow.plot(t, v)   
        axnow.set_xlim([min(t),max(t)])
        axnow.set_ylim(ylim)
        yticks = axnow.get_yticks()
        if len(yticks)>=6:
            axnow.set_yticks(yticks[::2])
        if s_range:
            axnow.axvspan(min(s_range), max(s_range), facecolor='r', alpha=0.3)
        if i == nstims-1:
            if xlabel:
                axnow.set_xlabel(xlabel, fontsize=14)
        else: 
            axnow.set_xticklabels('')
        if i >= nstims/2 and i< nstims/2+1:
            if ylabel:
                axnow.set_ylabel(ylabel, fontsize=14)
        i-=1
    return
    
def neutraj(spkrate, npc=3):
    ''' Calcualte neural response trajectory with PCA
    Keyword arguments:
    spkrate -- firing rate of data
    npc -- number of principal components to return
    Returns:
    pcdf -- principal components of firing rate data in a DataFrame
    '''    
    import sklearn.decomposition  
    meanrate = calmeanrate(spkrate)
    # stack stimulusid over to column
    meanrate.set_index(['stimulusid', 'unitid'], inplace = True)
    meanrate = meanrate.unstack(0) 
    meanrate.columns.names = ['binbeg','binend','stimulusid']
    
    # perform PCA
    pca = sklearn.decomposition.PCA(n_components = npc)
    pc = pca.fit_transform(meanrate.T.values)
    pcdf = pd.DataFrame(pc, index = meanrate.columns)
    pcdf = pcdf.swaplevel('binend','stimulusid',axis=0)
    pcdf = pcdf.swaplevel('binbeg','stimulusid',axis=0)
    pcdf = pcdf.unstack(0) 
    pcdf.columns.names = ['PC#', 'stimulusid']
    return pcdf

def plotneutraj(data, figsize=(7, 7)):
    ''' Plot neural response trajectory
    Keyword arguments:
    data -- neural response data to plot
    '''
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    stims = [1,5,9,13,17,21]
    freqs = ['sound 1', 'sound 2', 'sound 3', 'sound 4', 'sound 5', 'sound 6']
    for s in range(len(stims)):
        lineList = ax.plot(data[0,stims[s]].values, data[1,stims[s]].values,
                           data[2,stims[s]].values, label=freqs[s])
        ax.scatter(data[0,stims[s]].values[0], data[1,stims[s]].values[0],
                   data[2,stims[s]].values[0], s=20, c=lineList[0].get_color(),
                   marker='o', edgecolor='none')
    ax.set_xlabel('c1', fontsize = 14)
    ax.set_ylabel('c2', fontsize = 14)
    ax.set_zlabel('c3', fontsize = 14)
    
def tmpmatchtrain(xtrain, ytrain):
    ''' Train a template-match based classifier. 
    Keyword arguments:
    xtrain -- pandas DataFrame of training data
    ytrain -- label of training data
    Returns:
    xtemplate -- pandas DataFrame with template of each class
    ytemplate -- label of each class
    '''
    # take the training labels as the index of the DataFrame
    xtrain.index = ytrain
    # get the mean of each group
    xtemplate = xtrain.groupby(xtrain.index).mean()
    ytemplate = xtemplate.index
    return xtemplate, ytemplate

def tmpmatchtest(xtemplate, ytemplate, xtest, ytest, dist = 'Euclidean'):
    ''' Test a template-match based classifer.
    Keyword arguments:
    xtemplate -- class templates given as pandas DataFrame
    ytemplate -- label of class templates
    xtest -- test data given in pandas DataFrame
    ytest -- label of test data
    dist -- string defining distance metric
    Returns:
    ypred -- predicted test labels 
    accuracy -- percentage correct classification
    '''
    from scipy.spatial.distance import cdist
    
    distmat = pd.DataFrame(cdist(xtest.values, xtemplate.values, metric=dist),\
                          index = xtest.index, columns = ytemplate)
    ypred = distmat.idxmin(axis=1)
    accuracy = sum(ypred.values == ytest.values)/len(ypred.values)
    
    return ypred, accuracy

def neuraldecode(spkrate, testsize = 6, numevals = 10):
    ''' Performs the neural decoding analysis at different time points.
    Keyword arguments: 
    spkrate -- pandas DataFrame containing the dataset
    test_size -- the number of test samples 
    numevals -- the number of times to evaluate the decoding (cross validations)
    Returns: 
    accuracy -- pandas Series of average decoding accuracy over multiple evaluations
    '''
    import sklearn.cross_validation
    
    accraw = np.empty([numevals, spkrate.shape[1]])
    j=0
    for colname, x in spkrate.iteritems():
        x = x.unstack(0) # reshape the matrix so that each unit becomes a feature
        x.fillna(method = 'bfill', inplace = True)
        overrepon, overrepoff = (21,26)
        x = x.drop(np.arange(overrepon, overrepoff, 1),level='repetition')
        x.reset_index(level = 'repetition', drop = 'True', inplace = 'True')
        for i in range(numevals): # evaluate the decoding multiple times for cross validation
            xtrain, xtest, ytrain, ytest = sklearn.cross_validation.train_test_split\
            (x, x.index, test_size=testsize,stratify=x.index)
            xtmp, ytmp = tmpmatchtrain(xtrain, ytrain) # train the classifier
            ypred, acc = tmpmatchtest(xtmp, ytmp, xtest, ytest, dist = 'Euclidean') # test
            accraw[i,j] = acc
        j += 1
    accuracy = pd.Series(accraw.mean(axis=0), index = spkrate.columns)
    accuracy.index.names = ['binbeg', 'binend']
    accuracy.name = 'accuracy'
    return accuracy 
def plotaccuracy(accuracy, figsize=(8, 5), s_range=None, chance_level = None):
    ''' Plot prediction accuracy
    '''
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    acc = accuracy.reset_index(['binbeg', 'binend'])
    t = (acc['binbeg']+acc['binend'])/2
    ax.plot(t,acc['accuracy'])
    if chance_level:
        ax.plot([min(t), max(t)], [chance_level, chance_level], '--',c='k')
    ax.set_xlim([min(t), max(t)])
    ax.set_ylim([0,1])
    ax.set_xlabel('Time (ms)', fontsize=14)
    ax.set_ylabel('Prediction accuracy', fontsize=14)
    if s_range:
        ax.axvspan(min(s_range), max(s_range), facecolor='r', alpha=0.3)
