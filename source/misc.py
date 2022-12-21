import numpy as np
import itertools
from scipy import interpolate
import matplotlib.pyplot as plt

def format_numbers(number, Ndigits=3, exit_on_error=True):
    '''
        Take a number or a string representing a number and add a number of 0 in front so that
        it has a fixed representation in terms of digits. Negative numbers will also
        be handled such as the string remains of Ndigits length
        The output is necessarily a string
        number: The number to be processed. Must be an int, float or a string
        Ndigits: The number of digits of the string. eg. '000','001.1' is for Ndigits=3
    '''
    passed=False
    if type(number) == int or type(number) == float :
        passed=True
        n=number
        sign=np.sign(n)
        n=np.abs(n)
        sn=str(n)
    if type(number) == str:
        passed=True
        n=float(number)
        sign=np.sign(n)
        n=np.abs(n)
        if n-int(n) == 0:
            sn=str(int(n))
        else:
            sn=str(n)
    if sign == -1:
        Nsign=1
    else:
        Nsign=0
    if passed == False:
        print('Error in format_numbers: The type of the input is not recognized.')
        print('    This function accepts only int,float and string')
        exit()
    Nzeros=0  
    while n >= 1: # Identify the power of the number
        n=n/10
        Nzeros=Nzeros+1
    if Nzeros <= Ndigits:
        str_n=''
        if n !=0:
            for j in range(0,Ndigits-Nzeros):
                str_n=str_n + '0'
            str_n=str_n + sn
        else:
            str_n = '000'
    else:
        print('Warning: Ndigits smaller than the number of digits within the number')
        if exit_on_error == False:
            print('         The initial number will be returned')
            str_n=str(sn)
        else:
            exit()
    ## Handle the sign:
    if sign == -1:
        str_n='-' + str_n
    return str_n

def make_stats(samples, confidence=[2.25,16,50,84,97.75]):
	N=len(samples)
	s=np.sort(samples)
	cdf = 100.*np.array(range(N))/float(N) # in %
	r=np.interp(confidence, cdf, s)
	return r

def reduce_samples(smcmc, lpdf, lp=[], verbose=None):
    '''
        look at the occurence of lpdf values (due to the fact that ~25% only are accepted)
        and return the occurence rate of each values along with reduced vectors smcmc_reduced and lpdf_reduced
        smcmc: samples of all the parameters that need to be reduce. 2d array of dim (Nsamples, Nparams)
        lpdf: log-posterior. 1d array of dim (Nsamples)
        lp (optional): log-prior. 1d array of dim (Nsamples)
        verbose if: 
            -  None, do not show any outputs
            - [value], print the sample at which we are at i modulo [value]
    '''
    if len(smcmc[:,0]) != len(lpdf):
        print(' WARNING: SIZE OF SMCMC AND LPDF NOT THE SAME')
        print('len(lpdf) = ', len(lpdf))
        print('len(smcmc) =' , len(smcmc[:,0]))
        print(' We will assume a minor bug that has yet to be fixed in getstats or bin2txt')
        print(' Where the mismatch is of 1 sample only')
        lpdf=lpdf[0:-1]    
        lpdf_tmp=lpdf
        if len(lp) != 0:
            lp=lp[0:-1]
            lp_tmp=lp
    else:
        lpdf_tmp=lpdf
    #
    smcmc_tmp=smcmc
    #
    lpdf_reduced=np.empty(len(lpdf))
    smcmc_reduced=np.empty([len(smcmc[:,0]),len(smcmc[0,:])])
    recur=np.empty(len(lpdf))
    #
    if len(lp) != 0:
        lp_tmp=lp
        lp_reduced=np.empty(len(lp))
	#
    i=0
    print("reducing arrays...")
    while len(lpdf_tmp) != 0: # process the elements
        pos_r=np.where(lpdf_tmp == lpdf_tmp[0])
        r=len(pos_r[0])
        lpdf_reduced[i]=lpdf_tmp[0]
        smcmc_reduced[i,:]=smcmc_tmp[0,:]
        recur[i]=r
        #
        lpdf_tmp=np.delete(lpdf_tmp, pos_r)
        smcmc_tmp=np.delete(smcmc_tmp, pos_r, axis=0)
        #
        if lp !=[]:
            lp_reduced[i]=lp_tmp[0]
            lp_tmp=np.delete(lp_tmp, pos_r)
        if i%verbose == 0:
            print("i=",i)
        i=i+1
	#
    pos_del=np.arange(len(lpdf) - i) + i
    lpdf_reduced=np.delete(lpdf_reduced, pos_del)
    lp_reduced=np.delete(lp_reduced, pos_del)
    smcmc_reduced=np.delete(smcmc_reduced, pos_del, axis=0)
    recur=np.delete(recur, pos_del)
	#
    if lp == []:
        return np.array(smcmc_reduced), np.array(lpdf_reduced), np.array(recur)
    return np.array(smcmc_reduced), np.array(lpdf_reduced), np.array(lp_reduced), np.array(recur)


def list_indexes_many(seq,item):
	'''
		Function that find the indexes of a given input within a list
		seq: A list of elements inside which we perform the search
		item: The element we look for within seq
		locs (output): The indexes at which 'item' was found inside 'seq'
	'''
	start_at = -1
	locs = []
	while True:
		try:
			loc = seq.index(item,start_at+1)
		except ValueError:
			break
		else:
			locs.append(loc)
			start_at = loc
	return locs

def select_params(samples, isfixed, lpdf, lp, labels, indexes=[], names=[], variable_only=True, do_reduce_samples=True):
    '''
        This function uses user-rules in order to define which pdf are shown on the jpg.
        The parameters must not be constants.
        samples: The full set of sampels as read by bin2txt
        isfixed: Boolean array specifying if the parameters are fixed of variable
        labels: The full set of labels for all of the parameters (variables + constants)
        indexes: A list of indexes for the parameters that need to be shown. CANOT BE USED WITH name
        names: A list of name for the parameters that need to be shown. See config_default.cfg (TAMCMC code) for a full list of valid names.
        variable_only: Keep only the variables
        reduce_samples: If true, only keep the samples that resulted from a move in the parameter space (no duplicate values)
    '''
    if do_reduce_samples == True:
        print('... Reduce the variables dimension by removing stationary moves...')
        samples_reduced, lpdf_reduced, lp_reduced, recurence=reduce_samples(samples, lpdf, lp=lp, verbose=len(samples[:,0]/10))
        samples_fullset=samples_reduced  # This is for the getmodel 
        lpdf=lpdf_reduced # This is the log-posterior
        lp=lp_reduced  # This is the log-prior
        print('   Remaining samples:', len(samples_fullset[:,0]))
    else:
        samples_fullset=samples  # This is for the getmodel        
    #
    if indexes !=[] and names !=[]:
        print(' Error: You are not allowed to use both the indexes and names option!')
        print('        Please use only one of these options')
        print('        The program will exit now')
        exit()
    if indexes !=[] and names ==[]:
        print('Selecting by index...', indexes)
        samples4pdf=samples_fullset[:, indexes]
        isfixed=np.array(isfixed)
        isfixed=isfixed[indexes]
        labels_new=[]
        for ind in indexes:
            labels_new.append(labels[ind])
        labels=labels_new
    if indexes ==[] and names !=[]:
        print('Selecting by name...')
        indexes=[]
        for n in names:
            ind = list_indexes_many(labels,n)
            if ind == []:
                print('Error: Could not find indexes for the parameter name ', n)
                print('       Check your spelling and if the name is in use wiht the fitting model')
                exit()
            indexes.append(ind)
        indexes=list(itertools.chain.from_iterable(indexes))
        samples4pdf=samples_fullset[:, indexes]
        isfixed=np.array(isfixed)
        isfixed=isfixed[indexes]
        labels_new=[]
        for ind in indexes:
            labels_new.append(labels[ind])
        labels=labels_new
    if indexes ==[] and names ==[]:
        print('     Warning: No indexes or names provided. ALL PARAMETERS WILL BE SELECTED FOR VISUALISATION')
        samples4pdf=samples_fullset
        names=labels
    #
    if variable_only == True:
        print(' .  ... Removing constants, if any...')
        posFalse=np.asarray(np.where(np.array(isfixed) == False)[0], dtype=int)
        posTrue=np.asarray(np.where(np.array(isfixed) == True)[0], dtype=int)
        if posTrue !=[]:
            constants=samples4pdf[:, posTrue]
            print('  Removed Constants from the set:')
            for p in posTrue:
                print(' ', labels[p],  '   : ', samples4pdf[0, p])
        else:
            print('  No constants in the sample set')
            constants=np.array([])    
        samples4pdf=samples4pdf[:, posFalse]
        labels_variables=[]
        labels_constants=[]
        for p in posFalse:
            labels_variables.append(labels[p])
        for p in posTrue:
            labels_constants.append(labels[p])
        names=labels_variables # Set the variables only as names, even if the user specifically used names
    return samples4pdf, samples_fullset, lpdf, lp, names


def make_hist(samples, stats, ax=None, intervals=[True,True], binning=30, color=['black', 'gray', 'darkgray'], alpha=None, rotate=False, max_norm=None):
	'''
		histograms with confidence intervals shown withtray areas
		samples: The data samples
		stats: Vector with [-2s, -1s, med, 1s, 2s]
		intervals: Bolean Vector [1s, 2s]:
			If 1s is True: The 1sigma interval is shown
			If 2s is True: The 2sigma interval is shown
		binning: Control the binning of the pdf
		color: Choose the set of color to show the (1) line of the pdf, (2) the 2sigma confidence interval and (3) the 1sigma confidence interval
		rotate: If set to true, invert the x and y axis 
		max_norm: Must be 'None' or any Real value. If it is a real value, the pdf will be normalised such that 'max_norm' is the maximum value of the pdf 
	'''
	if rotate == True:
		orientation='horizontal'
	else:
		orientation='vertical'
	if ax == None:
		fig_1d, ax = plt.subplots(1, figsize=(12, 6))
	if max_norm == None:
		yvals, xvals, patches=ax.hist(samples,linestyle='-', bins=binning,histtype='step', color=color[0], density=True, orientation=orientation)
	else:
		fig_1d, ax_dummy = plt.subplots(1, figsize=(12, 6))
		yvals, xvals = np.histogram(samples,bins=np.linspace(np.min(samples), np.max(samples), binning))
		yvals=yvals*max_norm/np.max(yvals)
		ax.plot(xvals[0:-1], yvals,linestyle='-', ds='steps-mid', color=color[0])	
    #		
	xvals=xvals[0:len(yvals)]
	pos_1sig_interval=np.where(np.bitwise_and(xvals >= stats[1], xvals<=stats[3]))
	pos_2sig_interval=np.where(np.bitwise_and(xvals >= stats[0], xvals<=stats[4]))
	if rotate == False:
		if intervals[1] == True:
				ax.fill_between(xvals[pos_2sig_interval],yvals[pos_2sig_interval], color=color[1], alpha=alpha, step='post', interpolate=True)
		if intervals[0] == True:
			ax.fill_between(xvals[pos_1sig_interval],yvals[pos_1sig_interval], color=color[2], alpha=alpha,  step='post', interpolate=True)
		f = interpolate.interp1d(xvals, yvals, kind='cubic', bounds_error=False, fill_value=stats[2])
		ax.plot([stats[2], stats[2]], [0, f(stats[2])], color=color[0], linestyle='--')
		ax.set_ylim(0, max(yvals)*1.2)
	else:
		if intervals[1] == True:
				ax.fill_betweenx(xvals[pos_2sig_interval],yvals[pos_2sig_interval], color=color[1], alpha=alpha, step='post', interpolate=True)
		if intervals[0] == True:
			ax.fill_betweenx(xvals[pos_1sig_interval],yvals[pos_1sig_interval], color=color[2], alpha=alpha,  step='post', interpolate=True)
		f = interpolate.interp1d(xvals, yvals, kind='cubic', bounds_error=False, fill_value=stats[2])
		ax.plot([0, f(stats[2])],[stats[2], stats[2]], color=color[0], linestyle='--')
		ax.set_xlim(max(yvals)*1.2, 0)
	return xvals, yvals
