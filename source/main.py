import numpy as np
import glob
import cv2
from read_outputs_tamcmc import *
from subprocess import call
from scipy.ndimage import gaussian_filter
from function_rot import amplitude_ratio
from harvey import noise_harvey
#from matplotlib.patches import Rectangle
from misc import *
from show_whiskers import show_whiskers

def jpg2movie(dir_jpg, fps, file_out='output.avi', extension='jpg'):
    '''
        Make a movie from a sequence of jpg images. Images must be ordered by index so that 
        they appear in order by eg a command like ls. The best is to use unambiguous filenames:
        000.jpg, 001.jpg, ..., 999.jpg
        dir_jpg: directory that contains the jpg. You can add the root name of the file as well (eg '[dir]/image_')
        file_out (optional): name of the output file for the AVI. If not provided, it will be 'output.avi'
    '''
    img_array = []
    files=sorted(glob.glob(dir_jpg+'/*.'+extension))
    if files ==[]:
        print('Error: No file found that match the provided extension in the requested directory')
        print('       File Extension: ', extension)
        print('       Searched Directory: ', dir_jpg)
        exit()
    for filename in files:
        img = cv2.imread(filename)
        height, width, layers = img.shape
        img_array.append(img)
    size = (width,height)
    #
    out = cv2.VideoWriter(file_out,cv2.VideoWriter_fourcc(*'DIVX'), fps, size) 
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


def show_pdfs(samples, params_names, stats=[], vhighlight=None, file_out=None):
    '''
        From provided samples, construct and show all pdfs in a single jpg
        samples: Samples in a 2d array of dim (Nsamples, Nparams)
        stats: Statistical summary of the samples
        params_names: 1d array with names of the parameters. Of dim (Nparams)
        vhighlight (optional): If specified, must be a 2d array of dim (Nparams,3).
            It will highlight in red a specific value vhighlight[i,0] within the range of each samples[:,i]. 
            The term vhighlight[i,1] will be shown below (eg. The recurence of that value within the whole set of samples)
            vhighlight[i,2] contains the position within the sample of the (first) occurence of that value
            The values are shown in the pdf on top right.

    '''
    # -- Setup --
    confidence=[2.25,16,50,84,97.75]
    Nparams=len(samples[0,:])
    Ncols=int(np.ceil(np.sqrt(Nparams))) # This is to get a square matrix of pdfs
    Nrows=int(np.ceil(np.sqrt(Nparams)))
    while Ncols*Nrows - Nparams >= Ncols: # We readjust the number of rows by removing uncessary ones, if any.
        Nrows=Nrows-1
    binning=30
    intervals=[True,True]
    colors=['black', 'gray', 'darkgray']
    vhighlight_col='red'
    dpi_jpg=300
    # -----------
    if stats == []:
        Nconfidence=len(confidence)
        Nparams=len(samples[0,:])
        stats=np.zeros((Nparams, Nconfidence))
        for j in range(0, Nparams):
            stats[j,:]=make_stats(samples[:,j], confidence=confidence)
          
    #
	# Setting the ploting area
    fig, axes = plt.subplots(Nrows, Ncols, figsize=(12, 12), num=1, clear=True)
    #fig.suptitle('PDFs t = {0:d}'.format(vhighlight[i,2]))
    #
    col=0
    row=0
    print('Nparams = ', Nparams)
    print('Ncols = ', Ncols)
    print('Nrows = ', Nrows)
    for i in range(Nparams):
        print(i, '   ', params_names[i], '   ', stats[i,:])
        #print( ' (col, row) = (', col, ',', row, ')')
        axes[row,col].set_title(params_names[i])
        xvals, yvals=make_hist(samples[:,i], stats[i,:], ax=axes[row, col], intervals=intervals, binning=binning, color=colors, alpha=None)
        axes[row,col].axvline(vhighlight[i,0], linestyle='-', color=vhighlight_col)
        if np.median(samples[i,:] < 10):
            fmt='{0:0.4f}'
        else:
            fmt='{0:0.2f}'
        axes[row,col].text(0.96, 0.92, fmt.format(vhighlight[i,0]) , verticalalignment='bottom', horizontalalignment='right', transform=axes[row, col].transAxes, color=vhighlight_col, fontsize=10)#, fontweight='bold')
        #axes[row,col].text(0.95, 0.88, 'r = {0:d}'.format(int(vhighlight[i,1])) , verticalalignment='bottom', horizontalalignment='right', transform=axes[row, col].transAxes, color=vhighlight_col, fontsize=10)#, fontweight='bold')
        if col < Ncols-1:
            col=col+1
        else:
            row = row+1
            col=0
    #
    if file_out !=None:
        fig.savefig(file_out, dpi=dpi_jpg)
        print('Saved files with syntax: ', file_out)
        plt.close()
    else:
        plt.show()

def show_stats(lpdf, ppdf, lpdf_ref=None, ppdf_ref=None, indexes=[], range_samples=[], highlight_index=None, ax=None):
    '''
        Show the log-posterior and log-prior.
        lpdf: log-posterior. 1d array of dim (Nsamples)
        ppdf: log-prior. 1d array of dim (Nsamples)
        lpdf_ref: A reference value for the log-posterior (eg. at the median of the parameters)
        ppdf_ref: A reference value for the log-prior (eg. at the median of the parameters) 
        range_samples (optional): If specified, show only a specified range of samples. 
            If not specified, it will show the full range.
        ax (optionnal): If the plot zone is specified, use it. Otherwise, create a new plot
    '''
    if (range_samples !=[] and indexes == []) or (range_samples == [] and indexes !=[]):
        print('Error : You should provide both optional parameters indexes and range_samples OR none')
        print("        Current values in indexes: ", indexes)
        print("        Current values in range_samples: ", range_samples)
        exit()
    do_show=False
    if ax == None:
        fig, ax = plt.subplots(1, figsize=(12, 12))
        do_show=True
    ax2 = ax.twinx()
    if range_samples !=[] and indexes !=[]:
        pos=np.where(np.bitwise_and(indexes<=range_samples[0], indexes>=range_samples[1]))[0]
        lpdf_data=lpdf[pos]
        ppdf_data=ppdf[pos]
    else:
        lpdf_data=lpdf
        ppdf_data=ppdf
    x=np.linspace(0, len(lpdf_data)-1, len(lpdf_data))
    #
    if lpdf_data == [] or ppdf_data ==[]:
        print('Error : You cannot have empty arrays for lpdf_data and ppdf_data')
        exit()
    #
    ax.set_xlabel('reduced iteration')
    ax.set_ylabel('logPosterior', fontsize=7)
    ax2.set_ylabel('logPrior', fontsize=7)
    ax2.title.set_color('red')
    ax2.yaxis.label.set_color('red')
    ax2.tick_params(axis='y', colors='red', labelsize=7)
    ax2.spines['right'].set_color('red')
    ax.plot(x, lpdf_data, color='black', alpha=0.9, linewidth=0.7)
    ax2.plot(x, ppdf_data, color='red', alpha=0.75, linewidth=0.7)
    #ax2.set_ylim(np.min(ppdf)*0.85, np.max(ppdf)*1.2)
    if highlight_index !=None:
        ax.scatter(x[highlight_index], lpdf_data[highlight_index], marker='o', color='gray', s=75)
        ax2.scatter(x[highlight_index], ppdf_data[highlight_index], marker='o', color='orange', s=75)
        ax.text(0.87, 1.03, 'lpdf : {0:.2f}'.format(lpdf_data[highlight_index]) , verticalalignment='bottom', horizontalalignment='right', transform=ax.transAxes, color='black', fontsize=6, fontweight='bold')
        ax.text(1.00, 1.03, 'ppdf : {0:.2f}'.format(ppdf_data[highlight_index]) , verticalalignment='bottom', horizontalalignment='right', transform=ax.transAxes, color='red', fontsize=6, fontweight='bold')
        if lpdf_ref !=None:
            ax.text(0.18, 1.03, 'lpdf_ref : {0:.2f}'.format(lpdf_ref) , verticalalignment='bottom', horizontalalignment='right', transform=ax.transAxes, color='black', fontsize=6, fontweight='bold')
            ax.text(0.53, 1.03, '$\Delta(lpdf)$ : {0:.2f}'.format(lpdf_data[highlight_index] - lpdf_ref) , verticalalignment='bottom', horizontalalignment='right', transform=ax.transAxes, color='black', fontsize=6, fontweight='bold')
        if ppdf_ref != None:
            ax.text(0.35, 1.03, 'ppdf_ref : {0:.2f}'.format(ppdf_ref) , verticalalignment='bottom', horizontalalignment='right', transform=ax.transAxes, color='red', fontsize=6, fontweight='bold')
            ax.text(0.67, 1.03, '$\Delta(ppdf)$ : {0:.2f}'.format(ppdf_data[highlight_index] - ppdf_ref) , verticalalignment='bottom', horizontalalignment='right', transform=ax.transAxes, color='red', fontsize=6, fontweight='bold')
    else:
        ax.set_title('log Stats')
        ax.scatter(x[-1], lpdf_data[-1], marker='o', color='gray', s=250)
        ax2.scatter(x[-1], ppdf_data[-1], marker='o', color='orange', s=250)
    if do_show == True:
        plt.show()

def show_mixedmodes_diags(mixed_modes_dic, ax=None, show_Hl=True, show_Wl=True):
    '''
        This function shows some of the diagnostics of the mixed modes such
        mixed_modes_dic: sub-Dictionary read by read_output_tamcmc::read_paramsfile_getmodel() 
            and contained in the params.model when mixed modes were written by the model
        ax: The ploting zone
        show_H1: If true, show the (normalized to H0) height profiles in addition to the ksi function
        show_Wl: If true, show the (normalized to W) width profiles in addition to the ksi function [ NOT IMPLEMENTED ]
    '''
    from matplotlib.lines import Line2D
    if ax == None:
        fig, ax = plt.subplots(1, figsize=(12, 12))
    ymax=1.05
    ax.set_ylim(0,ymax)
    #ax.tick_params(left = False, right = False , labelleft = False , labelbottom = False, bottom = False) # Turn off ticks
    ax.tick_params(axis="y",direction="in", pad=-15, labelsize=6) # Puting them inside

    col_included=mixed_modes_dic["labels"].index("included?") # Indicate wether the frequency calculated by the ARMM was eventually included in the final model or rejected (out of fmin/fmax bounds)
    col_freq=mixed_modes_dic["labels"].index("fl1_asymptotic") # The asymptotic value as calculated by the ARMM
    col_spline=mixed_modes_dic["labels"].index("spline_corr") # A spline correction, if any. The final frequency is the sum of the asymptotic and of the spline
    col_ksi=mixed_modes_dic["labels"].index("ksi_pg") # The ksi function
    col_h1p=mixed_modes_dic["labels"].index("Hl1p_all") # The frequencies of Height for p-modes determined by interpolation of H(l=0) heights
    col_h1=mixed_modes_dic["labels"].index("Hl1_all") # The frequencies of Height for mixed modes
    col_wl1=mixed_modes_dic["labels"].index("Wl1_all")
    x=mixed_modes_dic["data"][:,col_freq] + mixed_modes_dic["data"][:,col_spline]
    y=mixed_modes_dic["data"][:, col_ksi]
    #
    pos_NOTincluded=np.where(mixed_modes_dic["data"][:,col_included] == 0)[0] # Detect when a frequency was not included in the final model to highlight it
    # The plot of ksi
    ax.plot(x,np.repeat(1, len(x)), color='red', linestyle='--', linewidth=0.8)
    ax.plot(x,y, color='red', marker="o", linewidth=1, markersize=4)
    # Some custom legend outside the ploting zone for the zeta function
    font_legend=6
    ydelta=0.05 # Define by how much we shift a legend
    ydelta_cpt=0
    line = Line2D([-0.145,-0.09], [0.98,0.98], linewidth=1, color='red', transform=ax.transAxes, linestyle='--')
    line.set_clip_on(False)
    ax.add_line(line)
    ax.text(-0.088, 0.98 - ydelta_cpt*ydelta, '$\zeta$', verticalalignment='center', horizontalalignment='left', transform=ax.transAxes, color='red', fontsize=font_legend, fontweight='bold')
    if show_Hl == True:
        # Some custom legend outside the ploting zone for the Height H1/H0
        ydelta_cpt=ydelta_cpt+1
        line = Line2D([-0.145,-0.09], [0.98-ydelta_cpt*ydelta,0.98-ydelta_cpt*ydelta], linewidth=1, color='black', transform=ax.transAxes, linestyle='--')
        line.set_clip_on(False)
        ax.add_line(line)
        ax.text(-0.088, 0.98-ydelta_cpt*ydelta, '$H_1/H_0$', verticalalignment='center', horizontalalignment='left', transform=ax.transAxes, color='black', fontsize=font_legend, fontweight='bold')
    ydelta_cpt=ydelta_cpt+1
    ax.text(-0.145, 0.98-ydelta_cpt*ydelta, 'p-mode', verticalalignment='center', horizontalalignment='left', transform=ax.transAxes, color='blue', fontsize=font_legend, fontweight='bold')
    ydelta_cpt=ydelta_cpt+1
    ax.text(-0.145, 0.98-ydelta_cpt*ydelta, 'g-mode', verticalalignment='center', horizontalalignment='left', transform=ax.transAxes, color='orange', fontsize=font_legend, fontweight='bold')
              
    if pos_NOTincluded != []:
        ax.plot(x[pos_NOTincluded], y[pos_NOTincluded], linestyle = 'None',marker="o", color='red', markersize=6)
    # Showing positions of p and g modes:
    for pm in mixed_modes_dic["nu_p"]:
        ax.plot([pm,pm], [0, ymax], linestyle='--', color='blue', linewidth=0.75)
    for gm in mixed_modes_dic["nu_g"]:
        ax.plot([gm,gm], [0, ymax], linestyle='--', color='orange', linewidth=0.75)  
    # Handling conditional plots
    if show_Hl == True:
        y=mixed_modes_dic["data"][:,col_h1]/mixed_modes_dic["data"][:,col_h1p]
        ax.plot(x, y, color='black', linestyle='--', linewidth=0.8, marker='o', markersize=3)
    
    #if show_Wl == True:
    #    y=mixed_modes_dic["data"][col_Wl1,:]
    #    ax.plot(x, y, color='black', linestyle='--')



def show_model(freq, spec, xmodel, model, params_dic, xmodel_ref=[], model_ref=[], params_dic_ref=[], freq_range=['Auto'], smooth_fact=None, ax=[], tag_modes_on_axis=False):
    '''
        Function that shows the model superimposed on data. 
        freq: x-values. Typically Frequencies
        spec: y-values. Typically Power in ppm^2/microHz
        xmodel and model: The x and y-axis for the model. Will be superimposed on (freq,spec)
        xmodel_ref and model_ref (optional): If given, show a reference model superimposed on (freq, spec) and on (xmodel, model). 
        params_dic: Full set of parameters, as read out from a params.model file created by getmodel. These are available structured 
                    so that it is easy to extract frequencies, heights, etc... Used for tagging the modes and for defining the visualisation
                    range in the Auto mode.
        stats_dic: Dictionary for the statistics for the Full set of parameters (in the 'raw' key) + some computed combination of those parameters in the 'extra' key (eg. Dnu_p for mixed modes models)
        params_dic_ref (optional): Full set of parameters for the reference model
        smooth_fact: factor of smoothing over the x-axis for the observational data. Given in unit of the x-axis (eg. microHz)
        ax (optional): If given, must be a list of 2 axes created by matplotlib. This will define the drawing zones for the model and for the diagnostic data
        tag_modes_on_axis: If True, put the frequencies of the modes in the x-axis instead of on the modes. Could give a clearer plot when (fmax-fmax)/Dnu is large
    '''
    # --- Temporary Variables ---
    Nmed= [-1, -1, -1, -1]

    # --- Setup ----
    do_show=False
    # Identify usefull columns and return error in case we do not find what we expect
    try:
        col_l=params_dic["modes"]["param_names"].index('degree')
        col_f=params_dic["modes"]["param_names"].index("freq")
        col_h=params_dic["modes"]["param_names"].index("H")
        col_w=params_dic["modes"]["param_names"].index("W")
        col_a1=params_dic["modes"]["param_names"].index("a1")
        col_inc=params_dic["modes"]["param_names"].index("inclination")
        #print(' col_l = ', col_l)
        #print(' col_f = ', col_f)
    except:
        print('Error: columns with degrees and/or with frequencies not identified in the params_dic within show_model()')
        print('       Expected (and used) keys in the dictionary: degree, freq, H, W, a1, inclination')
        print('       Check that your model use the correct names within the table created by getmodel')
        exit()
    # Linearize noise parameters so that later we can use the harvey calculation function to get the local noise
    if params_dic["noise"][0,0] == 0: # Case where the first Harvey is simply set to 0... no need to bother spending time to calculated it
        N=7
        i0=1
    else:
        i0=0
        N=10
    noise_params=np.zeros(N)
    c=0
    k=i0
    for i in range(N):
        #print(k,c)
        noise_params[i]=params_dic["noise"][k,c]
        if c >= 2:
            c=0
            k=k+1
        else:
            c=c+1    
    # Nmodes for l=0, 1, 2, 3 In the whole model
    pos_l=[np.where(params_dic["modes"]["params"][:,col_l] == 0)[0], np.where(params_dic["modes"]["params"][:,col_l] == 1)[0], 
        np.where(params_dic["modes"]["params"][:,col_l] == 2)[0], np.where(params_dic["modes"]["params"][:,col_l] == 3)[0]]
    Nmodes=[]
    Nmodes.append(len(pos_l[0])) 
    Nmodes.append(len(pos_l[1])) 
    Nmodes.append(len(pos_l[2])) 
    Nmodes.append(len(pos_l[3])) 
    # Calculate Dnu
    fl0=params_dic["modes"]["params"][pos_l[0],  col_f]   
    # Calculate Dnu
    #print('fl0 : ', fl0)
    #print('x   : ', np.linspace(0, len(fl0)-1, len(fl0)))
    xf=np.linspace(0, len(fl0)-1, len(fl0))
    fit,residu, rank, svd, rcond=np.polyfit(xf, fl0-np.min(fl0), 1, full=True)  # We compute the mean spacing through linear fiting.
    Dnu=fit[0]
    sigDnu=np.std(fl0 - np.min(fl0) - (fit[0]*xf + fit[1]))
    #print(Dnu, '  ',  sigDnu,  '  ', residu)
    # Define the ploting area
    if ax == []:
        fig, ax0 = plt.subplots(1, figsize=(12, 12))
        ax1=ax0
        do_show=True
    else:
        if len(ax) == 2:
            ax0=ax[0] # The zone for the model
            ax1=ax[1] # The zone for any text information (Dnu, DP, etc..)
        else:
            print('Error: ax is expected to contain two axis. ax0 and ax1')
            print('       show_model() cannot pursue')
            exit()
    if freq_range !=[]:
        if freq_range[0] == 'Auto': # Adjust the window to the min/max of the l=0 mode frequency minus/plus Dnu/2
            # Evaluate the automatic ranges
            freq_range= [np.min(fl0) - Dnu/2, np.max(fl0) + Dnu/2]
        pos=np.where(np.bitwise_and(freq>=freq_range[0], freq<=freq_range[1]))[0]
        x=freq[pos]
        y=spec[pos]
        pos=np.where(np.bitwise_and(xmodel>=freq_range[0], xmodel<=freq_range[1]))[0]
        xm=xmodel[pos]
        m=model[pos]
        # Number of modes within the specified shown window
        NWin=[]
        NWin.append(len(np.where(np.bitwise_and(params_dic["modes"]["params"][pos_l[0],  col_f] >=freq_range[0], params_dic["modes"]["params"][pos_l[0],  col_f]<=freq_range[1]))[0]))
        NWin.append(len(np.where(np.bitwise_and(params_dic["modes"]["params"][pos_l[1],  col_f] >=freq_range[0], params_dic["modes"]["params"][pos_l[1],  col_f]<=freq_range[1]))[0]))
        NWin.append(len(np.where(np.bitwise_and(params_dic["modes"]["params"][pos_l[2],  col_f] >=freq_range[0], params_dic["modes"]["params"][pos_l[2],  col_f]<=freq_range[1]))[0]))
        NWin.append(len(np.where(np.bitwise_and(params_dic["modes"]["params"][pos_l[3],  col_f] >=freq_range[0], params_dic["modes"]["params"][pos_l[3],  col_f]<=freq_range[1]))[0]))
    else:
        x=freq
        y=spec
        xm=xmodel
        m=model
    if smooth_fact !=None:
        if smooth_fact == 'Auto':
            # Use the average mode width within the show window to get an approximate ideal smooth factor
            f=params_dic["modes"]["params"][:,  col_f]
            w=params_dic["modes"]["params"][:,  col_w]
            pos=np.where(np.bitwise_and(f>=freq_range[0], f<=freq_range[1]))[0]
            smooth_fact=np.mean(w[pos])/4
        #
        resol=freq[1] - freq[0]
        sfactor=int(smooth_fact/resol) # Smooth over smooth_fact, in microHz
        y=gaussian_filter(y, sfactor, mode='mirror')
    # --- Create the main plots ---
    ymin=0.98*np.min(y) # used for the y-scale
    ymax=np.max(y) # used for the y-scale
    len_line_splitting=(ymax-ymin)*6/100 # Size of the lines that show the splitting for a given mode
    #
    ax0.set_xlabel('Frequency ('+r'$\mu$'+ 'Hz)', fontsize=7)
    ax0.set_ylabel('Power (ppm/'+r'$\mu$' + 'Hz)', fontsize=7)
    ax0.tick_params(axis='x', labelsize=7)
    ax0.tick_params(axis='y', labelsize=7)
    ax0.set_ylim(ymin, ymax)
    ax0.plot(x, y, color='gray', linewidth=0.8) # The data
    ax0.plot(xm, m, color='red', linewidth=0.8) # The model
    if xmodel_ref !=[] and model_ref !=[]:
        ax0.plot(xmodel_ref, model_ref, color='cyan', linewidth=0.6, linestyle='--', alpha=0.7) # The reference model, if specified
    ax0.set_xlim(np.min(x), np.max(x))
    # -------------------------------------------------
    # --- Deal with all of the text in the ax1 zone ---
    # -------------------------------------------------
    x0_txt=0.96
    delta_txt=0.125
    ax0.text(0.02, x0_txt, 's_fact : {0:.0f} nHz'.format(smooth_fact*1000) , verticalalignment='bottom', horizontalalignment='left', transform=ax0.transAxes, color='Black', fontsize=7)
    #ax1.text(0.94, x0_txt, 'smooth_fact : {0:.0f} nHz'.format(smooth_fact*1000) , verticalalignment='bottom', horizontalalignment='right', transform=ax1.transAxes, color='red', fontsize=6)
    #ax1.text(0.94, x0_txt-delta_txt, r'$\Delta\nu$ : {0:.3f} +/- {1:0.3f} $\mu$'.format(Dnu, sigDnu) + 'Hz', verticalalignment='bottom', horizontalalignment='right', transform=ax1.transAxes, color='red', fontsize=6)
    #  number of modes:
     # N: Number of mode, at: Median (Med.), Within Current Shown Window (Cur.Win), In Total in the Model (Cur.Tot.)
    #ax1.text(0.94, x0_txt-2*delta_txt, '$N_{l}$   Med.  Cur.Win.  Cur.Tot.', verticalalignment='bottom', horizontalalignment='right', transform=ax1.transAxes, color='blue', fontsize=6)
    #for k in range(len(Nmodes)): 
    #    ax1.text(0.94, x0_txt-(k+3)*delta_txt, '{0:2d}   {1:6}    {2:6}       {3:6}'.format(k, format_numbers(Nmed[k]), format_numbers(NWin[k]), format_numbers(Nmodes[k])), verticalalignment='bottom', horizontalalignment='right', transform=ax1.transAxes, color='blue', fontsize=6)    
    # -------------------------------------------------
    # --- Tagging of the modes ---
    # -------------------------------------------------
    Nrows=len(params_dic["modes"]["params"][:,  0])
    shift1_h=1.3
    shift2_h=0
    shift1_f=1.0
    shift2_f=0
    pc=['blue', 'red', 'purple', 'brown']
    count_l=[0,0,0,0] # Counter for l=0, ... , 3
    for n in range(0, Nrows):
        l=int(params_dic["modes"]["params"][n,  col_l])
        f=params_dic["modes"]["params"][n,  col_f]
        h=params_dic["modes"]["params"][n,  col_h]
        w=params_dic["modes"]["params"][n,  col_w]
        a1=params_dic["modes"]["params"][n,  col_a1]
        inc=params_dic["modes"]["params"][n,  col_inc]
        noise=noise_harvey(f, noise_params[0:-1])
        noise=noise + noise_params[-1]
        count_l[l]=count_l[l] + 1 # Mode by mode counter used for tagging them
        if f >= freq_range[0] and f<= freq_range[1]:
            hlm=amplitude_ratio(l, inc) # assume distribution of amplitudes according to the relation from Gizon between hlm and inclinations
            px=f*shift1_f + shift2_f
            py= h*np.max(hlm)*shift1_h + shift2_h + noise
            if py > ymax: # Readjusting if the text is out of the plot zone
                py=0.95*ymax
            if tag_modes_on_axis == False:
                ax0.text(px,py , '{0:.2f}'.format(f) , 
                    verticalalignment='bottom', horizontalalignment='center', color=pc[l], fontsize=4, rotation=90.)#, fontweight='bold')
            else:
                ax0.text(px,0 , '{0:.2f}'.format(f) , 
                    verticalalignment='top', horizontalalignment='center', color=pc[l], fontsize=4, rotation=75.)
            # Adding ticks proportional to the m-components heights at the bottom the plot and tagging them by index and m. 
            # eg. 1_{+} is the first mode (either l=0, l=1, l=2... using color code) with m=+1. ++ will means m=+2. - means m=-1
            for m in range(-l, l+1):
                sm=''
                is0_shift=0
                if m==0:
                    sm=count_l[l]
                    is0_shift=0.01*(ymax-ymin)
                if m >0:
                    for km in range(m):
                        sm=sm+'+'
                if m <0:
                    for km in range(np.abs(m)):
                        sm=sm+'-'
                px=f + m*a1
                py=ymin
                #py=h*hlm[m+l]*shift1_h + noise
                ax0.plot([px ,px], [py, py + len_line_splitting*hlm[m+l]], color=pc[l], linewidth=1.)
                ax0.text(px,py + len_line_splitting*hlm[m+l] + 0.01*(ymax-ymin) + is0_shift,  "{}".format(sm), 
                    verticalalignment='center', horizontalalignment='center', color=pc[l], fontsize=4)

        if inc < 20:
            ax0.text(0.97, 0.82, 'WARNING: inc < 20 deg' , verticalalignment='bottom', horizontalalignment='right', transform=ax0.transAxes, color='blue', fontsize=12, fontweight='bold')
    # ---------------------------
    if do_show == True:
        plt.show()

def show_Nmodes(iteration, params_dic, freq_range, Nmodes_all,  l_list=[0,1,2,3], pdf_only=False, ax=[], fileout=None):
    '''
        This function has for purpose to:
            - Summarize the number of modes present within the whole model and within a specified window
            - Show in the form of a time-plot and/or a pdf the distribution of that number of modes
        params_dic: The parameter dictionary for a specific mcmc model, as read with getmodel and returned with the python-mcmc bridge (read_output_tamcmc.py)
        freq_range: The range of frequencies that will be used to calculate the number of modes within a specified frequency window
        Nmodes_all: The dictionary that contains all of the number of modes within the model (key : "total") and within the window (key : "window")
                    that will be used to calculate the pdf and show the time-dependence (key : "iter") of the number of modes
                    WARNING: This dictionary is GLOBAL IN PYTHON: CHANGES ARE MADE AND REFELECTED GLOBALY
                    EG. I update here Nmodes_all['iter'] ['total'] and ['window'] by adding the current sample. 
        iteration: Current iteration number
        pdf_only: If True, shows only the pdf of the number of modes. If False (default), shows both the time-dependence and the pdfs (projection plot)
        l_list: List of degree that is shown. 
        ax: Define the ploting zone. If ax = None, a new figure is going to be created and shown on-screen (fileout = None) or written on-file (fileout = [name of a file])
        fileout : If ax is not provided, you can provide a filename where the image will be made.
    '''
    # --- Constants/Initialisation ---
    x0_txt=0.90
    delta_txt=0.125
    Nstats_min=5
    binning=10
    col_Ntot='blue'
    col_Nwin='red'
    do_show=False
    #
    Nmodes_iter=Nmodes_all["iter"] # Used to make the time-dependent plots. UPDATE OF Nmodes_all ALSO HAPPENS HEREAFTER
    Nmodes_total=Nmodes_all["total"] # Used to make the time-dependent plots and the pdf. UPDATE OF Nmodes_all ALSO HAPPENS HEREAFTER
    Nmodes_win=Nmodes_all["window"] # Used to make the time-dependent plots and the pdf. UPDATE OF Nmodes_all ALSO HAPPENS HEREAFTER
    col_l=params_dic["modes"]["param_names"].index('degree')
    col_f=params_dic["modes"]["param_names"].index("freq")
    # Nmodes for l=0, 1, 2, 3 In the whole model
    pos_l=[np.where(params_dic["modes"]["params"][:,col_l] == 0)[0], np.where(params_dic["modes"]["params"][:,col_l] == 1)[0], 
        np.where(params_dic["modes"]["params"][:,col_l] == 2)[0], np.where(params_dic["modes"]["params"][:,col_l] == 3)[0]]
    Nmodes=[]
    Nmodes.append(len(pos_l[0])) 
    Nmodes.append(len(pos_l[1])) 
    Nmodes.append(len(pos_l[2])) 
    Nmodes.append(len(pos_l[3])) 
    # Number of modes within the specified shown window
    NWin=[]
    NWin.append(len(np.where(np.bitwise_and(params_dic["modes"]["params"][pos_l[0],  col_f] >=freq_range[0], params_dic["modes"]["params"][pos_l[0],  col_f]<=freq_range[1]))[0]))
    NWin.append(len(np.where(np.bitwise_and(params_dic["modes"]["params"][pos_l[1],  col_f] >=freq_range[0], params_dic["modes"]["params"][pos_l[1],  col_f]<=freq_range[1]))[0]))
    NWin.append(len(np.where(np.bitwise_and(params_dic["modes"]["params"][pos_l[2],  col_f] >=freq_range[0], params_dic["modes"]["params"][pos_l[2],  col_f]<=freq_range[1]))[0]))
    NWin.append(len(np.where(np.bitwise_and(params_dic["modes"]["params"][pos_l[3],  col_f] >=freq_range[0], params_dic["modes"]["params"][pos_l[3],  col_f]<=freq_range[1]))[0]))
    if ax == []:
        if pdf_only == False:
            fig = plt.figure(layout=None, facecolor='0.9', num=1, clear=True)
            gs = fig.add_gridspec(nrows=1, ncols=4, left=0.1, right=0.90, hspace=0.05, wspace=0.1)
            ax0 = fig.add_subplot(gs[0, 0:3]) # ZONE FOR Time dependence
            ax1 = fig.add_subplot(gs[0, 3]) # ZONE For the projection
            ax1.tick_params(left = False, right = True , labelleft = False , labelright=True, labelbottom = True, bottom = True)
            ax0.tick_params(axis='y', labelsize=6)
            ax0.tick_params(axis='x', labelsize=6)
            ax1.tick_params(axis='x', labelsize=6)
            ax1.set_ylabel('Nmodes')
        else:
            fig, ax1 = plt.subplots(1, figsize=(12, 6), num=1, clear=True)
        do_show=True
    else:
        if pdf_only == False:
            ax0=ax[0]
            ax1=ax[1]
        else:
            ax1=ax[0]
    
    ax2=ax1.twiny() # duplicate axis to have two y-axis (based on two independent x-values)
    #ax1.set_xlabel('$N_{Total}($'+ 'l={})'.format(l) , color='blue', fontsize=6)
    #ax2.set_xlabel('$N_{Win}($'+ 'l={})'.format(l) , color='red', fontsize=6)
    ax1.yaxis.label.set_color(col_Ntot)
    ax2.yaxis.label.set_color(col_Nwin)
    ax1.tick_params(axis='y', labelsize=5, direction="in")
    ax1.tick_params(axis='x', colors=col_Ntot, labelsize=5, direction="in" , pad=-45) # pad=XX
    ax2.tick_params(axis='y', labelsize=5, direction="in")
    ax2.tick_params(axis='x', colors=col_Nwin, labelsize=5, direction="in", pad=1)
    Nmodes_iter.append(iteration)
    stats_total=[]
    stats_win=[]
    for el in range(len(l_list)):
        l=l_list[el]
        key="l="+str(l)
        Nmodes_total[key].append(Nmodes[l]) # We add the latest sample in the list to be able to show it
        Nmodes_win[key].append(NWin[l])
        Nmaxmodes_tot=np.max(Nmodes_total[key])
        Nmaxmodes_win=Nmaxmodes_tot #np.max(Nmodes_win[key])
        if pdf_only == False:
            ax0.plot(Nmodes_iter, Nmodes_total[key], label=key, linestyle='-', color=col_Ntot)
            ax0.plot(Nmodes_iter, Nmodes_win[key], label=key, linestyle='-', color=col_Nwin)
            if len(Nmodes_iter) > Nstats_min:
                stats_total.append(make_stats(Nmodes_total[key]))
                stats_win.append(make_stats(Nmodes_win[key]))
                yvals_tot, xvals_tot, patches=ax1.hist(Nmodes_total[key],linestyle='-', bins=binning,histtype='step', color=col_Ntot, density=False, orientation='vertical', align='left')
                yvals_win, xvals_win, patches=ax2.hist(Nmodes_win[key],linestyle='-', bins=binning,histtype='step', color=col_Nwin, density=False, orientation='vertical', align='left')
            else:
                ax1.text(0.5, 0.5, 'Getting stats...', transform=ax1.transAxes, color='red', fontsize=5)
        else:
            if len(Nmodes_iter) > Nstats_min:
                stats_total.append(make_stats(Nmodes_total[key]))
                stats_win.append(make_stats(Nmodes_win[key]))
                yvals_tot, xvals_tot, patches=ax1.hist(Nmodes_total[key],linestyle='-', bins=binning,histtype='step', color=col_Ntot, density=False, orientation='vertical', align='left')
                yvals_win, xvals_win, patches=ax2.hist(Nmodes_win[key],linestyle='-', bins=binning,histtype='step', color=col_Nwin, density=False, orientation='vertical', align='left')
            else:
                ax1.text(0.5, 0.5, 'Getting stats...', transform=ax1.transAxes, color='red', fontsize=5)           
    # Add a line for the current position
    el=1
    if len(Nmodes_iter) > Nstats_min:
        ymax=np.max([yvals_tot, yvals_win])
        xmin_tot=np.min(xvals_tot)
        xmax_tot=np.max(xvals_tot)
        xmin_win=np.min(xvals_win)
        xmax_win=np.max(xvals_win)    
    else:
        ymax=0
        xmin_tot=0
        xmax_tot=5
        xmin_win=0
        xmax_win=5
    ax1.set_xlim(xmin_tot - 1, xmax_tot + 1)
    ax2.set_xlim(xmin_win- 1, xmax_win + 1)
    ax1.set_ylim(0, ymax*1.2)
    ax2.set_ylim(0, ymax*1.2)
    ax1.plot([Nmodes[el], Nmodes[el]], [0,ymax*1.2], color=col_Ntot, linewidth=2, linestyle='--')
    ax2.plot([NWin[el], NWin[el]], [0,ymax*1.2], color=col_Nwin, linewidth=2, linestyle='--')
    
    ax1.text(0.02, x0_txt-2*delta_txt, 'Nt={}'.format(Nmodes[el]), verticalalignment='bottom', horizontalalignment='left', transform=ax1.transAxes, color=col_Ntot, fontsize=4)
    ax1.text(0.02, x0_txt-3*delta_txt, 'Nw={}'.format(NWin[el]), verticalalignment='bottom', horizontalalignment='left', transform=ax1.transAxes, color=col_Nwin, fontsize=4)
    
    if do_show == True:
        if fileout == None:
            plt.show()
        else:
            fig.savefig(fileout, dpi=300)
    Nmodes_current={"total":Nmodes, "window":NWin, "stats_total": stats_total, "stats_window": stats_win}
    return Nmodes_current

def make_model(model_name, params_model, plength, output_path='../tmp/model/', cpp_path='../cpp_prg/',  xrange_model=[0,5000,1000], get_params_only=False):
    '''
        Call getmodel of the TAMCMC program in order to build a specified model
        and with given parameters.
        params_model: parameters of the model. Must be consistent with expectation from model_name. This is therefore necessarily a full set of parameters
        model_name: name of the model.
        cpp_path: path where to find getmodel
        output_path: path for the outputs of getmodel
        get_params_only: If True, will set the resolution in xrange (the third term) to a dummy value just to get the computation done (quickly)
                         and will return only what is in the .model file. 
                         The .model file usually differs from the parameter vector because it calculates from the parameter vector
                         all of the frequencies, width, height, splitting and inclination, for each mode. This is particularly important
                         for mixed modes
    '''
    if get_params_only == True:
        xrange_model[3]=(xrange_model[1]-xrange_model[0])/5 # dummy resolution... to get fast computation while preserving any range-sensitive parameter
    # We perform a change of directory, so that we can be sure that getmodel will find properly the .list file within the cpp_path
    exec_path=os.getcwd()
    output_path=exec_path + '/' + output_path # Be sure that we account properly of the relative-to-execution path when determining the output path
    os.chdir(cpp_path)
    x, model, params_dic=getmodel_bin(model_name, params_model, plength, xrange_model, cpp_path='', outdir=output_path, read_output_params=True)
    # Going back to the working directory to avoid any problem later on
    os.chdir(exec_path)
    if get_params_only == True:
        return params_dic
    else:
        return x, model, params_dic

    
def gather_data(dir_mcmc, process_name, phase='A', chain=0, first_index=0, period=1):
    '''
        Function that collect all of the data from bin files created by the TAMCMC process
        and arange them in a suitable for for Python
        dir_mcmc: Directory that contains all of the mcmc results. This is similar to 'cfg_out_dir' in the TAMCMC config_presets.cfg
        process_name: Name of the process to compute the mcmc results. This will be the same as the name of .model file
        phase: Specify if we either read the 'B', 'L' or 'A' phase. See TAMCMC config_presets.cfg for more explanations
        chain: Specify which chain we read. 0 is the coolest chain. To know how many chains your analysis ran, refer to the TAMCMC config_default.cfg 
    '''
    cpp_path='../cpp_prg/'
    tmp_samples='../tmp/ascii/'
    #
    print('... Gather posterior samples and extract plength...')
    samples, labels, isfixed, plength=bin2txt(dir_mcmc, process_name, phase=phase, chain=chain, first_index=first_index, period=period, erase_tmp=True, cpp_path=cpp_path, outdir=tmp_samples, get_plength=True)
    #
    print('... Gather chain statistics...')
    ll, lp, lpdf=getstats_bin(dir_mcmc, process_name, phase=phase, chain=chain, first_index=first_index, period=period, erase_tmp=True, cpp_path=cpp_path, outdir=tmp_samples)

    return samples, labels, lp, lpdf, isfixed, plength


def make_frames(dir_frames, data, process_name, model_name, samples4pdf, samples_fullset, plength, labels4pdf, labels_fullset, lpdf, lp,  lpdf_med, lp_med, xmodel_med, model_med, stats_dic, xrange_model=['Auto']):
    '''
        The main function that creates frames in the form of a jpg files
        from the samples.
        dir_frames: output directory where frames are going to be written
        model_name: Name of the model that was used to perform the TAMCMC analysis. eg. model_RGB_asympt_a1etaa3_AppWidth_HarveyLike_v4
        samples4pdf: 2D array with samples of the parameters that are shown in the pdf plot. It should be a subset of samples_fullset in terms of parameters, but of same Nsamples value
        samples_fullset: 2D array with samples for all of the parameters of the model. It is used for computing the fitting model with the TAMCMC tool called getmodel
        plength: vector that allows to identify the type (eg. frequency) and number of the fullset of the parameter vector
        lpdf: log-posterior
        lp: log-prior
        lpdf_med: log-posterior for the median of the parameters
        lp_med: log-prior for the median of the parameters
        xmodel_med and model_med: The x and y-axis for the reference model (The median of the parameters)
        stats_dic: Statistical summary for all of the parameters in the 'raw' key and for any extra parameters (eg a combination of params) into the 'extra' key. 
            Labels are in the ['raw']['labels'] and in ['extra']['labels']
        labels4pdf: names of the parameters that are shown in the pdf plot
        xrange_model (optional): range and resolution on which the model are calculated using getmodel. If not given, use the Auto mode, that calculate the model for the whole data x-range
        This function form 3 series of frames:
            - The pdfs along the evolution of sampling over time for a set of parameters that are defined by the user (labels)
            - The log-posterior and log-prior evolution during the sampling with the model fit over the data, within a range specified by the user (xrange_model)
            - The evolution of the number of frequencies that are computed by the model and their distribution. In case of a MS model, this should be a constant.
              For mixed modes, it is possible that the number of modes included in the fit varies. This is then a diagnostic of that variation 
    '''
    get_params_only=False # This is required to get both the parameters and the model using getmodel
    if xrange_model[0] == 'Auto':
        xrange_model=[np.min(data[:,0]) , np.max(data[:,1])]
    i0=0
    imax=600
    step=5
    Nparams=len(samples4pdf[0,:])
    Dnu_p=[]
    epsilon_p=[]
    # Define a dictionary that will contain the number of modes in total in the model
    # and the number of modes within the model window as per specified by xrange_model
    Nmodes_all={"iter":[], "total":{"l=0":[], "l=1":[], "l=2":[], "l=3":[]}, "window":{"l=0":[], "l=1":[], "l=2":[], "l=3":[]}}
    for i in range(i0,imax, step):
        params_current=samples_fullset[i,:]
        #
        vhighlight_i=np.zeros((Nparams, 2))
        vhighlight_i[:,0]=samples4pdf[i,:]
        #file_out=dir_frames + '/pdfs/pdfs_' + str(i) + '.jpg'
        file_out=dir_frames + '/pdfs/pdfs_' + format_numbers(i, Ndigits=len(str(imax))) + '.jpg'
        show_pdfs(samples4pdf, labels4pdf, vhighlight=vhighlight_i, file_out=file_out)	
        xmodel, model, params_dic=make_model(model_name, samples_fullset[i,:], plength, output_path='../tmp/model/', cpp_path='../cpp_prg/',  xrange_model=xrange_model, get_params_only=get_params_only)
        #
        file_out=dir_frames + '/model/model_'+ format_numbers(i, Ndigits=len(str(imax))) + '.jpg'
        # Working Zone for all the plots
        fig = plt.figure(layout=None, facecolor='0.9', num=1, clear=True)
        gs = fig.add_gridspec(nrows=11, ncols=8, left=0.1, right=0.90, hspace=0.05, wspace=0.0)
        ax0 = fig.add_subplot(gs[0:2, :]) # ZONE FOR STATS
        ax0.tick_params(axis='y', labelsize=6)
        ax_spec = fig.add_subplot(gs[2:11, :-2]) # ZONE FOR SPECTRUM
        ax_text = fig.add_subplot(gs[2:4, -2:]) # ZONE FOR TEXT
        #ax_text.tick_params(left = False, right = False , labelleft = False , labelbottom = False, bottom = False)
        ax_text.spines['top'].set_visible(False)
        ax_text.spines['right'].set_visible(False)
        ax_text.spines['bottom'].set_visible(False)
        ax_text.spines['left'].set_visible(False)
        ax_whiskers_col1=[]
        ax_whiskers_col2=[]
        for k in range(4,11):  # ZONES FOR WHISKERS (DP1, alpha, q, ...). Up to 18 whiskers (9x2 area)
                ax_whiskers_col1.append(fig.add_subplot(gs[k, -2:-1]))
                ax_whiskers_col2.append(fig.add_subplot(gs[k, -1:]))
        for k in range(0,len(ax_whiskers_col1)): # Turn off labels and axes and remove frame
                ax_whiskers_col1[k].tick_params(left = False, right = False , labelleft = False , labelbottom = False, bottom = False)
                ax_whiskers_col2[k].tick_params(left = False, right = False , labelleft = False , labelbottom = False, bottom = False)
                ax_whiskers_col1[k].spines['top'].set_visible(False)
                ax_whiskers_col1[k].spines['right'].set_visible(False)
                ax_whiskers_col1[k].spines['bottom'].set_visible(False)
                ax_whiskers_col1[k].spines['left'].set_visible(False)
                ax_whiskers_col2[k].spines['top'].set_visible(False)
                ax_whiskers_col2[k].spines['right'].set_visible(False)
                ax_whiskers_col2[k].spines['bottom'].set_visible(False)
                #ax_whiskers_col2[i].spines['left'].set_visible(False)
        # Show the log priors and log posterior
        show_stats(lpdf[i0:-1], lp[i0:-1], lpdf_ref=lpdf_med, ppdf_ref=lp_med, indexes=[], range_samples=[], highlight_index=i, ax=ax0)
        # Show the data, the model and the reference model (using the median of the parameters)
        show_model(data[:,0], data[:,1], xmodel, model, params_dic, xmodel_ref=xmodel_med, model_ref=model_med, 
                freq_range=xrange_model, smooth_fact='Auto', ax=[ax_spec,ax_text], tag_modes_on_axis=True)
        # Show the mixed modes information such as the zeta function, the H1/H0 ratio and the nu_p and nu_g positions
        ax_mm_diags= ax_spec.twinx() # Left axis on the power spectrum to show the mixed modes diagnostics
        if params_dic["mixedmodes"] != []:
            show_mixedmodes_diags(params_dic["mixedmodes"], ax=ax_mm_diags, show_Hl=True, show_Wl=True)
            Dnu_p.append(params_dic["mixedmodes"]["Dnu_p"])
            epsilon_p.append(params_dic["mixedmodes"]["epsilon_p"])
        # Show a statistical summary in the form of whiskers plot. Also show estimates of the whiskers for Dnu_p and epsilon_p
        show_whiskers(params_current, stats_dic, labels_fullset, model_name, [ax_whiskers_col1, ax_whiskers_col2], Dnu_p=Dnu_p, epsilon_p=epsilon_p) # This section can handle up to 18 whisker data summary for DP1, alpha_g, delta01, etc...
        Nmodes_current=show_Nmodes(i,params_dic, xrange_model, Nmodes_all, ax=[ax_text], l_list=[1], pdf_only=True, fileout='test_Nmodes.jpg')
        #print(Nmodes_current)
        #print(Nmodes_all)
        #exit()
        fig.savefig(dir_frames + file_out, dpi=400)
        #plt.close()

def main(frame_cadence=1, dir_mcmc="", process_name="", model_name="", phase='A', chain=0, erase_frames=False):
    '''
        The main function that create a replay in video of the MCMC process.
    '''
    #------
    confidence=[2.25,16,50,84,97.75]
    dir_mcmc='../data/tests/mcmc_tests/asymp/'
    process_name='7268385_3'
    model_name='model_RGB_asympt_a1etaa3_AppWidth_HarveyLike_v4'
    phase='A'
    chain=0
    xrange_model=[180, 240, 'native'] # Native, means we use the data resolution. Otherwise, use the user-specified value
    lpdf_med=-10000
    lp_med=-60
    #------
    file_data=dir_mcmc + process_name + "/inputs_backup/data_backup.zip" # We use the backup data file in zip, which is saved by TAMCMC to make things easier
    dir_frames='../frames/'
    dir_mov='../movies/'
    period=50
    first_index=0
    indexes=[]
    names=[]
    names=['DP1', 'alpha_g', 'q']
    '''
    print('------------------------------------')
    print('0. Gathering data from binary files... ', dir_frames)
    print('------------------------------------')
    samples, labels, lp, lpdf, isfixed, plength=gather_data(dir_mcmc, process_name, phase=phase, chain=chain, first_index=first_index, period=period)
    data, hdr=read_datafile(file_data) # The input data file
    # 
    resol=data[1, 0]-data[0, 0]
    if xrange_model[2] == 'native':
        xrange_model[2]=resol
    print('--------------------------------------------------------------------')
    print('1. Apply selection rules for PDFs and extract statistical summary...')
    print('--------------------------------------------------------------------')
    samples_selected4pdf, samples_selected_fullset, lpdf_selected, lp_selected, labels_selected4pdf=select_params(samples, isfixed, lpdf, lp, labels, indexes=indexes, names=names, variable_only=True, do_reduce_samples=True)
	# Extract basic statistics from the samples
    #print("		Get stats using the samples of all of the parameters...")
    Nconfidence=len(confidence)
    Nparams=len(samples_selected_fullset[0,:])
    stats_selected_fullset=np.zeros((Nparams, Nconfidence))
    for j in range(0, Nparams):
        stats_selected_fullset[j,:]=make_stats(samples[:,j], confidence=confidence)
    stats_dic={'raw':{'labels':labels, 'stats':stats_selected_fullset}, 'extra':{'labels':[], 'stats':[]}}
    #
    print('------------------------------------')
    print('2. Creating frames in ', dir_frames)
    print('------------------------------------')
    #print('    a. Generating a reference model based on the median of the parameters...')
    #params_median=np.zeros(len(samples_selected_fullset[0,:]))
    #for i in range(len(samples_selected_fullset[0,:])):
    #    params_median[i]=np.median(samples_selected_fullset[:,i])
    params_median=stats_selected_fullset[:,2]
    xmodel_med, model_med, params_dic=make_model(model_name, params_median, plength,  xrange_model=xrange_model)
    make_frames(dir_frames, data, process_name, model_name, samples_selected4pdf, samples_selected_fullset, 
        plength, labels_selected4pdf, labels, lpdf_selected, lp_selected, lpdf_med, lp_med, xmodel_med, model_med, stats_dic, xrange_model=xrange_model)
    # I still need to:
    #    - Add mode tagging in the show_model() function
    #    - Test that within a sequence of images and see if everything is created properly
    #    - Will need to write some program that read the .model and extract the data_range... This is required when determining the list of frequencies generated within the range
    #    - Write the routines that will (1) extract the frequencies from the params_dic and (2) make images of the number of modes within the window of fit
    #    - Makes a test of movie
    '''
    print('------------------------------------')
    print('3. Converting frames into a movie...')
    print('------------------------------------')
    movie_file=dir_mov + '/models.avi'
    jpg2movie(dir_frames + '/model/', frame_cadence, file_out=movie_file)
    print('Output movie file for the models: ', movie_file )
	#
    movie_file=dir_mov + '/pdfs.avi'
    jpg2movie(dir_frames + '/pdfs/', frame_cadence, file_out=movie_file)
    print('Output movie file for the pdfs: ', movie_file )
	
    if erase_frames == True:
        print('Erasing of the temporary frames (jpg) requested...')
        call(["rm", dir_frames, '*.jpg'], stdout=PIPE, stderr=PIPE)
        print('Erasing complete')
            
