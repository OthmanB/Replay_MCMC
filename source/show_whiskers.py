from whisker import Draw_BoxAndWiskers_horizontal
from misc import make_stats

def show_whiskers(params_current, stats_dic, labels, modelname, axes, Dnu_p=[], epsilon_p=[], confidence=[2.25, 16, 50, 84, 97.75]):
    '''
        Function that takes the axes area defined for the general diagnostic screen 
        and plot the whiskers of some key MCMC parameters within the 18 zone allocated zones
    '''
    # --- Handling supported models ---
    warning=False
    if modelname != "model_RGB_asympt_a1etaa3_AppWidth_HarveyLike_v4":
        warning=True
    #
    if warning == True:
        print('Warning: show_whiskers() is not configured to handle such a model!')
        print('         Provided model: ', modelname)
        print('         Whiskers plots will not be shown... requires support implementation')
        print('         Currently handled models:')
        print('         - model_RGB_asympt_a1etaa3_AppWidth_HarveyLike_v4')
    # --- Starting the ploting ---
    y0=0.65
    color=['red', 'blue', 'Green']
    symbol_extra=['o', 1]
    color_extra='black'
    linestyle_extra='-'
    show_extra_txt=False # We don't show the value on the whisker. We will show it next to the name of the variable
    #
    param_name='DP1'
    col=0 # either 0 or 1 as we have two columns
    i=0 # One of the 7 zones/col: i=[0,6]
    pos_current=labels.index(param_name)
    pos_stats=stats_dic["raw"]['labels'].index(param_name)
    axes[col][i].set_ylim(0,1)
    axes[col][i].text(0.5 ,0.88, param_name + ': {0:0.3f}'.format(params_current[pos_current]), verticalalignment='center', horizontalalignment='center', transform=axes[col][i].transAxes, color='blue', fontsize=4.5)
    extra=[[params_current[pos_current], color_extra, symbol_extra, linestyle_extra, show_extra_txt]]
    Draw_BoxAndWiskers_horizontal(y0, stats_dic['raw']["stats"][pos_stats, :], 
        extra=extra, color=color, fill=True, width=0.175, ax=axes[col][i], show_stats=True, fontsize=3.5, linewidth=0.7)
    #
    param_name='q'
    col=0 # either 0 or 1 as we have two columns
    i=1 # One of the 7 zones/col: i=[0,6]
    pos_current=labels.index(param_name)
    pos_stats=stats_dic["raw"]['labels'].index(param_name)
    axes[col][i].set_ylim(0,1)
    axes[col][i].text(0.5 ,0.88, param_name  + ': {0:0.3f}'.format(params_current[pos_current]), verticalalignment='center', horizontalalignment='center', transform=axes[col][i].transAxes, color='blue', fontsize=4.5)
    extra=[[params_current[pos_current], color_extra, symbol_extra, linestyle_extra, show_extra_txt]]
    Draw_BoxAndWiskers_horizontal(y0, stats_dic['raw']["stats"][pos_stats, :], 
        extra=extra, color=color, fill=True, width=0.175, ax=axes[col][i], show_stats=True, fontsize=3.5, linewidth=0.7)
    #
    param_name='alpha_g'
    col=0 # either 0 or 1 as we have two columns
    i=2 # One of the 7 zones/col: i=[0,6]
    pos_current=labels.index(param_name)
    pos_stats=stats_dic["raw"]['labels'].index(param_name)
    axes[col][i].set_ylim(0,1)
    axes[col][i].text(0.5 ,0.88, param_name  + ': {0:0.3f}'.format(params_current[pos_current]), verticalalignment='center', horizontalalignment='center', transform=axes[col][i].transAxes, color='blue', fontsize=4.5)
    extra=[[params_current[pos_current], color_extra, symbol_extra, linestyle_extra, show_extra_txt]]
    Draw_BoxAndWiskers_horizontal(y0, stats_dic['raw']["stats"][pos_stats, :], 
        extra=extra, color=color, fill=True, width=0.175, ax=axes[col][i], show_stats=True, fontsize=3.5, linewidth=0.7)
    #
    param_name='delta01'
    col=0 # either 0 or 1 as we have two columns
    i=3 # One of the 7 zones/col: i=[0,6]
    pos_current=labels.index(param_name)
    pos_stats=stats_dic["raw"]['labels'].index(param_name)
    axes[col][i].set_ylim(0,1)
    axes[col][i].text(0.5 ,0.88, param_name  + ': {0:0.3f}'.format(params_current[pos_current]), verticalalignment='center', horizontalalignment='center', transform=axes[col][i].transAxes, color='blue', fontsize=4.5)
    extra=[[params_current[pos_current], color_extra, symbol_extra, linestyle_extra, show_extra_txt]]
    Draw_BoxAndWiskers_horizontal(y0, stats_dic['raw']["stats"][pos_stats, :], 
        extra=extra, color=color, fill=True, width=0.175, ax=axes[col][i], show_stats=True, fontsize=3.5, linewidth=0.7)
    #
    param_name='Dnu_p'
    col=0 # either 0 or 1 as we have two columns
    i=4 # One of the 7 zones/col: i=[0,6]
    Nmin=10
    axes[col][i].set_ylim(0,1)
    axes[col][i].text(0.5 ,0.88, param_name , verticalalignment='center', horizontalalignment='center', transform=axes[col][i].transAxes, color='blue', fontsize=4.5)
    if Dnu_p != []:
        if len(Dnu_p) > Nmin:
            Dnu_p_stats=make_stats(Dnu_p, confidence=confidence)
            #axes[col][i].text(0.5 ,0.5, 'Put Dnu_p here' , verticalalignment='center', horizontalalignment='center', transform=axes[col][i].transAxes, color='blue', fontsize=5)
            extra=[[Dnu_p[-1], color_extra, symbol_extra, linestyle_extra, show_extra_txt]]
            Draw_BoxAndWiskers_horizontal(y0, Dnu_p_stats, 
                extra=extra, color=color, fill=True, width=0.175, ax=axes[col][i], show_stats=True, fontsize=3.5, linewidth=0.7)
        else:
            axes[col][i].text(0.5 ,0.5, 'Getting stats' , verticalalignment='center', horizontalalignment='center', transform=axes[col][i].transAxes, color='blue', fontsize=5)
    else:
        axes[col][i].text(0.5 ,0.5, 'No Stats available' , verticalalignment='center', horizontalalignment='center', transform=axes[col][i].transAxes, color='blue', fontsize=5)
    #
    param_name='epsilon_p'
    col=0 # either 0 or 1 as we have two columns
    i=5 # One of the 7 zones/col: i=[0,6]
    Nmin=10
    axes[col][i].set_ylim(0,1)
    axes[col][i].text(0.5 ,0.88, param_name , verticalalignment='center', horizontalalignment='center', transform=axes[col][i].transAxes, color='blue', fontsize=4.5)
    if epsilon_p != []:
        #axes[col][i].text(0.5 ,0.5, 'Put epsilon here' , verticalalignment='center', horizontalalignment='center', transform=axes[col][i].transAxes, color='blue', fontsize=5)
        extra=[[epsilon_p[-1], color_extra, symbol_extra, linestyle_extra, show_extra_txt]]
        if len(epsilon_p) > Nmin:
            epsilon_p_stats=make_stats(epsilon_p, confidence=confidence)
            Draw_BoxAndWiskers_horizontal(y0, epsilon_p_stats, 
                extra=extra, color=color, fill=True, width=0.175, ax=axes[col][i], show_stats=True, fontsize=3.5, linewidth=0.7)
        else:
            axes[col][i].text(0.5 ,0.5, 'Getting stats' , verticalalignment='center', horizontalalignment='center', transform=axes[col][i].transAxes, color='blue', fontsize=5)
    else:
        axes[col][i].text(0.5 ,0.5, 'No Stats available' , verticalalignment='center', horizontalalignment='center', transform=axes[col][i].transAxes, color='blue', fontsize=5)
    #
    param_name='rot_env'
    col=1 # either 0 or 1 as we have two columns
    i=0 # One of the 7 zones/col: i=[0,6]
    pos_current=labels.index(param_name)
    pos_stats=stats_dic["raw"]['labels'].index(param_name)
    axes[col][i].set_ylim(0,1)
    axes[col][i].text(0.5 ,0.88, param_name  + ': {0:0.3f}'.format(params_current[pos_current]), verticalalignment='center', horizontalalignment='center', transform=axes[col][i].transAxes, color='blue', fontsize=4.5)
    extra=[[params_current[pos_current], color_extra, symbol_extra, linestyle_extra, show_extra_txt]]
    Draw_BoxAndWiskers_horizontal(y0, stats_dic['raw']["stats"][pos_stats, :], 
        extra=extra, color=color, fill=True, width=0.175, ax=axes[col][i], show_stats=True, fontsize=3.5, linewidth=0.7)
    #
    param_name='rot_core'
    col=1 # either 0 or 1 as we have two columns
    i=1 # One of the 7 zones/col: i=[0,6]
    pos_current=labels.index(param_name)
    pos_stats=stats_dic["raw"]['labels'].index(param_name)
    axes[col][i].set_ylim(0,1)
    axes[col][i].text(0.5 ,0.88, param_name  + ': {0:0.3f}'.format(params_current[pos_current]), verticalalignment='center', horizontalalignment='center', transform=axes[col][i].transAxes, color='blue', fontsize=4.5)
    extra=[[params_current[pos_current], color_extra, symbol_extra, linestyle_extra, show_extra_txt]]
    Draw_BoxAndWiskers_horizontal(y0, stats_dic['raw']["stats"][pos_stats, :], 
        extra=extra, color=color, fill=True, width=0.175, ax=axes[col][i], show_stats=True, fontsize=3.5, linewidth=0.7)
    #
    param_name='Inclination'
    col=1 # either 0 or 1 as we have two columns
    i=2 # One of the 7 zones/col: i=[0,6]
    pos_current=labels.index(param_name)
    pos_stats=stats_dic["raw"]['labels'].index(param_name)
    axes[col][i].set_ylim(0,1)
    axes[col][i].text(0.5 ,0.88, param_name  + ': {0:0.3f}'.format(params_current[pos_current]), verticalalignment='center', horizontalalignment='center', transform=axes[col][i].transAxes, color='blue', fontsize=4.5)
    extra=[[params_current[pos_current], color_extra, symbol_extra, linestyle_extra, show_extra_txt]]
    Draw_BoxAndWiskers_horizontal(y0, stats_dic['raw']["stats"][pos_stats, :], 
        extra=extra, color=color, fill=True, width=0.175, ax=axes[col][i], show_stats=True, fontsize=3.5, linewidth=0.7)
    #
    param_name='Wfactor'
    col=1 # either 0 or 1 as we have two columns
    i=3 # One of the 7 zones/col: i=[0,6]
    pos_current=labels.index(param_name)
    pos_stats=stats_dic["raw"]['labels'].index(param_name)
    axes[col][i].set_ylim(0,1)
    axes[col][i].text(0.5 ,0.88, param_name  + ': {0:0.3f}'.format(params_current[pos_current]), verticalalignment='center', horizontalalignment='center', transform=axes[col][i].transAxes, color='blue', fontsize=4.5)
    extra=[[params_current[pos_current], color_extra, symbol_extra, linestyle_extra, show_extra_txt]]
    Draw_BoxAndWiskers_horizontal(y0, stats_dic['raw']["stats"][pos_stats, :], 
        extra=extra, color=color, fill=True, width=0.175, ax=axes[col][i], show_stats=True, fontsize=3.5, linewidth=0.7)
    #
    param_name='Hfactor'
    col=1 # either 0 or 1 as we have two columns
    i=4 # One of the 7 zones/col: i=[0,6]
    pos_current=labels.index(param_name)
    pos_stats=stats_dic["raw"]['labels'].index(param_name)
    axes[col][i].set_ylim(0,1)
    axes[col][i].text(0.5 ,0.88, param_name  + ': {0:0.3f}'.format(params_current[pos_current]), verticalalignment='center', horizontalalignment='center', transform=axes[col][i].transAxes, color='blue', fontsize=4.5)
    extra=[[params_current[pos_current], color_extra, symbol_extra, linestyle_extra, show_extra_txt]]
    Draw_BoxAndWiskers_horizontal(y0, stats_dic['raw']["stats"][pos_stats, :], 
        extra=extra, color=color, fill=True, width=0.175, ax=axes[col][i], show_stats=True, fontsize=3.5, linewidth=0.7)
    #
    param_name='Visibility_l1'
    col=0 # either 0 or 1 as we have two columns
    i=6 # One of the 7 zones/col: i=[0,6]
    pos_current=labels.index(param_name)
    pos_stats=stats_dic["raw"]['labels'].index(param_name)
    axes[col][i].set_ylim(0,1)
    axes[col][i].text(0.5 ,0.88, param_name  + ': {0:0.3f}'.format(params_current[pos_current]), verticalalignment='center', horizontalalignment='center', transform=axes[col][i].transAxes, color='blue', fontsize=4.5)
    extra=[[params_current[pos_current], color_extra, symbol_extra, linestyle_extra, show_extra_txt]]
    Draw_BoxAndWiskers_horizontal(y0, stats_dic['raw']["stats"][pos_stats, :], 
        extra=extra, color=color, fill=True, width=0.175, ax=axes[col][i], show_stats=True, fontsize=3.5, linewidth=0.7)
    #
    param_name='Visibility_l2'
    col=1 # either 0 or 1 as we have two columns
    i=6 # One of the 7 zones/col: i=[0,6]
    pos_current=labels.index(param_name)
    pos_stats=stats_dic["raw"]['labels'].index(param_name)
    axes[col][i].set_ylim(0,1)
    axes[col][i].text(0.5 ,0.88, param_name  + ': {0:0.3f}'.format(params_current[pos_current]), verticalalignment='center', horizontalalignment='center', transform=axes[col][i].transAxes, color='blue', fontsize=4.5)
    extra=[[params_current[pos_current], color_extra, symbol_extra, linestyle_extra, show_extra_txt]]
    Draw_BoxAndWiskers_horizontal(y0, stats_dic['raw']["stats"][pos_stats, :], 
        extra=extra, color=color, fill=True, width=0.175, ax=axes[col][i], show_stats=True, fontsize=3.5, linewidth=0.7)

    '''
    # Product of some model...
    # Dnu_p
    try:
        pos_Dnu_p=-1
    except:
        pos_Dnu_p=-1
    #epsilon_p
    try:
        pos_Dnu_p=-1
    except:
        pos_Dnu_p=-1
    '''
