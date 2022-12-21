import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

def Draw_BoxAndWiskers_horizontal(y, stats, extra=[], color=['black', 'black', 'black'], 
                                  fill=None, width=0.05, linewidth=1, ax=None, show_stats=False, fontsize=12):
    '''  
        A simplified version for the horizontal whiskers plot.
        The one given by matplotlib use the whole data which
        may be slow in a loop. This function takes directly the statistical
        information given by the user in order to make the wiskers plot
        y: y-position. Note that the x-position is given by the stats values
        stats: statistical information: [s0, s1, m, s2, s3]
        extra: bars with symbol as additional marker.
                must be list of list of this kind: [[valx, color, [symbol, symbol_size], line, show_value_True_False]]
                if both symbol is set to [] and line is set to None, the option is deactivated.
                |         --------------            |      |
                |--------|      |        |----------O------|
                |         --------------            |      |
                s0       s1     m       s2      extra      s3
        0 ----- Dx1 ---------------------------------------Dx2-------- 1
        color: color of the plot. Nust contain 3 values: The color of the box, of (s1 - s0) and (s3-s2), and of the median (m)   
        width: width of the plot, in unit of the ploting zone
        linewidth : size of all the lines in the plot
        ax: ploting zone
        show_stats: If True, show the values of s0, s1, m, s2, extra (if element 4 is True) and s3
    '''
    # Check the parameters.
    if ax == None:
        fig, ax = plt.subplots(1, figsize=(8, 8))
        ax.set_ylim(y-5*width,  y + 5*width)
        ax.set_xlim(np.min(stats)*0.7, np.max(stats)*1.3)     
    s0=stats[0] 
    s1=stats[1] 
    m=stats[2]
    s2=stats[3]
    s3=stats[4]
    #Draw the box.
    ax.add_patch(Rectangle((s1, y-width/2), s2-s1, width, facecolor=color[0], edgecolor=color[0], fill=fill, alpha=0.7, linewidth=0)) # (x0, y0), Dx, Dy
    ax.plot([m,m], [y-width/2, y+width/2], color=color[2], linewidth=linewidth)
    #Draw the whiskers.
    ax.plot([s0, s1], [y, y], color=color[1], linewidth=linewidth)
    ax.plot([s2, s3], [y,y], color=color[1], linewidth=linewidth)
    ax.plot([s0,s0], [y-width/2, y+width/2], color=color[1], linewidth=linewidth)
    ax.plot([s3,s3], [y-width/2, y+width/2], color=color[1], linewidth=linewidth)
    #
    if extra != []:
        for e in extra:
            if e[2] != []:
                ax.plot([e[0]], [y], color=e[1], marker=e[2][0], markersize=e[2][1])
            if e[3] != None:
                ax.plot([e[0], e[0]], [y-width/2- 0.2*width, y+width/2 + 0.2*width], color=e[1], linewidth=linewidth, linestyle=e[3])
    y_txt03=y-width/2 - 1.25*width
    y_txt12=y-width/2 - 1.25*width
    y_txtm=y-width/2 - 1.25*width
    y_txt_extra=y+width/2 + 1.25*width
    if show_stats == True:
        str_f=eval_precision_txt(s0)
        ax.text(s0, y_txt03, str_f.format(s0) , verticalalignment='center', horizontalalignment='center', color=color[1], fontsize=fontsize, rotation=80)
        ax.text(s1, y_txt12, str_f.format(s1) , verticalalignment='center', horizontalalignment='center', color=color[1], fontsize=fontsize, rotation=80)
        ax.text(m, y_txtm, str_f.format(m) , verticalalignment='center', horizontalalignment='center',  color=color[2], fontsize=fontsize, rotation=80)
        ax.text(s2, y_txt12, str_f.format(s2) , verticalalignment='center', horizontalalignment='center', color=color[1], fontsize=fontsize, rotation=80)
        ax.text(s3, y_txt03, str_f.format(s3) , verticalalignment='center', horizontalalignment='center', color=color[1], fontsize=fontsize, rotation=80)
        if extra != []:
            for e in extra:
                if e[4] == True:
                    str_f=eval_precision_txt(e[0])
                    ax.text(e[0], y_txt_extra, str_f.format(e[0]) , verticalalignment='center', horizontalalignment='center', color='red', fontsize=fontsize)

def Draw_BoxAndWiskers_vertical(x, stats, extra=[], color=['black', 'black', 'black'], 
                                  fill=None, width=0.05, linewidth=1, ax=None, show_stats=False, fontsize=12):
    '''  
        A simplified version for the horizontal whiskers plot.
        The one given by matplotlib use the whole data which
        may be slow in a loop. This function takes directly the statistical
        information given by the user in order to make the wiskers plot
        x: x-position. Note that the y-position is given by the stats values
        stats: statistical information: [s0, s1, m, s2, s3]
        extra: bars with symbol as additional marker.
                must be list of list of this kind: [[valx, color, [symbol, symbol_size], line, show_value_True_False]]
                if both symbol is set to [] and line is set to None, the option is deactivated.
                |         --------------            |      |
                |--------|      |        |----------O------|
                |         --------------            |      |
                s0       s1     m       s2      extra      s3
        0 ----- Dy1 ---------------------------------------Dy2-------- 1
        color: color of the plot. Nust contain 3 values: The color of the box, of (s1 - s0) and (s3-s2), and of the median (m)   
        width: width of the plot, in unit of the ploting zone
        linewidth : size of all the lines in the plot
        ax: ploting zone
        show_stats: If True, show the values of s0, s1, m, s2, extra (if element 4 is True) and s3
    '''
    # Check the parameters.
    if ax == None:
        fig, ax = plt.subplots(1, figsize=(8, 8))
        ax.set_xlim(x-5*width,  x + 5*width)
        ax.set_ylim(np.min(stats)*0.7, np.max(stats)*1.3)     
    s0=stats[0] 
    s1=stats[1] 
    m=stats[2]
    s2=stats[3]
    s3=stats[4]
    #Draw the box.
    ax.add_patch(Rectangle((x-width/2, s1), width, s2-s1, facecolor=color[0], edgecolor=color[0], fill=fill, alpha=0.7, linewidth=0)) # (x0, y0), Dx, Dy
    ax.plot([x-width/2, x+width/2], [m,m], color=color[2], linewidth=linewidth)
    #Draw the whiskers.
    ax.plot([x, x], [s0, s1], color=color[1], linewidth=linewidth)
    ax.plot([x,x],[s2, s3],  color=color[1], linewidth=linewidth)
    ax.plot([x-width/2, x+width/2], [s0,s0], color=color[1], linewidth=linewidth)
    ax.plot([x-width/2, x+width/2], [s3,s3],  color=color[1], linewidth=linewidth)
    #
    if extra != []:
        for e in extra:
            if e[2] != []:
                ax.plot([x], [e[0]], color=e[1], marker=e[2][0], markersize=e[2][1])
            if e[3] != None:
                ax.plot([x-width/2- 0.2*width, x+width/2 + 0.2*width], [e[0], e[0]], color=e[1], linewidth=linewidth, linestyle=e[3])
    x_txt03=x-width/2 - 1.25*width
    x_txt12=x-width/2 - 1.25*width
    x_txtm=x-width/2 - 1.25*width
    x_txt_extra=x+width/2 + 1.25*width
    if show_stats == True:
        str_f=eval_precision_txt(s0)
        ax.text(x_txt03, s0,  str_f.format(s0) , verticalalignment='center', horizontalalignment='center', color=color[1], fontsize=fontsize, rotation=0)
        ax.text(x_txt12, s1, str_f.format(s1) , verticalalignment='center', horizontalalignment='center', color=color[1], fontsize=fontsize, rotation=0)
        ax.text(x_txtm, m, str_f.format(m) , verticalalignment='center', horizontalalignment='center',  color=color[2], fontsize=fontsize, rotation=0)
        ax.text(x_txt12,s2,  str_f.format(s2) , verticalalignment='center', horizontalalignment='center', color=color[1], fontsize=fontsize, rotation=0)
        ax.text(x_txt03, s3,  str_f.format(s3) , verticalalignment='center', horizontalalignment='center', color=color[1], fontsize=fontsize, rotation=0)
        if extra != []:
            for e in extra:
                if e[4] == True:
                    str_f=eval_precision_txt(e[0])
                    ax.text(x_txt_extra, e[0],  str_f.format(e[0]) , verticalalignment='center', horizontalalignment='center', color='red', fontsize=fontsize)

def eval_precision_txt(s):
    if len(str(np.fix(s)))<=3:
        str_f='{0:.3f}'
    else:
        str_f='{0:.2f}'
    return str_f

def tests_1():
    y=1
    stats=[2, 3, 5, 5.5, 8]
    Draw_BoxAndWiskers_horizontal(y, stats)
    plt.show()


def tests_2():
    y=1
    stats=[2, 3, 5, 5.5, 8]
    color=['red', 'blue', 'Green']
    fill=True
    Draw_BoxAndWiskers_horizontal(y, stats, color=color, fill=fill)
    plt.show()

def tests_3():
    y=1
    stats=[2, 3, 5, 5.5, 8]
    color=['red', 'blue', 'Green']
    fill=True
    #extra=[[9, 'cyan', ['o', 20] , None]]
    extra=[[9, 'cyan', ['o', 20] , '-',True]]
    Draw_BoxAndWiskers_horizontal(y, stats, color=color, fill=fill, extra=extra, show_stats=True)
    plt.show()

def tests_4():
    x=1
    stats=[2, 3, 5, 5.5, 8]
    color=['red', 'blue', 'Green']
    fill=True
    #extra=[[9, 'cyan', ['o', 20] , None]]
    extra=[[9, 'cyan', ['o', 20] , '-', True]]
    Draw_BoxAndWiskers_vertical(x, stats, color=color, fill=fill, extra=extra, show_stats=True)
    plt.show()

#tests_4()