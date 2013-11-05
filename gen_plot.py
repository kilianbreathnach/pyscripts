import subprocess
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.stats.mstats import mquantiles

"""
Trying to make a general plotting script that I can call to make most any plot
I will have to make in my academic career, adding to it whenever new plots are
required. Can also provide plot objects, for instance axes that are more
particular, which can then be fed back in to provide the end plot.
"""


def add_date():
    """
    Uses the linux shell the script is called in to determine the date and
    return a string of the form "DD MON YYYY", where the month is in letters
    rather than digits.
    """
    process = subprocess.Popen(['date', '+%d-%b-%Y'], stdout=subprocess.PIPE)
    date, err = process.communicate()
    date = date.split()[0].split("-")

    return "{0} {1} {2}".format(*date)


def mk_hist():
    pass


def hist_quant(ax, x, y, xlog, ylog, xmin, xmax,
               Nxbins, Nybins, xbinedges, ybinedges, dath):

    if not (xmin):
        xmin = np.min(x)
    if not (xmax):
        xmax = np.max(x)

    # Find the bin edges for the x axis
    if not xbinedges:
        if not Nxbins:
            Nxbins = 20
        if xlog:
            xbinedges = np.logspace(np.log10(xmin), np.log10(xmax), Nxbins + 1)
        else:
            xbinedges = np.linspace(xmin, xmax, Nxbins + 1)

    # Now make the arrays for the quantiles in the y direction
    if not ybinedges:

        if not Nybins:
            Nybins = 10

        ybinedges = np.array(range(Nybins + 1)) / float(Nybins)

    # Final array is
    quantiles = np.zeros((Nxbins, Nybins))

    # ...and crank!
    for i in range(len(x)):

        id_near = (np.abs(xbinedges - x[i])).argmin()

        if not ((x[i] - xbinedges[id_near] >= 0) and (id_near != Nxbins)):
            id_near -= 1

        qu_near = (np.abs(ybinedges - y[i])).argmin()

        if not ((y[i] - ybinedges[qu_near] >= 0) and (qu_near != Nybins)):
            qu_near -= 1

        quantiles[id_near, qu_near] += 1

    # Now have to normalise over each x value
    heatmap = np.zeros(quantiles.shape)
    norm = 1.0 / len(y)

    for xcol in range(Nxbins):

        ycol = quantiles[xcol, :]

        for i in range(len(ycol)):
            heatmap[xcol, i] = norm * (np.sum(ycol[:i + 1]))

    # And plot
    mappable = ax.imshow(heatmap, cmap=mpl.get_cmap(dath),
                         interpolation='nearest')

    return mappable


class custom_ax:

    def __init__(self, ):

        self.fig, self.axes = plt.figure()


from matplotlib import rc
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)


def plotz(x, y, z=None,
          xvarn=None, yvarn=None, zvarn=None,
          style="line", axdim=None, overplot=False,
          xlog=False, ylog=False, Nxbins=None, Nybins=None,
          xmin=None, xmax=None, ymin=None, ymax=None, zmin=None, zmax=None,
          xbinedges=None, ybinedges=None, invert=None,
          colour_bar=False, dath="binary", labels=None, title=None,
          date=True, datepos=[0.6, 0.05],
          name="./plot.png"):
    """
    This is the main function to be called by a script in which I am plotting
    quantities.

    Parameters
    ----------

    x, y: 1D, 2D numpy arrays
        are points for a line or scatter plot, or several arrays for
        multiplots.
    z: 1D, 2D numpy array
        This is an optional 3rd dataset for 3D plots.
    {x, y, z}varn: string or list of strings with length of second dimension of
    above arrays
        These are strings containing the variable names (and units) for each
        axis, given in string form
    style: string
        This indicates the kind of plot, can be
          * line: see how many there are to determine no. of overplots
            or subplots.
          * scatter: just a load of points
          * hist: histogram, yup
          * hist_quant: plots a 2D colourmap with binning on xaxis and yaxis,
            where yaxis binning is cumulative along the length of the
            array.
          * error_ax: plots a model fit as line with data points plotted with
            errorbars and a residual plot beneath
    axdim: list
        a 2-element list of integers [Nrows, Ncols] that gives the
        subplot shape; each data array pair will be added in order along rows
        then columns
    overplot: Bool
        dictates whether an array of lines are all to be plotted on the
      same set of axes
    {x, y}log: Bool
        set scale of axes to logspace; if xlog is set and the style is hist,
        the binning will be done in logspace
    {x, y, z}min/max: float
        sets the min/max values to plot between if interested in a
        certain region of the data or values to count between for
        histograms
    invert: list or list of lists of bools
        list of bools for each axis in the plot; list of such lists for several
        plots going in order of left to right, top to bottom
    labels: string or list of strings
        (list of) string(s) of the same dimension as the data array 2nd column;
        if the plot is an overplot this will be interpreted to be the legend
        labels, otherwise it will be the axes headings
    title: string
        Figure title for the whole plot
    """

    if style == "error_ax":
        pass

    elif style == "overplot":
        pass

    else:
        if axdim:
            fig, axes = plt.subplots(axdim[0], axdim[1])
        else:

            # This will give us one row of axes with simple lines if there
            # are no special arguments.
            fig, axes = plt.subplots(1, x.shape[1])

        for i, ax in enumerate(axes):

            x = np.atleast_2d(x)
            y = np.atleast_2d(y)

            if invert:
                if invert[i][0]:
                    ax.invert_xaxis()
                if invert[i][1]:
                    ax.invert_yaxis()

            if style == "line":
                ax.plot(x[i], y[i])
            elif style == "scatter":
                ax.scatter(x[i], y[i])
            elif style == "hist":
                mk_hist(ax, x[i], y[i])
            elif style == "hist_quant":
                mappable = hist_quant(ax, x[i], y[i], xlog, ylog,
                                      xmin, xmax, ymin, ymax,
                                      Nxbins, Nybins, xbinedges, ybinedges,
                                      dath)
                fig.colorbar(mappable, ax=ax)

            if xvarn:
                ax.set_xlabel(r"{0}".format(xvarn[i]))
            if yvarn:
                ax.set_ylabel(r"{0}".format(yvarn[i]))

            if labels:
                ax.set_title(r"{0}".format(labels[i]))

    if not axdim:
        Nrows = 1
        Ncols = len(np.atleast_2d(x))

    else:
        Nrows = axdim[0]
        Ncols = axdim[1]

    fig.subplots_adjust(left=(0.33 / (Ncols + 0.5)),
                        bottom=(0.33 / (Nrows + 0.8)),
                        right=(0.17 / (Ncols + 0.5)),
                        top=(0.47 / (Nrows + 0.8)),
                        wspace=(0.17 / (Ncols + 0.5)),
                        hspace=(0.33 / (Nrows + 0.8)))

    if title:
        fig.suptitle(r"{0}".format(title))

    if date:
        fig.text(datepos[0], datepos[1], add_date(), alpha=0.5)

    fig.savefig(name)
