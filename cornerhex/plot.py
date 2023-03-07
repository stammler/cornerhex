import matplotlib
from matplotlib import colormaps
from matplotlib.colors import Colormap
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
from scipy.special import erf
from typing import Tuple


def cornerplot(
    data: np.ndarray,
    cmap="Blues",
    correlation_textcolor=None,
    dpi=100.,
    hex_gridsize=30,
    highlight=None,
    highlight_linecolor=None,
    highlight_markercolor=None,
    hist_backgroundcolor=None,
    hist_bins=20,
    hist_edgecolor=None,
    hist_facecolor=None,
    labels=None,
    scatter_alpha=0.5,
    scatter_markercolor=None,
    scatter_outside_sigma=None,
    show_correlations=False,
    sigma_levels=None,
    sigma_linecolor=None,
    sigma_smooth=3.,
    title_quantiles=None,
    width=3.
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """
    Function creates hexbin corner plot matrix to visualize multidimensional data.

    Parameters:
    -----------
    data : (N_sample, N_dims) array-like
        Two-dimensional data array to be visualized. First dimension are the samples,
        second dimension the features.
    cmap : str or matplotlib.colors.Colormap, optional, default: "Blues"
        Either a string of a valid matplotlib colormap or a custom Colormap object
        to be used in the plot.
    correlation_textcolor : str, optional, default: None
        Color of correlation text.
        Defaults to a value from the chosen colormap.
    dpi : float, optional, default: 100.
        Dots per inch of the figure
    hex_gridsize : int, optional, default: 30
        Number of hexagons ins x-direction
    highlight : (N_dims) array-like, optional, default: None
        If not None, array of values to be highlighted in the plot. Typically
        the truth values.
    highlight_linecolor : str or tuple, optional, default: None
        If not None linecolor of the highlighted values.
        Defaults to a value from the chosen colormap.
    highlight_markercolor : str or tuple, optional, default: None
        If not None markercolor of the highlighted values.
        Defaults to a value from the chosen colormap.
    hist_backgroundcolor : str or tuple, optional, default: None
        If not None background color of the axes object.
        Defaults to a value from the chosen colormap.
    hist_bins : int, optinal, default: 20
        Number of bins in histograms.
    hist_edgecolor : str or tuple, optional, default: None
        If not None edgecolor of the histgram bars.
        Defaults to a value from the chosen colormap.
    hist_facecolor : str or tuple, optional, default: None
        If not None face of the histgram bars.
        Defaults to a value from the chosen colormap.
    labels : (N_dims) list, optional, default: None
        If not None list of strings with the feature names
        to be used as axis labels or in the axis titles.
    scatter_alpha : float, optional, default: 0.5
        Alpha transparency value of scatter plot
    scatter_markercolor : str, optional, default: None
        If not None markercolor of scatter plot markers.
        Defaults to a value from the chosen colormap.
    scatter_outside_sigma : float, optional, default: None
        If not None displays scatter plot of individual data
        points outside of given sigma contour.
    show_correlations : boolean, optional, default: False
        If True show Pearson's correlation coeffiction in each tile.
    sigma_levels : array_like, optional, default: None
        If not None contour levels to be plotted in
        units of the standard deviation.
    sigma_linecolor : str or tuple, optional, default: None
        If not None linecolor of the sigma contour lines.
        Defaults to a value from the chosen colormap.
    sigma_smooth : float, optional, default: 3.
        Smoothing factor for hexbin plot to smooth
        out contour lines.
    title_quantiles : (1,) or (3,) array-like, optional, default: None
        One-dimensional array of either size one or size three with
        the feature quantiles to be plotted as histogram titles.
    width : float, optional, default: 3.
        Width of a single tile of the corner plot.

    Returns:
    --------
    (fig, ax) : tuple
        Tuple containing the figure and axes objects.
    """

    # Number of dimensions
    _, Nd = data.shape

    # Validate colormap
    if isinstance(cmap, str):
        cm = colormaps[cmap]
    elif isinstance(cmap, Colormap):
        cm = cmap
    else:
        raise ValueError(
            "'cmap' has to be either type str or LinearSegmentedColormap.")

    # Setting colors
    hl_lc = highlight_linecolor if highlight_linecolor is not None else cm(1.)
    hl_mc = highlight_markercolor if highlight_markercolor is not None \
        else cm(0.)
    sig_lc = sigma_linecolor if sigma_linecolor is not None else cm(1.)
    hist_bc = hist_backgroundcolor if hist_backgroundcolor is not None \
        else cm(0.)
    hist_ec = hist_edgecolor if hist_edgecolor is not None else cm(1.)
    hist_fc = hist_facecolor if hist_facecolor is not None else cm(0.5)
    scat_mc = scatter_markercolor if scatter_markercolor is not None else cm(
        0.5)

    # Validate labels
    set_labels = False if labels is None else True
    if set_labels:
        if len(labels) != Nd:
            raise ValueError(
                "Size of 'labels' does not match number of dimensions.")

    # Validate highlights
    set_highlights = False if highlight is None else True
    if set_highlights:
        if len(highlight) != Nd:
            raise ValueError(
                "Size of 'highlight' does not match number of dimensions.")

    # Produce levels
    if sigma_levels is not None:
        # Converting to 1d quantiles
        quants = []
        for s in sigma_levels:
            q = sigma_to_quantile(s)
            quants.extend([50.-q, 50.+q])
        quants = np.array(quants)
        quants.sort()
        # Converting to 2d sigma levels
        levels2d = 1. - np.exp(-0.5*np.array(sigma_levels)**2)

    # Threshold for scatter plot
    if scatter_outside_sigma is not None:
        scatter_thr = 1. - np.exp(-0.5*np.array(scatter_outside_sigma)**2)

    # Validate title quantiles
    if title_quantiles is not None:
        if len(title_quantiles) not in [1, 3]:
            raise ValueError(
                "'title_quantiles' has to have either size 1 or size 3.")
        title_quantiles = np.array(title_quantiles)
        title_quantiles.sort()

    # Compute correlation coefficients
    if show_correlations:
        rho = np.corrcoef(data.T)
        cor_tc = correlation_textcolor if correlation_textcolor is not None else cm(
            1.)

    fig, ax = plt.subplots(nrows=Nd, ncols=Nd, figsize=(
        Nd*width, Nd*width), dpi=dpi)

    for i in range(Nd**2):
        # x and y coordinates of subplots
        ix, iy = np.divmod(i, Nd)

        if iy > ix:
            # Remove axes outside of corner plot
            fig.delaxes(ax[ix, iy])
        elif ix == iy:
            # Histograms
            ax[ix, iy].set_facecolor(hist_bc)
            ax[ix, iy].hist(data[:, ix], bins=hist_bins, color=cm(1.),
                            facecolor=hist_fc, edgecolor=hist_ec)
            if sigma_levels is not None:
                perc = np.percentile(data[:, ix], quants)
                for p in perc:
                    ax[ix, iy].axvline(p, ls="--", lw=1, c=sig_lc)
            if set_highlights:
                ax[ix, iy].axvline(highlight[iy], ls="-", color=hl_lc)
            if title_quantiles is not None:
                perc = np.percentile(data[:, ix], title_quantiles)
                if len(title_quantiles) == 1:
                    if labels is not None:
                        prefix = "{} $= ".format(labels[iy])
                    else:
                        prefix = "$"
                    msg = "{}{{{:.2f}}}$".format(
                        prefix, perc[0])
                else:
                    diff = np.diff(perc)
                    if labels is not None:
                        prefix = "{} $= ".format(labels[iy])
                    else:
                        prefix = "$"
                    msg = "{}{{{:.2f}}}^{{+{:.2f}}}_{{-{:.2f}}}$".format(
                        prefix, perc[1], diff[1], diff[0])
                ax[ix, iy].set_title(msg, fontsize="x-large")
        else:
            # Hexbin plots
            h = ax[ix, iy].hexbin(data[:, iy], data[:, ix], cmap=cm,
                                  gridsize=hex_gridsize, linewidths=0.5, edgecolor=cm(0))

            # Data gridding if sigma_levels given or scatter plot required
            if sigma_levels is not None or scatter_outside_sigma is not None:

                # Get positions and values from hexbin plots
                xy = h.get_offsets()
                z = h.get_array()

                # Compute gridded data for contour plot
                grid_x, grid_y = np.mgrid[
                    xy[:, 0].min():xy[:, 0].max():3*hex_gridsize*1j,
                    xy[:, 1].min():xy[:, 1].max():3*hex_gridsize*1j
                ]
                zi = griddata(xy, z, (grid_x, grid_y), method="linear")
                # Smooth data to forget about hexplot grid
                zi = gaussian_filter(zi, sigma_smooth)

                # Convert sigma levels to data levels
                z_flattened = zi.flatten()
                z_ordered = z_flattened[z_flattened.argsort()[::-1]]
                cumsum = z_ordered.cumsum()
                cumsum /= cumsum[-1]

            # Contourplot
            if sigma_levels is not None:
                # Compute levels
                levels = np.empty(len(levels2d))
                for i, s in enumerate(levels2d):
                    levels[i] = z_ordered[np.abs(cumsum-s).argmin()]
                levels.sort()
                # Draw plot
                ax[ix, iy].contour(
                    grid_x, grid_y, zi,
                    levels=levels,
                    colors=[sig_lc],
                    linewidths=1.,
                    linestyles="--"
                )

            # Scatter plot
            if scatter_outside_sigma is not None:
                # Compute threshold value
                scat_thr = z_ordered[np.abs(cumsum-scatter_thr).argmin()]
                # Nearest neighbor search
                NNDI = RegularGridInterpolator(
                    (grid_x[:, 0], grid_y[0, :]),
                    zi,
                    method="linear",
                    bounds_error=False,
                    fill_value=0.
                )
                mask = NNDI((data[:, iy], data[:, ix])) < scat_thr
                ax[ix, iy].scatter(
                    data[mask, iy], data[mask, ix],
                    marker=".",
                    s=0.5,
                    alpha=scatter_alpha,
                    c=[scat_mc])

            # Correlation coefficient
            if show_correlations:
                msg = r"$\rho = {{{:.2f}}}$".format(rho[ix, iy])
                ax[ix, iy].text(
                    0.02, 0.98, msg,
                    va="top", ha="left",
                    transform=ax[ix, iy].transAxes,
                    c=cor_tc
                )

            # Set the highlights
            if set_highlights:
                ax[ix, iy].axvline(highlight[iy], ls="-", color=hl_lc)
                ax[ix, iy].axhline(highlight[ix], ls="-", color=hl_lc)
                ax[ix, iy].plot(
                    highlight[iy], highlight[ix],
                    marker=(6, 0, 0),
                    markersize=8,
                    c=hl_mc,
                    markeredgecolor=hl_lc
                )

            ax[ix, iy].set_xlim(data[:, iy].min(), data[:, iy].max())
            ax[ix, iy].set_ylim(data[:, ix].min(), data[:, ix].max())

        # Set y-labels
        if ix > 0 and iy == 0:
            if set_labels:
                ax[ix, iy].set_ylabel(labels[ix], fontsize="x-large")
        elif iy < ix:
            ax[ix, iy].sharey(ax[ix, 0])
            plt.setp(ax[ix, iy].get_yticklabels(), visible=False)
        else:
            plt.setp(ax[ix, iy].get_yticklabels(), visible=False)

        # Set x-labels
        if ix == Nd-1:
            if set_labels:
                ax[ix, iy].set_xlabel(labels[iy], fontsize="x-large")
        else:
            ax[ix, iy].sharex(ax[Nd-1, iy])
            plt.setp(ax[ix, iy].get_xticklabels(), visible=False)

    fig.tight_layout()

    plt.show()

    return fig, ax


def sigma_to_quantile(sig: float) -> float:
    """
    Function converts standard deviation of
    a normal distribution to quantile.

    Parameters:
    -----------
    sig : float
        Standard deviation

    Returns:
    --------
    q : float
        Quantile
    """
    return _gaussian_primitive(-sig, 0., 1.)*100


def _gaussian_primitive(x: float, mu: float, sig: float) -> float:
    """
    Function returns the primitive of the normal distribution

    Parameters:
    -----------
    x : float, array-like
        Evaluation coordinate
    mu : float
        Mean of normal distribution
    sig : float
        Standard deviation of normal distribution

    Returns:
    --------
    F(x) : float, array-like
        Primitive at x
    """

    return -0.5*erf((x-mu)/(np.sqrt(2.)*sig))
