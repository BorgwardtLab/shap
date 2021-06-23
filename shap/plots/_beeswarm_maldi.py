""" Summary plots of SHAP values across a whole dataset.
"""

from __future__ import division

import warnings
import numpy as np
import scipy as sp
from scipy.stats import gaussian_kde
try:
    import matplotlib.pyplot as pl
except ImportError:
    warnings.warn("matplotlib could not be loaded!")
    pass
import matplotlib.cm as cm
from ._labels import labels
from . import colors
from ..utils import safe_isinstance, OpChain, format_value
from ._utils import convert_ordering, convert_color, merge_nodes, get_sort_order, sort_inds
from .. import Explanation



def shorten_text(text, length_limit):
    if len(text) > length_limit:
        return text[:length_limit - 3] + "..."
    else:
        return text


def is_color_map(color):
    safe_isinstance(color, "matplotlib.colors.Colormap")


def summary_legacy(shap_values, features=None, feature_names=None, max_display=None,
                 color=None, axis_color="#333333", title=None, alpha=1, show=True,
                 color_bar=True, plot_size="auto", layered_violin_max_num_bins=20, class_names=None,
                 class_inds=None,
                 color_bar_label=labels["FEATURE_VALUE"],
                 cmap=colors.red_blue):
    """Create a SHAP beeswarm plot, colored by feature values when they are provided.

    Parameters
    ----------
    shap_values : numpy.array
        For single output explanations this is a matrix of SHAP values (# samples x # features).
        For multi-output explanations this is a list of such matrices of SHAP values.

    features : numpy.array or pandas.DataFrame or list
        Matrix of feature values (# samples x # features) or a feature_names list as shorthand

    feature_names : list
        Names of the features (length # features)

    max_display : int
        How many top features to include in the plot (default is 20, or 7 for interaction plots)

    plot_size : "auto" (default), float, (float, float), or None
        What size to make the plot. By default the size is auto-scaled based on the number of
        features that are being displayed. Passing a single float will cause each row to be that 
        many inches high. Passing a pair of floats will scale the plot by that
        number of inches. If None is passed then the size of the current figure will be left
        unchanged.
    """

    # support passing an explanation object
    if str(type(shap_values)).endswith("Explanation'>"):
        shap_exp = shap_values
        base_value = shap_exp.base_values
        shap_values = shap_exp.values
        if features is None:
            features = shap_exp.data
        if feature_names is None:
            feature_names = shap_exp.feature_names

    assert len(shap_values.shape) != 1, "Summary plots need a matrix of shap_values, not a vector."

    # default color:
    if color is None:
        color = colors.blue_rgb

    idx2cat = None
    # convert from a DataFrame or other types
    if str(type(features)) == "<class 'pandas.core.frame.DataFrame'>":
        if feature_names is None:
            feature_names = features.columns
        # feature index to category flag
        idx2cat = features.dtypes.astype(str).isin(["object", "category"]).tolist()
        features = features.values
    elif isinstance(features, list):
        if feature_names is None:
            feature_names = features
        features = None
    elif (features is not None) and len(features.shape) == 1 and feature_names is None:
        feature_names = features
        features = None

    num_features = shap_values.shape[1]

    if features is not None:
        shape_msg = "The shape of the shap_values matrix does not match the shape of the " \
                    "provided data matrix."
        if num_features - 1 == features.shape[1]:
            assert False, shape_msg + " Perhaps the extra column in the shap_values matrix is the " \
                          "constant offset? Of so just pass shap_values[:,:-1]."
        else:
            assert num_features == features.shape[1], shape_msg

    if feature_names is None:
        feature_names = np.array([labels['FEATURE'] % str(i) for i in range(num_features)])

    if max_display is None:
        max_display = 20

    # order features by the sum of their effect magnitudes
    feature_order = np.argsort(np.sum(np.abs(shap_values), axis=0))
    feature_order = feature_order[-min(max_display, len(feature_order)):]


    pl.figure(figsize=(20,15))
    fig, ax = pl.subplot_mosaic(
        """
        011
        """,
        figsize=(15,10)
    )
    row_height = 0.4

    for i, type_ in enumerate(['bar', 'dot']):
        i = str(i)

        if type_ == 'dot':
            ax[i].axvline(x=0, color="#999999", zorder=-1)

            for pos, idx in enumerate(feature_order):
                ax[i].axhline(y=pos, color="#cccccc", lw=0.5, dashes=(1, 5), zorder=-1)

                shaps = shap_values[:, idx]
                values = None if features is None else features[:, idx]
                inds = np.arange(len(shaps))
                np.random.shuffle(inds)
                if values is not None:
                    values = values[inds]
                shaps = shaps[inds]
                colored_feature = True

                try:
                    if idx2cat is not None and idx2cat[idx]: # check categorical feature
                        colored_feature = False
                    else:
                        values = np.array(values, dtype=np.float64)  # make sure this can be numeric
                except:
                    colored_feature = False

                N = len(shaps)
                nbins = 100
                quant = np.round(nbins * (shaps - np.min(shaps)) / (np.max(shaps) - np.min(shaps) + 1e-8))
                inds = np.argsort(quant + np.random.randn(N) * 1e-6)
                layer = 0
                last_bin = -1
                ys = np.zeros(N)
                for ind in inds:
                    if quant[ind] != last_bin:
                        layer = 0
                    ys[ind] = np.ceil(layer / 2) * ((layer % 2) * 2 - 1)
                    layer += 1
                    last_bin = quant[ind]
                ys *= 0.9 * (row_height / np.max(ys + 1))

                # trim the color range, but prevent the color range from collapsing
                vmin = np.nanpercentile(values, 5)
                vmax = np.nanpercentile(values, 95)
                if vmin == vmax:
                    vmin = np.nanpercentile(values, 1)
                    vmax = np.nanpercentile(values, 99)
                    if vmin == vmax:
                        vmin = np.min(values)
                        vmax = np.max(values)
                if vmin > vmax: # fixes rare numerical precision issues
                    vmin = vmax

                assert features.shape[0] == len(shaps), "Feature and SHAP matrices must have the same number of rows!"

                # plot the nan values in the interaction feature as grey
                nan_mask = np.isnan(values)
                ax[i].scatter(shaps[nan_mask], pos + ys[nan_mask], color="#777777", vmin=vmin,
                           vmax=vmax, s=16, alpha=alpha, linewidth=0,
                           zorder=3, rasterized=len(shaps) > 500)

                # plot the non-nan values colored by the trimmed feature value
                cvals = values[np.invert(nan_mask)].astype(np.float64)
                cvals_imp = cvals.copy()
                cvals_imp[np.isnan(cvals)] = (vmin + vmax) / 2.0
                cvals[cvals_imp > vmax] = vmax
                cvals[cvals_imp < vmin] = vmin
                ax[i].scatter(shaps[np.invert(nan_mask)], pos + ys[np.invert(nan_mask)],
                           cmap=cmap, vmin=vmin, vmax=vmax, s=16,
                           c=cvals, alpha=alpha, linewidth=0,
                           zorder=3, rasterized=len(shaps) > 500)

            ax[i].set_yticklabels(['' for i in feature_inds])

            # draw the color bar
            m = cm.ScalarMappable(cmap=cmap)
            m.set_array([0, 1])
            cb = pl.colorbar(m, ax=ax[i], ticks=[0, 1], aspect=1000)
            cb.set_ticklabels([labels['FEATURE_VALUE_LOW'], labels['FEATURE_VALUE_HIGH']])
            cb.set_label(color_bar_label, size=12, labelpad=0, rotation=270)
            cb.ax.tick_params(labelsize=11, length=0)
            cb.set_alpha(1)
            cb.outline.set_visible(False)
            bbox = cb.ax.get_window_extent().transformed(pl.gcf().dpi_scale_trans.inverted())
            cb.ax.set_aspect((bbox.height - 0.9) * 20)

        if type_ == 'bar':
            # plot baseline
            ax[i].axvline(x=0, color="#999999", zorder=-1)

            feature_inds = feature_order[:max_display]
            y_pos = np.arange(max_display)
            global_shap_values = np.abs(shap_values).mean(0)

            ax[i].barh(y_pos, global_shap_values[feature_inds], 0.7, align='center', color=color)
            ax[i].set_yticks(y_pos)
            ax[i].set_yticklabels([feature_names[i] for i in feature_inds])

    print('plotted both plots')
    for i, type_ in enumerate(['bar', 'dot']):
        i = str(i)
        ax[i].xaxis.set_ticks_position('bottom')
        ax[i].yaxis.set_ticks_position('none')
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['left'].set_visible(False)
        ax[i].tick_params(color=axis_color, labelcolor=axis_color)
        ax[i].set_yticks(range(len(feature_order)), [feature_names[i] for i in feature_order])

        if type_ != "bar":
            ax[i].tick_params('y', length=20, width=0.5, which='major')

        ax[i].tick_params('x', labelsize=11)
        ax[i].set_ylim(-1, len(feature_order))

        if type_ == "bar":
            ax[i].set_xlabel(labels['GLOBAL_VALUE'])
        else:
            ax[i].set_xlabel(labels['VALUE'])
