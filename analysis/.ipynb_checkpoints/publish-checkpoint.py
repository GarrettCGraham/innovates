import numpy as np
import os
import pandas as pd
import pickle

from copy import deepcopy
from sklearn.dummy import DummyClassifier
from sklearn import metrics
from matplotlib import patches
from matplotlib import pyplot
from matplotlib.colors import LogNorm

from dynomics import dynomics
from dynomics import misc
from dynomics import models
from dynomics import transformations


# All experiments over which we classified.
EOI_ALL = [
    1711, 1726, 1742, 1778, 1784, 1794,
    1798, 1808, 1822, 1823, 1824, 1835,
    1836, 1838, 1844, 1848, 1924, 1952
]


def unpickle(filename, data_dir):
    with open(data_dir+filename+".pickle", "rb") as pickled_file:
        unpickled_file = pickle.load(pickled_file)
    return unpickled_file


def unpickle_EOI_feats_targs():
    pickle_cabinet = "data/20190419_cache_features_targets_for_quick_reload/"
    EOI_LIST = unpickle("EOI_LIST", pickle_cabinet)
    features = unpickle("features", pickle_cabinet)
    targets = unpickle("targets", pickle_cabinet)
    return EOI_LIST, features, targets


def check_mkdirs(*paths):
    for path in paths:
        if not os.path.isdir(path):
            os.makedirs(path)
    return


def create_exp_strain_position_mapping(exp):
    """
    Create a strain-name-to-list-of-device-positions mapping for a single experiment. 
    
    parameters
    ----------
    exp: dynomics.Experiment
        A dynomics.Experiment object whose strain-positions should be tabulated.
    
    returns
    -------
    strain_position_mapping: dict(list)))
        A dictionary of lists, indexed by gene_name,
        containing integer positions for the indexed strain for the specified
        experiment.
    """
    lr = exp.loading_record[['device_position', 'gene_name']].dropna()
    lr.index  = lr.device_position
    lr = lr.drop('device_position', axis=1)
    strain_position_mapping = {
        gene_name : lr.index[lr.gene_name == gene_name].tolist()
        for gene_name in lr.gene_name.unique()
    }
    return strain_position_mapping



def create_strain_position_mapping(exp_list):
    """
    Create a strain-name-to-list-of-device-positions mapping. This dictionary
    can be especially useful when plotting by strain.
    
    parameters
    ----------
    exp_list: list(dynomics.Experiment)
        A list of Experiment objects whose strain-positions should be tabulated.
    
    returns
    -------
    strain_position_mapping: dict(dict(list)))
        A dictionary of dictionaries of lists, indexed by exp.idx, then gene_name,
        containing integer positions for the indexed strain for the specified
        experiment.
        
    """
    strain_position_mapping = { exp.idx: {} for exp in exp_list }
    exp = exp_list[0]
    strain_position_mapping[exp.idx] =\
        create_exp_strain_position_mapping(exp)
    _lr = exp.loading_record['gene_name'].dropna().values
    _exp_idx = exp.idx
    
    for exp in exp_list[1:]:
        lr = exp.loading_record['gene_name'].dropna().values
        if (~(_lr == lr)).sum() == 0:
            strain_position_mapping[exp.idx] =\
                deepcopy(strain_position_mapping[_exp_idx])
        else:
            strain_position_mapping[exp.idx] =\
                create_exp_strain_position_mapping(exp)
        _lr = lr.copy()
        _exp_idx = exp.idx
    return strain_position_mapping

def plot_stacked_timeseries(
    strain_name,
    channel_list,
    exp_list,
    strain_position_mapping,
    plot_kwargs,
    y_axis_kwargs
):

    """
    Defines a plotting function that outputs vertically aligned subplots of 
    dynomics timeseries data. For example, you could produce a plot that has 
    smoothed_firstdiff_gfpffc_bulb_1 timeseries in the top plot and the corresponding 
    gfpffc_bulb_1 timeseries in the bottom plot.
    """
    
    # Get the number of subplots needed.
    num_subplots = len(channel_list)
    
    # Create a list of axes-indices and the corresponding channel whose data
    #  will be displayed in that subplot.
    channel_tuples_list = [
        (ax_idx, channel) for ax_idx, channel in enumerate(channel_list)
    ]
    
    # Loop through list of experiment objects and plot the desired data on an
    # array of appropriately-sized subplots.
    for exp in exp_list:
        
        # Initialize the figure and axes objects with a sufficient figure size.
        fig, ax = plt.subplots(num_subplots, 1, figsize=(10,6*num_subplots))
        
        # Add a title to the overall figure.
        fig.suptitle(
            'Exp {idx}: {strain_name}'.format(
                idx=exp.idx,
                strain_name=strain_name,
            ),
            fontsize=32,
            **{'y':1.05}
        )
        
        # Get the device positions of the desired strain for
        # the current experiment.
        strain_dp = strain_position_mapping[exp.idx][strain_name]
        
        # Loop through all desired channels and plot the corresponding data
        # on the appropriate subplot.
        for ax_idx, channel in channel_tuples_list:
            exp.plot(
                strain_dp,
                traj_type=[channel],
                ax=ax[ax_idx],
                exp_idx=channel,
                **plot_kwargs
            )
            
        # Annotate the y-axii.
        ylabel = ax[ax_idx].get_ylabel()
        annotate_axis(
            ax[ax_idx].yaxis, 
            tick_labels=ax[ax_idx].get_yticks(), 
            axis_label=ylabel,
            **y_axis_kwargs
        )
        
        # Let pyplot work its plot-formatting magic.
        plt.tight_layout()
        
        # Display the plot!
        plt.show()
        
        return


def get_featureset_targets_splits(featureset_name=None, as_class=True, as_1d=True):
    
    # Define list of feature sets on which to train.
    # featureset_dir_names_list = [d for d in os.listdir('data/') if '20181210' in d]
    # _featureset_names_list = ['_'.join(n.split('_')[1:]) for n in featureset_dir_names ]
    featureset_names_list = [
        'transform_derivative',
        'smooth`',
        'z_scored_smooth`_norm_mean_abs',
        'smooth_z_scored_transform_derivative_norm_mean_abs'
    ]

    if featureset_name is None:
        featureset_name = 'z_scored_smooth`_norm_mean_abs'
    elif featureset_name not in featureset_names_list:
        print("That feature does not exist. Use one of the following:")
        for feature in featureset_names_list:
            print(feature)
        return

    # Define list of exp_idx.
    EOI_ALL = [
        1711, 1726, 1742, 1778, 1784, 1794,
        1798, 1808, 1822, 1823, 1824, 1835,
        1836, 1838, 1844, 1848, 1924, 1952
    ]

    date = '20181210_'

    # Load featureset exp list.
    featuresets_exp_list = [
        misc.pickle_load(
            'data/' + date + featureset_name + 
            '/exp_{0!s}_'.format(exp_idx) + featureset_name + '.pickle'
        ) for exp_idx in EOI_ALL
    ]

    # Redefine list with featuresets that have their growth trimmed
    # and are trimmed of BAD and SJ inductions.
    featuresets_exp_list = [
        drop_bad_inductions(
            drop_sj_inductions(
                exp, filter_following_ind=True
            )[0], filter_following_ind=True
        )[0].trim_growth()
        for exp in featuresets_exp_list
    ]

    # Fix exp 1824's loading record.
    featuresets_exp_list = [
        exp if exp.idx != 1824 else fix_exp_1824_loading_record(exp)
        for exp in featuresets_exp_list
    ]

    # Create features, targets, and leave-one-out splits
    # using models.extract_features_targets_splits().
    features, targets, splits = models.extract_features_targets_splits(
        featuresets_exp_list,
        dropna=True,
        as_class=as_class,
        as_1d=as_1d,
    )
    targets = targets[~targets.index.duplicated(keep='first')]

    return features, targets, splits


def plot_confusion_matrix(
        cm_df,
        title=None,
        cmap=pyplot.cm.Blues,
        fontsize=6,
        fontcolor=None,
        num_round=4,
        plot_top=0.88,
        cbar_ticks=None,
        cbar_min_divisor=2,
        figsize=None
):
    """
    Create and return a matplotlib figure representing a confusion matrix.

    Input:
        cm_df : pandas.DataFrame
            a pandas dataframe representing a confusion matrix
        title : str
            a plot title
        cmap : color map
            some pyplot colormap to use in plotting
        fontsize : int
            how large the text in each posititon of the matrix should be
        fontcolor : str
            the color that the text in each position of the matrix
    Return: pyplot.figure
        a figure object representing the plot
"""

    # Set figure title.
    if title is None:
        title = 'Confusion matrix'

    # Set figure fontcolor.
    if fontcolor is None:
        fontcolor = "black"
    
    if figsize is None:
        figsize = (14, 10)

    conf_mat = cm_df.as_matrix()
    conf_mat_nozeros = cm_df.copy()
    #     conf_mat_nozeros['Sum'] = 0
    #     conf_mat_nozeros.loc['Sum'] = 0
    conf_mat_nozeros = conf_mat_nozeros.as_matrix()

    # Get class names.
    classes = cm_df.index

    # Set color bar ticks and format their labels.
    if cbar_ticks is None:
        cbar_ticks = [0.001, 0.01, 0.1, 0.2, 0.4, 0.8, 1.0]
    cbar_tick_labels = [str(label) for label in cbar_ticks]

    # Set color bar minimum and maximum.
    cbar_min = np.min(
        [i for i in cm_df.values.ravel() if i > 0]) / cbar_min_divisor
    cbar_max = np.max([i for i in cm_df.values.ravel() if i < 1])

    # Eliminate actual zeros from plotting data.
    for i, row in enumerate(conf_mat):
        for j, col in enumerate(row):
            if col < cbar_min:
                conf_mat_nozeros[i, j] = cbar_min

    # Initialize figure and axes objects and plot colored cells.
    fig, ax = pyplot.subplots(
        1,
        1,
        figsize=figsize
    )
    cax = ax.imshow(
        conf_mat_nozeros,
        interpolation='nearest',
        cmap=cmap,
        norm=LogNorm(vmin=cbar_min, vmax=cbar_max)
    )

    # Add color bar, figure title, labels, and axis ticks.
    cbar = fig.colorbar(cax, ax=ax, ticks=cbar_ticks)
    cbar.ax.set_yticklabels(cbar_tick_labels)
    fig.suptitle(
        title,
        **{'x': 0.53, 'y': 0.97}
    )
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    ticks = list(range(len(classes)))
    pyplot.xticks(ticks, classes, rotation=45)
    pyplot.yticks(ticks, classes)
    ax.tick_params(axis=u'both', which=u'both', length=0)

    # Add numerical values to the matrix's cells.
    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            ax.text(
                j,
                i,
                # conf_mat[i, j],
                '{results:.{digits}f}'.format(
                    results=conf_mat[i, j],
                    digits=num_round
                    ),
                horizontalalignment="center",
                color=fontcolor,
                fontsize=fontsize
            )

#     # Add borders to the summation row and column.
#     p = patches.Rectangle(
#         (-0.45, 6.5),
#         7.9,
#         0.95,
#         fill=False,
#         linewidth=3,
#         edgecolor='black'
#     )
#     ax.add_patch(p)
#     p = patches.Rectangle(
#         (6.5, -0.45),
#         0.95,
#         7.9,
#         fill=False,
#         linewidth=3,
#         edgecolor='black'
#     )
#     ax.add_patch(p)

    # Format final image for saving.
    pyplot.tight_layout()
    pyplot.subplots_adjust(top=plot_top)

    return fig, ax

def fix_exp_1824_loading_record(exp_1824):
    """Fix the problem-child: experiment 1824. It's the one experiment where 
    Nick rotated the chip just before spotting to test if the fluid-flow on
    the two different halves of the chip were responsible for the different
    categories of zntA responses."""
    exp_1798 = dynomics.Experiment(1798)

    # Rename the cell_trap's index to look like all the other experiments.
    ct_df = exp_1824.cell_trap.copy()
    ct_df.index = exp_1798.cell_trap.index.copy()
    new_exp_1824 = exp_1824.set(cell_trap = ct_df)

    # Replace its loading record, just in case.
    new_exp_1824 = new_exp_1824.set(
        loading_record = exp_1798.loading_record.copy()
    )
    return new_exp_1824


def drop_sj_inductions(exp, filter_following_ind=False, traj_type=None):

    if traj_type is None:
        traj_type = 'cell_trap'

    # Define SJ River water stock indices.
    sjr_stock_indices_set = {1502, 1497, 1655, 1667}

    # Make a copy of the experiment's induction record.
    IR = exp.induction_record.copy()

    # Check whether experiment included San Juan River inductions.
    # If not, then return the original experiment object.
    if len(
        sjr_stock_indices_set.intersection(
            set(IR.stock_idx.unique())
        )
    ) == 0:
        exp_cleaned = deepcopy(exp)

        return exp_cleaned, []

    else:
        # Get the list of the induction indices for all inductions where
        # San Juan River water was NOT present. Subset the experiment to
        # only these inductions and return the new experiment object.
        sj_induction_idx_series =\
            IR.stock_idx.isin(sjr_stock_indices_set)

        sj_inductions_list = IR[
                sj_induction_idx_series
            ].index.tolist()

        not_sj_inductions_list = IR[
                ~sj_induction_idx_series
            ].index.tolist()

        # If only dropping the SJ inductions, then make a list of all
        # non-SJ induction indices and return a new experiment object
        # with only those inductions.
        if not filter_following_ind:
            exp_cleaned = exp.get_induction(traj_type, *not_sj_inductions_list)
        # If dropping SJ AND the following uninductions, make two lists of
        # SJ induction indices and the following induction indices.
        else:

            sj_next_inductions_position_list = [
                np.where(
                    IR.index.values == ind_idx
                )[0][0]+1
                for ind_idx in sj_inductions_list
            ]

            sj_next_inductions_list =\
                IR.iloc[
                    sj_next_inductions_position_list
                ].index.tolist()

            sj_related_inds_todrop_list =\
                sj_inductions_list + sj_next_inductions_list

            not_sj_related_inductions_list = [
                ind_idx for ind_idx in IR.index.tolist()
                if ind_idx not in sj_related_inds_todrop_list
            ]
            exp_cleaned = exp.get_induction(
                traj_type, *not_sj_related_inductions_list
            )

        return exp_cleaned, sj_inductions_list


def drop_bad_inductions(exp, filter_following_ind=False, traj_type=None):

    if traj_type is None:
        traj_type = 'cell_trap'

    # Make a copy of the experiment's induction record.
    IR = exp.induction_record.copy()

    # If all inductions are good, then return the original experiment.
    if len(IR.good_induction.unique()) == 1:
        return exp, []

    # Get a pandas.Series of booleans indicating the induction indices for
    # all GOOD inductions.
    good_induction_idx_series =\
        IR.good_induction.isin([1])

    # If only dropping the BAD inductions, then make a list of all
    # GOOD induction indices and return a new experiment object
    # with only those inductions.
    if not filter_following_ind:
        good_inductions_list = IR[
            good_induction_idx_series
        ].index.tolist()
        return exp.get_induction(
            traj_type, *good_inductions_list
            ), [i for i in exp.inductions if i not in good_inductions_list]

    # If dropping both BAD and the following uninductions, make two lists.
    else:
        bad_inductions_list = IR[
            ~good_induction_idx_series
        ].index.tolist()

        bad_next_inductions_position_list = [
            np.where(
                IR.index.values == ind_idx
            )[0][0]+1
            for ind_idx in bad_inductions_list
        ]

        # Try to get a list of all BAD/BAD-related induction
        # indices. If the LAST induction is BAD, an IndexError
        # will occur and the 'except' commands will be executed.
        try:
            bad_next_inductions_list =\
                IR.iloc[
                    bad_next_inductions_position_list
                ].index.tolist()
        except IndexError:
            bad_next_inductions_list =\
                IR.iloc[
                    bad_next_inductions_position_list[:-1]
                ].index.tolist()

        # Combine all BAD-related inductions and then use them
        # to filter the GOOD-only inductions.
        bad_related_inds_todrop_list =\
            bad_inductions_list + bad_next_inductions_list
        not_bad_related_inductions_list = [
            ind_idx for ind_idx in IR.index.tolist()
            if ind_idx not in bad_related_inds_todrop_list
        ]
        return exp.get_induction(
            traj_type, *not_bad_related_inductions_list
            ), bad_related_inds_todrop_list


def random_guessing(
    features,
    targets,
    num_repeats=None,
    dummy_classifier_strategies=None,
    f1_averaging_methods=None,
    sig_digits=5,
):

    if num_repeats is None:
        num_repeats = 100
    if dummy_classifier_strategies is None:
        dummy_classifier_strategies = ['most_frequent']
    if f1_averaging_methods is None:
        f1_averaging_methods = ['micro', 'weighted']


    dummy_f1_scores = {
        strategy : {f1_method : 0 for f1_method in f1_averaging_methods}
        for strategy in dummy_classifier_strategies
    }
    dummy_accuracy_scores = {
        strategy : 0
        for strategy in dummy_classifier_strategies
    }
    if len(np.unique(targets)) == 2:
        dummy_roc_auc_scores = {
            strategy : 0
            for strategy in dummy_classifier_strategies
        }

    for dummy_classifier_strategy in dummy_classifier_strategies:

            mean_dummy_f1_score_micro = 0
            mean_dummy_f1_score_weighted = 0
            mean_dummy_roc_auc_score = 0
            mean_dummy_accuracy_score = 0

            for i in range(num_repeats):

                # Initialize a DummyClassifier and fit it.
                clf = DummyClassifier(
                    strategy=dummy_classifier_strategy
                ).fit(
                    features,
                    targets
                )

                # Make random predictions based on the class distributions
                # in the training data.
                predictions = clf.predict(features)

                # Calculate an F1_micro score from the data for comparison with the
                # actual classifiers.
                mean_dummy_f1_score_micro += metrics.f1_score(
                    targets,
                    predictions,
                    labels=targets,
                    average='micro'
                )
                mean_dummy_f1_score_weighted += metrics.f1_score(
                    targets,
                    predictions,
                    labels=targets,
                    average='weighted'
                )
                mean_dummy_accuracy_score += metrics.accuracy_score(
                    targets,
                    predictions
                )
                if len(np.unique(targets)) == 2:
                    mean_dummy_roc_auc_score += metrics.roc_auc_score(
                        targets,
                        predictions,
                        average='weighted'
                    )

            dummy_f1_scores[dummy_classifier_strategy]['micro'] =\
                mean_dummy_f1_score_micro / num_repeats
            dummy_f1_scores[dummy_classifier_strategy]['weighted'] =\
                mean_dummy_f1_score_weighted / num_repeats
            dummy_accuracy_scores[dummy_classifier_strategy] =\
                mean_dummy_accuracy_score / num_repeats
            if len(np.unique(targets)) == 2:
                dummy_roc_auc_scores[dummy_classifier_strategy] =\
                    mean_dummy_roc_auc_score / num_repeats

    results_list = []
    for dc_strategy in dummy_classifier_strategies:

        for f1_method in f1_averaging_methods:

                results_list.append(
                    '{0: <42}'.format(
                        '{}, F1-score, {}-average:'.format(dc_strategy, f1_method)
                    )
                )
                results_list.append(
                    '{number:.{digits}f}'.format(
                        number=dummy_f1_scores[dc_strategy][f1_method],
                        digits=sig_digits,
                    )
                )

        results_list.append(
            '{0: <42}'.format(
                '{}, accuracy-score:'.format(dc_strategy)
            )
        )
        results_list.append(
            '{number:.{digits}f}'.format(
            number=dummy_accuracy_scores[dc_strategy],
            digits=sig_digits,
            )
        )

        if len(np.unique(targets)) == 2:
            results_list.append(
                '{0: <42}'.format(
                    '{}, ROC_AUC-score:'.format(dc_strategy)
                )
            )
            results_list.append(
                '{number:.{digits}f}'.format(
                number=dummy_roc_auc_scores[dc_strategy],
                digits=sig_digits,
                )
            )

    return results_list


def print_full_classification_metrics(ground_truth, predictions, sig_digits=5):

    results_list = []
    results_list.append(
        '{0: <42}'.format('classifier, Kappa-score:')
    )
    results_list.append(
        '{number:.{digits}f}'.format(
            number=metrics.cohen_kappa_score(ground_truth, predictions),
            digits=sig_digits,
        )
    )
    results_list.append(
        '{0: <42}'.format('classifier, F1-score, micro-average')
    )
    results_list.append(
        '{number:.{digits}f}'.format(
            number=metrics.f1_score(ground_truth, predictions, average='micro'),
            digits=sig_digits
        )
    )
    results_list.append(
        '{0: <42}'.format('classifier, F1-score, weighted-average:')
    )
    results_list.append(
        '{number:.{digits}f}'.format(
            number=metrics.f1_score(ground_truth, predictions, average='weighted'),
            digits=sig_digits
        )
    )
    results_list.append(
        '{0: <42}'.format('classifier, accuracy-score:')
    )
    results_list.append(
        '{number:.{digits}f}'.format(
            number=metrics.accuracy_score(ground_truth, predictions),
            digits=sig_digits
        )
    )
    if len(np.unique(ground_truth))==2:
        results_list.append(
            '{0: <42}'.format('classifier, ROC_AUC-score:')
        )
        results_list.append(
            '{number:.{digits}f}'.format(
                number=metrics.roc_auc_score(ground_truth, predictions, average='weighted'),
                digits=sig_digits
            )
        )

    return results_list


def pickel_top_k_strains(k, exps=None, feature_type=None, data_save_dir=None):
    """
    k : int
        Number of top-k strains to select.
    exps : [dynomics.Experiment], default None
        List of Experiments from which to select top-k strains.
        Needs to be a keyword argument for parallel processing.
    data_save_dir: str, default None
        Path to directory where data needs to be saved. Needs to
        be keywork argument for parallel processing.
    """

    exp = exps[0]
    if feature_type is None:
        feature_type = 'smooth``^2'

    top_strains = exp.top_k_strains(
        other_exps=exps[1:],
        tf=transformations.feature_types[feature_type],
        k=k)

    # Save the top-k strains.
    misc.pickle_dump(
        data_save_dir + 'top_{k}_strains.pickle'.format(k=k),
        top_strains
    )

    return
