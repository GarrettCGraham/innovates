#!/Users/garrettgraham/anaconda3/bin/python


"""
20210428. C
"""


import copy
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from datetime import datetime

import warnings
warnings.filterwarnings('ignore')


##############################################################################
     ####################### DEFINE PARAMETERS #######################
##############################################################################

PROJECT_DIR = "/Users/garrettgraham/repos/soilqc_anomaly_detection"

TAGS_SET = {
    'Acclima-Zero', 'Acclima-Toohigh', 'Acclima-Too high', 'Acclima-NoPrcpResponse', 
    'Acclima-FrozenRecovery', 'Acclima-Noise', 'Acclima-Failure',
    'Acclima-Spike', 'Acclima-DiurnalNoise', 'Acclima-Erratic', 
    'Acclima-Static'
}


##############################################################################
       ####################### DEFINE FUNCTIONS #######################
##############################################################################

def clean_tags_dataframe(df_targets):
    
    """
    This function takes a target data frame and replaces the tags with their cleaned-up, space-less versions.
    """
    
    # Make a copy of the dataframe so we don't overwrite the original.
    df_targets_cleaned = copy.deepcopy(df_targets)
    
    # Loop through all the cleaned versions of the tags and replace the 
    # original versions, which have extra whitespace pre-pended to them, with
    # the cleaned versions.
    for tag in TAGS_SET:
        df_targets_cleaned.replace(
            to_replace=" "+tag,
            value=tag,
            inplace=True,
        )
    
    # Replace "None" tags with an empty string.
    df_targets_cleaned.replace(
        to_replace=[None],
        value=[""],
        inplace=True,
    )
    
    return df_targets_cleaned
    
    
def rename_tags_in_df(df_targets):
    """
    Replaces 'Acclima-Spike' with 'spike' and noise-related Acclima tags with 
    'noise'. Returns a dataframe with renamed tags.
    """
    df_targets_renamed = copy.deepcopy(df_targets)
    
    # Rename SPIKES.
    df_targets_renamed.replace(
        to_replace="Acclima-Spike",
        value="spike",
        inplace=True,
    )
    # Rename NOISE.
    noise_tag_list = [
        "Acclima-Noise",
        "Acclima-Diurnal Noise", 
        "Acclima-FrozenRecovery", 
        "Acclima-Erratic",
    ]
    for noise_tag in noise_tag_list:
        df_targets_renamed.replace(
            to_replace=noise_tag,
            value="noise",
            inplace=True,
        )
    return df_targets_renamed


def get_station_dataframe(station_id_num, df_acclima):
    # Subset down to the single ACCLIMA station of interest.
    df_station = df_acclima[df_acclima.WBANNO.eq(station_id_num)]
    return df_station


def reduce_station_df_and_convert_dates(df_station):
    
    # Subset down to just the columns of interest.
    df_station =\
        df_station[
            [
                "UTC_START", "NAME", "VALUE", 
                "TAGS_00", "TAGS_01", "TAGS_02", "TAGS_03"
            ]
        ]
    
    # Convert all datetimes to actual datetime datatypes.
    df_station.UTC_START =\
        pd.to_datetime(df_station.UTC_START, format="%Y-%m-%d %H:%M:%S")
    
    return df_station


def get_station_targets(df_station):

    # Get just the station's targets.
    df_station_targets = df_station[
        ["TAGS_00", "TAGS_01", "TAGS_02", "TAGS_03"]
    ]

    # Clean up and then rename the targets.
    df_station_targets =\
            rename_tags_in_df(
                clean_tags_dataframe(
                    df_station[["TAGS_00", "TAGS_01", "TAGS_02", "TAGS_03"]]
                )
            )
    
    return df_station_targets


def get_filtered_targets(df_station_targets):
    
    # Isolate the last three columns of targets. If there's a row where they're 
    # non-empty, then that's indicative of a multilabel example. Our goal here 
    # is to eliminate all multilabel examples.
    array_multilabel_targets = df_station_targets.iloc[:, 1:].values

    # Iterate through the rows of multilabel targets and concatenate all tags 
    # into a single string. For rows that don't have any multilabel tag, the 
    # resulting entry will be an empty string of length 0. For multilabel rows, 
    # there will be a string with non-zero length.
    arr_tags_concatenated = np.array(
        ["".join(row) for row in array_multilabel_targets]
    )

    # Iterate through the concatenated tags and calculate their lengths. 
    # These lengths will be stored in the new array defined below.
    arr_concattags_lengths = np.array(
        [l for l in map(len, arr_tags_concatenated)]
    )

    # Find all zero-length elements of the array. These entries are the rows in 
    # the original targets dataframe that we want to keep, since they are the 
    # single-label (ie, non-multilabel) rows.
    arr_tags_to_keep = (arr_concattags_lengths == 0)

    # Reduce the targets dataframe to the first column.
    # This column represents all of the single-label targets.
    df_tags_reduced = df_station_targets[arr_tags_to_keep].iloc[:,0]

    # Get the final set of targets by filtering out anything that's not a 
    # spike, noise, or normal.
    df_station_targets_reduced_final =\
        df_tags_reduced[df_tags_reduced.isin(["", "spike", "noise"])]
    
    return df_station_targets_reduced_final, df_tags_reduced, arr_tags_to_keep
    
    
def get_filtered_features(df_station, df_station_targets_reduced_final):
    # Get the final set of station features by using the indices of the 
    # remaining targets.
    df_station_features =\
        df_station.loc[
        df_station_targets_reduced_final.index, ["UTC_START", "NAME", "VALUE"]
    ]
    return df_station_features


def locate_all_multilabel_rows(df_pivoted_targets_clean):
    # Concatenate all the values in each row and check for non-unique labels.
    # Ie, check for time points where there's normal/spike, etc.
    arr_tags_concatenated =\
        np.array(
            ["".join(row) for row in df_pivoted_targets_clean.values]
        )

    # Use set operations to the ID unique tags in each row.
    unique_tags_by_row = [
        list(set(row)) for row in df_pivoted_targets_clean.values
    ]

    # WELLLLL, ACTUALLY, I'm not concerned with multilabel rows that have both 
    # normal and then one tag. I'm concerned with multilabel rows that have 
    # "spike" and "noise". So, I'll concatenate the unique labels together. 
    # Anything over length 5 ( len("spike")=5 and len("noise")=5 ) will be a 
    # multilabel instance, since len("spikenoise")=10.  

    # Find all the multilabel row locations via some string method trickery.
    multilabel_row_locations = np.array(
        [
            len(
                "".join(row)    # Join all the unique tags in each row 
                                # together; ie, ["", "noise"] --> "noise", 
                                # while ["noise", "spike"] --> "noisespike"
            ) > 5               # Check for anything that has length > 5. This 
                                # will only occur where "".join(row) --> 
                                # "spikenoise" or "noisespike".
            for row in unique_tags_by_row
        ]
    )
    
    return multilabel_row_locations


def convert_targs_to_single_column(df_pivoted_targets_singlelabel):
    
    array_pivoted_targets_singlelabel = df_pivoted_targets_singlelabel.values

    # Use the list(set()) trick to filter down to the unique entries in each 
    # row of the targets array.
    unique_tags_by_row = [
        list(set(row)) for row in array_pivoted_targets_singlelabel
    ]

    # Join these unique entries together to form a single entry per target row.
    # Since each row only has either {""}, {"", "spike"} or {"", "noise"}, the 
    # result will be a single label per row.
    multiclass_targets_array = np.char.array(
        [
            "".join(row)              # Join together the unique single-labels;
                                      # ie, ["","spike"] --> "spike" 
                                      # and ["", "noise"] --> "noise".
            for row in unique_tags_by_row
        ]
    )

    # Recombine the newly-filtered multiclass targets with their original 
    # datetime index.
    df_targets_singlelabel = pd.DataFrame(
        data=multiclass_targets_array,
        columns=["TAGS"], 
        index=df_pivoted_targets_singlelabel.index, 
    )

    return df_targets_singlelabel


##############################################################################
     ####################### PROGRAM MAIN BODY #######################
##############################################################################


if __name__=="__main__":
    
    # Load the list of stations ID #s.
    data_dir = PROJECT_DIR + "/data/stations/"
    station_filenames_list = [
        filename for filename in os.listdir(path=data_dir)
        if filename!=".DS_Store"
    ]

    # Load just the list of ACCLIMA station IDs.
    file_path = PROJECT_DIR + "/data/acclima_stations_id_list.txt"
    acclima_stations_list =\
        pd.read_csv(file_path, header=None).iloc[:,0].values.tolist()
    
    # This operation takes a couple minutes, so only do it if you really need
    # to reload the stuff.
    df_expanded = pd.read_pickle(
        PROJECT_DIR + "/data/acclima_soil_water_rleeper_1214.pickle"
    )

    # The previous line loads column names as values in the first row. Set them
    # as the actual column names and then delete the first row.
    df_expanded.columns = df_expanded.iloc[0].values
    df_expanded = df_expanded.iloc[1:, :]
    
    # Subset the expanded 2020-12-14 dataset to ACCLIMA only
    # This takes about 1 minute to run, so only run when necessary.
    # Filter the expanded dataset down to just the ACCLIMA stations so that
    # it's easier to wield in memory.
    df_acclima = df_expanded.isin({"WBANNO":acclima_stations_list})
    df_acclima = df_expanded.iloc[df_acclima.WBANNO.values]

    # Delete df_expanded to free up some dang memory.
    del(df_expanded)
    
    # Rename the ACCLIMA DF's TAG columns to not be "TAGS, NaN, NaN, NaN".
    df_acclima.columns =\
            df_acclima.columns[:9].tolist() +\
            ["TAGS_00", "TAGS_01", "TAGS_02", "TAGS_03",]
    
    ####################### THE MEAT #######################
    # Loop through all the stations, process their features and targets, and 
    # then print out their resulting stats (number of unique targets, target 
    # kinds, etc.) so I can look for problems later. Finally, cache the 
    # features and the targets for later use.
    for station_list_idx in range(len(acclima_stations_list)):

        station_id_num = acclima_stations_list[station_list_idx]
        print(
            "########################", 
            station_list_idx, ":", station_id_num,
            "########################"
        )

        # Get just the station of interest.
        df_station = get_station_dataframe(station_id_num, df_acclima)

        # Cut the station data down to just the columns-of-interest.
        # Convert the date-times in the UTC_START column to datetime objects.
        df_station = reduce_station_df_and_convert_dates(df_station)

        # Isolate the station targets for filtering.
        df_station_targets = get_station_targets(df_station)

        # Print sanity-check statistics.
        print("Original set of targets:                                         ", np.unique(df_station_targets.values))
        print("Original number of targets:                                      ", df_station_targets.shape[0])

        # Filter the targets down to single-label targets-of-interest (ie, just normal, "spike" and "noise").
        df_station_targets_reduced_final, df_tags_reduced, arr_tags_to_keep = get_filtered_targets(df_station_targets)

        # Print some sanity statistics.
        print()
        print("Remaining unique labels:                                         ", df_tags_reduced.unique())
        print("Original number of targets:                                      ", df_station_targets.shape[0])
        print("Number of reduced targets:                                       ", df_tags_reduced.shape[0])
        print("Number of reduced targets plus number of dropped targets:        ", (~arr_tags_to_keep).sum() + df_tags_reduced.shape[0])

        # Get the final feature-set by filtering to feature-rows that have labels remaining after the labels were filtered.
        df_station_features = get_filtered_features(df_station, df_station_targets_reduced_final)

        print()
        print("Final set of unique labels:                                      ", df_station_targets_reduced_final.unique())
        print("Number of final labels:                                          ", df_station_features.shape[0])
        print("Number of final features:                                        ",df_station_targets_reduced_final.shape[0])

        # Combine features and targets for station into single DF, then pivot the targets, isolate to just the sensors
        # by dropping the precipitation and temperature fields, and then convert any resulting NaN values in the targets
        # pivot DF to the normal label ''.
        df_station_combined = pd.concat([df_station_features, df_station_targets_reduced_final],axis=1)
        df_pivoted_targets = df_station_combined.pivot(index="UTC_START", columns="NAME", values="TAGS_00")
        df_pivoted_targets = df_pivoted_targets.drop(["p_official", "t_official"], axis=1)
        df_pivoted_targets_clean =\
            clean_tags_dataframe(df_pivoted_targets) # Clean the tags up; ie, convert all NaN to ''.

        # Drop the multilabel timepoints from the analysis. First, filter the pivoted 
        # targets DF of any multilabel row locations (ie, locations that are co-labeled "spike" and "noise").
        multilabel_row_locations = locate_all_multilabel_rows(df_pivoted_targets_clean)
        df_pivoted_targets_singlelabel =\
            df_pivoted_targets[~multilabel_row_locations]

        # Use the filtered targets dataframe's index to filter the pivoted features dataframe.
        # That way, we have only feature locations with single-label multiclass features.
        df_pivoted_features_singlelabel = df_station_features.pivot(
            index="UTC_START", columns="NAME", values="VALUE"            # Create a pivoted DF of the features.
        ).loc[
            df_pivoted_targets_singlelabel.index                         # Filter the pivoted features DF using the datetimes of the remaining targets.
        ]

        # Filter the pivoted single-label targets DF using the remaining feature DF datetime indices.
        df_pivoted_targets_singlelabel =\
            df_pivoted_targets_singlelabel.loc[
                df_pivoted_features_singlelabel.index
            ]

        # Now both the pivoted targets and the pivoted features have the same datetime indices.
        # Now I need to go back and reformat the remaining targets so that they're a 
        # single-column series, rather than a multi-dimensional dataframe.
        # Get just the remaining single-label target values.
        # I'll use these to get down to one label entry per datetime row.
        df_pivoted_targets_singlelabel =\
            clean_tags_dataframe(df_pivoted_targets_singlelabel)
        df_targets_singlelabel =\
            convert_targs_to_single_column(df_pivoted_targets_singlelabel)

        # Drop any and all feature rows that have NaN values.
        df_pivoted_features_singlelabel = df_pivoted_features_singlelabel.dropna(how="any", axis=0)

        # Filter the targets based on the remaining datetime
        # indices from the NaN-filtered pivoted features.
        df_targets_singlelabel =\
            df_targets_singlelabel.loc[
                df_pivoted_features_singlelabel.index
            ]

        print()
        print("Final set of unique labels in targets' pd.series object:         ", df_targets_singlelabel["TAGS"].unique())
        print("Number of final labels after pivoting and matching:              ", df_targets_singlelabel.shape[0])
        print("Number of final pivoted features:                                ", df_pivoted_features_singlelabel.shape[0])
        print()
        print()
        
        # Cache all the features/targets in the multiclass data directory.
        MC_DATA_DIR = PROJECT_DIR + "/data/multiclass/"
        df_pivoted_features_singlelabel.to_pickle(
            MC_DATA_DIR+"mc_features_"+str(station_id_num)+".pickle"
        )
        df_targets_singlelabel.to_pickle(
            MC_DATA_DIR+"mc_targets_"+str(station_id_num)+".pickle"
        )
