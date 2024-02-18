###########################################################################
#                                                                         #
# This file is part of RobustBrachy.                                      #
# Copyright (C) 2024 Andrew Kennedy                                       #
#                                                                         #
# RobustBrachy is free software: you can redistribute it and/or modify    #
# it under the terms of the GNU General Public License as published by    #
#  the Free Software Foundation, either version 3 of the License, or      #
#  (at your option) any later version.                                    #
#                                                                         #
# This program is distributed in the hope that it will be useful,         #
#  but WITHOUT ANY WARRANTY; without even the implied warranty of         #
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the          #
#  GNU General Public License for more details.                           #
#                                                                         #
#  You should have received a copy of the GNU General Public License      #
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.  #
#                                                                         #
###########################################################################

import numpy as np
import pandas as pd


def volume_from_slices_vectorised_cpu(structure):
    """Obtain volume from 2D slice contours then mulitply by the slice width. units of images and slices need to be in mm.

    Args:
        structure (numpy array): shape is (no. of slices, 3 {for x y z}, vector of x or y or z coordinates) Array should be in mm.

    Returns:
        float: volume of contour in cm^3 (cc)
    """

    # get the slice width
    slice_width = np.abs(structure[0, 2, 0] - structure[1, 2, 0])

    # need the polygon that is the slice contour to be enclosed with the starting and ending point being the same.
    # This adds the first point to the end of the array. Also, this takes only the X and Y coordinates, deleting the Z coordinates.
    poly_2D_in_3D_slices = np.append(
        structure[:, :2, :],
        structure[:, :2, 0].reshape(structure.shape[0], 2, 1),
        axis=2,
    )

    # using the "Shoelace formula" to find the area (https://en.wikipedia.org/wiki/Shoelace_formula)
    # find the determinate of the X and Y coordinates
    det = poly_2D_in_3D_slices[:, 0] * np.roll(
        poly_2D_in_3D_slices[:, 1], 1, axis=1
    ) - poly_2D_in_3D_slices[:, 1] * np.roll(poly_2D_in_3D_slices[:, 0], 1, axis=1)

    # then sum the determinates, find the absolute values, multiply by 0.5
    Areas = np.absolute(0.5 * (np.sum(det, axis=1)))

    # find the average of adjacent slice areas, sum them all together, and finally multiply by the thickness (slice width)
    volume = 0.5 * np.sum(Areas[:-1] + Areas[1:]) * slice_width

    return volume / 1000  # change from mm to cc


def get_metrics_for_population_cpu(
    all_DVHs_pop,
    volumes,
    prescribed_dose,
    df_dose_metrics_to_include,
    df_volume_metrics_to_include,
):
    metrics = np.zeros(
        (
            (
                len(df_dose_metrics_to_include.index)
                + len(df_volume_metrics_to_include.index)
            ),
            np.size(all_DVHs_pop, 0),
        )
    )
    for i in range(len(df_dose_metrics_to_include.index)):
        if np.isnan(df_dose_metrics_to_include.iloc[i, 0]) != True:
            # prostate
            if df_dose_metrics_to_include.iloc[i, 0] == 0:
                if df_dose_metrics_to_include.iloc[i, 2] == "%":
                    Dmetric = get_population_dose_metrics_cpu(
                        np.array([int(df_dose_metrics_to_include.iloc[i, 1])]),
                        all_DVHs_pop[:, 0],
                    )

                elif df_dose_metrics_to_include.iloc[i, 2] == "cc":
                    Dmetric = get_population_dose_metrics_cpu(
                        df_dose_metrics_to_include.iloc[i, 1] / volumes[:, 0] * 100,
                        all_DVHs_pop[:, 0],
                    )

            # Urethra
            elif df_dose_metrics_to_include.iloc[i, 0] == 1:
                if df_dose_metrics_to_include.iloc[i, 2] == "%":
                    Dmetric = get_population_dose_metrics_cpu(
                        np.array([int(df_dose_metrics_to_include.iloc[i, 1])]),
                        all_DVHs_pop[:, 1],
                    )

                elif df_dose_metrics_to_include.iloc[i, 2] == "cc":
                    Dmetric = get_population_dose_metrics_cpu(
                        df_dose_metrics_to_include.iloc[i, 1] / volumes[:, 1] * 100,
                        all_DVHs_pop[:, 1],
                    )

                # Rectum
            elif df_dose_metrics_to_include.iloc[i, 0] == 2:
                if df_dose_metrics_to_include.iloc[i, 2] == "%":
                    Dmetric = get_population_dose_metrics_cpu(
                        np.array([int(df_dose_metrics_to_include.iloc[i, 1])]),
                        all_DVHs_pop[:, 2],
                    )

                elif df_dose_metrics_to_include.iloc[i, 2] == "cc":
                    Dmetric = get_population_dose_metrics_cpu(
                        df_dose_metrics_to_include.iloc[i, 1] / volumes[:, 2] * 100,
                        all_DVHs_pop[:, 2],
                    )

            metrics[i, :] = np.round(Dmetric, 2)
    j = i + 1

    for i in range(len(df_volume_metrics_to_include.index)):
        position = j + i
        if np.isnan(df_volume_metrics_to_include.iloc[i, 0]) != True:
            # prostate
            if df_volume_metrics_to_include.iloc[i, 0] == 0:
                if df_volume_metrics_to_include.iloc[i, 2] == "%":
                    Vmetric = get_population_volume_metrics_cpu(
                        np.array(
                            [
                                int(df_volume_metrics_to_include.iloc[i, 1])
                                / 100
                                * prescribed_dose
                            ]
                        ),
                        all_DVHs_pop[:, 0],
                    )
                    if df_volume_metrics_to_include.iloc[i, 5] == "cc":
                        Vmetric = Vmetric / 100 * volumes[:, 0]

                elif df_volume_metrics_to_include.iloc[i, 2] == "Gy":
                    Vmetric = get_population_volume_metrics_cpu(
                        np.array([df_volume_metrics_to_include.iloc[i, 1]]),
                        all_DVHs_pop[:, 0],
                    )
                    if df_volume_metrics_to_include.iloc[i, 5] == "cc":
                        Vmetric = Vmetric / 100 * volumes[:, 0]

            # Urethra
            if df_volume_metrics_to_include.iloc[i, 0] == 1:
                if df_volume_metrics_to_include.iloc[i, 2] == "%":
                    Vmetric = get_population_volume_metrics_cpu(
                        np.array(
                            [
                                int(df_volume_metrics_to_include.iloc[i, 1])
                                / 100
                                * prescribed_dose
                            ]
                        ),
                        all_DVHs_pop[:, 1],
                    )
                    if df_volume_metrics_to_include.iloc[i, 5] == "cc":
                        Vmetric = Vmetric / 100 * volumes[:, 1]

                elif df_volume_metrics_to_include.iloc[i, 2] == "Gy":
                    Vmetric = get_population_volume_metrics_cpu(
                        np.array([df_volume_metrics_to_include.iloc[i, 1]]),
                        all_DVHs_pop[:, 1],
                    )
                    if df_volume_metrics_to_include.iloc[i, 5] == "cc":
                        Vmetric = Vmetric / 100 * volumes[:, 1]

            # Rectum
            if df_volume_metrics_to_include.iloc[i, 0] == 2:
                if df_volume_metrics_to_include.iloc[i, 2] == "%":
                    Vmetric = get_population_volume_metrics_cpu(
                        np.array(
                            [
                                int(df_volume_metrics_to_include.iloc[i, 1])
                                / 100
                                * prescribed_dose
                            ]
                        ),
                        all_DVHs_pop[:, 2],
                    )

                    if df_volume_metrics_to_include.iloc[i, 5] == "cc":
                        Vmetric = Vmetric / 100 * volumes[:, 2]

                elif df_volume_metrics_to_include.iloc[i, 2] == "Gy":
                    Vmetric = get_population_volume_metrics_cpu(
                        np.array([df_volume_metrics_to_include.iloc[i, 1]]),
                        all_DVHs_pop[:, 2],
                    )
                    if df_volume_metrics_to_include.iloc[i, 5] == "cc":
                        Vmetric = Vmetric / 100 * volumes[:, 2]

            metrics[position, :] = np.round(Vmetric, 2)

    return metrics


def get_population_dose_metrics_cpu(volume_percentage, DVH_test):
    # needs to be in the shape (pop_size, 1, 2, DVH_size): DVH_size used is generally 1000 or 100
    # linear interpolates between values
    # can be used for a whole population of uncertainty scenarios or just one
    # volume_percentage should be in %
    if len(volume_percentage) == 1:
        val_large = np.repeat(
            volume_percentage, (DVH_test[:, 0]).shape[0] * (DVH_test[:, 0]).shape[1]
        ).reshape(DVH_test[:, 1].shape)

    else:
        val_large = np.repeat(
            volume_percentage, (DVH_test[:, 0]).shape[1], axis=0
        ).reshape(DVH_test[:, 1].shape)

    # indices that are closest to volume_percentage
    min_index = np.absolute(DVH_test[:, 1] - val_large).argmin(axis=1)

    # the clostest volume values to volume_percentage
    Y1_V = (DVH_test[:, 1])[np.arange(len(DVH_test[:, 1])), min_index]

    # masks for values that are an under estimate or over estimate of 90 (linear interpolaiton below)
    Y1_V_low_mask = (Y1_V < volume_percentage).astype(np.bool_)
    Y1_V_exact_mask = (Y1_V == volume_percentage).astype(np.bool_)
    Y1_V_high_mask = ~(Y1_V_low_mask + Y1_V_exact_mask).astype(np.bool_)

    # the clostest dose values corresponging to Y1_V above
    X1_D = (DVH_test[:, 0])[np.arange(len(DVH_test[:, 0])), min_index]

    # the next dose and volume values above
    # the previous dose and volume values below
    Y0_V = (DVH_test[:, 1])[np.arange(len(DVH_test[:, 1])), min_index - 1]
    X0_D = (DVH_test[:, 0])[np.arange(len(DVH_test[:, 0])), min_index - 1]

    mask = (min_index + 1 >= (DVH_test[:, 1].shape[1])).astype(np.bool_)
    min_index[mask] = DVH_test[:, 1].shape[1] - 2

    Y2_V = (DVH_test[:, 1])[np.arange(len(DVH_test[:, 1])), min_index + 1]
    X2_D = (DVH_test[:, 0])[np.arange(len(DVH_test[:, 0])), min_index + 1]

    slope = (Y0_V - Y1_V) / (X0_D - X1_D)
    slope_mask = (slope == 0).astype(np.bool_)

    Dmetric_pop_below = (
        np.nan_to_num(((1 / slope) * (volume_percentage - Y0_V) + X0_D) * (~slope_mask))
        + ((X0_D + X1_D) / 2) * slope_mask
    )

    slope = (Y2_V - Y1_V) / (X2_D - X1_D)
    slope_mask = (slope == 0).astype(np.bool_)

    Dmetric_pop_above = (
        np.nan_to_num(((1 / slope) * (volume_percentage - Y2_V) + X2_D) * (~slope_mask))
        + ((X2_D + X1_D) / 2) * slope_mask
    )

    Dmetric_pop = (
        Dmetric_pop_above * Y1_V_high_mask
        + Dmetric_pop_below * Y1_V_low_mask
        + X1_D * Y1_V_exact_mask
    )

    return Dmetric_pop


def get_population_volume_metrics_cpu(dose_of_interest, DVH_test):
    # needs to be in the shape (pop_size, 1, 2, DVH_size): DVH_size used is generally 1000 or 100
    # linear interpolates between values
    # can be used for a whole population of uncertainty scenarios or just one
    # dose_of_interest should be in Gy
    if len(dose_of_interest) == 1:
        val_large = np.repeat(
            dose_of_interest, (DVH_test[:, 0]).shape[0] * (DVH_test[:, 0]).shape[1]
        ).reshape(DVH_test[:, 1].shape)

    else:
        val_large = np.repeat(
            dose_of_interest, (DVH_test[:, 0]).shape[1], axis=0
        ).reshape(DVH_test[:, 1].shape)

    # indices that are closest to 90

    min_index = np.absolute(DVH_test[:, 0] - dose_of_interest).argmin(axis=1)

    # the closest volume values to 90
    Y1_D = (DVH_test[:, 0])[np.arange(len(DVH_test[:, 0])), min_index]

    # masks for values that are an under estimate or over estimate of 90 (linear interpolaiton below)
    Y1_D_low_mask = (Y1_D < dose_of_interest).astype(np.bool_)
    Y1_D_exact_mask = (Y1_D == dose_of_interest).astype(np.bool_)
    Y1_D_high_mask = ~(Y1_D_low_mask + Y1_D_exact_mask).astype(np.bool_)

    # the closest dose values corresponding to Y1_D above
    X1_V = (DVH_test[:, 1])[np.arange(len(DVH_test[:, 1])), min_index]

    # the next dose and volume values above
    # the previous dose and volume values below
    Y0_D = (DVH_test[:, 0])[np.arange(len(DVH_test[:, 0])), min_index - 1]
    X0_V = (DVH_test[:, 1])[np.arange(len(DVH_test[:, 1])), min_index - 1]

    mask = (min_index + 1 >= (DVH_test[:, 0].shape[1])).astype(np.bool_)
    min_index[mask] = DVH_test[:, 0].shape[1] - 2

    Y2_D = (DVH_test[:, 0])[np.arange(len(DVH_test[:, 0])), min_index + 1]
    X2_V = (DVH_test[:, 1])[np.arange(len(DVH_test[:, 1])), min_index + 1]

    slope = (Y0_D - Y1_D) / (X0_V - X1_V)
    slope_mask = (slope == 0).astype(np.bool_)

    Vmetric_pop_below = (
        np.nan_to_num(((1 / slope) * (dose_of_interest - Y0_D) + X0_V) * (~slope_mask))
        + ((X0_V + X1_V) / 2) * slope_mask
    )

    slope = (Y2_D - Y1_D) / (X2_V - X1_V)
    slope_mask = (slope == 0).astype(np.bool_)

    Vmetric_pop_above = (
        np.nan_to_num(((1 / slope) * (dose_of_interest - Y2_D) + X2_V) * (~slope_mask))
        + ((X2_V + X1_V) / 2) * slope_mask
    )

    Vmetric_pop = (
        Vmetric_pop_above * Y1_D_high_mask
        + Vmetric_pop_below * Y1_D_low_mask
        + X1_V * Y1_D_exact_mask
    )
    Vmetric_pop = np.array(Vmetric_pop, dtype=np.float32)

    return Vmetric_pop


# makes dose and volume metric labels for the dataframe tables
def make_dvh_metric_labels(df_dose_metrics, df_volume_metrics):
    passrates_labels_D = []
    metric_labels_to_print_D = []
    metric_labels_D = []
    passrates_labels_V = []
    metric_labels_to_print_V = []
    metric_labels_V = []

    # produce labels for the results tables
    for i in range(len(df_dose_metrics.index)):
        if np.isnan(df_dose_metrics.iloc[i, 0]) != True:
            if df_dose_metrics.iloc[i, 0] == 0:
                structure = "Prostate"
            elif df_dose_metrics.iloc[i, 0] == 1:
                structure = "Urethra"
            elif df_dose_metrics.iloc[i, 0] == 2:
                structure = "Rectum"
        else:
            continue
        if df_dose_metrics.iloc[i, 2] == "%":
            D = int(df_dose_metrics.iloc[i, 1])

        elif df_dose_metrics.iloc[i, 2] == "cc":
            D = df_dose_metrics.iloc[i, 1]

        if df_dose_metrics.iloc[i, 5] == "%":
            constraint = int(df_dose_metrics.iloc[i, 4])

        elif df_dose_metrics.iloc[i, 5] == "Gy":
            constraint = df_dose_metrics.iloc[i, 4]
        metric_labels_D.append(
            str(structure[:1])
            + ": D_"
            + str(D)
            + str(df_dose_metrics.iloc[i, 2])
            + " "
            + df_dose_metrics.iloc[i, 3]
            + " "
            + str(constraint)
            + str(df_dose_metrics.iloc[i, 5])
            + "; units: (Gy)"
        )
        metric_labels_to_print_D.append(
            "<b>"
            + str(structure)
            + "</b>: D<sub>"
            + str(D)
            + " "
            + str(df_dose_metrics.iloc[i, 2])
            + "</sub> "
            + df_dose_metrics.iloc[i, 3]
            + " "
            + str(constraint)
            + " "
            + str(df_dose_metrics.iloc[i, 5])
        )
        passrates_labels_D.append("Pass-rate (%): D_" + str(D) + ", " + str(structure))

    for i in range(len(df_volume_metrics.index)):
        if np.isnan(df_volume_metrics.iloc[i, 0]) != True:
            if df_volume_metrics.iloc[i, 0] == 0:
                structure = "Prostate"
            elif df_volume_metrics.iloc[i, 0] == 1:
                structure = "Urethra"
            elif df_volume_metrics.iloc[i, 0] == 2:
                structure = "Rectum"
        else:
            continue

        if df_volume_metrics.iloc[i, 2] == "%":
            V = int(df_volume_metrics.iloc[i, 1])

        elif df_volume_metrics.iloc[i, 2] == "cc":
            V = df_volume_metrics.iloc[i, 1]

        if df_volume_metrics.iloc[i, 5] == "%":
            constraint = int(df_volume_metrics.iloc[i, 4])

        elif df_volume_metrics.iloc[i, 5] == "cc":
            constraint = df_volume_metrics.iloc[i, 4]

        metric_labels_V.append(
            str(structure[:1])
            + ": V_"
            + str(V)
            + str(df_volume_metrics.iloc[i, 2])
            + " "
            + df_volume_metrics.iloc[i, 3]
            + " "
            + str(constraint)
            + str(df_volume_metrics.iloc[i, 5])
            + "; units: ("
            + str(df_volume_metrics.iloc[i, 5])
            + ")"
        )
        metric_labels_to_print_V.append(
            "<b>"
            + str(structure)
            + "</b>: V<sub>"
            + str(V)
            + " "
            + str(df_volume_metrics.iloc[i, 2])
            + "</sub> "
            + df_volume_metrics.iloc[i, 3]
            + " "
            + str(constraint)
            + " "
            + str(df_volume_metrics.iloc[i, 5])
        )
        passrates_labels_V.append("Pass-rate (%): V_" + str(V) + ", " + str(structure))
    return (
        passrates_labels_D,
        metric_labels_D,
        metric_labels_to_print_D,
        passrates_labels_V,
        metric_labels_V,
        metric_labels_to_print_V,
    )


# ** calculate parameter variable SD **
def calculate_parameter_variable_SDs(uncertainty_magnitudes):
    # Parameter 1 Standard Deviation: Dwells points move along the needle
    dwells_shift_along_needle = np.round(
        np.sqrt(
            float(uncertainty_magnitudes["P1_needle_recon"]) ** 2
            + float(uncertainty_magnitudes["P1_needle_rigid"]) ** 2
            + float(uncertainty_magnitudes["P1_source_position"]) ** 2
        ),
        1,
    )

    # Parameter 2 Standard Deviation: Prostate boundary expansion/contraction from geometric mean (prostate centre) in 3D
    mm_change_shrink_enlarge_prostate = np.round(
        float(uncertainty_magnitudes["P2_prostate_contour_expand_shrink"]), 1
    )

    # Parameter 3a Standard Deviation: Urethra boundary expanded/contracted in each transverse slice
    mm_change_shrink_enlarge_urethra = np.round(
        float(uncertainty_magnitudes["P3a_urethra_contour_expand_shrink"]), 2
    )

    # Parameter 3b Standard Deviation: Rectum boundary expanded/contracted in each transverse slice
    mm_change_shrink_enlarge_rectum = np.round(
        float(uncertainty_magnitudes["P3b_rectum_contour_expand_shrink"]), 1
    )

    # Parameter 4 Standard Deviation: Needle transverse plane movements
    mm_change_dwells_random_2D = np.round(
        np.sqrt(
            float(uncertainty_magnitudes["P4_needle_recon_2D"]) ** 2
            + float(uncertainty_magnitudes["P4_needle_rigid_2D"]) ** 2
        ),
        1,
    )

    # Parameter 5a Standard Deviation: Dwell times add constant value
    dwell_time_increase_decrease = np.round(
        float(uncertainty_magnitudes["P5a_dwell_time_precision"]), 2
    )

    # Parameter 5b Standard Deviation: Dwell times percentage change
    dwell_time_change_percentage = np.round(
        np.sqrt(
            float(uncertainty_magnitudes["P5b_dose_calc_med_percent"]) ** 2
            + float(uncertainty_magnitudes["P5b_dose_calc_TPS_percent"]) ** 2
            + float(uncertainty_magnitudes["P5b_source_activity"]) ** 2
        ),
        1,
    )

    # Parameter 6 Standard Deviation: Prostate and Urethra (+ needles) anterior-posterior movement
    mm_change_X_rigid = np.round(float(uncertainty_magnitudes["P6_LR_rigid"]), 2)
    mm_change_Y_rigid = np.round(float(uncertainty_magnitudes["P6_AP_rigid"]), 2)
    mm_change_Z_rigid = np.round(float(uncertainty_magnitudes["P6_SI_rigid"]), 2)

    # creating dictionary of uncertainties
    uncertainties_SD = {
        "dwells_shift_along_needle": dwells_shift_along_needle,
        "mm_change_shrink_enlarge_prostate": mm_change_shrink_enlarge_prostate,
        "mm_change_shrink_enlarge_urethra": mm_change_shrink_enlarge_urethra,
        "mm_change_shrink_enlarge_rectum": mm_change_shrink_enlarge_rectum,
        "mm_change_dwells_random_2D": mm_change_dwells_random_2D,
        "dwell_time_increase_decrease": dwell_time_increase_decrease,
        "dwell_time_change_percentage": dwell_time_change_percentage,
        "mm_change_Y_rigid": mm_change_Y_rigid,
        "mm_change_X_rigid": mm_change_X_rigid,
        "mm_change_Z_rigid": mm_change_Z_rigid,
    }
    return uncertainties_SD


def calculate_all_nominal_metrics_cpu(
    df_dose_metrics,
    df_volume_metrics,
    metric_labels_D,
    metric_labels_V,
    all_nominal_dvhs,
    plan_parameters,
):
    metrics = []
    for i in range(len(df_dose_metrics.index)):
        if np.isnan(df_dose_metrics.iloc[i, 0]) != True:
            # prostate
            if df_dose_metrics.iloc[i, 0] == 0:
                if df_dose_metrics.iloc[i, 2] == "%":
                    D = int(df_dose_metrics.iloc[i, 1])

                elif df_dose_metrics.iloc[i, 2] == "cc":
                    D = (
                        df_dose_metrics.iloc[i, 1]
                        / plan_parameters["prostate_vol"]
                        * 100
                    )

                Dmetric = get_population_dose_metrics_cpu(
                    np.array([D]), np.array([all_nominal_dvhs[0][0]])
                )

            elif df_dose_metrics.iloc[i, 0] == 1:
                # Urethra
                if df_dose_metrics.iloc[i, 2] == "%":
                    D = int(df_dose_metrics.iloc[i, 1])

                elif df_dose_metrics.iloc[i, 2] == "cc":
                    D = (
                        df_dose_metrics.iloc[i, 1]
                        / plan_parameters["urethra_vol"]
                        * 100
                    )

                Dmetric = get_population_dose_metrics_cpu(
                    np.array([D]), np.array([all_nominal_dvhs[0][1]])
                )

            elif df_dose_metrics.iloc[i, 0] == 2:
                # Rectum
                if df_dose_metrics.iloc[i, 2] == "%":
                    D = int(df_dose_metrics.iloc[i, 1])

                elif df_dose_metrics.iloc[i, 2] == "cc":
                    D = df_dose_metrics.iloc[i, 1] / plan_parameters["rectum_vol"] * 100

                Dmetric = get_population_dose_metrics_cpu(
                    np.array([D]), np.array([all_nominal_dvhs[0][2]])
                )

            metrics.append(round(Dmetric[0].astype(float), 2))

    for i in range(len(df_volume_metrics.index)):
        if np.isnan(df_volume_metrics.iloc[i, 0]) != True:
            # prostate
            if df_volume_metrics.iloc[i, 0] == 0:
                if df_volume_metrics.iloc[i, 2] == "%":
                    V = (
                        int(df_volume_metrics.iloc[i, 1])
                        / 100
                        * plan_parameters["prescribed_dose"]
                    )
                    Vmetric = get_population_volume_metrics_cpu(
                        np.array([V]), np.array([all_nominal_dvhs[0][0]])
                    )
                    if df_volume_metrics.iloc[i, 5] == "cc":
                        Vmetric = Vmetric[0] / 100 * plan_parameters["prostate_vol"]
                    else:
                        Vmetric = Vmetric[0]
                elif df_volume_metrics.iloc[i, 2] == "Gy":
                    V = df_volume_metrics.iloc[i, 1]
                    Vmetric = get_population_volume_metrics_cpu(
                        np.array([V]), np.array([all_nominal_dvhs[0][0]])
                    )  # / 100 * prostate_vol
                    if df_volume_metrics.iloc[i, 5] == "cc":
                        Vmetric = Vmetric[0] / 100 * plan_parameters["prostate_vol"]
                    else:
                        Vmetric = Vmetric[0]

            elif df_volume_metrics.iloc[i, 0] == 1:
                # Urethra
                if df_volume_metrics.iloc[i, 2] == "%":
                    V = (
                        int(df_volume_metrics.iloc[i, 1])
                        / 100
                        * plan_parameters["prescribed_dose"]
                    )
                    Vmetric = get_population_volume_metrics_cpu(
                        np.array([V]), np.array([all_nominal_dvhs[0][1]])
                    )
                    if df_volume_metrics.iloc[i, 5] == "cc":
                        Vmetric = Vmetric[0] / 100 * ["urethra_vol"]
                    else:
                        Vmetric = Vmetric[0]
                elif df_volume_metrics.iloc[i, 2] == "Gy":
                    V = df_volume_metrics.iloc[i, 1]
                    Vmetric = get_population_volume_metrics_cpu(
                        np.array([V]), np.array([all_nominal_dvhs[0][1]])
                    )  # / 100 * urethra_vol
                    if df_volume_metrics.iloc[i, 5] == "cc":
                        Vmetric = Vmetric[0] / 100 * plan_parameters["urethra_vol"]
                    else:
                        Vmetric = Vmetric[0]

            elif df_volume_metrics.iloc[i, 0] == 2:
                # Rectum
                if df_volume_metrics.iloc[i, 2] == "%":
                    V = (
                        int(df_volume_metrics.iloc[i, 1])
                        / 100
                        * plan_parameters["prescribed_dose"]
                    )

                    Vmetric = get_population_volume_metrics_cpu(
                        np.array([V]), np.array([all_nominal_dvhs[0][2]])
                    )
                    if df_volume_metrics.iloc[i, 5] == "cc":
                        Vmetric = Vmetric[0] / 100 * plan_parameters["rectum_vol"]
                    else:
                        Vmetric = Vmetric[0]
                elif df_volume_metrics.iloc[i, 2] == "Gy":
                    V = df_volume_metrics.iloc[i, 1]
                    Vmetric = get_population_volume_metrics_cpu(
                        np.array([V]), np.array([all_nominal_dvhs[0][2]])
                    )  # / 100 * rectum_vol
                    if df_volume_metrics.iloc[i, 5] == "cc":
                        Vmetric = Vmetric[0] / 100 * plan_parameters["rectum_vol"]
                    else:
                        Vmetric = Vmetric[0]

            metrics.append(round(Vmetric.astype(float), 2))
    df_nominal_metric_data = pd.DataFrame(
        data=[metrics], columns=[*metric_labels_D, *metric_labels_V]
    )
    return df_nominal_metric_data


def copy_structures_cpu(plan_parameters, arr_of_dwells_times=False):
    changed_prostate_contour_pts = np.copy(
        plan_parameters["prostate_contour_pts"]
    ).astype(np.float32)
    changed_urethra_contour_pts = np.copy(
        plan_parameters["urethra_contour_pts"]
    ).astype(np.float32)
    changed_rectum_contour_pts = np.copy(plan_parameters["rectum_contour_pts"]).astype(
        np.float32
    )
    if isinstance(arr_of_dwells_times, bool):
        changed_dwell_coords = np.copy(plan_parameters["dwell_coords_TPS"]).astype(
            np.float32
        )
        changed_dwell_pts_source_end_inf = np.copy(
            plan_parameters["dwell_pts_source_end_inf_TPS"]
        ).astype(np.float32)
        changed_dwell_pts_source_end_sup = np.copy(
            plan_parameters["dwell_pts_source_end_sup_TPS"]
        ).astype(np.float32)
        changed_dwell_times = np.copy(plan_parameters["dwell_times_TPS"]).astype(
            np.float32
        )

        changed_structures = {
            "changed_dwell_coords": changed_dwell_coords,
            "changed_dwell_pts_source_end_inf": changed_dwell_pts_source_end_inf,
            "changed_dwell_pts_source_end_sup": changed_dwell_pts_source_end_sup,
            "changed_dwell_times": changed_dwell_times,
            "changed_prostate_contour_pts": changed_prostate_contour_pts,
            "changed_urethra_contour_pts": changed_urethra_contour_pts,
            "changed_rectum_contour_pts": changed_rectum_contour_pts,
        }

    else:
        changed_arr_of_dwells_times = np.copy(arr_of_dwells_times).astype(np.float32)
        changed_dwell_coords = np.copy(
            plan_parameters["dwell_coords_optimisation"]
        ).astype(np.float32)
        changed_dwell_pts_source_end_inf = np.copy(
            plan_parameters["dwell_pts_source_end_inf_optimisation"]
        ).astype(np.float32)
        changed_dwell_pts_source_end_sup = np.copy(
            plan_parameters["dwell_pts_source_end_sup_optimisation"]
        ).astype(np.float32)

        changed_structures = {
            "changed_dwell_coords": changed_dwell_coords,
            "changed_dwell_pts_source_end_inf": changed_dwell_pts_source_end_inf,
            "changed_dwell_pts_source_end_sup": changed_dwell_pts_source_end_sup,
            "changed_arr_of_dwells_times": changed_arr_of_dwells_times,
            "changed_prostate_contour_pts": changed_prostate_contour_pts,
            "changed_urethra_contour_pts": changed_urethra_contour_pts,
            "changed_rectum_contour_pts": changed_rectum_contour_pts,
        }

    return changed_structures

def arrays_to_numpy(plan_parameters):
    # float32 is used since it is more memory efficient, and GPU memory is generally limited.
    for key, value in plan_parameters.items():
        if key in [
            "prostate_contour_pts",
            "urethra_contour_pts",
            "rectum_contour_pts",
            "F_interp",
            "g_interp",
            "bezier_arc_length_points_optimisation",
            "bezier_arc_length_dists_optimisation",
            "bezier_arc_length_points_TPS",
            "bezier_arc_length_dists_TPS",
            "dwell_coords_optimisation",
            "dwell_coords_TPS",
            "dwell_times_TPS",
            "dwell_pts_source_end_sup_TPS",
            "dwell_pts_source_end_inf_TPS",
            "dwell_pts_source_end_sup_optimisation",
            "dwell_pts_source_end_inf_optimisation",
        ]:
            plan_parameters[key] = np.asarray(
                value,
                dtype=np.float32,
            )

        elif key in [
            "mask_dwell_coords_TPS",
            "mask_dwell_times_TPS",
            "mask_dwell_coords_optimisation",
            "mask_dwell_times_optimisation",
        ]:
            plan_parameters[key] = np.asarray(
                value,
                dtype=np.bool_,
            )

    return plan_parameters

__all__ = [
    "get_metrics_for_population_cpu",
    "volume_from_slices_vectorised_cpu",
    "get_population_dose_metrics_cpu",
    "get_population_volume_metrics_cpu",
    "calculate_parameter_variable_SDs",
    "make_dvh_metric_labels",
    "calculate_all_nominal_metrics_cpu",
    "copy_structures_cpu",
    "arrays_to_numpy"
]
