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
import cupy as cp
import time

from robustbrachy.robustevaluation.simulations_gpu import *
from robustbrachy.robustevaluation.fast_TG43_gpu import *
from robustbrachy.robustevaluation.utils_gpu import *

#####################################
###       Robust Evaluation       ###
#####################################


# GPU function


def probabilistic_robust_measure_gpu(
    no_of_runs,
    plan_parameters,
    df_dose_metrics_to_include,
    df_volume_metrics_to_include,
    uncertainties_SD,
    progress_bar=False,
):
    # single plan robust evaluation ## Probablisitic ##

    # calculate the nominal dvh

    # make a copy of each structure array.
    # This is needed so the structures and plan_parameters are separated,
    # which is needed in the robust evaluation use of this function.
    copied_structures = copy_structures_gpu(plan_parameters)

    # calculate the nominal DVHs for each structure (dose calculation)
    all_nominal_dvhs = fast_TG43_gpu(
        copied_structures,
        plan_parameters,
        1,
    )

    all_nominal_dvhs = cp.array([all_nominal_dvhs])
    min_vals = cp.ceil(cp.amin(all_nominal_dvhs[:, :, 1, :], axis=2))

    # do metrics need volumes calculated? This sorts it out
    prostate_volume_needed = False
    urethra_volume_needed = False
    rectum_volume_needed = False

    for i in range(len(df_dose_metrics_to_include.index)):
        # prostate
        if df_dose_metrics_to_include.iloc[i, 0] == 0:
            if df_dose_metrics_to_include.iloc[i, 2] == "cc":
                prostate_volume_needed = True

        # urethra
        if df_dose_metrics_to_include.iloc[i, 0] == 1:
            if df_dose_metrics_to_include.iloc[i, 2] == "cc":
                urethra_volume_needed = True

        # rectum
        if df_dose_metrics_to_include.iloc[i, 0] == 2:
            if df_dose_metrics_to_include.iloc[i, 2] == "cc":
                rectum_volume_needed = True

    for i in range(len(df_volume_metrics_to_include.index)):
        # prostate
        if df_volume_metrics_to_include.iloc[i, 0] == 0:
            if df_volume_metrics_to_include.iloc[i, 2] == "cc":
                prostate_volume_needed = True

        # urethra
        if df_volume_metrics_to_include.iloc[i, 0] == 1:
            if df_volume_metrics_to_include.iloc[i, 2] == "cc":
                urethra_volume_needed = True

        # rectum
        if df_volume_metrics_to_include.iloc[i, 0] == 2:
            if df_volume_metrics_to_include.iloc[i, 2] == "cc":
                rectum_volume_needed = True

    # no_of_runs can be a list of multiple possible number of runs, so when an single int is inputted, it needs to be changed to a list
    if isinstance(no_of_runs, int):
        no_of_runs = list([no_of_runs])

    # all the runs requested, say [10, 100, 500, 1000] for example will run 1000 times and then pick 500, 100, and 10 at random.
    # utility of this is so that you can see how the robust eval changes with the number of run times
    all_no_of_runs = no_of_runs

    # get highest
    no_of_runs = max(all_no_of_runs)

    # store the run times
    run_times_arr = []

    # this initiates an array of all DVHs for all trials
    all_DVHs = cp.zeros((no_of_runs, 3, 2, 1000))

    # need to store all volumes also
    all_volumes = cp.zeros((no_of_runs, 3))

    # store the DVHs but
    dose_per_vol = cp.empty((1, 3, 110, no_of_runs))
    dose_per_vol[:] = cp.nan

    # iterate through the uncertainty scenarios
    for sim_num in range(0, no_of_runs):
        start = time.time()

        # sets up the random change arrays that are used in the simulation function to apply the changes to structures, dwell times and dwell points
        sim_magnitudes = simulated_change_arrays(uncertainties_SD, plan_parameters)

        # make a copy of each array so that the structure gets reset each run.
        changed_structures = copy_structures_gpu(plan_parameters)

        # carries out scenario movements and then calculates dose values, dose metrics, and dvh arrays
        (changed_structures) = simulations_gpu(
            sim_magnitudes, changed_structures, plan_parameters
        )

        # calculate the DVHs for each structure
        DVHs = fast_TG43_gpu(
            changed_structures,
            plan_parameters,
            1,
        )

        # calculate volumes
        volumes = cp.zeros((3))
        if prostate_volume_needed == True:
            prostate_volume_uncert_senario = volume_from_slices_vectorised_gpu(
                changed_structures["changed_prostate_contour_pts"]
            )
            volumes[0] = prostate_volume_uncert_senario
        else:
            volumes[0] = cp.nan

        if urethra_volume_needed == True:
            urethra_volume_uncert_senario = volume_from_slices_vectorised_gpu(
                changed_structures["changed_urethra_contour_pts"]
            )
            volumes[1] = urethra_volume_uncert_senario
        else:
            volumes[1] = cp.nan

        if rectum_volume_needed == True:
            rectum_volume_uncert_senario = volume_from_slices_vectorised_gpu(
                changed_structures["changed_rectum_contour_pts"]
            )
            volumes[2] = rectum_volume_uncert_senario
        else:
            volumes[2] = cp.nan

        # store DVHs and volumes from uncertainty scenario
        all_DVHs[sim_num, :, :, :] = DVHs
        all_volumes[sim_num, :] = volumes

        DVHs = cp.array([DVHs])

        dose = DVHs[:, :, 0]
        vol = DVHs[:, :, 1]

        vol_array = cp.tile(
            cp.flip(
                cp.concatenate(
                    (
                        cp.arange(100),
                        cp.array(
                            [99.1, 99.2, 99.3, 99.4, 99.5, 99.6, 99.7, 99.8, 99.9, 100]
                        ),
                    )
                )
            ),
            len(dose) * len(dose[0]),
        ).reshape(len(dose), len(dose[0]), -1)

        mask = (vol_array >= min_vals[:, :, cp.newaxis]).astype(cp.bool_).flatten()

        idx_to_nearest_vol = cp.abs(
            vol[:, :, cp.newaxis, :] - vol_array[:, :, :, cp.newaxis]
        ).argmin(axis=3)
        vol_array_args = cp.argwhere(idx_to_nearest_vol > -100)

        dose_per_vol[
            vol_array_args[:, 0][mask],
            vol_array_args[:, 1][mask],
            vol_array_args[:, 2][mask],
            sim_num,
        ] = dose[
            vol_array_args[:, 0][mask],
            vol_array_args[:, 1][mask],
            idx_to_nearest_vol.flatten()[mask],
        ]

        run_times_arr.append(time.time() - start)

        # print out or progress bar
        if (int(sim_num) + 1) % 10 == 0:
            run_times_avg = np.mean(np.array(run_times_arr))

            if progress_bar != False:
                progress_bar.progress(
                    sim_num / no_of_runs,
                    text="Up to probabilistic senario "
                    + str(int(sim_num) + 1)
                    + " out of "
                    + str(no_of_runs)
                    + ", average runtime of "
                    + str(round(run_times_avg, 4))
                    + " seconds.",
                )

            else:
                print(
                    "Up to probabilistic senario "
                    + str(int(sim_num) + 1)
                    + ", completed last 20 runs in an mean time of "
                    + str(run_times_avg)
                )

    if progress_bar != False:
        run_times_avg = np.mean(np.array(run_times_arr))
        progress_bar.progress(
            1.0,
            text="Finished with an average runtime of "
            + str(round(run_times_avg, 4))
            + " seconds per scenario.",
        )

    # obtain DVH metrics
    metrics = get_metrics_for_population_gpu(
        all_DVHs,
        all_volumes,
        plan_parameters["prescribed_dose"],
        df_dose_metrics_to_include,
        df_volume_metrics_to_include,
    )

    # calculate pass rates
    metric_pass_rates = []
    all_mask_pass_rates = cp.zeros(metrics.shape)

    for i in range(len(df_dose_metrics_to_include.index)):
        if np.isnan(df_dose_metrics_to_include.iloc[i, 0]) != True:
            if df_dose_metrics_to_include.iloc[i, 3] == ">":
                if df_dose_metrics_to_include.iloc[i, 5] == "%":
                    mask_pass_rates = (
                        (
                            df_dose_metrics_to_include.iloc[i, 4]
                            / 100
                            * plan_parameters["prescribed_dose"]
                            - metrics[i, :]
                        )
                        <= 0
                    ).astype(cp.bool_)

                else:
                    mask_pass_rates = (
                        (df_dose_metrics_to_include.iloc[i, 4] - metrics[i, :]) <= 0
                    ).astype(cp.bool_)


            else:
                if df_dose_metrics_to_include.iloc[i, 5] == "%":
                    mask_pass_rates = (
                        (
                            metrics[i, :]
                            - df_dose_metrics_to_include.iloc[i, 4]
                            / 100
                            * plan_parameters["prescribed_dose"]
                        )
                        <= 0
                    ).astype(cp.bool_)

                else:
                    mask_pass_rates = (
                        (metrics[i, :] - df_dose_metrics_to_include.iloc[i, 4]) <= 0
                    ).astype(cp.bool_)


            pass_rate = cp.sum(mask_pass_rates).get() / len(mask_pass_rates) * 100
            all_mask_pass_rates[i, :] = mask_pass_rates
            metric_pass_rates.append(pass_rate)

    for j in range(len(df_volume_metrics_to_include.index)):
        if np.isnan(df_volume_metrics_to_include.iloc[j, 0]) != True:
            if df_volume_metrics_to_include.iloc[j, 3] == ">":
                mask_pass_rates = (
                    (df_volume_metrics_to_include.iloc[j, 4] - metrics[j + i + 1, :])
                    <= 0
                ).astype(cp.bool_)

            else:
                mask_pass_rates = (
                    (metrics[j + i + 1, :] - df_volume_metrics_to_include.iloc[j, 4])
                    <= 0
                ).astype(cp.bool_)

            pass_rate = cp.sum(mask_pass_rates).get() / len(mask_pass_rates) * 100
            all_mask_pass_rates[i + 1 + j, :] = mask_pass_rates
            metric_pass_rates.append(pass_rate)

    overall_pass_rate = (
        cp.sum(cp.prod(all_mask_pass_rates, axis=0)).get()
        / len(all_mask_pass_rates[0])
        * 100
    )

    # calculate DVH robustness curves (obtaining the confidence intervals for robustness along the dose axis)
    dose_per_vol[dose_per_vol == 0] = cp.nan
    mu = cp.nanmean(dose_per_vol, axis=3)
    std = cp.nanstd(dose_per_vol, axis=3)

    vols = cp.flip(
        cp.concatenate(
            (
                cp.arange(100),
                cp.array([99.1, 99.2, 99.3, 99.4, 99.5, 99.6, 99.7, 99.8, 99.9, 100]),
            )
        )
    )
    all_robust_dvhs = cp.array(
        [
            mu,
            mu - std,
            mu + std,
            mu - 1.96 * std,
            mu + 1.96 * std,
            mu - 2.576 * std,
            mu + 2.576 * std,
        ]
    )
    vol_array = cp.tile(
        vols,
        (all_robust_dvhs.shape[0], all_robust_dvhs.shape[1], all_robust_dvhs.shape[2]),
    ).reshape(all_robust_dvhs.shape)
    all_robust_dvh_summary = cp.stack((all_robust_dvhs, vol_array), axis=3)

    return (
        metric_pass_rates,
        overall_pass_rate,
        all_robust_dvh_summary.get(),
        all_nominal_dvhs.get(),
    )


def simulated_change_arrays(
    uncertainties_SD, plan_parameters, arr_of_dwells_times=False
):
    ################################# Uncertainty Values to Specify #############################################
    # Note if values are zero, passes over uncertainty senario movement

    # allocate structure coordinates dependent on single set of dwell times or array of 2D dwell times
    if isinstance(arr_of_dwells_times, bool):
        dwell_coords = plan_parameters["dwell_coords_TPS"]

    else:
        dwell_coords = plan_parameters["dwell_coords_optimisation"]

    ########## Parameter Variable 1 ##########
    ## Description: Dwells points move along the needle. Needles shift randomly Sup-Inf direction but along the needle ##
    # this is a matrix that contains a different value change per needle and applies that change to each dwell point in the needle
    # output --> array(num_needles, num_dwells_per_needle)

    # random sampling assuming normal approximation for the distribution of the uncertainty
    dwells_shift_along_needle = get_values_between_90CI(
        mu=0,
        sd=uncertainties_SD["dwells_shift_along_needle"],
        amount=cp.size(dwell_coords, 0),
    )

    # increase the number of values so that the values are equal to the number of dwells in each needle
    dwells_shift_along_needle = cp.repeat(
        dwells_shift_along_needle, cp.size(dwell_coords, 1)
    ).reshape(
        cp.size(dwell_coords, 0),
        cp.size(dwell_coords, 1),
    )

    ########## Parameter Variable 2 ##########
    ## Description: Shrink/enlarge Prostate from the 3D geometric centre point of the prostate
    # output --> array(scalar)
    mm_change_shrink_enlarge_prostate = get_values_between_90CI(
        mu=0, sd=uncertainties_SD["mm_change_shrink_enlarge_prostate"], amount=1
    ).item()

    ########## Parameter Variable 3a ##########
    ## Description: Shrink/enlarge Urethra from the 2D geometric centre point in each slice
    # output --> array(scalar)
    mm_change_shrink_enlarge_urethra = get_values_between_90CI(
        mu=0, sd=uncertainties_SD["mm_change_shrink_enlarge_urethra"], amount=1
    ).item()

    ########## Parameter Variable 3b ##########
    ## Description: Shrink/enlarge Rectum from the 2D geometric centre point in each slice
    # output --> array(scalar)
    mm_change_shrink_enlarge_rectum = get_values_between_90CI(
        mu=0, sd=uncertainties_SD["mm_change_shrink_enlarge_rectum"], amount=1
    ).item()

    ########## Parameter Variable 4 ##########
    # this is a matrix that contains a different value change per needle and applies that change to each dwell point in the needle
    # output --> array(num_needles, num_dwells_per_needle, len([X,Y,Z]))
    # generate random radius lengths, one per needle

    ## Worst-case scenario: Moving dwells out from prostate centre line - This is not a probabilistic simulation but a worst-case scenario needs to be set to zero to pass over in simulations function.
    dwells_move_out_from_prostate_centre_line = 0  # negaitve means move inwards

    ## Probabilistic scenario: needles shift randomly ##
    # this is a matrix that contains a different value change per needle and applies that change to each dwell point in the needle

    # generate random radius lengths, one per needle
    mm_change_dwells_random_2D_radius = get_values_between_90CI(
        mu=0,
        sd=uncertainties_SD["mm_change_dwells_random_2D"],
        amount=cp.size(dwell_coords, 0),
    )
    # generate set of random angles
    mm_change_dwells_random_2D_angle = cp.random.uniform(
        0, 2 * cp.pi, cp.size(dwell_coords, 0)
    )
    # get set of X and Y changes using the angle and radius
    mm_change_dwells_random_2D_X = cp.multiply(
        mm_change_dwells_random_2D_radius,
        cp.cos(mm_change_dwells_random_2D_angle),
    )
    mm_change_dwells_random_2D_Y = cp.multiply(
        mm_change_dwells_random_2D_radius,
        cp.sin(mm_change_dwells_random_2D_angle),
    )
    # extend random changes form one per needle to one per dwell point
    mm_change_dwells_random_2D_X = cp.repeat(
        mm_change_dwells_random_2D_X,
        cp.size(dwell_coords, 1) * cp.size(dwell_coords, 2),
    ).reshape(
        cp.size(dwell_coords, 0),
        cp.size(dwell_coords, 1),
        cp.size(dwell_coords, 2),
    )
    mm_change_dwells_random_2D_Y = cp.repeat(
        mm_change_dwells_random_2D_Y,
        cp.size(dwell_coords, 1) * cp.size(dwell_coords, 2),
    ).reshape(
        cp.size(dwell_coords, 0),
        cp.size(dwell_coords, 1),
        cp.size(dwell_coords, 2),
    )
    # transfer changes into one array of the form (X, Y, Z=0)
    selector_vector_X = cp.tile(
        cp.array([1, 0, 0], dtype=cp.float32),
        cp.size(dwell_coords, 0) * cp.size(dwell_coords, 1),
    ).reshape(
        cp.size(dwell_coords, 0),
        cp.size(dwell_coords, 1),
        cp.size(dwell_coords, 2),
    )
    selector_vector_Y = cp.tile(
        cp.array([0, 1, 0], dtype=cp.float32),
        cp.size(dwell_coords, 0) * cp.size(dwell_coords, 1),
    ).reshape(
        cp.size(dwell_coords, 0),
        cp.size(dwell_coords, 1),
        cp.size(dwell_coords, 2),
    )
    mm_change_dwells_random_2D = (
        mm_change_dwells_random_2D_X * selector_vector_X
        + mm_change_dwells_random_2D_Y * selector_vector_Y
    )

    ########## Parameter Variable 5a ##########
    ## Description: Change dwell times by adding constant value, different per dwell time
    # different based on if there is a single set or 2D array of mulitple dwell times.
    if isinstance(arr_of_dwells_times, bool):
        # output --> array(num_needles, num_dwells_per_needle)
        dwell_time_increase_decrease = get_values_between_90CI(
            mu=0,
            sd=uncertainties_SD["dwell_time_increase_decrease"],
            amount=cp.size(plan_parameters["dwell_times_TPS"]),
        )
        dwell_time_increase_decrease = dwell_time_increase_decrease.reshape(
            cp.size(plan_parameters["dwell_times_TPS"], 0),
            cp.size(plan_parameters["dwell_times_TPS"], 1),
        ).astype(cp.float32)
        dwell_time_mask = (plan_parameters["dwell_times_TPS"] > 0).astype(cp.int32)
        dwell_time_increase_decrease = cp.multiply(
            dwell_time_increase_decrease, dwell_time_mask
        )
    else:
        # output --> array(num_needles, num_dwells_per_needle, number of sets of dwell times)
        dwell_time_increase_decrease = get_values_between_90CI(
            mu=0,
            sd=uncertainties_SD["dwell_time_increase_decrease"],
            amount=cp.size(arr_of_dwells_times),
        )
        dwell_time_increase_decrease = dwell_time_increase_decrease.reshape(
            cp.size(arr_of_dwells_times, 0),
            cp.size(arr_of_dwells_times, 1),
        ).astype(cp.float32)
        dwell_time_mask = (arr_of_dwells_times > 0).astype(cp.int32)
        dwell_time_increase_decrease = cp.multiply(
            dwell_time_increase_decrease, dwell_time_mask
        )

    ########## Parameter Variable 5b ##########
    ## Description: Change dwell times by %/100, all dwell times changed by one fixed percentage
    # output --> array(num_needles, num_dwells_per_needle)
    dwell_time_change_percentage = (
        get_values_between_90CI(
            mu=0, sd=uncertainties_SD["dwell_time_change_percentage"], amount=1
        )
        / 100
    )

    ########## Parameter Variable 6 (6a, 6b, 6c) ##########
    ## Description: ridged move of Prostate, urethra, and needles

    # Left-right (X) value change, positive moves right (axis left to right is -63 to 0 mm)
    # output --> array(scalar)
    mm_change_X_rigid = get_values_between_90CI(
        mu=0, sd=uncertainties_SD["mm_change_X_rigid"], amount=1
    ).item()

    # Anterior-Posterior (Y) value change, positive moves down (axis top to bottom is -75 to -29 mm)
    # output --> array(scalar)
    mm_change_Y_rigid = get_values_between_90CI(
        mu=0, sd=uncertainties_SD["mm_change_Y_rigid"], amount=1
    ).item()

    # Sup-Inf (Z) value change, positive moves superior (axis top to bottom is 10 to -45 mm)
    # output --> array(scalar)
    mm_change_Z_rigid = get_values_between_90CI(
        mu=0, sd=uncertainties_SD["mm_change_Z_rigid"], amount=1
    ).item()

    sim_magnitudes = {
        "dwells_shift_along_needle": dwells_shift_along_needle,
        "mm_change_shrink_enlarge_prostate": mm_change_shrink_enlarge_prostate,
        "mm_change_shrink_enlarge_urethra": mm_change_shrink_enlarge_urethra,
        "mm_change_shrink_enlarge_rectum": mm_change_shrink_enlarge_rectum,
        "dwells_move_out_from_prostate_centre_line": dwells_move_out_from_prostate_centre_line,
        "mm_change_dwells_random_2D": mm_change_dwells_random_2D,
        "dwell_time_increase_decrease": dwell_time_increase_decrease,
        "dwell_time_change_percentage": dwell_time_change_percentage,
        "mm_change_X_rigid": mm_change_X_rigid,
        "mm_change_Y_rigid": mm_change_Y_rigid,
        "mm_change_Z_rigid": mm_change_Z_rigid,
    }

    return sim_magnitudes


def get_values_between_90CI(mu, sd, amount):
    # generate random values
    set_of_values = cp.random.normal(mu, sd, amount).astype(cp.float32)

    # want the middle 90%CI of values so cycle through until only middle 90%CI values in array
    # so test how many are within 90% CI
    CI90Low = mu - sd * 1.644854
    CI90High = mu + sd * 1.644854
    num_between = cp.count_nonzero(
        (set_of_values < (CI90Low)) | (set_of_values > (CI90High))
    )

    # keep generating more until all are within 90%CI range
    while num_between > 0:
        # replace the ones that are not within the limit with new random values
        set_of_values[
            (set_of_values < (CI90Low)) | (set_of_values > (CI90High))
        ] = cp.random.normal(mu, sd, num_between.get()).astype(cp.float32)

        # test how many are now within the limits
        num_between = cp.count_nonzero(
            (set_of_values < (CI90Low)) | (set_of_values > (CI90High))
        )

    return set_of_values


__all__ = ["simulated_change_arrays", "probabilistic_robust_measure_gpu"]
