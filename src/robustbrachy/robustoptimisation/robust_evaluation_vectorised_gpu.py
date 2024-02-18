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

from robustbrachy.robustevaluation.utils_gpu import *
from robustbrachy.robustevaluation.robust_evaluation_gpu import *
from robustbrachy.robustevaluation.simulations_gpu import *
from robustbrachy.robustevaluation.fast_TG43_gpu import *
from robustbrachy.robustoptimisation.dose_per_dwell_gpu import *
from robustbrachy.robustoptimisation.evaluation_gpu import *


def robust_measure_array_gpu(
    optimiser_parametres,
    plan_parameters,
    dwell_times_pareto_front,
    all_nominal_dvhs,
    progress_bar=False,
):
    run_times_arr = []

    # creating dictionary of uncertainties. This can be changed here, but the optimiser objective functions are modelled on these uncertainties.
    uncertainties_SD = {
        "dwells_shift_along_needle": 1.5,
        "mm_change_shrink_enlarge_prostate": 1.0,
        "mm_change_shrink_enlarge_urethra": 0.5,
        "mm_change_shrink_enlarge_rectum": 0.25,
        "mm_change_dwells_random_2D": 1.5,
        "dwell_time_increase_decrease": 0.06,
        "dwell_time_change_percentage": 0.044,
        "mm_change_Y_rigid": 0.5,
        "mm_change_X_rigid": 0.1,
        "mm_change_Z_rigid": 0.0,
    }

    # deal with memory limits, the number of iterations being done in parallel is split
    split_amount = optimiser_parametres["no_of_runs_internal"]  # 100
    no_outer_itr = np.ceil(optimiser_parametres["no_of_runs"] / split_amount)
    num_of_itr_split = np.ones(int(no_outer_itr))
    num_of_itr_split[:-1] = split_amount
    if optimiser_parametres["no_of_runs"] % split_amount != 0:
        num_of_itr_split[-1] = optimiser_parametres["no_of_runs"] % split_amount
    else:
        num_of_itr_split[-1] = split_amount

    # arrays initiated
    all_D90_pop = cp.zeros(
        (optimiser_parametres["no_of_runs"], len(dwell_times_pareto_front)),
        dtype=cp.float32,
    )
    all_DmaxU_pop = cp.zeros(
        (optimiser_parametres["no_of_runs"], len(dwell_times_pareto_front)),
        dtype=cp.float32,
    )
    all_DmaxR_pop = cp.zeros(
        (optimiser_parametres["no_of_runs"], len(dwell_times_pareto_front)),
        dtype=cp.float32,
    )
    dose_per_vol = cp.empty(
        (len(dwell_times_pareto_front), 3, 110, optimiser_parametres["no_of_runs"]),
        dtype=cp.float32,
    )
    dose_per_vol[:] = np.nan
    min_vals = cp.ceil(cp.amin(all_nominal_dvhs[:, :, 1, :], axis=2))

    # dwell time array with only 1s in it, this is for making the dose per dwell point per dose point array
    dwell_times_unity = cp.empty(
        shape=(
            plan_parameters["dwell_coords_optimisation"].shape[0],
            plan_parameters["dwell_coords_optimisation"].shape[1],
        ),
        dtype=cp.float32,
    )
    dwell_times_unity.fill(1.0)
    dwell_times_unity[
        plan_parameters["dwell_coords_optimisation"][:, :, 0] == -100
    ] = -100

    total_sim_num = 0
    last_itr = 0
    for outer_itr in range(0, int(no_outer_itr)):
        no_of_inner_itr = int(num_of_itr_split[outer_itr])
        all_DVHs = cp.zeros(
            (no_of_inner_itr, len(dwell_times_pareto_front), 3, 2, 1000),
            dtype=cp.float32,
        )
        all_volumes = cp.zeros(
            (no_of_inner_itr, len(dwell_times_pareto_front), 2), dtype=cp.float32
        )

        for inner_itr in range(0, no_of_inner_itr):
            start = time.time()

            # sets up the random change arrays that are used in the simulation function to apply the changes to structures, dwell times and dwell points
            sim_magnitudes = simulated_change_arrays(
                uncertainties_SD, plan_parameters, dwell_times_pareto_front
            )

            # make a copy of each array so that the structure gets reset each run.
            changed_structures = copy_structures_gpu(
                plan_parameters, dwell_times_pareto_front
            )

            # carries out scenario movements and then calculates dose values, dose metrics, and dvh arrays
            (changed_structures) = simulations_gpu(
                sim_magnitudes, changed_structures, plan_parameters, optimisation=True
            )

            ########################################
            ### Construct dose rate arrays ###
            ########################################

            dose_calc_pts_prostate = get_dose_volume_pts_gpu(
                changed_structures["changed_prostate_contour_pts"],
                changed_structures["changed_urethra_contour_pts"],
                voxel_size=1,
            )
            dose_calc_pts_urethra = get_dose_volume_pts_gpu(
                changed_structures["changed_urethra_contour_pts"], voxel_size=1
            )
            dose_calc_pts_rectum = get_dose_volume_pts_gpu(
                changed_structures["changed_rectum_contour_pts"], voxel_size=1
            )

            dose_per_dwell_per_vol_pt_prostate_robust_eval = (
                TG43calc_dose_per_dwell_per_vol_pt_gpu(
                    dose_calc_pts_prostate,
                    dwell_times_unity,
                    plan_parameters,
                    changed_structures["changed_dwell_coords"],
                    changed_structures["changed_dwell_pts_source_end_sup"],
                    changed_structures["changed_dwell_pts_source_end_inf"],
                )
            )

            dose_per_dwell_per_vol_pt_urethra_robust_eval = (
                TG43calc_dose_per_dwell_per_vol_pt_gpu(
                    dose_calc_pts_urethra,
                    dwell_times_unity,
                    plan_parameters,
                    changed_structures["changed_dwell_coords"],
                    changed_structures["changed_dwell_pts_source_end_sup"],
                    changed_structures["changed_dwell_pts_source_end_inf"],
                )
            )

            dose_per_dwell_per_vol_pt_rectum_robust_eval = (
                TG43calc_dose_per_dwell_per_vol_pt_gpu(
                    dose_calc_pts_rectum,
                    dwell_times_unity,
                    plan_parameters,
                    changed_structures["changed_dwell_coords"],
                    changed_structures["changed_dwell_pts_source_end_sup"],
                    changed_structures["changed_dwell_pts_source_end_inf"],
                )
            )

            del changed_structures["changed_dwell_coords"]
            del changed_structures["changed_dwell_pts_source_end_sup"]
            del changed_structures["changed_dwell_pts_source_end_inf"]

            ########################################
            ### Calculate Dose metrics of interest ###
            ########################################

            # get all DVHs
            all_DVHs_pop = get_dvh_for_population_optimisation_gpu(
                changed_structures["changed_arr_of_dwells_times"],
                [
                    dose_per_dwell_per_vol_pt_prostate_robust_eval,
                    dose_per_dwell_per_vol_pt_urethra_robust_eval,
                    dose_per_dwell_per_vol_pt_rectum_robust_eval,
                ],
                plan_parameters["prescribed_dose"],
            )
            all_DVHs_pop = cp.array(all_DVHs_pop, dtype=cp.float32)
            # get volumes of OAR so we can get max doses and repeat into an array
            urethra_volume = volume_from_slices_vectorised_gpu(
                changed_structures["changed_urethra_contour_pts"]
            )
            rectum_volume = volume_from_slices_vectorised_gpu(
                changed_structures["changed_rectum_contour_pts"]
            )
            volumes = cp.array([urethra_volume, rectum_volume])
            volumes = volumes * cp.ones(
                (len(changed_structures["changed_arr_of_dwells_times"]), 2)
            )
            volumes = cp.array(volumes, dtype=cp.float32)

            all_DVHs[inner_itr, :, :, :, :] = all_DVHs_pop
            dose = all_DVHs_pop[:, :, 0]
            vol = all_DVHs_pop[:, :, 1]
            dose = cp.array(dose, dtype=cp.float32)
            vol = cp.array(vol, dtype=cp.float32)

            vol_array = cp.tile(
                cp.flip(
                    cp.concatenate(
                        (
                            cp.arange(100),
                            cp.array(
                                [
                                    99.1,
                                    99.2,
                                    99.3,
                                    99.4,
                                    99.5,
                                    99.6,
                                    99.7,
                                    99.8,
                                    99.9,
                                    100,
                                ]
                            ),
                        )
                    )
                ),
                len(dose) * len(dose[0]),
            ).reshape(len(dose), len(dose[0]), -1)
            vol_array = cp.array(vol_array, dtype=cp.float32)
            mask = (vol_array >= min_vals[:, :, cp.newaxis]).astype(cp.bool_).flatten()

            idx_to_nearest_vol = cp.abs(
                vol[:, :, cp.newaxis, :] - vol_array[:, :, :, cp.newaxis]
            ).argmin(axis=3)
            vol_array_args = cp.argwhere(idx_to_nearest_vol > -100)

            dose_per_vol[
                vol_array_args[:, 0][mask],
                vol_array_args[:, 1][mask],
                vol_array_args[:, 2][mask],
                total_sim_num,
            ] = dose[
                vol_array_args[:, 0][mask],
                vol_array_args[:, 1][mask],
                idx_to_nearest_vol.flatten()[mask],
            ]

            all_volumes[inner_itr, :, :] = volumes

            end = time.time()
            run_time = end - start

            sim_num = int(total_sim_num)
            run_times_arr.append(run_time)

            if sim_num % 10 == 0:
                run_times_avg = np.mean(np.array(run_times_arr))

                if progress_bar != False:
                    progress_bar.progress(
                        sim_num / optimiser_parametres["no_of_runs"],
                        text="Up to probabilistic senario "
                        + str(int(sim_num) + 1)
                        + " out of "
                        + str(optimiser_parametres["no_of_runs"])
                        + ", average runtime of "
                        + str(round(run_times_avg, 4))
                        + " seconds.",
                    )

                else:
                    print(
                        "Probabilistic scenario "
                        + str(sim_num)
                        + " for patient completed last 10 in an mean time of "
                        + str(run_times_avg)
                        + "."
                    )
                # run_times_arr = []

            total_sim_num += 1

        # get the metric values in every outer iteration
        all_DVHs = all_DVHs.reshape(
            no_of_inner_itr * len(dwell_times_pareto_front), 3, 2, 1000
        ).astype(cp.float32)
        all_volumes = all_volumes.reshape(
            no_of_inner_itr * len(dwell_times_pareto_front), 2
        ).astype(cp.float32)

        D90_pop, DmaxU_pop, DmaxR_pop = get_metrics_for_population_RE_array_gpu(
            all_DVHs, all_volumes, plan_parameters["prescribed_dose"]
        )

        D90_pop = D90_pop.reshape(
            no_of_inner_itr, len(dwell_times_pareto_front)
        ).astype(cp.float32)
        DmaxU_pop = DmaxU_pop.reshape(
            no_of_inner_itr, len(dwell_times_pareto_front)
        ).astype(cp.float32)
        DmaxR_pop = DmaxR_pop.reshape(
            no_of_inner_itr, len(dwell_times_pareto_front)
        ).astype(cp.float32)

        last_itr = total_sim_num

    # calculate passrates
    mask_passed_D90 = ((plan_parameters["prescribed_dose"] - D90_pop) <= 0).astype(
        cp.bool_
    )
    mask_passed_DmaxR = ((DmaxR_pop - 13) <= 0).astype(cp.bool_)
    mask_passed_DmaxU = (
        (DmaxU_pop - plan_parameters["prescribed_dose"] * 1.10) <= 0
    ).astype(cp.bool_)

    mask_passed_all = mask_passed_D90 * mask_passed_DmaxU * mask_passed_DmaxR

    D90_passrate = (
        cp.sum(mask_passed_D90, axis=0).get().astype(np.float32) / len(D90_pop) * 100
    )
    DmaxU_passrate = (
        cp.sum(mask_passed_DmaxU, axis=0).get().astype(np.float32)
        / len(DmaxU_pop)
        * 100
    )
    DmaxR_passrate = (
        cp.sum(mask_passed_DmaxR, axis=0).get().astype(np.float32)
        / len(DmaxR_pop)
        * 100
    )

    all_passrate = (
        cp.sum(mask_passed_all, axis=0).get().astype(np.float32) / len(DmaxR_pop) * 100
    )

    # obtaining the confidence intervals for robustness along the dose axis
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

    if progress_bar != False:
        run_times_avg = np.mean(np.array(run_times_arr))
        progress_bar.progress(
            1.0,
            text="Finished with an average runtime of "
            + str(round(run_times_avg, 4))
            + " seconds per scenario.",
        )

    return (
        all_passrate,
        D90_passrate,
        DmaxU_passrate,
        DmaxR_passrate,
        all_robust_dvh_summary,
    )
