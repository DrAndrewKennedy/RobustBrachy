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

from robustbrachy.robustevaluation.utils_gpu import *
from robustbrachy.robustoptimisation.dose_per_dwell_gpu import *


def get_dvh_for_population_optimisation_gpu(
    pop_dwell_times, all_dose_per_dwell_per_vol_pt, prescribed_dose
):
    # cycle through the three structures
    for structure, structure_dose_per_pt_arr in enumerate(
        all_dose_per_dwell_per_vol_pt
    ):
        pop_dwell_times = pop_dwell_times.astype(cp.float32)

        # get structures dose per dwell pt per dose calc point array
        dose_per_dwell_per_vol_pt = structure_dose_per_pt_arr.astype(cp.float32)

        # get dose per volume point due to all dwell points, for all sets of dwell point (for all treatment plans)
        pop_dose = cp.einsum("ij,jk-> ik", pop_dwell_times, dose_per_dwell_per_vol_pt)

        # calculate all DVHs for the population
        histogram_data = cp.apply_along_axis(
            histogram_function_gpu, 1, pop_dose, [prescribed_dose]
        )

        # split data into bin edge values and counts in each dose bin
        bin_edges = cp.vstack(histogram_data[:, 1])
        counts_per_dose_bin_histogram = cp.vstack(histogram_data[:, 0])

        bin_width = bin_edges[0][1] - bin_edges[0][0]

        # DVHs are found for the range of 0 Gy to 3 x prescribed dose.
        # Particularly for the prostate, a proportion of the dose in the prostate will be higher than 3 x prescribed dose.
        # this calculates that fraction/proportion so it can be included into the volume percentage of the DVH
        fraction_not_included_in_dvh = cp.apply_along_axis(
            fraction_not_in_dvh_gpu, 1, pop_dose, [prescribed_dose]
        )

        # calculates the cumulative histogram for each bin
        volume_percentages = cp.apply_along_axis(
            volumes_from_dvh_gpu, 1, counts_per_dose_bin_histogram, [bin_width]
        )

        # adds the fraction that is not included in the DVHs
        volume_percentages = (
            fraction_not_included_in_dvh[:, cp.newaxis] + volume_percentages
        )

        # calculates the percentage volume recieving each amount of dose
        volume_percentages = (
            volume_percentages
            / cp.amax(volume_percentages, axis=1)[:, cp.newaxis]
            * 100
        )

        # packages the DVHs together into a consistent manner
        pop_dose = cp.apply_along_axis(
            put_vols_and_bins_together_gpu, 1, volume_percentages, [bin_edges[0]]
        ).reshape(len(pop_dwell_times), 1, 2, 1000)

        # incase this is only for one structure
        if structure == 0:
            all_DVHs = pop_dose
        else:
            all_DVHs = cp.concatenate((all_DVHs, pop_dose), axis=1)

    all_DVHs = cp.array(all_DVHs, dtype=cp.float32)

    return all_DVHs


def histogram_function_gpu(a, *prescribed_dose):
    # calculating the DVH data. The function gets called for each ordered set of dwell times in a somewhat vectorised manner.
    hist, bin_edges = cp.histogram(
        a, bins=1000, range=(0, prescribed_dose[0][0] * 3), density=True
    )

    return cp.stack([hist, bin_edges[:-1]])


def fraction_not_in_dvh_gpu(a, *prescribed_dose):
    # DVHs are found for the range of 0 Gy to 3 x prescribed dose.
    # Particularly for the prostate, a proportion of the dose in the prostate will be higher than 3 x prescribed dose.
    # this calculates that fraction/proportion so it can be included into the volume percentage of the DVH
    try:
        frac = 1 - len(a[a < prescribed_dose[0][0] * 3]) / len(a)
    except:
        frac = 1

    return frac


def put_vols_and_bins_together_gpu(a, *bin_edges):
    return cp.array([bin_edges[0][0], cp.around(a, 2)], dtype=cp.float32)


def volumes_from_dvh_gpu(a, *bin_width):
    # calculates the cumulative histogram for each bin
    return cp.flip(cp.cumsum(cp.flip(a))) * (bin_width[0][0])


def get_metrics_for_population_RE_array_gpu(all_DVHs_pop, volumes, prescribed_dose):
    D90_pop = get_population_dose_metrics_gpu(cp.array([90.0]), all_DVHs_pop[:, 0])

    DmaxU_pop = get_population_dose_metrics_gpu(
        0.01 / volumes[:, 0] * 100, all_DVHs_pop[:, 1]
    )

    DmaxR_pop = get_population_dose_metrics_gpu(
        0.1 / volumes[:, 1] * 100, all_DVHs_pop[:, 2]
    )

    return D90_pop, DmaxU_pop, DmaxR_pop


def get_metrics_for_population_optimisation_gpu(
    all_DVHs_pop, urethra_volume, rectum_volume
):
    # using the functions from Robust_Evaluation.py file
    D90_pop = get_population_dose_metrics_gpu(cp.array([90.0]), all_DVHs_pop[:, 0])
    DmaxU_pop = get_population_dose_metrics_gpu(
        cp.array([0.01 / urethra_volume * 100]), all_DVHs_pop[:, 1]
    )
    DmaxR_pop = get_population_dose_metrics_gpu(
        cp.array([0.1 / rectum_volume * 100]), all_DVHs_pop[:, 2]
    )

    return D90_pop, DmaxU_pop, DmaxR_pop


def get_nominal_metrics_array_gpu(
    dwell_times_pareto_front,
    dose_per_dwell_per_vol_pt_prostate,
    dose_per_dwell_per_vol_pt_urethra,
    dose_per_dwell_per_vol_pt_rectum,
    prescribed_dose,
    nominal_urethra_volume,
    nominal_rectum_volume,
):
    # get nominal metric values
    all_nominal_dvhs = get_dvh_for_population_optimisation_gpu(
        dwell_times_pareto_front,
        [
            dose_per_dwell_per_vol_pt_prostate,
            dose_per_dwell_per_vol_pt_urethra,
            dose_per_dwell_per_vol_pt_rectum,
        ],
        prescribed_dose,
    )

    volume_percentage = cp.array([90], dtype=cp.float32)
    nominal_D90_pop = get_population_dose_metrics_gpu(
        volume_percentage, all_nominal_dvhs[:, 0]
    )

    volume_percentage = cp.array(
        [0.01 / nominal_urethra_volume * 100], dtype=cp.float32
    )
    nominal_DmaxU_pop = get_population_dose_metrics_gpu(
        volume_percentage, all_nominal_dvhs[:, 1]
    )

    volume_percentage = cp.array([0.1 / nominal_rectum_volume * 100], dtype=cp.float32)
    nominal_DmaxR_pop = get_population_dose_metrics_gpu(
        volume_percentage, all_nominal_dvhs[:, 2]
    )

    # V100
    dose_of_interest = cp.array([prescribed_dose], dtype=cp.float32)
    nominal_V100_pop = get_population_volume_metrics_gpu(
        dose_of_interest, all_nominal_dvhs[:, 0]
    )

    # V150
    dose_of_interest = cp.array([prescribed_dose * 1.5], dtype=cp.float32)
    nominal_V150_pop = get_population_volume_metrics_gpu(
        dose_of_interest, all_nominal_dvhs[:, 0]
    )

    # V200
    dose_of_interest = cp.array([prescribed_dose * 2.0], dtype=cp.float32)
    nominal_V200_pop = get_population_volume_metrics_gpu(
        dose_of_interest, all_nominal_dvhs[:, 0]
    )

    # D10
    volume_percentage = cp.array([10], dtype=cp.float32)
    nominal_D10_pop = get_population_dose_metrics_gpu(
        volume_percentage, all_nominal_dvhs[:, 1]
    )

    # V75
    dose_of_interest = cp.array([prescribed_dose * 0.75], dtype=cp.float32)
    nominal_V75_pop = get_population_volume_metrics_gpu(
        dose_of_interest, all_nominal_dvhs[:, 2]
    )
    nominal_V75_pop = nominal_V75_pop / 100 * float(nominal_rectum_volume)

    nominal_metrics = np.array(
        [
            nominal_D90_pop.get(),
            nominal_V100_pop.get(),
            nominal_V150_pop.get(),
            nominal_V200_pop.get(),
            nominal_D10_pop.get(),
            nominal_DmaxU_pop.get(),
            nominal_V75_pop.get(),
            nominal_DmaxR_pop.get(),
        ]
    )

    return nominal_metrics, all_nominal_dvhs.get()


def get_dose_grid_gpu(
    plan_parameters,
    dwell_coords,
    dwell_pts_source_end_sup,
    dwell_pts_source_end_inf,
    voxel_size=1,
):
    # get full dose grid
    z_vector = cp.flip(
        cp.arange(
            (
                cp.amin(
                    cp.array(
                        [
                            cp.amin(plan_parameters["prostate_contour_pts"][:, 2, 0]),
                            cp.amin(plan_parameters["urethra_contour_pts"][:, 2, 0]),
                            cp.amin(plan_parameters["rectum_contour_pts"][:, 2, 0]),
                        ]
                    )
                )
            ).get(),
            (
                cp.amax(
                    cp.array(
                        [
                            cp.amin(plan_parameters["prostate_contour_pts"][:, 2, 0]),
                            cp.amax(plan_parameters["urethra_contour_pts"][:, 2, 0]),
                            cp.amax(plan_parameters["rectum_contour_pts"][:, 2, 0]),
                        ]
                    )
                )
            ).get()
            + voxel_size,
            voxel_size,
            dtype=cp.float32,
        )
    )

    x_vector = cp.arange(
        (
            cp.amin(
                cp.array(
                    [
                        cp.amin(plan_parameters["prostate_contour_pts"][:, 0, :]),
                        cp.amin(plan_parameters["urethra_contour_pts"][:, 0, :]),
                        cp.amin(plan_parameters["rectum_contour_pts"][:, 0, :]),
                    ]
                )
            )
            - 1
        ).get(),
        (
            cp.amax(
                cp.array(
                    [
                        cp.amax(plan_parameters["prostate_contour_pts"][:, 0, :]),
                        cp.amax(plan_parameters["urethra_contour_pts"][:, 0, :]),
                        cp.amax(plan_parameters["rectum_contour_pts"][:, 0, :]),
                    ]
                )
            )
            + 2
        ).get(),
        voxel_size,
        dtype=cp.float32,
    )

    y_vector = cp.arange(
        (
            cp.amin(
                cp.asarray(
                    [
                        cp.amin(plan_parameters["prostate_contour_pts"][:, 1, :]),
                        cp.amin(plan_parameters["urethra_contour_pts"][:, 1, :]),
                        cp.amin(plan_parameters["rectum_contour_pts"][:, 1, :]),
                    ]
                )
            )
            - 1
        ).get(),
        (
            cp.amax(
                cp.asarray(
                    [
                        cp.amax(plan_parameters["prostate_contour_pts"][:, 1, :]),
                        cp.amax(plan_parameters["urethra_contour_pts"][:, 1, :]),
                        cp.amax(plan_parameters["rectum_contour_pts"][:, 1, :]),
                    ]
                )
            )
            + 1
        ).get(),
        voxel_size,
        dtype=cp.float32,
    )

    xyz_meshgrid_gpu = cp.array(
        cp.meshgrid(x_vector, y_vector, z_vector), dtype=cp.float32
    )

    # all dose calculation ponints in a 3D grid
    dose_calc_pts = cp.vstack(xyz_meshgrid_gpu).reshape(3, -1).T

    # form all dwell times = 1 second in ordr to get dose rate grid
    dwell_times = cp.empty(
        shape=(dwell_coords.shape[0], dwell_coords.shape[1]), dtype=cp.float32
    )
    dwell_times.fill(1.0)
    dwell_times[dwell_coords[:, :, 0] == -100] = -100

    # get the dose rate per volume dose calculation point

    dose_per_dwell_per_vol_pt = TG43calc_dose_per_dwell_per_vol_pt_gpu(
        dose_calc_pts,
        cp.array(dwell_times),
        plan_parameters,
        cp.array(dwell_coords),
        cp.array(dwell_pts_source_end_sup),
        cp.array(dwell_pts_source_end_inf),
    )

    return dose_per_dwell_per_vol_pt.get(), dose_calc_pts.get()


__all__ = [
    "get_dose_grid_gpu",
    "get_nominal_metrics_array_gpu",
    "get_dvh_for_population_optimisation_gpu",
    "get_metrics_for_population_RE_array_gpu",
    "get_metrics_for_population_optimisation_gpu",
]
