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


def TG43calc_dose_per_dwell_per_vol_pt_gpu(
    dose_calc_pts,
    dwell_times,
    plan_parameters,
    dwell_coords,
    dwell_pts_source_end_sup,
    dwell_pts_source_end_inf,
):
    # from the TPS there was negative times and times of 0, remove these from the array.
    # also flatten the array into an array of just 3D points, they were in needle sub arrays

    # make sure we have just positive dwell times (should be an array of just 1.0 seconds) and corresponding dwell pts, by making two masks
    active_dwell_times_mask = (dwell_times > 0).astype(cp.bool_)
    active_dwell_pts_mask = (cp.repeat(active_dwell_times_mask, 3, axis=1)).reshape(
        len(active_dwell_times_mask), -1, 3
    )

    # so get the active dwell times, points, and both source ends only
    active_dwell_times = dwell_times[active_dwell_times_mask]

    active_dwell_points_source_middle = (
        dwell_coords[active_dwell_pts_mask].reshape(-1, 3) / 10
    )
    active_dwell_points_source_sup_end = (
        dwell_pts_source_end_sup[active_dwell_pts_mask].reshape(-1, 3) / 10
    )
    active_dwell_points_source_inf_end = (
        dwell_pts_source_end_inf[active_dwell_pts_mask].reshape(-1, 3) / 10
    )

    # mm to cm
    dose_calc_pts = dose_calc_pts / 10

    # if there is no dose points, spit out an error
    if not dose_calc_pts.size:
        return "Error"

    dose_calc_pts_tiled = cp.tile(
        dose_calc_pts[None, :, :], (active_dwell_points_source_middle.shape[0], 1, 1)
    )

    # Make an array of copies of the dwell times, mid, sup end, inf end. Many copies of the same list of points or times.
    active_dwell_points_source_middle_tiled = cp.tile(
        active_dwell_points_source_middle[:, None, :], (1, dose_calc_pts.shape[0], 1)
    )
    active_dwell_points_source_inf_end_tiled = cp.tile(
        active_dwell_points_source_inf_end[:, None, :], (1, dose_calc_pts.shape[0], 1)
    )
    active_dwell_points_source_sup_end_tiled = cp.tile(
        active_dwell_points_source_sup_end[:, None, :], (1, dose_calc_pts.shape[0], 1)
    )

    # find the distances between the arrays. This is the r values in TG43, DoseCalnpts are the P in TG43
    # and SourceMidddle is the centre of the source points, the dwell points
    r = cp.sqrt(
        cp.sum(
            cp.subtract(dose_calc_pts_tiled, active_dwell_points_source_middle_tiled)
            ** 2,
            axis=2,
        )
    )

    # obtains the angle between all Pts (dose calc points) and the midddle of the source
    # for all dwell points in one array. The dot product between line source and vector
    # between P and source middle pt is calculated than inverse cos to calculate the angle
    # theta in TG43. theta = cos^(-1) [vec1 (dot) vec2 / (length(vec1) x length(vec2))]
    # here we find the dot product by multiplying x,y,z components and summing

    theta_dot_product = cp.divide(
        cp.sum(
            cp.multiply(
                (
                    active_dwell_points_source_sup_end_tiled
                    - active_dwell_points_source_inf_end_tiled
                ),
                (dose_calc_pts - active_dwell_points_source_middle_tiled),
            ),
            axis=2,
        ),
        r * plan_parameters["L"],
    )

    # some values are above and some below the limits of -1 to 1. It is thought due to rounding
    # a large number of values and summing these up. +/-3% above/below +/-1 and about 1% of values.

    positive_angle_mask = (theta_dot_product > 1).astype(cp.int32)
    negative_angle_mask = (theta_dot_product < -1).astype(cp.int32)
    other_angle_mask = (
        ~cp.array(positive_angle_mask + negative_angle_mask, dtype=cp.float32).astype(
            cp.bool_
        )
    ).astype(cp.int32)

    theta_dot_product = (
        cp.multiply(0.9999999999, positive_angle_mask)
        + cp.multiply(-0.9999999999, negative_angle_mask)
        + cp.multiply(theta_dot_product, other_angle_mask)
    )

    theta = cp.arccos(theta_dot_product)

    # # To find the angle beta in TG43,  minus the angles between each source end
    # # and the Point P. Some working out but in terms of L =  source length, theta found above and r above.

    beta = cp.arctan2(
        r * cp.cos(theta) - plan_parameters["L"] / 2.0, r * cp.sin(theta)
    ) - cp.arctan2(r * cp.cos(theta) + plan_parameters["L"] / 2.0, r * cp.sin(theta))

    # G = the geometric function in TG43. For small angles close to 0 degrees or 180 degrees beta and theta
    # get close to zero, so  division of zero. At these angles, it is similar to a point source (1/r^2).
    # G for small and large angles are found in two martix operations. Than two masks are used to select the
    # angles which are actually small and those that are not extreme (otheranglesmask for Large angles).

    G = cp.multiply(
        1.0 / (r * r - (plan_parameters["L"] * plan_parameters["L"]) / 4.0),
        ((cp.pi - theta) < 0.003).astype(cp.int32) + ((theta) < 0.003).astype(cp.int32),
    ) + cp.multiply(
        beta / (plan_parameters["L"] * r * cp.sin(theta)),
        (
            ~cp.array(
                (
                    ((cp.pi - theta) < 0.003).astype(cp.int32)
                    + ((theta) < 0.003).astype(cp.int32)
                ).astype(cp.bool_)
            )
        ).astype(cp.int32),
    )

    G_mask = (G > 0).astype(cp.bool_)
    G[G_mask] = -G[G_mask]

    # this is the TG43 equation F_interp and g_interp are extracted from the source excel data sheet and
    # interpolated as explained else where. Sk is the air kerma, DoseRateConstant, G0 are all extracted from
    # the treatment plan files. g and F point to two functions which find the values in the interpolated
    # array for all r or r & theta. It does this for all values at once. Summing accross axis zero adds all dose
    # contributions from source at one point per sum, resulting in the dose calculated at all dose points.

    r_high_value = plan_parameters["F_interp"][
        (cp.rint(cp.degrees(theta) * 10) - 1).astype(cp.int32),
        (cp.rint(100 * 10) - 1).astype(cp.int32),
    ]

    second_indx = (cp.rint(r * 100) - 1).astype(cp.int32)
    second_indx[second_indx > 999] = 999

    r_low_value = plan_parameters["F_interp"][
        (cp.rint(cp.degrees(theta) * 10) - 1).astype(cp.int32),
        second_indx.astype(cp.int32),
    ]

    f_dose = cp.select([r > 10, r <= 10], [r_high_value, r_low_value])

    r_high_value = plan_parameters["g_interp"][
        (cp.rint(8 * 100) - 1).astype(cp.int32)
    ] * cp.exp(
        (r - 8)
        / (10 - 8)
        * (
            cp.log(
                plan_parameters["g_interp"][(cp.rint(10 * 100) - 1).astype(cp.int32)]
            )
            - cp.log(
                plan_parameters["g_interp"][(cp.rint(8 * 100) - 1).astype(cp.int32)]
            )
        )
    )

    r_low_value = plan_parameters["g_interp"][
        (cp.rint(0.15 * 100) - 1).astype(cp.int32)
    ]

    r_middle_source = plan_parameters["g_interp"][second_indx.astype(cp.int32)]

    g_dose = cp.select(
        [r > 10, r < 0.15, (r > 0.15) ^ (r > 10)],
        [r_high_value, r_low_value, r_middle_source],
    )

    active_dwell_times = cp.tile(
        active_dwell_times[:, None], (1, dose_calc_pts.shape[0])
    )

    # TG43 dose calculation at each dose calculation point
    dose_per_dwell_per_vol_pt = cp.einsum(
        " ij, ij, ij, ij -> ij", G, g_dose, f_dose, active_dwell_times
    ) * (
        (plan_parameters["air_kerma_rate"] * plan_parameters["dose_rate_constant"])
        / (plan_parameters["G0"] * 360000)
    )

    return dose_per_dwell_per_vol_pt


__all__ = ["TG43calc_dose_per_dwell_per_vol_pt_gpu"]
