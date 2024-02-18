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

#####################################
###       Dose Calculation        ###
#####################################


# first, CPU dose calc functions
def fast_TG43_cpu(
    changed_structures,
    plan_parameters,
    voxel_size=1,
):
    dose_calc_pts_prostate = get_dose_volume_pts_cpu(
        changed_structures["changed_prostate_contour_pts"],
        structure_excluded=changed_structures["changed_urethra_contour_pts"],
        voxel_size=voxel_size,
    )

    dose_calc_pts_urethra = get_dose_volume_pts_cpu(
        changed_structures["changed_urethra_contour_pts"], voxel_size=voxel_size
    )

    dose_calc_pts_rectum = get_dose_volume_pts_cpu(
        changed_structures["changed_rectum_contour_pts"], voxel_size=voxel_size
    )

    all_dose_pts = np.zeros(
        (
            len(dose_calc_pts_prostate)
            + len(dose_calc_pts_urethra)
            + len(dose_calc_pts_rectum),
            3,
        ),
        dtype=np.float32,
    )

    all_dose_pts[: len(dose_calc_pts_prostate), :] = dose_calc_pts_prostate
    all_dose_pts[
        len(dose_calc_pts_prostate) : (
            len(dose_calc_pts_prostate) + len(dose_calc_pts_urethra)
        ),
        :,
    ] = dose_calc_pts_urethra
    all_dose_pts[
        (len(dose_calc_pts_prostate) + len(dose_calc_pts_urethra)) : (
            len(dose_calc_pts_prostate)
            + len(dose_calc_pts_urethra)
            + len(dose_calc_pts_rectum)
        ),
        :,
    ] = dose_calc_pts_rectum

    mask_dose_pts_prostate = np.zeros(
        (
            len(dose_calc_pts_prostate)
            + len(dose_calc_pts_urethra)
            + len(dose_calc_pts_rectum)
        ),
        dtype=np.bool_,
    )
    mask_dose_pts_urethra = np.zeros(
        (
            len(dose_calc_pts_prostate)
            + len(dose_calc_pts_urethra)
            + len(dose_calc_pts_rectum)
        ),
        dtype=np.bool_,
    )
    mask_dose_pts_rectum = np.zeros(
        (
            len(dose_calc_pts_prostate)
            + len(dose_calc_pts_urethra)
            + len(dose_calc_pts_rectum)
        ),
        dtype=np.bool_,
    )

    mask_dose_pts_prostate[: len(dose_calc_pts_prostate)] = True
    mask_dose_pts_urethra[
        len(dose_calc_pts_prostate) : (
            len(dose_calc_pts_prostate) + len(dose_calc_pts_urethra)
        )
    ] = True
    mask_dose_pts_rectum[
        (len(dose_calc_pts_prostate) + len(dose_calc_pts_urethra)) : (
            len(dose_calc_pts_prostate)
            + len(dose_calc_pts_urethra)
            + len(dose_calc_pts_rectum)
        )
    ] = True

    dose = TG43_dose_calc_cpu(all_dose_pts, changed_structures, plan_parameters)

    # prostate DVH
    dose_prostate = dose[mask_dose_pts_prostate]
    hist_values, bin_edges = np.histogram(
        dose_prostate,
        bins=1000,
        range=(0, plan_parameters["prescribed_dose"] * 3),
        density=True,
    )

    hist_values = hist_values.astype(np.float32)
    bin_edges = bin_edges.astype(np.float32)

    bin_width = bin_edges[1] - bin_edges[0]
    try:

        fraction_not_included_in_dvh = 1 - len(
            dose_prostate[dose_prostate < plan_parameters["prescribed_dose"] * 3]
        ) / len(dose_prostate)

    except:
        fraction_not_included_in_dvh = 1
        
    volumes = (
        fraction_not_included_in_dvh
        + np.flip(np.cumsum(np.flip(hist_values))) * bin_width
    )

    volumes_prostate = volumes / np.amax(volumes) * 100

    # urethra DVH
    dose_urethra = dose[mask_dose_pts_urethra]
    hist_values, _ = np.histogram(
        dose_urethra,
        bins=1000,
        range=(0, plan_parameters["prescribed_dose"] * 3),
        density=True,
    )
    hist_values = hist_values.astype(np.float32)

    try:
        fraction_not_included_in_dvh = 1 - len(
            dose_urethra[dose_urethra < plan_parameters["prescribed_dose"] * 3]
        ) / len(dose_urethra)

    except:
        fraction_not_included_in_dvh = 1
    
    volumes = (
        fraction_not_included_in_dvh
        + np.flip(np.cumsum(np.flip(hist_values))) * bin_width
    )

    volumes_urethra = volumes / np.amax(volumes) * 100

    # rectum DVH
    dose_rectum = dose[mask_dose_pts_rectum]
    hist_values, _ = np.histogram(
        dose_rectum,
        bins=1000,
        range=(0, plan_parameters["prescribed_dose"] * 3),
        density=True,
    )
    hist_values = hist_values.astype(np.float32)

    try:
        fraction_not_included_in_dvh = 1 - len(
            dose_rectum[dose_rectum < plan_parameters["prescribed_dose"] * 3]
        ) / len(dose_rectum)
    except:
        fraction_not_included_in_dvh = 1

    volumes = (
        fraction_not_included_in_dvh
        + np.flip(np.cumsum(np.flip(hist_values))) * bin_width
    )

    volumes_rectum = volumes / np.amax(volumes) * 100

    DVH = np.zeros((3, 2, 1000))
    DVH[0, :, :] = np.array(
        [bin_edges[:-1], np.around(volumes_prostate, 2)], dtype=np.float32
    )
    DVH[1, :, :] = np.array(
        [bin_edges[:-1], np.around(volumes_urethra, 2)], dtype=np.float32
    )
    DVH[2, :, :] = np.array(
        [bin_edges[:-1], np.around(volumes_rectum, 2)], dtype=np.float32
    )

    return DVH


def get_dose_volume_pts_cpu(structure, structure_excluded=[False], voxel_size=1):
    # voxelisation of the structure is the first step. Getting the set of
    # all points which will have the dose calculated at. (separation of size voxel_size apart)
    # method for deciding if pts are in or out of structure is the authors own method following the maths from:
    # https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect/565282#565282
    # note that I've ignored the parallel cases as for using floats in my context would be VERY rare if they were parallel
    # and check would add computational inefficencies.

    slice_width = np.float32(
        np.absolute(structure[0, 2, 0] - structure[1, 2, 0])
    ).round(3)

    # this step reduces the number of slices to the desired voxel size, strutures are usually in 0.5 mm slices, this changes it to say 1.0 mm by selecting every second slice
    # if the desired voxel size is not divisable by the number of slices currently in the studture, this method isn't going to work so the strucuture needs to be resampled.
    if voxel_size % slice_width == 0:
        select_stepping = np.int32(voxel_size / slice_width)
        structure = structure[::select_stepping, :, :]

        if (
            len(structure_excluded) != 1
        ):  # assuming more than one slice in exclusion structure
            ind_start_structure = np.where(
                structure[0, 2, 0] == structure_excluded[:, 2, 0]
            )[0]

            if len(ind_start_structure) == 0:
                structure_excluded = extrapolate_structure_nearest_nbr_cpu(
                    structure, structure_excluded, "start"
                )
                ind_start_structure = np.where(
                    structure[0, 2, 0] == structure_excluded[:, 2, 0]
                )[0]

            ind_end_structure = (
                np.where(structure[-1, 2, 0] == structure_excluded[:, 2, 0])[0] + 1
            )

            if len(ind_end_structure) == 0:
                if np.abs(structure[-1, 2, 0] - structure[-2, 2, 0]) < np.abs(
                    structure[-2, 2, 0] - structure[-3, 2, 0]
                ):
                    structure[-1, 2, :] = structure[-2, 2, 0] - np.abs(
                        structure[-2, 2, 0] - structure[-3, 2, 0]
                    )
                    ind_end_structure = (
                        np.where(structure[-1, 2, 0] == structure_excluded[:, 2, 0])[0]
                        + 1
                    )

                    if len(ind_end_structure) == 0:
                        structure_excluded = extrapolate_structure_nearest_nbr_cpu(
                            structure, structure_excluded, "end"
                        )
                        ind_end_structure = (
                            np.where(
                                structure[-1, 2, 0] == structure_excluded[:, 2, 0]
                            )[0]
                            + 1
                        )
                else:
                    structure_excluded = extrapolate_structure_nearest_nbr_cpu(
                        structure, structure_excluded, "end"
                    )
                    ind_end_structure = (
                        np.where(structure[-1, 2, 0] == structure_excluded[:, 2, 0])[0]
                        + 1
                    )

            structure_excluded = structure_excluded[
                ind_start_structure[0] : ind_end_structure[0] : select_stepping, :, :
            ]

            x_vector = np.arange(
                np.amin(structure[:, 0, :]) - 0.5,
                np.amax(structure[:, 0, :]) + voxel_size,
                voxel_size,
                dtype=np.float32,
            )

            y_vector = np.arange(
                np.amin(structure[:, 1, :]) - 0.5,
                np.amax(structure[:, 1, :]) + voxel_size,
                voxel_size,
                dtype=np.float32,
            )

            XY_mesh_grid = np.array(np.meshgrid(x_vector, y_vector), dtype=np.float32)
            XY_grid = np.vstack(XY_mesh_grid).reshape(2, -1).T
            poly_2D = np.swapaxes(structure[:, :2, :], 1, 2)
            poly_2D_exclude = np.swapaxes(structure_excluded[:, :2, :], 1, 2)

            r_vox = np.tile(
                XY_grid,
                (1, poly_2D.shape[0], poly_2D.shape[1] + poly_2D_exclude.shape[1]),
            ).reshape(
                (poly_2D.shape[0]),
                len(XY_grid),
                (poly_2D.shape[1] + poly_2D_exclude.shape[1]),
                2,
            )

            shape_1 = (poly_2D.shape[0], len(XY_grid), poly_2D.shape[1], 2)

            shape_2 = (
                poly_2D_exclude.shape[0],
                len(XY_grid),
                poly_2D_exclude.shape[1],
                2,
            )

            b1 = np.repeat(poly_2D, len(XY_grid), axis=0).reshape(shape_1)
            b2 = np.repeat(np.roll(poly_2D, 1, axis=1), len(XY_grid), axis=0).reshape(
                shape_1
            )
            s = b2 - b1  # directional vector along line 1

            b1 = np.repeat(poly_2D_exclude, len(XY_grid), axis=0).reshape(shape_2)
            b2 = np.repeat(
                np.roll(poly_2D_exclude, 1, axis=1), len(XY_grid), axis=0
            ).reshape(shape_2)
            s2 = b2 - b1  # directional vector along line 1

            s = np.concatenate((s, s2), axis=-2)

            v = np.repeat(poly_2D, len(XY_grid), axis=0).reshape(shape_1)
            v2 = np.repeat(poly_2D_exclude, len(XY_grid), axis=0).reshape(shape_2)

            v = np.concatenate((v, v2), axis=-2)

        else:
            x_vector = np.arange(
                np.amin(structure[:, 0, :]) - 0.5,
                np.amax(structure[:, 0, :]) + voxel_size,
                voxel_size,
                dtype=np.float32,
            )
            y_vector = np.arange(
                np.amin(structure[:, 1, :]) - 0.5,
                np.amax(structure[:, 1, :]) + voxel_size,
                voxel_size,
                dtype=np.float32,
            )

            XY_mesh_grid = np.array(np.meshgrid(x_vector, y_vector), dtype=np.float32)
            XY_grid = np.vstack(XY_mesh_grid).reshape(2, -1).T

            poly_2D = np.swapaxes(structure[:, :2, :], 1, 2)

            r_vox = np.tile(XY_grid, (1, poly_2D.shape[0], poly_2D.shape[1])).reshape(
                poly_2D.shape[0], len(XY_grid), poly_2D.shape[1], 2
            )

            b1 = np.repeat(poly_2D, len(XY_grid), axis=0).reshape(r_vox.shape)

            b2 = np.repeat(np.roll(poly_2D, 1, axis=1), len(XY_grid), axis=0).reshape(
                r_vox.shape
            )
            s = b2 - b1  # directional vector along line 1

            v = np.repeat(poly_2D, len(XY_grid), axis=0).reshape(r_vox.shape)

        z_vector = structure[:, 2, 0]

        det_v_s = (
            v[:, :, :, 0] * s[:, :, :, 1] - v[:, :, :, 1] * s[:, :, :, 0]
        )  # 2D cross product of v and s = determinate
        det_v_r = (
            v[:, :, :, 0] * r_vox[:, :, :, 1] - v[:, :, :, 1] * r_vox[:, :, :, 0]
        )  # 2D cross product of v and r = determinate
        det_r_s = (
            r_vox[:, :, :, 0] * s[:, :, :, 1] - r_vox[:, :, :, 1] * s[:, :, :, 0]
        )  # 2D cross product of r and s = determinate

        # these are the length along the line 1 and line 2
        t = np.divide(det_v_s, det_r_s)
        u = np.divide(det_v_r, det_r_s)

        t_mask = ((t > 0) & (t < 1)).astype(np.int32)
        u_mask = ((u > 0) & (u < 1)).astype(np.int32)

        mask = t_mask * u_mask

        indexs = np.sum(
            mask, axis=2
        )  # if they are 0 or 2 the point is not in contour but if 1, 3, or 5 ... it is.

        indexs = ((indexs == 1) + (indexs == 3) + (indexs == 5)).astype(np.bool_)

        indexs_reshaped = indexs.reshape(
            indexs.shape[0], y_vector.shape[0], x_vector.shape[0]
        )

        XY_grid_slice = np.tile(
            np.array(XY_mesh_grid, dtype=np.float32), (indexs_reshaped.shape[0], 1, 1)
        ).reshape(
            indexs_reshaped.shape[0],
            2,
            indexs_reshaped.shape[1],
            indexs_reshaped.shape[2],
        )

        XY_grid_slice = np.swapaxes(XY_grid_slice, 0, 1)

        Z_tiled = np.repeat(
            z_vector, (indexs_reshaped.shape[1] * indexs_reshaped.shape[2])
        ).reshape(indexs_reshaped.shape)

        XYZ_grid_slice = np.array(
            [
                np.array(XY_grid_slice[0], dtype=np.float32)[indexs_reshaped],
                np.array(XY_grid_slice[1], dtype=np.float32)[indexs_reshaped],
                np.array(Z_tiled, dtype=np.float32)[indexs_reshaped],
            ]
        )

        dose_calc_pts = XYZ_grid_slice.T

    else:
        print(
            "Warning: voxel_size is not divisable by slice_width, resample structure to desired sliceWdith before doseCal"
        )
        print(
            "slice_width: "
            + str(slice_width)
            + ", voxel_size: "
            + str(voxel_size)
            + ", remainder (should be 0.0): "
            + str(voxel_size % slice_width)
        )
        dose_calc_pts = None

    return dose_calc_pts


def TG43_dose_calc_cpu(
    dose_calc_pts,
    changed_structures,
    plan_parameters,
):
    # from the TPS there was negative times and times of 0, remove these from the array.
    # also flatten the array into an array of just 3D points, they were in needle sub arrays ** needs rewording

    active_dwell_times_mask = (changed_structures["changed_dwell_times"] > 0).astype(
        np.bool_
    )
    active_dwell_pts_mask = (np.repeat(active_dwell_times_mask, 3, axis=1)).reshape(
        len(active_dwell_times_mask), -1, 3
    )

    active_dwell_times = changed_structures["changed_dwell_times"][
        active_dwell_times_mask
    ]

    active_dwell_points_source_middle = (
        changed_structures["changed_dwell_coords"][active_dwell_pts_mask].reshape(-1, 3)
        / 10
    )
    active_dwell_points_source_sup_end = (
        changed_structures["changed_dwell_pts_source_end_sup"][
            active_dwell_pts_mask
        ].reshape(-1, 3)
        / 10
    )
    active_dwell_points_source_inf_end = (
        changed_structures["changed_dwell_pts_source_end_inf"][
            active_dwell_pts_mask
        ].reshape(-1, 3)
        / 10
    )

    dose_calc_pts = dose_calc_pts / 10

    # if there is no dose points, give an error
    if not dose_calc_pts.size:
        return "Error"

    dose_calc_pts_tiled = np.tile(
        dose_calc_pts[None, :, :], (active_dwell_points_source_middle.shape[0], 1, 1)
    )

    # Make an array of copies of the dwell times, mid, sup end, inf end. Many copies of the same list of points or times.
    active_dwell_points_source_middle_tiled = np.tile(
        active_dwell_points_source_middle[:, None, :], (1, dose_calc_pts.shape[0], 1)
    )
    active_dwell_points_source_inf_end_tiled = np.tile(
        active_dwell_points_source_inf_end[:, None, :], (1, dose_calc_pts.shape[0], 1)
    )
    active_dwell_points_source_sup_end_tiled = np.tile(
        active_dwell_points_source_sup_end[:, None, :], (1, dose_calc_pts.shape[0], 1)
    )

    # find the distances between the arrays. This is the r values in TG43, DoseCalnpts are the P in TG43
    # and SourceMidddle is the centre of the source points, the dwell points
    r = np.sqrt(
        np.sum(
            np.subtract(dose_calc_pts_tiled, active_dwell_points_source_middle_tiled)
            ** 2,
            axis=2,
        )
    )

    # obtains the angle between all Pts (dose calc points) and the midddle of the source
    # for all dwell points in one array. The dot product between line source and vector
    # between P and source middle pt is calculated than inverse cos to calculate the angle
    # theta in TG43. theta = cos^(-1) [vec1 (dot) vec2 / (length(vec1) x length(vec2))]
    # here we find the dot product by multiplying x,y,z components and summing

    theta_dot_product = np.divide(
        np.sum(
            np.multiply(
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

    positive_angle_mask = (theta_dot_product > 1).astype(np.int32)
    negative_angle_mask = (theta_dot_product < -1).astype(np.int32)
    other_angle_mask = (
        ~np.array(positive_angle_mask + negative_angle_mask, dtype=np.float32).astype(
            np.bool_
        )
    ).astype(np.int32)

    theta_dot_product = (
        np.multiply(0.9999999999, positive_angle_mask)
        + np.multiply(-0.9999999999, negative_angle_mask)
        + np.multiply(theta_dot_product, other_angle_mask)
    )

    theta = np.arccos(theta_dot_product)

    # # To find the angle beta in TG43,  minus the angles between each source end
    # # and the Point P. Some working out but in terms of L =  source length, theta found above and r above.

    beta = np.arctan2(
        r * np.cos(theta) - plan_parameters["L"] / 2.0, r * np.sin(theta)
    ) - np.arctan2(r * np.cos(theta) + plan_parameters["L"] / 2.0, r * np.sin(theta))

    # G = the geometric function in TG43. For small angles close to 0 degrees or 180 degrees beta and theta
    # get close to zero, so  division of zero. At these angles, it is similar to a point source (1/r^2).
    # G for small and large angles are found in two martix operations. Than two masks are used to select the
    # angles which are actually small and those that are not extreme (otheranglesmask for Large angles).

    G = np.multiply(
        1.0 / (r * r - (plan_parameters["L"] * plan_parameters["L"]) / 4.0),
        ((np.pi - theta) < 0.003).astype(np.int32) + ((theta) < 0.003).astype(np.int32),
    ) + np.multiply(
        beta / (plan_parameters["L"] * r * np.sin(theta)),
        (
            ~np.array(
                (
                    ((np.pi - theta) < 0.003).astype(np.int32)
                    + ((theta) < 0.003).astype(np.int32)
                ).astype(np.bool_)
            )
        ).astype(np.int32),
    )

    G_mask = (G > 0).astype(np.bool_)
    G[G_mask] = -G[G_mask]

    # this is the TG43 equation F_interp and g_interp are extracted from the source excel data sheet and
    # interpolated as explained else where. Sk is the air kerma, DoseRateConstant, G0 are all extracted from
    # the treatment plan files. g and F point to two functions which find the values in the interpolated
    # array for all r or r & theta. It does this for all values at once. Summing accross axis zero adds all dose
    # contributions from source at one point per sum, resulting in the dose calculated at all dose points.

    r_high_value = plan_parameters["F_interp"][
        (np.rint(np.degrees(theta) * 10) - 1).astype(np.int32),
        (np.rint(100 * 10) - 1).astype(np.int32),
    ]

    r_low_value = plan_parameters["F_interp"][
        (np.rint(np.degrees(theta) * 10) - 1).astype(np.int32),
        (np.rint(r * 100) - 1).astype(np.int32),
    ]

    f_dose = np.select([r > 10, r <= 10], [r_high_value, r_low_value])

    r_high_value = plan_parameters["g_interp"][
        (np.rint(8 * 100) - 1).astype(np.int32)
    ] * np.exp(
        (r - 8)
        / (10 - 8)
        * (
            np.log(
                plan_parameters["g_interp"][(np.rint(10 * 100) - 1).astype(np.int32)]
            )
            - np.log(
                plan_parameters["g_interp"][(np.rint(8 * 100) - 1).astype(np.int32)]
            )
        )
    )

    r_low_value = plan_parameters["g_interp"][
        (np.rint(0.15 * 100) - 1).astype(np.int32)
    ]

    r_middle_source = plan_parameters["g_interp"][
        (np.rint(r * 100) - 1).astype(np.int32)
    ]

    g_dose = np.select(
        [r > 10, r < 0.15, (r > 0.15) ^ (r > 10)],
        [r_high_value, r_low_value, r_middle_source],
    )

    active_dwell_times = np.tile(
        active_dwell_times[:, None], (1, dose_calc_pts.shape[0])
    )

    dose = np.sum(
        (
            plan_parameters["air_kerma_rate"]
            * plan_parameters["dose_rate_constant"]
            * (G / plan_parameters["G0"])
            * g_dose
            * f_dose
            * active_dwell_times
            / 360000
        ),
        axis=0,
    )

    return dose


def extrapolate_structure_nearest_nbr_cpu(structure, structure_excluded, location):
    slice_width_2 = np.float32(
        np.absolute(structure_excluded[0, 2, 0] - structure_excluded[1, 2, 0])
    )

    if location == "start":
        num_needed = int(
            np.abs(structure[0, 2, 0] - structure_excluded[0, 2, 0]) / slice_width_2
        )

        extra_contours = np.repeat(
            np.array([structure_excluded[0, :, :]]), num_needed, axis=0
        )
        new_z = np.repeat(
            np.flip(
                np.arange(
                    structure_excluded[0, 2, 0] + slice_width_2,
                    structure[0, 2, 0] + slice_width_2,
                    slice_width_2,
                )
            ),
            extra_contours.shape[2],
            axis=0,
        ).reshape(-1, extra_contours.shape[2])

        extra_contours[:, 2, :] = new_z

        new_arr = np.zeros(
            (
                structure_excluded.shape[0] + extra_contours.shape[0],
                structure_excluded.shape[1],
                structure_excluded.shape[2],
            )
        )
        new_arr[0 : len(extra_contours), :, :] = extra_contours
        new_arr[len(extra_contours) :, :, :] = structure_excluded

    else:
        num_needed = int(
            (
                np.abs(structure[-1, 2, 0] - structure_excluded[-1, 2, 0])
                + slice_width_2 * 2
            )
            / slice_width_2
        )

        extra_contours = np.repeat(
            np.array([structure_excluded[-1, :, :]]), num_needed, axis=0
        )
        new_z = np.repeat(
            np.flip(
                np.arange(
                    structure[-1, 2, 0] - slice_width_2,
                    structure_excluded[-1, 2, 0] + slice_width_2,
                    slice_width_2,
                )
            ),
            extra_contours.shape[2],
            axis=0,
        ).reshape(-1, extra_contours.shape[2])

        extra_contours[:, 2, :] = new_z
        new_arr = np.zeros(
            (
                structure_excluded.shape[0] + extra_contours.shape[0],
                structure_excluded.shape[1],
                structure_excluded.shape[2],
            )
        )

        new_arr[: len(structure_excluded), :, :] = structure_excluded
        new_arr[len(structure_excluded) :, :, :] = extra_contours

    return new_arr


__all__ = ["fast_TG43_cpu", "get_dose_volume_pts_cpu"]
