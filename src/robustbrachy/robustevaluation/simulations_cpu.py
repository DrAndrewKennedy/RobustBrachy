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
###          SIMULATIONS          ###
#####################################


# CPU based simulations
def simulations_cpu(
    sim_magnitudes, changed_structures, plan_parameters, optimisation=False
):
    """Runs through the 6 changes to the nominal treatment plan. Passes over the function if the change value is zero.

    Args:
        sim_magnitudes (dict): All parameter change magnitudes in the form of arrays.
        changed_structures (dict): A copy of all structures that will be changed to form an uncertainty scenario.
        plan_parameters (dict): all plan parameters such as nominal structures, dwell times and points, source information
        optimisation (bool): if the algorithm is using TPS (False) or optimisaiton-defined (True) dwell coordinates
    """
    if optimisation != False:
        bezier_arc_length_points = plan_parameters[
            "bezier_arc_length_points_optimisation"
        ]
        bezier_arc_length_dists = plan_parameters[
            "bezier_arc_length_dists_optimisation"
        ]
        mask_dwell_coords = plan_parameters["mask_dwell_coords_optimisation"]
        mask_dwell_times = plan_parameters["mask_dwell_times_optimisation"]
    else:
        bezier_arc_length_points = plan_parameters["bezier_arc_length_points_TPS"]
        bezier_arc_length_dists = plan_parameters["bezier_arc_length_dists_TPS"]
        mask_dwell_coords = plan_parameters["mask_dwell_coords_TPS"]
        mask_dwell_times = plan_parameters["mask_dwell_times_TPS"]

    ### 1) Needle shift inf-sup direction along needle. Finds new points along bezier curves of each needle , replaces these new dwell coordinates to structure array
    if not np.all(np.array(sim_magnitudes["dwells_shift_along_needle"]) == 0):
        (
            changed_structures["changed_dwell_coords"],
            changed_structures["changed_dwell_pts_source_end_sup"],
            changed_structures["changed_dwell_pts_source_end_inf"],
        ) = move_along_channel_cpu(
            bezier_arc_length_points,
            bezier_arc_length_dists,
            changed_structures["changed_dwell_coords"],
            sim_magnitudes["dwells_shift_along_needle"],
            plan_parameters["L"],
            mask_dwell_coords,
            mask_dwell_times,
        )

    # 2) # Method: for worstcase only. Find prostate 2D centre per slice; interpolate to larger array; vector between centre and dwell pts; get new dwell pt; update dwell coordinates
    if sim_magnitudes["dwells_move_out_from_prostate_centre_line"] != 0:
        # get the line along the centre of the prostate in each slice and interpolate
        centre_of_prostate_line_sup_to_inf_interp = get_prostate_centre_line_cpu(
            changed_structures["changed_prostate_contour_pts"]
        )  # this fn gets the prostate centre line and also interpolates it to 1000 points, includes the avg radius per point for use in scaling the movement distance.

        # apply change to dwell points
        (
            changed_structures["changed_dwell_coords"],
            changed_structures["changed_dwell_pts_source_end_sup"],
            changed_structures["changed_dwell_pts_source_end_inf"],
        ) = move_dwells_out_cpu(
            centre_of_prostate_line_sup_to_inf_interp,
            changed_structures["changed_dwell_coords"],
            sim_magnitudes["dwells_move_out_from_prostate_centre_line"],
            changed_structures["changed_dwell_pts_source_end_sup"],
            changed_structures["changed_dwell_pts_source_end_inf"],
            mask_dwell_coords,
            mask_dwell_times,
        )

    # 3) # increase dwell times as a percentage
    if sim_magnitudes["dwell_time_change_percentage"] != 0:
        try:
            changed_structures["changed_arr_of_dwells_times"] = changed_structures[
                "changed_arr_of_dwells_times"
            ] * (1 + sim_magnitudes["dwell_time_change_percentage"])
        except:
            changed_structures["changed_dwell_times"] = changed_structures[
                "changed_dwell_times"
            ] * (1 + sim_magnitudes["dwell_time_change_percentage"])

    # 4) # shrink or enlarge prostate and then need to resample the structure
    if sim_magnitudes["mm_change_shrink_enlarge_prostate"] != 0:
        slice_width = np.float32(
            np.absolute(
                changed_structures["changed_prostate_contour_pts"][0, 2, 0]
                - changed_structures["changed_prostate_contour_pts"][1, 2, 0]
            )
        )

        changed_structures[
            "changed_prostate_contour_pts"
        ] = enlarge_shrink_structure_cpu(
            changed_structures["changed_prostate_contour_pts"],
            sim_magnitudes["mm_change_shrink_enlarge_prostate"],
        )

        changed_structures["changed_prostate_contour_pts"] = resample_structure_cpu(
            changed_structures["changed_prostate_contour_pts"], slice_width=slice_width
        )

    # 5) # Method: simplification with just moving whole prostate , dwells, and urethra
    if (
        np.absolute(sim_magnitudes["mm_change_X_rigid"])
        + np.absolute(sim_magnitudes["mm_change_Y_rigid"])
        + np.absolute(sim_magnitudes["mm_change_Y_rigid"])
    ) != 0:
        changed_structures["changed_prostate_contour_pts"] = rigid_move_cpu(
            changed_structures["changed_prostate_contour_pts"],
            sim_magnitudes["mm_change_X_rigid"],
            sim_magnitudes["mm_change_Y_rigid"],
            sim_magnitudes["mm_change_Y_rigid"],
        )

        changed_structures["changed_urethra_contour_pts"] = rigid_move_cpu(
            changed_structures["changed_urethra_contour_pts"],
            sim_magnitudes["mm_change_X_rigid"],
            sim_magnitudes["mm_change_Y_rigid"],
            sim_magnitudes["mm_change_Y_rigid"],
        )

        (
            changed_structures["changed_dwell_coords"],
            changed_structures["changed_dwell_pts_source_end_sup"],
            changed_structures["changed_dwell_pts_source_end_inf"],
        ) = move_dwells_rigid_cpu(
            changed_structures["changed_dwell_coords"],
            changed_structures["changed_dwell_pts_source_end_sup"],
            changed_structures["changed_dwell_pts_source_end_inf"],
            mask_dwell_coords,
            sim_magnitudes["mm_change_X_rigid"],
            sim_magnitudes["mm_change_Y_rigid"],
            sim_magnitudes["mm_change_Y_rigid"],
        )

    # 4a) Rectum, enlarge or shrink per slice#
    if sim_magnitudes["mm_change_shrink_enlarge_rectum"] != 0:
        changed_structures[
            "changed_rectum_contour_pts"
        ] = enlarge_shrink_structure_per_slice_cpu(
            changed_structures["changed_rectum_contour_pts"],
            sim_magnitudes["mm_change_shrink_enlarge_rectum"],
        )

    # 4b) Rectum, enlarge or shrink per slice#
    if sim_magnitudes["mm_change_shrink_enlarge_urethra"] != 0:
        changed_structures[
            "changed_urethra_contour_pts"
        ] = enlarge_shrink_structure_per_slice_cpu(
            changed_structures["changed_urethra_contour_pts"],
            sim_magnitudes["mm_change_shrink_enlarge_urethra"],
        )

    # 5) Dwell times randomly move in 2D transverse plane
    if not np.all(sim_magnitudes["mm_change_dwells_random_2D"] == 0):
        (
            changed_structures["changed_dwell_coords"],
            changed_structures["changed_dwell_pts_source_end_sup"],
            changed_structures["changed_dwell_pts_source_end_inf"],
        ) = move_dwells_randomly_2D_cpu(
            changed_structures["changed_dwell_coords"],
            sim_magnitudes["mm_change_dwells_random_2D"],
            changed_structures["changed_dwell_pts_source_end_sup"],
            changed_structures["changed_dwell_pts_source_end_inf"],
            mask_dwell_coords,
        )

    # 6) Dwell time precision. randomly increase, decrease by ms
    if not np.all(sim_magnitudes["dwell_time_increase_decrease"] == 0):
        try:
            changed_structures[
                "changed_arr_of_dwells_times"
            ] = dwell_time_increase_decrease_cpu(
                changed_structures["changed_arr_of_dwells_times"],
                sim_magnitudes["dwell_time_increase_decrease"],
            )
        except:
            changed_structures[
                "changed_dwell_times"
            ] = dwell_time_increase_decrease_cpu(
                changed_structures["changed_dwell_times"],
                sim_magnitudes["dwell_time_increase_decrease"],
            )

    # this makes sure any dwell times with -100 that were multiplied and changed are switched back to -100.
    if optimisation == False:
        changed_structures["changed_dwell_times"] = (
            changed_structures["changed_dwell_times"] * (~mask_dwell_times)
            + -100 * mask_dwell_times
        )

    return changed_structures


def move_along_channel_cpu(
    map_pts, map_dist, pts, move_magnitude, L, mask_dwell_coords, mask_dwell_times
):
    ##notes:
    # remember curve length should be 130 mm
    # search pts in map for the clostest two pt in array and interpolate mapped distance between using ratios
    # add move_magnitude and find closet two closest mapped point and find the ratio linear point between them
    # Useses nearest neigbour interpolartion

    # These arrays work in parallel by moving all dwell pt in all needles at once.
    # Idea is to find shortest distance between the interpolated Beizer curve for each needle and each dwell point
    # Get index of points in Beizer curve array, use index in distance along Beizer curve array
    # add the shift disance to these distances along
    # find new indexs in distance along Beizer curve arrays
    # use indexs to get corresponding points on beizer curves.

    # more info: two arrays that were interpolated to 1000-500 points, one has points along each needle
    # and the other the is the cumulative distance along the Beizer curve. The indexs of array
    # of the points correspond to the cum dist. array indexs, so they can be used to refer to eachother.

    step_1 = np.repeat(map_pts, len(pts[0]), axis=0).reshape(
        len(pts), len(pts[0]), len(map_pts[0]), 3
    )
    step_2 = np.repeat(pts, len(map_pts[0]), axis=1).reshape(
        len(pts), len(pts[0]), len(map_pts[0]), 3
    )

    min_indices = (
        (np.absolute(step_1 - step_2) * np.absolute(step_1 - step_2))
        .sum(axis=3)
        .argmin(axis=2)
        .reshape(len(pts), len(pts[0]), 1)
    )

    distance_along_curve = np.take_along_axis(
        map_dist.reshape(len(map_dist), len(map_dist[0]), 1), min_indices, axis=1
    )
    new_distance_along_curve = distance_along_curve + move_magnitude.reshape(
        len(move_magnitude), len(move_magnitude[0]), 1
    )
    dist_of_end_of_dwells_sup = (
        distance_along_curve
        + move_magnitude.reshape(len(move_magnitude), len(move_magnitude[0]), 1)
        - L / 2 * 10
    )
    dist_of_end_of_dwells_inf = (
        distance_along_curve
        + move_magnitude.reshape(len(move_magnitude), len(move_magnitude[0]), 1)
        + L / 2 * 10
    )

    # middle of source; the dwell points
    step_1 = np.repeat(map_dist, len(new_distance_along_curve[0]), axis=0).reshape(
        len(new_distance_along_curve),
        len(new_distance_along_curve[0]),
        len(map_dist[0]),
    )
    step_2 = np.repeat(new_distance_along_curve, len(map_dist[0]), axis=1).reshape(
        len(new_distance_along_curve),
        len(new_distance_along_curve[0]),
        len(map_dist[0]),
    )
    min_indices = (
        (np.absolute(step_1 - step_2))
        .argmin(axis=2)
        .reshape(len(new_distance_along_curve), len(new_distance_along_curve[0]), 1)
    )
    new_clostest_pts = (
        np.take_along_axis(map_pts, min_indices, axis=1)
        * (np.invert(mask_dwell_coords))
        + (-100) * mask_dwell_coords
    )

    # Sup end of source
    step_1 = np.repeat(map_dist, len(dist_of_end_of_dwells_sup[0]), axis=0).reshape(
        len(dist_of_end_of_dwells_sup),
        len(dist_of_end_of_dwells_sup[0]),
        len(map_dist[0]),
    )
    step_2 = np.repeat(dist_of_end_of_dwells_sup, len(map_dist[0]), axis=1).reshape(
        len(dist_of_end_of_dwells_sup),
        len(dist_of_end_of_dwells_sup[0]),
        len(map_dist[0]),
    )
    min_indices = (
        (np.absolute(step_1 - step_2))
        .argmin(axis=2)
        .reshape(len(dist_of_end_of_dwells_sup), len(dist_of_end_of_dwells_sup[0]), 1)
    )
    new_clostest_pts_end_of_source_sup = (
        np.take_along_axis(map_pts, min_indices, axis=1)
        * (np.invert(mask_dwell_coords))
        + (-100) * mask_dwell_coords
    )

    # Inf end of source
    step_1 = np.repeat(map_dist, len(dist_of_end_of_dwells_inf[0]), axis=0).reshape(
        len(dist_of_end_of_dwells_inf),
        len(dist_of_end_of_dwells_inf[0]),
        len(map_dist[0]),
    )
    step_2 = np.repeat(dist_of_end_of_dwells_inf, len(map_dist[0]), axis=1).reshape(
        len(dist_of_end_of_dwells_inf),
        len(dist_of_end_of_dwells_inf[0]),
        len(map_dist[0]),
    )
    min_indices = (
        (np.absolute(step_1 - step_2))
        .argmin(axis=2)
        .reshape(len(dist_of_end_of_dwells_inf), len(dist_of_end_of_dwells_inf[0]), 1)
    )
    new_clostest_pts_end_of_source_inf = (
        np.take_along_axis(map_pts, min_indices, axis=1)
        * (np.invert(mask_dwell_coords))
        + (-100) * mask_dwell_coords
    )

    return (
        new_clostest_pts,
        new_clostest_pts_end_of_source_sup,
        new_clostest_pts_end_of_source_inf,
    )


def get_prostate_centre_line_cpu(changed_prostate_contour_pts):
    # gets the geometric mean value from each slice in the prostate, forming a line
    centre_line_array = np.mean(changed_prostate_contour_pts, axis=2)

    # gets the average distance (radius) per slice between each contoured point on the 2D boundry in a slice and the centre of the prostate in that slice
    average_radius = np.linalg.norm(
        changed_prostate_contour_pts
        - np.repeat(
            centre_line_array, len(changed_prostate_contour_pts[1][0]), axis=1
        ).reshape(len(changed_prostate_contour_pts), 3, -1),
        axis=1,
    ).mean(axis=1)

    # my own linear interpolation method. values are not evenly spaced but they are evenly spaced between original points and there is an even number of them between original points.
    centre_line_array_interp = np.append(
        np.hstack(
            np.array(
                (
                    (centre_line_array - np.roll(centre_line_array, 1, axis=0))
                    / (100 + 1)
                ),
                dtype=np.float32,
            )[1:, :, np.newaxis]
            * np.arange(0, 100 + 1, 1, dtype=np.float32)
            + np.roll(centre_line_array, 1, axis=0)[1:, :, np.newaxis]
        ).T,
        [centre_line_array[-1]],
        axis=0,
    )

    average_radius_interp = np.append(
        np.hstack(
            np.array(
                ((average_radius - np.roll(average_radius, 1, axis=0)) / (100 + 1)),
                dtype=np.float32,
            )[1:, np.newaxis]
            * np.arange(0, 100 + 1, 1, dtype=np.float32)
            + np.roll(average_radius, 1, axis=0)[1:, np.newaxis]
        ).T,
        [average_radius[-1]],
        axis=0,
    )

    # puts the interpolated points and avg radius in that transverse direction into one array
    centre_line_array_interp_with_average_radius = np.concatenate((
        centre_line_array_interp,
        np.array([average_radius_interp], dtype=np.float32).T),
        axis=1,
    ).T

    return centre_line_array_interp_with_average_radius


def move_dwells_out_cpu(
    centre_line_array_interp_with_average_radius,
    dwell_pts,
    mm_change,
    dwell_pts_source_end_sup,
    dwell_pts_source_end_inf,
    mask_dwell_coords,
    mask_dwell_times,
):
    # This has been vectorised.
    # get z point on centre_line_array_interp_with_average_radius that is closest to z coord in needle
    # use vector between centre_line_array_interp_with_average_radius pt and needle pt (2d is fine) to move out 'mm_change' (-ve value moves in)

    # find closest points along the prostate centre line. finds the closest Z point between prostate centre line array
    # and the array of all dwell points.
    step_0 = np.c_[
        centre_line_array_interp_with_average_radius[0, :],
        centre_line_array_interp_with_average_radius[1, :],
        centre_line_array_interp_with_average_radius[2, :],
        centre_line_array_interp_with_average_radius[3, :],
    ]
    step_1 = np.tile(step_0[:, 2], (len(dwell_pts), len(dwell_pts[0]), 1))
    step_2 = np.repeat(dwell_pts[:, :, 2], len(step_0)).reshape(
        len(dwell_pts), len(dwell_pts[0]), -1
    )

    min_indices = (np.absolute(step_1 - step_2)).argmin(axis=2)
    closest_pts = step_0[min_indices]

    # find the corresponding radius of the prostate at that Z location for all dwell points
    step_1 = np.repeat(step_0[:, 3], len(min_indices), axis=0).reshape(
        len(min_indices), len(step_0)
    )
    max_radius = np.take_along_axis(step_1, min_indices, axis=1)

    # finds the distance between the dwell point and the prostate centreline as a unit vector for direction (v_norm) and than its magnitude (norm)
    v_norm = np.array(dwell_pts - closest_pts[:, :, 0:3], dtype=np.float32)
    norm_cpu = np.linalg.norm(v_norm, axis=2)

    # find the fraction of how far out that dwell point is from the prostate centre line with reference to the prostates
    # radius at that Z plane. if the fraction is greater than the intended change movement, it sets it to the intended
    # movement. The fraction is to simulate how oedema pushes needles out with greater displacment if they a further
    # from prostate centre line.
    fraction_of_radius = norm_cpu / max_radius
    fraction_of_radius[fraction_of_radius[:, :] > np.absolute(mm_change)] = np.absolute(
        mm_change
    )  # maximum change should be the intended shift amount

    # get the direction normal vector along the intended movement direction (radial) then scales it to the correct size.
    v_norm_direction = v_norm / norm_cpu[:, :, np.newaxis]
    change_amount = (
        v_norm_direction
        * np.array(fraction_of_radius * mm_change, dtype=np.float32)[:, :, np.newaxis]
    )

    # applies the change to the dewll points to get the new dwell points
    new_dwell_pts = dwell_pts + change_amount
    dwell_pts_source_end_sup = dwell_pts_source_end_sup + change_amount
    dwell_pts_source_end_inf = dwell_pts_source_end_inf + change_amount

    new_dwell_pts = (
        new_dwell_pts * (np.invert(mask_dwell_coords)) + (-100) * mask_dwell_coords
    )
    dwell_pts_source_end_sup = (
        dwell_pts_source_end_sup * (np.invert(mask_dwell_coords))
        + (-100) * mask_dwell_coords
    )
    dwell_pts_source_end_inf = (
        dwell_pts_source_end_inf * (np.invert(mask_dwell_coords))
        + (-100) * mask_dwell_coords
    )

    return new_dwell_pts, dwell_pts_source_end_sup, dwell_pts_source_end_inf


def enlarge_shrink_structure_cpu(prostate_contour_pts, mm_change=0):
    #   1) find prostate centre (x,y,z)
    if len(prostate_contour_pts) % 2:  # Odd
        centre = np.mean(prostate_contour_pts[len(prostate_contour_pts) // 2], axis=1)

    else:  # Even
        centre = np.mean(
            (
                prostate_contour_pts[len(prostate_contour_pts) // 2 - 1]
                + prostate_contour_pts[len(prostate_contour_pts) // 2]
            )
            / 2,
            axis=1,
        )

    #   2) find distance between prostate centre and all points in 3D
    # a) subtract to get vector between centre and each point, need to tile the centre first so it has as many points as the prostate

    step_2 = np.tile(
        np.array(
            [
                np.repeat(
                    np.array([centre[0]], dtype=np.float32),
                    len(prostate_contour_pts[0][0]),
                    axis=0,
                ),
                np.repeat(
                    np.array([centre[1]], dtype=np.float32),
                    len(prostate_contour_pts[0][0]),
                    axis=0,
                ),
                np.repeat(
                    np.array([centre[2]], dtype=np.float32),
                    len(prostate_contour_pts[0][0]),
                    axis=0,
                ),
            ],
            dtype=np.float32,
        ),
        (len(prostate_contour_pts), 1, 1),
    ).reshape(len(prostate_contour_pts), 3, -1)

    direction_vector = np.array(prostate_contour_pts - step_2, dtype=np.float32)

    # b) find the norm (distance/magnitude) of all difference vectors
    distance = np.linalg.norm(direction_vector, axis=1)

    #   3) Find directional unit vector
    unit_vector = direction_vector / distance[:, np.newaxis, :]

    #   4) Find magnitude change in correct direction by scalling direction unit vectors
    change_magnitude_and_direction = unit_vector * mm_change

    #   5) add change to current point
    new_prostate_contour_pts = change_magnitude_and_direction + prostate_contour_pts

    return new_prostate_contour_pts


def resample_structure_cpu(structure, slice_width):
    # 1) this part reorders the structure points so nearest Z coordinates are located at the
    # same position on list of points in each slice. Essentially the points above and below
    # in each slice are at the same point in the slice array.

    # It also interpolate the number of points in each slice so there is more points
    # to choose from.

    # Reordering is important when swaping axis from XY-slices to Z-radial slices.

    # create a new array to store ordered points and
    # put first slice values into newly created array

    structure_ordered = np.array([np.copy(structure[0])], dtype=np.float32)

    # we are going to go through slice be slice here. item is the current number of iterations
    for item, val in enumerate(structure):
        # if statment basically ignores the first slice since it is already in the reodered
        # array. The remaining slices will be ordered based on the first point in the first
        # array.
        if (item + 1) < (len(structure)):
            # grabs the next slice so that it can be compared to the last one in strutOrdered
            next_slice = structure[item + 1].T

            # this is a vectorised linear interploation method. not even points around contour outline, but even points (50) between each old point, even dist apart
            # places 50 points evenly between old points in a 2D slice
            next_slice = (
                np.hstack(
                    np.array(
                        (next_slice - np.roll(next_slice, 1, axis=0)) / (50 + 1),
                        dtype=np.float32,
                    )[:, :, np.newaxis]
                    * np.arange(0, 50 + 1, 1, dtype=np.float32)
                    + np.roll(next_slice, 1, axis=0)[:, :, np.newaxis]
                ).T
            ).T

            x_interp = next_slice[0, :]
            y_interp = next_slice[1, :]
            z_interp = next_slice[2, :]

            # This part basically finds the shortest distance between all points in the next
            # slice and all values in the previous slice (each slice in TPS is orginally 32 (or 16)
            # points long). It then selects the points in the interpolate array that line up
            # with the 32 points in the last slice.
            # Make copies of the last slice array such that every point in the prevous slice
            # can be subtracted from every point in the interpolated next slice
            # (of length len(x_fine))

            first_pts_array = np.repeat(
                structure_ordered[item].T, len(x_interp), axis=1
            ).reshape(len(structure_ordered[item].T), 3, len(x_interp))

            # also need to make 32 copies of the interpolated next slice
            next_slice = np.tile(
                next_slice, (len(structure_ordered[item].T), 1, 1)
            ).reshape(len(structure_ordered[item].T), 3, len(x_interp))

            # now find all min distances between all points of 32 points in the last array
            # and all points in the interpolated next array. What does this acheive? A list
            # of indexs in the interpolated next array that are directly above (below) the 32
            # points in the last slice array.
            minDistIndex_cpu = np.linalg.norm(
                (first_pts_array - next_slice), axis=1
            ).argmin(axis=1)

            # This selects the closest points to the last array and stores them.

            structure_ordered = np.append(
                structure_ordered,
                np.array(
                    [
                        [
                            x_interp[minDistIndex_cpu],
                            y_interp[minDistIndex_cpu],
                            z_interp[minDistIndex_cpu],
                        ]
                    ],
                    dtype=np.float32,
                ),
                axis=0,
            )

            # each slice is looped over to the end of all slices

    # 2) This is the second major part of the process. Here we turn the slices into curves
    # along the sup inf direction and interpolate each curve then select values along each
    # curve that is the correct slice distance apart.
    # want the slices to be 32 curves along the sup inf direction rather than closed axial
    # curves.
    structure_reordered_along_z = np.swapaxes(structure_ordered, 0, 2)

    # get max and min values in the sup and inf directions and calculate the length of the
    # structure in the sup inf direciton.
    max_z = structure_reordered_along_z[
        0, 2, :
    ].max()  # will be negative Z to more negative Z
    min_z = structure_reordered_along_z[-1, 2, :].min()

    # calculate the number slices in the new structure. It rounds to whole names since they
    # are whole numbers in the TPS that we are using.
    slices = np.flip(
        np.arange(
            np.round(min_z, 0).astype(np.int32),
            np.round(max_z, 0).astype(np.int32) + slice_width,
            slice_width,
        )
    )

    first_slice_diff = structure[0, 2, :].min() - slices[0]
    last_slice_diff = structure[-1, 2, :].max() - slices[-1]

    # this step deletes the first and/or last slice Z value in slices if they are beyond the avaliable data
    if first_slice_diff < 0:
        slices = slices[(slices != slices[0])]
    if last_slice_diff > 0:
        slices = slices[(slices != slices[-1])]

    # similar process as the interpolated process above. for each of the sup-inf curves,
    # interpolate the array then find the points in the curve that line up with the Z values
    # in the 'slices' array above. The 'slices' array is an array of Z values that you want
    # the axial slices to be at. These are found by the min distance between the 'slices'
    # array values (Z values for the slice) and Z coordinates of the points in each curve of
    # the 'StrutctureZ' array. ('structure_reordered_along_z' array is not just Z coordinates but all points
    # sorted such that they a curves along the sup inf direction, Z direction)

    # linear interpolation, authors own method since scipy couldn't be vectorised

    structure_reordered_along_z_interp = np.swapaxes(
        np.array(
            (
                np.swapaxes(structure_reordered_along_z, 1, 2)
                - np.swapaxes(np.roll(structure_reordered_along_z, 1, axis=2), 1, 2)
            )
            / (50 + 1),
            dtype=np.float32,
        )[:, 1:, :, np.newaxis]
        * np.arange(0, 50 + 1, 1, dtype=np.float32)
        + np.swapaxes(np.roll(structure_reordered_along_z, 1, axis=2), 1, 2)[
            :, 1:, :, np.newaxis
        ],
        1,
        2,
    ).reshape(
        structure_reordered_along_z.shape[0], structure_reordered_along_z.shape[1], -1
    )

    step_1 = np.tile(
        structure_reordered_along_z_interp[:, 2, :], (slices.shape[0])
    ).reshape(structure_reordered_along_z_interp[:, 2, :].shape[0], slices.shape[0], -1)
    step_2 = np.hstack(
        np.tile(
            slices,
            (
                1,
                structure_reordered_along_z_interp[:, 2, :].shape[1],
                structure_reordered_along_z_interp[:, 2, :].shape[0],
            ),
        )
    ).T.reshape(step_1.shape)

    min_indices = (
        np.absolute(step_1 - step_2)
        .argmin(axis=2)
        .reshape(len(structure_ordered[item].T), 1, -1)
    )

    resampled_structure = np.take_along_axis(
        structure_reordered_along_z_interp, min_indices, axis=2
    )

    # round the Z values in the structure
    resampled_structure[:, 2, :] = np.around(resampled_structure[:, 2, :], decimals=1)

    # swap back to axial slices rather than 32 curves along the sup inf direction
    resampled_structure = np.swapaxes(resampled_structure, 2, 0)

    return resampled_structure.astype(np.float32)


def rigid_move_cpu(
    structure, mm_change_X_rigid=0, mm_change_Y_rigid=0, mm_change_Z_rigid=0
):
    # makes a number of copies of the change in mm vector so there is one vector per point in
    # the structure
    mm_change_array = np.tile(
        np.array(
            [
                np.repeat(
                    np.array([mm_change_X_rigid], dtype=np.float32),
                    len(structure[0][0]),
                    axis=0,
                ),
                np.repeat(
                    np.array([mm_change_Y_rigid], dtype=np.float32),
                    len(structure[0][0]),
                    axis=0,
                ),
                np.repeat(
                    np.array([mm_change_Z_rigid], dtype=np.float32),
                    len(structure[0][0]),
                    axis=0,
                ),
            ],
            dtype=np.float32,
        ),
        (len(structure), 1, 1),
    ).reshape(len(structure), 3, -1)

    # makes the change
    structure = structure + mm_change_array

    return structure


def move_dwells_rigid_cpu(
    dwell_pts,
    dwell_pts_source_end_sup,
    dwell_pts_source_end_inf,
    mask_dwell_coords,
    mm_change_X_rigid=0,
    mm_change_Y_rigid=0,
    mm_change_Z_rigid=0,
):
    # sims a rigid move of the dwells WITH the prostate moving. Different from the
    # rigid_move method since it requires the masking of the arrays.
    mm_change_array = np.tile(
        np.array(
            [mm_change_X_rigid, mm_change_Y_rigid, mm_change_Z_rigid], dtype=np.float32
        ),
        (len(dwell_pts), len(dwell_pts[0]), 1),
    )

    dwell_pts = (np.array(dwell_pts, dtype=np.float32) + mm_change_array) * (
        np.invert(mask_dwell_coords)
    ) + (-100) * mask_dwell_coords
    dwell_pts_source_end_sup = (
        np.array(dwell_pts_source_end_sup, dtype=np.float32) + mm_change_array
    ) * (np.invert(mask_dwell_coords)) + (-100) * mask_dwell_coords
    dwell_pts_source_end_inf = (
        np.array(dwell_pts_source_end_inf, dtype=np.float32) + mm_change_array
    ) * (np.invert(mask_dwell_coords)) + (-100) * mask_dwell_coords

    return dwell_pts, dwell_pts_source_end_sup, dwell_pts_source_end_inf


def enlarge_shrink_structure_per_slice_cpu(structure, mm_change=0):
    # 1) find gradient vecotr along each 2D contour in each slice at once
    structure_gradient = np.gradient(structure, axis=2)

    # 2) Caclulate normal vector at each point (pointing out) for all slices at once
    structure_gradient = structure_gradient[:, [1, 0, 2], :]
    structure_gradient[:, 1, :] *= -1

    # 3) change normal vector to unit length
    structure_normal = np.linalg.norm(structure_gradient, axis=1)
    unit_vector = np.divide(structure_gradient, structure_normal[:, np.newaxis, :])

    # 4) scale normal unit vectors at each pt to desired movement value
    change_magnitude_and_direction = unit_vector * mm_change

    # 5) move all points in desired direction in orginal structure
    new_structure = change_magnitude_and_direction + structure

    return new_structure


def move_dwells_randomly_2D_cpu(
    dwell_pts,
    change_array,
    dwell_pts_source_end_sup,
    dwell_pts_source_end_inf,
    mask_dwell_coords,
):
    # adding the change array [delta_x, delta_y, delta_z] per needle to the all orginal dwell
    # points. Also done for both ends of the source positioned at each dwell location.
    new_dwell_pts = (dwell_pts + change_array) * (np.invert(mask_dwell_coords)) + (
        -100
    ) * mask_dwell_coords

    dwell_pts_source_end_sup = (dwell_pts_source_end_sup + change_array) * (
        np.invert(mask_dwell_coords)
    ) + (-100) * mask_dwell_coords

    dwell_pts_source_end_inf = (dwell_pts_source_end_inf + change_array) * (
        np.invert(mask_dwell_coords)
    ) + (-100) * mask_dwell_coords

    return new_dwell_pts, dwell_pts_source_end_sup, dwell_pts_source_end_inf


def dwell_time_increase_decrease_cpu(dwell_times, dwell_time_increase_decrease):
    dwell_times = dwell_times + dwell_time_increase_decrease

    return dwell_times


__all__ = [
    "simulations_cpu",
    "move_along_channel_cpu",
    "move_dwells_randomly_2D_cpu",
    "enlarge_shrink_structure_cpu",
    "resample_structure_cpu",
    "rigid_move_cpu",
    "move_dwells_rigid_cpu",
    "enlarge_shrink_structure_per_slice_cpu",
]
