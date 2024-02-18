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

from robustbrachy.robustoptimisation.evaluation_cpu import *


def generate_dwells_for_pop_optimisation(
    time_limit_lin_fn_urethra,
    time_limit_lin_fn_rectum,
    dwell_times,
    all_dose_calc_pts,
    all_dose_per_dwell_per_vol_pt,
    plan_parameters,
    optimiser_parametres,
):
    # this method is a gamma function to randomly select dwell times with a deterministic method as an upper threshold based on distance from closest pt on urethra contour (time_threshold = 3 * dist_from_urethra - 3). This equation was obtained emperically.
    # additional deterministic method was accepting solutions (ordered sets of dwell times) based on a range around the D90 metric
    # and also, a 3rd deterministic algorithm ensures only dwell points close to the prostate are activated.

    # first want to word with an order set of dwell times and points. (1-D array)
    dwell_coords_flat = plan_parameters["dwell_coords_optimisation"][
        plan_parameters["dwell_coords_optimisation"] != -100
    ].reshape(-1, 3)
    dwell_times_flat = dwell_times[dwell_times != -100].flatten()

    # Create a mask for all dwell points not close to a prostate volume point
    prostate_volume_pts = all_dose_calc_pts[0]
    dist_between_pros_vol_pts_and_dwells = np.linalg.norm(
        prostate_volume_pts[:, np.newaxis] - dwell_coords_flat[np.newaxis, :], axis=2
    )
    closest_dist_to_prostate_vol_pts = np.amin(
        dist_between_pros_vol_pts_and_dwells, axis=0
    )
    mask_dwells_over_x_mm_from_prostate = (
        closest_dist_to_prostate_vol_pts
        > optimiser_parametres["limit_on_dwells_outside_prostate_mm"]
    ).astype(np.bool_)

    # select only dwell points that are the correct distance from the prostate or inside the prostate.
    active_dwell_coords = dwell_coords_flat[~mask_dwells_over_x_mm_from_prostate]

    # set an intital starting set of dwell times to deliver an even dose distribution
    vector_between_all_dwell_pts = (
        dwell_coords_flat.T[np.newaxis, :, :] - dwell_coords_flat[:, :, np.newaxis]
    ).swapaxes(1, 2)
    dist_between_all_dwells = np.linalg.norm(vector_between_all_dwell_pts, axis=2)
    recipical_of_dist_between_all_dwells = np.reciprocal(
        dist_between_all_dwells, where=(dist_between_all_dwells != 0)
    )
    dwell_time_weights_even_dist = np.reciprocal(
        np.einsum(
            "ij,ij->j",
            recipical_of_dist_between_all_dwells,
            recipical_of_dist_between_all_dwells,
        )
    )
    dwell_time_weights_even_dist[mask_dwells_over_x_mm_from_prostate] = 0.0

    # this part finds an upper and lower limit for 'total dwell time' that would result in approximately the range [prescribed dose - 1 Gy, prescribed dose + 5 Gy].
    # total dwell time is approximately equal to total dose, so a range of total times equals a range of total dose and a certain total dose is needed to deliver at least the prescribed dose.
    dwell_times_flat_2 = np.copy(dwell_time_weights_even_dist)

    # this gives a 2D array of ordered sets of dwell times for each row (so a large population of treatment plans at each row)
    dwell_times_mulitplied = np.einsum(
        "i,j-> ij", np.arange(1, 10, 0.05), dwell_times_flat_2
    )

    # gets the total dwell times for each ordered set of dwell points (each treatment plan)
    total_dwell_times = dwell_times_mulitplied.sum(1)

    # function to get DVHs for each row of ordered set of dwell points (each treatment plan) in a vectorised way
    all_DVHs_pop = get_dvh_for_population_optimisation_cpu(
        dwell_times_mulitplied,
        all_dose_per_dwell_per_vol_pt,
        plan_parameters["prescribed_dose"],
    )

    # now get all metrics for each ordered set of dwell times
    D90_pop, _, _ = get_metrics_for_population_optimisation_cpu(
        all_DVHs_pop, plan_parameters["urethra_vol"], plan_parameters["rectum_vol"]
    )

    # transform the values so > 0 is higher than the lower target and < 0 is below lower target
    # this is so we can see which row a sign change occurs and then relate this to the corresponding row in the 'total_dwell_times'
    diff_from_D90_lower = plan_parameters["prescribed_dose"] - 1 - D90_pop
    diff_from_D90_lower[
        diff_from_D90_lower == 0
    ] = 0.01  # next step doesn't work if it is exactly zero
    sign_change_index_lower = np.where(
        diff_from_D90_lower[:-1] * diff_from_D90_lower[1:] < 0
    )[0]

    # same as above but for higher target
    diff_from_D90_upper = plan_parameters["prescribed_dose"] + 5 - D90_pop
    diff_from_D90_upper[diff_from_D90_upper == 0] = 0.01
    sign_change_index_upper = np.where(
        diff_from_D90_upper[:-1] * diff_from_D90_upper[1:] < 0
    )[0]

    # method above would fail on occusion for the lower sign change. Repeating the porcess once more fixed the issue. What causes the issue is unknown. itr upto 10 times just in case.
    itr = 0
    if len((total_dwell_times[sign_change_index_lower])) == 0:
        while len((total_dwell_times[sign_change_index_lower])) == 0 and itr < 10:
            dwell_coords_flat = plan_parameters["dwell_coords_optimisation"][
                plan_parameters["dwell_coords_optimisation"] != -100
            ].reshape(-1, 3)
            dwell_times_flat = dwell_times[dwell_times != -100].flatten()

            prostate_volume_pts = all_dose_calc_pts[0]

            dist_between_pros_vol_pts_and_dwells = np.linalg.norm(
                prostate_volume_pts[:, np.newaxis] - dwell_coords_flat[np.newaxis, :],
                axis=2,
            )

            closest_dist_to_prostate_vol_pts = np.amin(
                dist_between_pros_vol_pts_and_dwells, axis=0
            )

            mask_dwells_over_x_mm_from_prostate = (
                closest_dist_to_prostate_vol_pts
                > optimiser_parametres["limit_on_dwells_outside_prostate_mm"]
            ).astype(np.bool_)

            active_dwell_coords = dwell_coords_flat[
                ~mask_dwells_over_x_mm_from_prostate
            ]

            diff = (
                dwell_coords_flat.T[np.newaxis, :, :]
                - dwell_coords_flat[:, :, np.newaxis]
            ).swapaxes(1, 2)
            dist_between_all_dwells = np.linalg.norm(diff, axis=2)
            recipical_of_dist_between_all_dwells = np.reciprocal(
                dist_between_all_dwells, where=(dist_between_all_dwells != 0)
            )

            dwell_time_weights_even_dist = np.reciprocal(
                np.einsum(
                    "ij,ij->j",
                    recipical_of_dist_between_all_dwells,
                    recipical_of_dist_between_all_dwells,
                )
            )
            dwell_time_weights_even_dist[mask_dwells_over_x_mm_from_prostate] = 0.0

            dwell_times_flat_2 = np.copy(dwell_time_weights_even_dist)

            dwell_times_mulitplied = np.einsum(
                "i,j-> ij", np.arange(1, 10, 0.05), dwell_times_flat_2
            )
            total_dwell_times = dwell_times_mulitplied.sum(1)

            all_DVHs_pop = get_dvh_for_population_optimisation_cpu(
                dwell_times_mulitplied,
                all_dose_per_dwell_per_vol_pt,
                plan_parameters["prescribed_dose"],
            )

            D90_pop, DmaxU_pop, DmaxR_pop = get_metrics_for_population_optimisation_cpu(
                all_DVHs_pop,
                plan_parameters["urethra_vol"],
                plan_parameters["rectum_vol"],
            )

            diff_from_D90_lower = plan_parameters["prescribed_dose"] - 1 - D90_pop
            diff_from_D90_lower[diff_from_D90_lower == 0] = 0.01
            sign_change_index_lower = np.where(
                diff_from_D90_lower[:-1] * diff_from_D90_lower[1:] < 0
            )[0]

            diff_from_D90_upper = plan_parameters["prescribed_dose"] + 5 - D90_pop
            diff_from_D90_upper[diff_from_D90_upper == 0] = 0.01
            sign_change_index_upper = np.where(
                diff_from_D90_upper[:-1] * diff_from_D90_upper[1:] < 0
            )[0]

            itr += 1

    # get the corresponding total dwell times
    lower_bound_total_time = (total_dwell_times[sign_change_index_lower])[0]
    upper_bound_total_time = (total_dwell_times[sign_change_index_upper + 1])[0]

    # this part is to obtain a relative maximum dwell time per dwell to ensure urethra is cool
    # min distance between the urethra contour and dwell points
    all_pts_list_urethra = np.swapaxes(
        plan_parameters["urethra_contour_pts"], 1, 2
    ).reshape(-1, 3)
    dist_to_urethra_contour = np.linalg.norm(
        all_pts_list_urethra[:, np.newaxis, :] - dwell_coords_flat[np.newaxis, :, :],
        axis=2,
    )
    closest_dist_to_urethra_contour = np.amin(dist_to_urethra_contour, axis=0)

    # this part is to obtain a relative maximum dwell time per dwell to ensure rectum is cool
    # min distance between the urethra contour and dwell points

    all_pts_list_rectum = np.swapaxes(
        plan_parameters["rectum_contour_pts"], 1, 2
    ).reshape(-1, 3)
    dist_to_rectum_contour = np.linalg.norm(
        all_pts_list_rectum[:, np.newaxis, :] - dwell_coords_flat[np.newaxis, :, :],
        axis=2,
    )
    closest_dist_to_rectum_contour = np.amin(dist_to_rectum_contour, axis=0)

    # a set of dwell times that are very large per structure. this will be changed and lowered to reach the right time_limit_lin_fn_ limits for each structure
    # The lowest dwell times well then be accepted.
    upper_time_limit_urethra = np.ones(len(dwell_times_flat)) * 1000
    upper_time_limit_rectum = np.ones(len(dwell_times_flat)) * 1000
    upper_time_limit_prostate = np.ones(len(dwell_times_flat)) * 1000
    upper_time_limit_remaining = np.ones(len(dwell_times_flat)) * 1000

    # if the time_limit_lin_fn_rectum is all zeros, than don't consider limitating dwell times based on making sure rectum is cool.
    if not np.count_nonzero(time_limit_lin_fn_rectum):
        # get all dwell points which are closer to the urethra than a set distance (11 mm in the original research in which this was created)
        mask_dist_below_max_dist_urethra = (
            closest_dist_to_urethra_contour < time_limit_lin_fn_urethra[2]
        )

        # mask of all dwells greatter than 11 mm (opposite to above)
        mask_of_all_other = (
            ~mask_dist_below_max_dist_urethra & ~mask_dwells_over_x_mm_from_prostate
        )

        # applies the sloped linear function to all dwells < 11 mm
        #           slope             *  (                             x                                     -            uncertainty       ) +        intercept
        upper_time_limit_urethra[mask_dist_below_max_dist_urethra] = 1 * (
            time_limit_lin_fn_urethra[0]
            * (
                closest_dist_to_urethra_contour[mask_dist_below_max_dist_urethra]
                - time_limit_lin_fn_urethra[3]
            )
            + time_limit_lin_fn_urethra[1]
        )  # upper_time_limit_urethra[mask_dist_below_max_dist_urethra] * (time_limit_lin_fn_urethra[0] *  closest_dist_to_urethra_contour[mask_dist_below_max_dist_urethra] + time_limit_lin_fn_urethra[1])

        # set all times out of the prostate to zero
        upper_time_limit_prostate[mask_dwells_over_x_mm_from_prostate] = 0.0

        # set all other dwell times to the maximum time limit
        upper_time_limit_remaining[mask_of_all_other] = optimiser_parametres[
            "max_time_limit"
        ]

        # takes the smallest dwell times between the prostate limits and urethra limits
        upper_time_limit = (
            np.amin(
                np.array(
                    [
                        upper_time_limit_urethra,
                        upper_time_limit_prostate,
                        upper_time_limit_remaining,
                    ]
                ),
                axis=0,
            )
        ).round(1)

    else:
        # as above, but also includes limits for dwells close to the rectum
        mask_dist_below_max_dist_urethra = (
            closest_dist_to_urethra_contour < time_limit_lin_fn_urethra[2]
        )
        mask_dist_below_max_dist_rectum = (
            closest_dist_to_rectum_contour < time_limit_lin_fn_rectum[2]
        )
        mask_of_all_other = (
            ~mask_dist_below_max_dist_urethra
            & ~mask_dist_below_max_dist_rectum
            & ~mask_dwells_over_x_mm_from_prostate
        )
        #           slope             *  (                             x                                     -            uncertainty       ) +        intercept
        upper_time_limit_urethra[mask_dist_below_max_dist_urethra] = 1 * (
            time_limit_lin_fn_urethra[0]
            * (
                closest_dist_to_urethra_contour[mask_dist_below_max_dist_urethra]
                - time_limit_lin_fn_urethra[3]
            )
            + time_limit_lin_fn_urethra[1]
        )  # upper_time_limit_urethra[mask_dist_below_max_dist_urethra] * (time_limit_lin_fn_urethra[0] *  closest_dist_to_urethra_contour[mask_dist_below_max_dist_urethra] + time_limit_lin_fn_urethra[1])
        upper_time_limit_rectum[mask_dist_below_max_dist_rectum] = 1 * (
            time_limit_lin_fn_rectum[0]
            * (
                closest_dist_to_rectum_contour[mask_dist_below_max_dist_rectum]
                - time_limit_lin_fn_rectum[3]
            )
            + time_limit_lin_fn_rectum[1]
        )  # upper_time_limit_rectum[mask_dist_below_max_dist_rectum] * (time_limit_lin_fn_rectum[0] *  closest_dist_to_rectum_contour[mask_dist_below_max_dist_rectum] + time_limit_lin_fn_rectum[1])
        upper_time_limit_prostate[mask_dwells_over_x_mm_from_prostate] = 0.0
        upper_time_limit_remaining[mask_of_all_other] = optimiser_parametres[
            "max_time_limit"
        ]
        upper_time_limit = (
            np.amin(
                np.array(
                    [
                        upper_time_limit_urethra,
                        upper_time_limit_rectum,
                        upper_time_limit_prostate,
                        upper_time_limit_remaining,
                    ]
                ),
                axis=0,
            )
        ).round(1)

    # with the above deterministic processes complete, the random part of the dwell time population initialisation is completed

    # a list to save all dwell times
    pop_dwell_times_within_range = []

    # a gamma function has been modelled to randomly generate more lower dwell times than larger ones. the shape and scale values are choosen based on this.
    # https://numpy.org/doc/stable/reference/random/generated/numpy.random.gamma.html
    shape = 1.7
    scale = 3
    dont_have_Enough = True
    loops = 0
    while dont_have_Enough == True and loops < 100:
        # # generate population of random numbers using gamma function
        pop_dwell_times = np.random.default_rng().gamma(
            shape, scale, size=(optimiser_parametres["pop_size"], len(dwell_times_flat))
        )

        # # rescale to range from 0 to 1
        pop_dwell_times = pop_dwell_times / np.max(pop_dwell_times)

        # use upper time limit to scale dwell times from the values between 0 and 1 to 0 and upper limit for each dwell point
        pop_dwell_times = np.einsum("ij,j -> ij", pop_dwell_times, upper_time_limit)

        # minimum dwell time of 0.4 seconds. change any below 0.4 s to 0 s.
        pop_dwell_times[pop_dwell_times < 0.4] = 0.0

        # to add randomness, random dwell times are set to zero.
        # random uniform numbers between 0 and 1 for each dwell time in the population are created
        random_turn_off_dwell_times = np.random.uniform(
            size=optimiser_parametres["pop_size"] * len(dwell_times_flat)
        ).reshape(optimiser_parametres["pop_size"], len(dwell_times_flat))
        # keep all dwell times that got a random number below 0.7
        random_turn_off_dwell_times[random_turn_off_dwell_times < 0.7] = 1
        # set all dwell times that got a random number higher than 0.7 to 0 (opposite to above hence the ~)
        random_turn_off_dwell_times[~(random_turn_off_dwell_times == 1.0)] = 0
        # setting those dwell times to zero
        pop_dwell_times = np.einsum(
            "ij,ij -> ij", pop_dwell_times, random_turn_off_dwell_times
        )

        # finally round all numbers to 1 decimal point
        pop_dwell_times = pop_dwell_times.round(1)

        # collect dwell times within range
        # calculate total dwell time in each row
        total_dwell_times = pop_dwell_times.sum(1)
        # gets the index of all rows between the total dwell time range from above.
        index_dwells_within_range = (
            (total_dwell_times > lower_bound_total_time)
            & (total_dwell_times < upper_bound_total_time)
        ).astype(np.bool_)

        # add the generated treatment plans (rows) to the dwells that will make it to the optimiser
        pop_dwell_times_within_range.extend(pop_dwell_times[index_dwells_within_range])

        # want a large population to start with so the optimiser has a lot to choose from (increasing reproducibility)
        if len(pop_dwell_times_within_range) > optimiser_parametres["pop_size"] * 5:
            dont_have_Enough = False

        # incase a low number of solutions where within the range, increase the gamma function shape.
        # increase shape parameter = higher dwell times have a higher probability of being generated in the gamma function
        # decrease shape parameter has the opposite effect

        if np.mean(total_dwell_times) < lower_bound_total_time:
            shape += 0.2
        else:
            shape -= 0.2

        if shape < 0.01:
            shape = 0.01
        loops += 1

    pop_dwell_times_within_range = np.array(pop_dwell_times_within_range)

    print("Final pop size is " + str(len(pop_dwell_times_within_range)))

    return (
        pop_dwell_times_within_range,
        mask_dwells_over_x_mm_from_prostate,
        upper_time_limit,
    )


__all__ = ["generate_dwells_for_pop_optimisation"]
