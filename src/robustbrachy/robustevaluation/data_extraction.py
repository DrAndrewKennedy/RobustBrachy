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
from pydicom import datadict, dcmread
import pandas as pd
from scipy.interpolate import interp1d, interp2d


#####################################
### INITIAL DICOM DATA EXTRACTION ###
#####################################


def extract_all_needed_data(
    file_rp,
    file_rs,
    file_rd,
    file_source_data,
):
    # *** extract data  ***

    # loading DICOM rt files
    rp = dcmread(file_rp)
    rs = dcmread(file_rs)
    rd = dcmread(file_rd)

    # Extract values from source data excel file
    F_interp, g_interp, G0, L, dose_rate_constant = get_source_data_array(
        file_source_data
    )

    # import all structures
    all_structures, structure_names = get_structures(rs)

    # Extract the Air Kerma Rate and prescribed dose from the treatment plan file
    air_kerma_rate = rp.SourceSequence[0].ReferenceAirKermaRate
    prescribed_dose = float((rp.DoseReferenceSequence[0][0x300A, 0x26].value))

    # Extracts the dwell points, dwell times, source end points per dwell location. The needle structures are extracted
    # as an interpolated array of points in 3D mapped to an accummulated length along the needle at the same index (BezierArcLengthMap_t_to_mm).
    (
        bezier_arc_length_points_TPS,
        bezier_arc_length_dists_TPS,
        dwell_coords_TPS,
        dwell_times_TPS,
        dwell_pts_source_end_sup_TPS,
        dwell_pts_source_end_inf_TPS,
        mask_dwell_coords_TPS,
        mask_dwell_times_TPS,
    ) = get_dwell_points(all_structures, L, rp, structure_names)

    # allocates dwell points. start = 5 mm from needle tip then 3.5 mm after that
    (
        dwell_coords_optimisation,
        dwell_pts_source_end_sup_optimisation,
        dwell_pts_source_end_inf_optimisation,
        bezier_arc_length_points_optimisation,
        bezier_arc_length_dists_optimisation,
        mask_dwell_coords_optimisation,
        mask_dwell_times_optimisation,
    ) = generate_dwell_coords_optimisation(all_structures, L, rp, structure_names)

    # structure volumes
    prostate_vol = rd.DVHSequence[0].DVHData[1]
    urethra_vol = rd.DVHSequence[1].DVHData[1]
    rectum_vol = rd.DVHSequence[2].DVHData[1]

    # creating dictionary of uncertainties
    plan_parameters = {
        "F_interp": F_interp,
        "g_interp": g_interp,
        "G0": G0,
        "L": L,
        "dose_rate_constant": dose_rate_constant,
        "air_kerma_rate": air_kerma_rate,
        "prescribed_dose": prescribed_dose,
        "bezier_arc_length_points_TPS": bezier_arc_length_points_TPS,
        "bezier_arc_length_dists_TPS": bezier_arc_length_dists_TPS,
        "dwell_coords_TPS": dwell_coords_TPS,
        "dwell_times_TPS": dwell_times_TPS,
        "dwell_pts_source_end_sup_TPS": dwell_pts_source_end_sup_TPS,
        "dwell_pts_source_end_inf_TPS": dwell_pts_source_end_inf_TPS,
        "mask_dwell_coords_TPS": mask_dwell_coords_TPS,
        "mask_dwell_times_TPS": mask_dwell_times_TPS,
        "dwell_coords_optimisation": dwell_coords_optimisation,
        "dwell_pts_source_end_sup_optimisation": dwell_pts_source_end_sup_optimisation,
        "dwell_pts_source_end_inf_optimisation": dwell_pts_source_end_inf_optimisation,
        "bezier_arc_length_points_optimisation": bezier_arc_length_points_optimisation,
        "bezier_arc_length_dists_optimisation": bezier_arc_length_dists_optimisation,
        "mask_dwell_coords_optimisation": mask_dwell_coords_optimisation,
        "mask_dwell_times_optimisation": mask_dwell_times_optimisation,
        "prostate_vol": prostate_vol,
        "urethra_vol": urethra_vol,
        "rectum_vol": rectum_vol,
        "prostate_contour_pts": all_structures[0],
        "urethra_contour_pts": all_structures[1],
        "rectum_contour_pts": all_structures[2],
        "all_structures": all_structures,
        "structure_names": structure_names,
    }
    return plan_parameters, rp, rs, rd


def generate_dwell_coords_optimisation(all_structures, L, rp, structure_names):
    # gets Bezier points for needles, allocate dwell points based on distance from tip

    # this length of Z is to be used for interpolating the beizer curve
    temp = []
    [temp.extend(x[0][2].tolist()) for x in all_structures[3:]]
    max_z = max(temp)
    min_z = min(temp)
    length_z = abs(round((max_z - min_z) * 100))

    # here we are extracting: 1) needle_coord: any points along each needle (from rs and rp files) for use in interpolation
    #                         2) dwell_coords: just the dwell coordinates from rp file
    #                         3) dwell_times: dwell times from rp file
    needle_coord = []
    struture_sup_coords = []
    struture_inf_coords = []
    j = 0
    while (
        (structure_names[j] == "Prostate")
        or (structure_names[j] == "Urethra")
        or (structure_names[j] == "Rectum")
        or (structure_names[j] == "PTV Boost")
        or (structure_names[j] == "PTV Boost2")
        or (structure_names[j] == "PTVboost2")
    ):
        j += 1
    for i, pt_set in enumerate(all_structures[j:]):
        temp_coords = []

        for pt in pt_set:
            temp_coords.extend(pt.T)

            struture_sup_coords.append(pt.T[0])
            struture_inf_coords.append(pt.T[-1])

        for dwell_pt in rp[0x300A, 0x230][0][0x300A, 0x280][i][0x300A, 0x2D0]:
            temp_coords.append(np.array(dwell_pt[0x300A, 0x2D4].value))

        temp_coords.sort(key=lambda x: x[2], reverse=True)
        needle_coord.append([str(structure_names[i + 3]), temp_coords])

    # obtaining an interpolated array for each needle (list of 3D points) and a corresponding length
    # along the needle using Bezier curves

    bezier_arc_length_points = []
    bezier_arc_length_dists = []
    bezier_control_points = []

    # does the interpolation using Bezier curves. Bezier curves are used by the TPS to the best of our knowledge.
    for i in needle_coord:
        P0, P1, P2, P3 = get_bezier_contorl_pts(i[1])
        [Q, cum_dist] = map_points_on_curve_to_mm(P0, P1, P2, P3, length_z)
        bezier_arc_length_points.append(Q)
        bezier_arc_length_dists.append(cum_dist)
        bezier_control_points.append([i[0], [P0, P1, P2, P3]])

    # find dist of first point and then every point 5 mm after it
    struture_sup_coords_large = (
        np.repeat(np.array(struture_sup_coords), len(bezier_arc_length_points[0]))
        .reshape(len(bezier_arc_length_points), 3, -1)
        .swapaxes(1, 2)
    )
    min_indices = np.linalg.norm(
        struture_sup_coords_large - np.array(bezier_arc_length_points), axis=2
    ).argmin(axis=1)
    sup_needle_end_length_from_BC_end = np.take_along_axis(
        np.array(bezier_arc_length_dists), min_indices.reshape(-1, 1), axis=1
    )
    dwell_dist = (
        np.repeat(np.arange(3.5, length_z / 100, 5), len(bezier_arc_length_points))
        .reshape(-1, len(bezier_arc_length_points))
        .T
    )

    dwell_dist_along_BC = dwell_dist + np.repeat(
        sup_needle_end_length_from_BC_end, len(dwell_dist[0])
    ).reshape(dwell_dist.shape)

    step_1 = np.repeat(
        bezier_arc_length_dists, len(dwell_dist_along_BC[0]), axis=0
    ).reshape(
        len(dwell_dist_along_BC),
        len(dwell_dist_along_BC[0]),
        len(bezier_arc_length_dists[0]),
    )
    step_2 = np.repeat(
        dwell_dist_along_BC, len(bezier_arc_length_dists[0]), axis=1
    ).reshape(
        len(dwell_dist_along_BC),
        len(dwell_dist_along_BC[0]),
        len(bezier_arc_length_dists[0]),
    )
    min_indices = (
        (np.absolute(step_1 - step_2))
        .argmin(axis=2)
        .reshape(len(dwell_dist_along_BC), len(dwell_dist_along_BC[0]), 1)
    )

    out_of_arr = (
        min_indices == (np.array(bezier_arc_length_points).shape[1] - 1)
    ).astype(np.bool_)
    out_of_arr = np.repeat(out_of_arr, 3, axis=2)
    dwell_coords = np.take_along_axis(
        np.array(bezier_arc_length_points), min_indices, axis=1
    )
    dwell_coords[out_of_arr] = -100

    pts_needle = []

    for i in needle_coord:
        pts_needle.extend(i[1])

    bezier_arc_length_dists = np.array(bezier_arc_length_dists, dtype=np.float32)
    bezier_arc_length_points = np.array(bezier_arc_length_points, dtype=np.float32)

    dwell_coords = np.array(dwell_coords, dtype=np.float32)

    # array mapping to find ends of each source at each dwell
    map_pts = bezier_arc_length_points
    map_dist = bezier_arc_length_dists
    step_1 = np.repeat(map_pts, len(dwell_coords[0]), axis=0).reshape(
        len(dwell_coords), len(dwell_coords[0]), len(map_pts[0]), 3
    )
    step_2 = np.repeat(dwell_coords, len(map_pts[0]), axis=1).reshape(
        len(dwell_coords), len(dwell_coords[0]), len(map_pts[0]), 3
    )
    min_indices = (
        (np.absolute(step_1 - step_2) * np.absolute(step_1 - step_2))
        .sum(axis=3)
        .argmin(axis=2)
        .reshape(len(dwell_coords), len(dwell_coords[0]), 1)
    )

    distance_along_curve = np.take_along_axis(
        map_dist.reshape(len(map_dist), len(map_dist[0]), 1), min_indices, axis=1
    )
    dist_along_curve_for_sup = distance_along_curve - L / 2 * 10
    dist_along_curve_for_inf = distance_along_curve + L / 2 * 10

    # Sup end of Dwells
    step_1 = np.repeat(map_dist, len(dist_along_curve_for_sup[0]), axis=0).reshape(
        len(dist_along_curve_for_sup),
        len(dist_along_curve_for_sup[0]),
        len(map_dist[0]),
    )
    step_2 = np.repeat(dist_along_curve_for_sup, len(map_dist[0]), axis=1).reshape(
        len(dist_along_curve_for_sup),
        len(dist_along_curve_for_sup[0]),
        len(map_dist[0]),
    )
    min_indices = (
        (np.absolute(step_1 - step_2))
        .argmin(axis=2)
        .reshape(len(dist_along_curve_for_sup), len(dist_along_curve_for_sup[0]), 1)
    )
    dwell_pts_source_end_sup = np.take_along_axis(map_pts, min_indices, axis=1)

    # Inf end of Dwells
    step_1 = np.repeat(map_dist, len(dist_along_curve_for_inf[0]), axis=0).reshape(
        len(dist_along_curve_for_inf),
        len(dist_along_curve_for_inf[0]),
        len(map_dist[0]),
    )
    step_2 = np.repeat(dist_along_curve_for_inf, len(map_dist[0]), axis=1).reshape(
        len(dist_along_curve_for_inf),
        len(dist_along_curve_for_inf[0]),
        len(map_dist[0]),
    )
    min_indices = (
        (np.absolute(step_1 - step_2))
        .argmin(axis=2)
        .reshape(len(dist_along_curve_for_inf), len(dist_along_curve_for_inf[0]), 1)
    )
    dwell_pts_source_end_inf = np.take_along_axis(map_pts, min_indices, axis=1)

    # need mask for both dwell times and points created during the initialisation of robust optimisation
    mask_dwell_coords = np.isin(np.array(dwell_coords), [-100]).astype(np.bool_)

    dwell_times = np.empty(
        shape=(
            dwell_coords.shape[0],
            dwell_coords.shape[1],
        ),
        dtype=np.bool,
    )
    dwell_times.fill(True)
    dwell_times[dwell_coords[:, :, 0] == -100] = False

    mask_dwell_times = np.isin(dwell_times, [-100]).astype(dtype=np.bool_)

    return (
        dwell_coords,
        dwell_pts_source_end_sup,
        dwell_pts_source_end_inf,
        bezier_arc_length_points,
        bezier_arc_length_dists,
        mask_dwell_coords,
        mask_dwell_times,
    )


def get_structures(rs):
    """Gets all the points and names of all structures from the rs file"""
    structs_np_fast = [
        fast_structure_coordinates(contour)
        for contour in rs.ROIContourSequence
        if hasattr(contour, "ContourSequence")
    ]
    StructureNames = []
    i = 0
    while i < len(structs_np_fast):
        StructureNames.append(rs[0x3006, 0x20][i][0x3006, 0x26].value)
        i += 1
    return structs_np_fast, StructureNames


# Extracting relvant information from the source data file
def get_source_data_array(uploaded_file_source_data):
    xls = pd.read_excel(uploaded_file_source_data)

    # source length, L
    L = xls.iloc[8, 2]

    # Source dose rate constant (in TG43)
    dose_rate_constant = xls.iloc[3, 2]

    Fr = np.array(xls.iloc[9, 5:23]).astype(np.float32)
    Ft = np.array(xls.iloc[10:49, 4]).astype(np.float32)
    Fi = np.array(xls.iloc[10:49, 5:23]).astype(np.float32)
    gi = np.array(xls.iloc[10:24, 1:3]).astype(np.float32)

    # the rest is a modification from pyTG43 and interpolates the function
    f = interp2d(Fr, Ft, Fi, kind="linear")
    g = interp1d(gi[:, 0], gi[:, 1])
    Fr_interp = np.linspace(0, 10.00, 1000)
    Ft_interp = np.linspace(0, 180.0, 1800)
    F_interp = f(Fr_interp, Ft_interp)
    g_interp = g(Fr_interp)

    # calculate the geometric function constant, in TG43
    G0 = (np.arctan2(-L / 2.0, 1.0) - np.arctan2(L / 2.0, 1.0)) / L

    return F_interp, g_interp, G0, L, dose_rate_constant


# load dwell points, needle arrays, and dwell times
def get_dwell_points(all_structures, L, rp, structure_names):
    # get total length of needles =. this length of Z is to be used for interpolating the beizer curve
    # the PTV boost regions will also be included in the max and min calc but won't impact on largest/smallest Z coordinate for needles.
    temp = []
    [temp.extend(x[0][2].tolist()) for x in all_structures[3:]]
    length_Z = abs(round((max(temp) - min(temp)) * 100))

    # here we are extracting: 1) needle_coord: any points along each needle (from rs and rp files) for use in interpolation
    #                         2) dwell_coords: just the dwell coordinates from rp file
    #                         3) dwell_times: dwell times from rp file
    needle_coords = []
    dwell_coords = []
    dwell_times = []

    j = 0
    while (
        (structure_names[j] == "Prostate")
        or (structure_names[j] == "Urethra")
        or (structure_names[j] == "Rectum")
        or (structure_names[j] == "PTV Boost")
        or (structure_names[j] == "PTV Boost2")
        or (structure_names[j] == "PTVboost2")
    ):
        j += 1
    for i, pt_set in enumerate(all_structures[j:]):
        temp_coords = []
        temp_dwells = []
        temp_dwell_times = []

        for pt in pt_set:
            temp_coords.extend(pt.T)

        channel_total_time = rp[0x300A, 0x230][0][0x300A, 0x280][i][0x300A, 0x286].value
        final_cumulative_time_weight = rp[0x300A, 0x230][0][0x300A, 0x280][i][
            0x300A, 0x2C8
        ].value

        for dwell_pt in rp[0x300A, 0x230][0][0x300A, 0x280][i][0x300A, 0x2D0]:
            temp_coords.append(np.array(dwell_pt[0x300A, 0x2D4].value))
            temp_dwells.append(np.array(dwell_pt[0x300A, 0x2D4].value))
            if float(final_cumulative_time_weight) == 0:
                dwell_time = 0

            else:
                dwell_time = (
                    dwell_pt[0x300A, 0x2D6].value / final_cumulative_time_weight
                ) * channel_total_time

            temp_dwell_times.append(dwell_time)

        temp_dwell_times = (
            np.array(temp_dwell_times) - np.roll(np.array(temp_dwell_times), 1)
        ).tolist()
        temp_coords.sort(key=lambda x: x[2], reverse=True)
        needle_coords.append([str(structure_names[i + 3]), temp_coords])
        dwell_coords.append(temp_dwells)
        dwell_times.append(temp_dwell_times)

    # obtaining an interpolated array for each needle (list of 3D points) and a corresponding length
    # along the needle using Bezier curves
    # BezierArcLengthMap_t_to_mm = []
    bezier_arc_length_points = []
    bezier_arc_length_dists = []
    bezier_control_points = []

    # Array has varing number of dwell points in the needle. numpy works best with constant dimensions.
    # code below mask all needle dwells a constant number by filling with float('NaN') values for extract
    # points to fill array to same dimensions as other needles. It than uses a mask array to ignore the NaN values.
    # NaN values do not do anything with in calcaluations.

    m = len(max(dwell_coords, key=len))
    dwell_coords = np.array(
        [
            v + [[float(-100), float(-100), float(-100)]] * (m - len(v))
            for v in dwell_coords
        ]
    )

    # dwellCoords = np.ma.array(dwellCoords, mask=np.isnan(dwellCoords))
    mask_dwell_coords = np.isin(dwell_coords, [-100])

    m = len(max(dwell_times, key=len))
    new_dwell_times = np.array([v + [float(-100)] * (m - len(v)) for v in dwell_times])
    mask_dwell_times = np.isin(new_dwell_times, [-100])

    # NewdwellTimes = np.ma.array(NewdwellTimes, mask=np.isnan(NewdwellTimes))

    # does the interpolation using Bezier curves. Bezier curves are used by the TPS to the best of our knowledge.
    for i in needle_coords:
        P0, P1, P2, P3 = get_bezier_contorl_pts(i[1])
        [Q, cum_dist] = map_points_on_curve_to_mm(P0, P1, P2, P3, length_Z)
        # BezierArcLengthMap_t_to_mm.append([Q,cumDist]) # name of needle(deleted), Points on curve (500), distance from distil end
        bezier_arc_length_points.append(Q)
        bezier_arc_length_dists.append(cum_dist)
        bezier_control_points.append([i[0], [P0, P1, P2, P3]])

    bezier_arc_length_dists = np.array(bezier_arc_length_dists, dtype=np.float32)
    bezier_arc_length_points = np.array(bezier_arc_length_points, dtype=np.float32)
    new_dwell_times = np.array(new_dwell_times, dtype=np.float32)
    dwell_coords = np.array(dwell_coords, dtype=np.float32)
    mask_dwell_times = np.array(mask_dwell_times, dtype=np.int32)
    mask_dwell_coords = np.array(mask_dwell_coords, dtype=np.int32)

    # array mapping to find ends of each source at each dwell
    map_pts = bezier_arc_length_points
    map_dist = bezier_arc_length_dists
    step_1 = np.repeat(map_pts, len(dwell_coords[0]), axis=0).reshape(
        len(dwell_coords), len(dwell_coords[0]), len(map_pts[0]), 3
    )
    step_2 = np.repeat(dwell_coords, len(map_pts[0]), axis=1).reshape(
        len(dwell_coords), len(dwell_coords[0]), len(map_pts[0]), 3
    )
    min_indices = (
        (np.absolute(step_1 - step_2) * np.absolute(step_1 - step_2))
        .sum(axis=3)
        .argmin(axis=2)
        .reshape(len(dwell_coords), len(dwell_coords[0]), 1)
    )

    dist_along_curve = np.take_along_axis(
        map_dist.reshape(len(map_dist), len(map_dist[0]), 1), min_indices, axis=1
    )
    dist_along_curve_for_sup = dist_along_curve - L / 2 * 10
    dist_along_curve_for_Inf = dist_along_curve + L / 2 * 10

    # Sup end of Dwells
    step_1 = np.repeat(map_dist, len(dist_along_curve_for_sup[0]), axis=0).reshape(
        len(dist_along_curve_for_sup),
        len(dist_along_curve_for_sup[0]),
        len(map_dist[0]),
    )
    step_2 = np.repeat(dist_along_curve_for_sup, len(map_dist[0]), axis=1).reshape(
        len(dist_along_curve_for_sup),
        len(dist_along_curve_for_sup[0]),
        len(map_dist[0]),
    )
    min_indices = (
        (np.absolute(step_1 - step_2))
        .argmin(axis=2)
        .reshape(len(dist_along_curve_for_sup), len(dist_along_curve_for_sup[0]), 1)
    )
    dwell_pts_source_end_sup = (
        np.take_along_axis(map_pts, min_indices, axis=1)
        * (np.invert(mask_dwell_coords))
        + (-100) * mask_dwell_coords
    )

    # Inf end of Dwells
    step_1 = np.repeat(map_dist, len(dist_along_curve_for_Inf[0]), axis=0).reshape(
        len(dist_along_curve_for_Inf),
        len(dist_along_curve_for_Inf[0]),
        len(map_dist[0]),
    )
    step_2 = np.repeat(dist_along_curve_for_Inf, len(map_dist[0]), axis=1).reshape(
        len(dist_along_curve_for_Inf),
        len(dist_along_curve_for_Inf[0]),
        len(map_dist[0]),
    )
    min_indices = (
        (np.absolute(step_1 - step_2))
        .argmin(axis=2)
        .reshape(len(dist_along_curve_for_Inf), len(dist_along_curve_for_Inf[0]), 1)
    )
    dwell_pts_source_end_inf = (
        np.take_along_axis(map_pts, min_indices, axis=1)
        * (np.invert(mask_dwell_coords))
        + (-100) * mask_dwell_coords
    )

    return (
        bezier_arc_length_points,
        bezier_arc_length_dists,
        dwell_coords,
        new_dwell_times,
        dwell_pts_source_end_sup,
        dwell_pts_source_end_inf,
        mask_dwell_coords,
        mask_dwell_times,
    )


# Bezier function for needle array interpolation
def get_bezier_contorl_pts(coords):
    # should already be order based on z coord
    # Set P0 = start point; P3 end point
    P0 = coords[0]
    P3 = coords[-1]
    # if length of coords < 4, display error message

    # initialise variables
    A1 = 0
    A2 = 0
    A12 = 0
    C1 = 0
    C2 = 0
    t, _ = chord_length_per_point(coords)

    for i in range(1, len(coords) - 1):
        B0 = (1 - float(t[i])) ** 3
        B1 = 3 * float(t[i]) * (1 - float(t[i])) ** 2
        B2 = 3 * float(t[i]) ** 2 * (1 - float(t[i]))
        B3 = float(t[i]) ** 3

        A1 = A1 + B1**2
        A2 = A2 + B2**2
        A12 = A12 + B1 * B2
        C1 = C1 + B1 * (coords[i] - B0 * P0 - B3 * P3)
        C2 = C2 + B2 * (coords[i] - B0 * P0 - B3 * P3)

    common_denominator = A1 * A2 - A12 * A12
    if common_denominator == 0:
        P1 = P0
        P2 = P3
    else:
        P1 = (A2 * C1 - A12 * C2) / common_denominator
        P2 = (A1 * C2 - A12 * C1) / common_denominator

    return P0, P1, P2, P3


# Bezier function for needle array interpolation
def chord_length_per_point(chords):
    total_distance = [0]
    cum_sum = 0
    for i in range(0, len(chords) - 1):
        cum_sum = cum_sum + np.sqrt(
            ((chords[i + 1][0] - chords[i][0]) ** 2)
            + ((chords[i + 1][1] - chords[i][1]) ** 2)
            + ((chords[i + 1][2] - chords[i][2]) ** 2)
        )
        total_distance.append(cum_sum)
    t = [0]
    cum_dist = [0]
    for i in range(1, len(chords) - 1):
        t.append(total_distance[i] / cum_sum)
        cum_dist.append(total_distance[i])
    t.append(1)
    cum_dist.append(total_distance[-1])

    return t, cum_dist


# Bezier function for needle array interpolation
def points_on_bezier_curve(P0, P1, P2, P3, length_Z):
    # Gets 100 points at equal intervals along the BezierCurve
    t = np.linspace(
        -0.2, 1.2, num=(length_Z + 1)
    )  # careful changing this, lower numbers have lower accruacy, check on dvhcoordinates when changign
    M = np.array(
        [[-1, 3, -3, 1], [3, -6, 3, 0], [-3, 3, 0, 0], [1, 0, 0, 0]]
    )  # coefficients
    P = np.array([P0, P1, P2, P3])
    Q = []
    for i in range(0, len(t)):
        Q.append(np.matmul(np.matmul(np.array([t[i] ** 3, t[i] ** 2, t[i], 1]), M), P))
    return t, Q


# Bezier function for needle array interpolation
def map_points_on_curve_to_mm(P0, P1, P2, P3, length_Z):
    t, Q = points_on_bezier_curve(P0, P1, P2, P3, length_Z)
    t2, cum_dist = chord_length_per_point(Q)
    max_value = max(cum_dist)
    bezier_t_to_mm = [Q, cum_dist]

    return bezier_t_to_mm


#####################################################################
###   contributor pydicom methods for fast obtaining structures   ###
###      https://github.com/pydicom/contrib-pydicom/issues/10     ###
#####################################################################


def fast_num_string(parent_dataset, child_tag):
    """Returns a numpy array for decimal string or integer string values.

    parent_dataset: a pydicom Dataset
    child_tag:      a tag or keyword for the numeric string lookup
    """

    if type(child_tag) is str:
        child_tag = datadict.tag_for_keyword(child_tag)
    val = parent_dataset.__getitem__(child_tag).value
    vr = datadict.dictionary_VR(child_tag)
    if vr == "IS":
        np_dtype = "i8"
    elif vr == "DS":
        np_dtype = "f8"
    else:
        raise ValueError("Must be IS or DS: {} is {}.".format(child_tag, vr))
    try:
        num_string = val.decode(encoding="utf-8")
        return np.fromstring(num_string, dtype=np_dtype, sep=chr(92))  # 92:'/'
    except AttributeError:  # 'MultiValue'  has no 'decode' (bytes does)
        # It's already been converted to doubles and cached
        return np.array(parent_dataset.__getitem__(child_tag).value, dtype=np_dtype)


def fast_structure_coordinates(contour):
    """Returns a list of numpy arrays. Each element in the list is a loop
    from the structure. Each loop is given as a numpy array where each column
    are the x, y, z coordinates of a point on the loop.

    contour: input an item from a structure set ROIContourSequence."""
    return [
        np.reshape(fast_num_string(loop, "ContourData"), (3, -1), order="F")
        for loop in contour.ContourSequence
    ]


__all__ = [
    "get_source_data_array",
    "get_structures",
    "get_dwell_points",
    "extract_all_needed_data",
]
