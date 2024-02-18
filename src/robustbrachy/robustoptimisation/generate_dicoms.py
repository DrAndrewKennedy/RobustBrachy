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
from pydicom.sequence import Sequence
from pydicom.dataset import Dataset
from copy import deepcopy

from robustbrachy.robustoptimisation.dose_per_dwell_cpu import *
from robustbrachy.robustevaluation.fast_TG43_cpu import *


def dose_from_1_dwell_at_1_pt(
    dwell_pt,
    field_pt,
    dwell_time,
    plan_parameters
):
    field_pt = np.array([field_pt]) / 10
    dwell_pt = np.array([dwell_pt]) / 10
    dwell_pts_source_end_sup_optimisation = (
        np.array([plan_parameters["dwell_pts_source_end_sup_optimisation"]]) / 10
    )
    dwell_pts_source_end_inf_optimisation = (
        np.array([plan_parameters["dwell_pts_source_end_inf_optimisation"]]) / 10
    )

    r = np.sqrt(np.sum(np.subtract(field_pt, dwell_pt) ** 2))

    stage_1 = np.multiply(
        (dwell_pts_source_end_sup_optimisation - dwell_pts_source_end_inf_optimisation),
        (field_pt - dwell_pt),
    )

    stage_2 = np.sum(stage_1)

    theta_dot_product = np.divide(stage_2, r * plan_parameters["L"])

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

    Stage_3a = r * np.cos(theta) - plan_parameters["L"] / 2.0
    Stage_3b = r * np.sin(theta)
    Stage_4 = np.arctan2(Stage_3a, Stage_3b)

    Stage_5a = r * np.cos(theta) + plan_parameters["L"] / 2.0
    Stage_5b = r * np.sin(theta)
    Stage_6 = np.arctan2(Stage_5a, Stage_5b)

    beta = Stage_4 - Stage_6

    G_small_angle = 1.0 / (r * r - (plan_parameters["L"] * plan_parameters["L"]) / 4.0)
    G_large_angle = beta / (plan_parameters["L"] * r * np.sin(theta))

    small_angle_mask = ((np.pi - theta) < 0.003).astype(np.int32) + (
        (theta) < 0.003
    ).astype(np.int32)
    other_angle_mask = (~np.array(small_angle_mask.astype(np.bool_))).astype(np.int32)

    G = np.multiply(G_small_angle, small_angle_mask) + np.multiply(
        G_large_angle, other_angle_mask
    )

    if G > 0:
        G = G * -1

    r_high = plan_parameters["F_interp"][
        (np.rint(np.degrees(theta) * 10) - 1).astype(np.int32),
        (np.rint(10 * 100) - 1).astype(np.int32),
    ]

    r_low = plan_parameters["F_interp"][
        (np.rint(np.degrees(theta) * 10) - 1).astype(np.int32),
        (np.rint(r * 100) - 1).astype(np.int32),
    ]

    f_dose = np.select([r > 10, r <= 10], [r_high, r_low])

    r_high = plan_parameters["g_interp"][
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

    r_low = plan_parameters["g_interp"][(np.rint(0.15 * 100) - 1).astype(np.int32)]

    r_middle = plan_parameters["g_interp"][
        np.array((np.rint(r * 100) - 1).astype(np.int32))
    ]

    g_dose = np.select(
        [r > 10, r < 0.15, (r > 0.15) ^ (r > 10)], [r_high, r_low, r_middle]
    )  # ^ means exclude the (r > 10) values. so all r greater than 0.15 and less than 10

    dose_per_dwell = (
        plan_parameters["air_kerma_rate"]
        * plan_parameters["dose_rate_constant"]
        * (G / plan_parameters["G0"])
        * g_dose
        * f_dose
        * np.array([dwell_time])
        / 360000
    )

    return dose_per_dwell, (dose_per_dwell / plan_parameters["prescribed_dose"])


def generate_DICOM_plan_files(
    rp,
    rs,
    rd,
    plan_parameters,
    dwell_times_nominal_robust,
):
    # this is for the original structure of the dwell coordinates
    dwell_structure_array = (
        plan_parameters["dwell_coords_optimisation"][:, :, 0] != -100
    ).astype(np.bool_)

    # update dicom file using own coordinates 3.5 mm from tip then 5 mm after that

    all_oreintations_own_dwells = np.divide(
        (
            plan_parameters["dwell_pts_source_end_sup_optimisation"]
            - plan_parameters["dwell_pts_source_end_inf_optimisation"]
        ),
        np.array(
            np.linalg.norm(
                (
                    plan_parameters["dwell_pts_source_end_sup_optimisation"]
                    - plan_parameters["dwell_pts_source_end_inf_optimisation"]
                ),
                axis=2,
            )
        )[:, :, np.newaxis],
    )

    # collect the dose reference points from the TPs plan
    ref_pts = []
    for ref in rp[0x300A, 0x010]:
        if ref[0x300A, 0x014].value == "COORDINATES":
            ref_pts.append(ref[0x300A, 0x018].value)

    ref_pts = np.array(ref_pts)

    # put dwell times within the correct shape
    dwell_times = np.empty(
        shape=(dwell_structure_array.shape[0], dwell_structure_array.shape[1]),
        dtype=np.float32,
    )
    count = 0
    for k, val in enumerate(dwell_structure_array):
        for d, val2 in enumerate(val):
            if val2 == True:
                dwell_times[k, d] = dwell_times_nominal_robust[count]
                count += 1
            else:
                dwell_times[k, d] = -100

    dwell_times = np.array(dwell_times)

    # update dwell coordinates
    rp_robust = deepcopy(rp)

    # changing the TRAK 'total reference air kerma' which is the product total dwell time and reference air kerma
    rp_robust[0x300A, 0x230][0][0x300A, 0x250].value = round(
        np.sum(dwell_times_nominal_robust[dwell_times_nominal_robust > 0])
        / (60 * 60)
        * plan_parameters["air_kerma_rate"],
        2,
    )

    # find the index of the first needle
    j = 0
    while (
        (plan_parameters["structure_names"][j] == "Prostate")
        or (plan_parameters["structure_names"][j] == "Urethra")
        or (plan_parameters["structure_names"][j] == "Rectum")
        or (plan_parameters["structure_names"][j] == "PTV Boost")
        or (plan_parameters["structure_names"][j] == "PTV Boost2")
        or (plan_parameters["structure_names"][j] == "PTVboost2")
    ):
        j += 1

    # get needle tips from structure files
    needle_control_pts = []
    for needles in plan_parameters["all_structures"][j:]:
        needle_control_pts.append(list(needles[0][:, 0]))
    needle_tips = np.array(needle_control_pts)

    dist_from_tip_all_needles = []
    for n, needle in enumerate(plan_parameters["dwell_coords_optimisation"]):
        needle_tip = needle_tips[n]

        dist_of_tip = plan_parameters["bezier_arc_length_dists_optimisation"][n][
            np.linalg.norm(
                needle_tip[None, :]
                - plan_parameters["bezier_arc_length_points_optimisation"][n],
                axis=1,
            ).argmin()
        ]

        needle = np.array(needle[needle[:, 0] != -100])

        dist_from_tip = (
            plan_parameters["bezier_arc_length_dists_optimisation"][n][
                np.linalg.norm(
                    needle[:, None, :]
                    - plan_parameters["bezier_arc_length_points_optimisation"][n][
                        None, :, :
                    ],
                    axis=2,
                ).argmin(axis=1)
            ]
        ) - dist_of_tip
        dist_from_tip_all_needles.append(dist_from_tip.round(1))

    new_FinalCumulativeTimeWeight = 0

    # goes through each needle in turn
    for k, pt_set in enumerate(plan_parameters["all_structures"][j:]):
        # get dwell coordinates and times along the needle
        new_needle_dwell_times = dwell_times[k][dwell_times[k] != -100]
        new_needle_dwell_coords = plan_parameters["dwell_coords_optimisation"][k][
            plan_parameters["dwell_coords_optimisation"][k] != -100
        ].reshape(-1, 3)
        needle_dist_from_tip = dist_from_tip_all_needles[k]
        needle_orientations = all_oreintations_own_dwells[k]

        # cut start and end of array with dwell times = 0, dwells in the middle = 0 stay in array
        if np.sum(new_needle_dwell_times) == 0.0:
            new_needle_dwell_times = new_needle_dwell_times[:2]
            new_needle_dwell_coords = new_needle_dwell_coords[:2]
            needle_dist_from_tip = needle_dist_from_tip[:2]
            needle_orientations = needle_orientations[:2]
        else:
            idx = np.where(new_needle_dwell_times != 0)[0]
            new_needle_dwell_times = new_needle_dwell_times[idx[0] : 1 + idx[-1]]
            new_needle_dwell_coords = new_needle_dwell_coords[idx[0] : 1 + idx[-1]]
            needle_dist_from_tip = needle_dist_from_tip[idx[0] : 1 + idx[-1]]
            needle_orientations = needle_orientations[idx[0] : 1 + idx[-1]]

        # in dicom at each dwell point there are two bracy control point (1) a stop with zero time, (2) a wait for dwell time
        # so this doubles the dwell time array and allocates dwell times to every second array
        needle_dwells_arr = np.zeros(2 * len(new_needle_dwell_times), dtype=np.float32)
        needle_dwells_arr[1::2] = new_needle_dwell_times

        needle_dist_from_tip = np.repeat(needle_dist_from_tip, 2)
        needle_orientations = np.repeat(needle_orientations, 2, axis=0)
        new_needle_dwell_coords = np.repeat(new_needle_dwell_coords, 2, axis=0)

        # calculate total time in needle
        new_channel_total_time = round(np.sum(new_needle_dwell_times), 1)

        # caluculate the brachy reference doses
        # dose reference 1 is the prostate "Target". Here is how the cummulative dose ref coeff is worked out for a "SITE" type.
        # 1) fraction of 1 needed for each needle (1/needle_number):
        total_cum_ref_target_per_needle = 1 / len(
            plan_parameters["dwell_coords_optimisation"]
        )

        # 2) find ratio of dwell weighting per dwell time
        needle_weight_per_dwell = needle_dwells_arr / new_channel_total_time

        # 3) use ratio to calculate contribution per dwell time
        needle_weight_per_dwell = (
            needle_weight_per_dwell * total_cum_ref_target_per_needle
        )

        # finally calculate the cummulative amount per dwell
        dose_ref_target_per_dwell = np.cumsum(needle_weight_per_dwell)

        # get the dose rate at each dose reference point
        dose_ref_per_dwell = []
        for dpt, dtime in zip(new_needle_dwell_coords, needle_dwells_arr):
            if dtime == 0:
                dose_ref_per_dwell.append(np.zeros(len(ref_pts)))
            else:
                temp = []
                for r in ref_pts:
                    dose, dose_ratio = dose_from_1_dwell_at_1_pt(
                        dpt,
                        r,
                        dtime,
                        plan_parameters
                    )
                    temp.append(dose_ratio[0])
                dose_ref_per_dwell.append(temp)

        dose_ref_per_dwell = np.array(dose_ref_per_dwell).T
        dose_ref_per_dwell = np.cumsum(dose_ref_per_dwell, axis=1).T

        ####### DICOM CHANGES ######
        ## whole channel/needle changeds

        # 1) change "number of control points"
        number_of_control_points = int(len(needle_dwells_arr))
        rp_robust[0x300A, 0x230][0][0x300A, 0x280][k][0x300A, 0x110].value = str(
            number_of_control_points
        )

        # 2) change "channel total time"
        new_FinalCumulativeTimeWeight = new_channel_total_time
        rp_robust[0x300A, 0x230][0][0x300A, 0x280][k][0x300A, 0x286].value = str(
            new_channel_total_time
        )

        # 3) change "Final cumulative time weight"
        rp_robust[0x300A, 0x230][0][0x300A, 0x280][k][0x300A, 0x2C8].value = str(
            new_FinalCumulativeTimeWeight
        )

        # create new squence of brachy control points
        new_sequence = []
        for d in range(number_of_control_points):
            new_ds = Dataset()
            new_ds.ControlPointIndex = str(d)
            new_ds.ControlPointRelativePosition = str(needle_dist_from_tip[d])
            new_ds.ControlPoint3DPosition = list(new_needle_dwell_coords[d])
            new_ds.CumulativeTimeWeight = str(needle_dwells_arr[d])
            new_ds.ControlPointOrientation = list(needle_orientations[d])

            # new sequence of dose reference points
            new_brachy_ref_pts = []
            for ref in range(len(ref_pts) + 1):
                if ref == 0:
                    new_ds_2 = Dataset()
                    new_ds_2.CumulativeDoseReferenceCoefficient = str(
                        dose_ref_target_per_dwell[d]
                    )
                    new_ds_2.ReferencedDoseReferenceNumber = str(ref + 1)

                else:
                    new_ds_2 = Dataset()

                    new_ds_2.CumulativeDoseReferenceCoefficient = str(
                        dose_ref_per_dwell[d][ref - 1]
                    )
                    new_ds_2.ReferencedDoseReferenceNumber = str(ref + 1)

                new_brachy_ref_pts.append(new_ds_2)

            new_ds.BrachyReferencedDoseReferenceSequence = Sequence(new_brachy_ref_pts)

            new_sequence.append(new_ds)

        # save the channel sequence to rp file
        rp_robust[0x300A, 0x230][0][0x300A, 0x280][k][0x300A, 0x2D0].value = Sequence(
            new_sequence
        )

    ##########################
    # now for rd file
    ########################
    rd_robust = deepcopy(rd)

    # get points to calculate dose. these are the pixel array coordinates in the rd file
    XCols = rd.Columns
    YRows = rd.Rows
    Xspacing = rd.PixelSpacing[0]
    Yspacing = rd.PixelSpacing[1]
    ZOffSetvector = np.array(rd.GridFrameOffsetVector)

    Origin = rd.ImagePositionPatient
    Xvector = Origin[0] + np.linspace(0, Xspacing * XCols - 1, XCols)
    Yvector = Origin[1] + np.linspace(0, Yspacing * YRows - 1, YRows)
    Zvector_FULLrange = Origin[2] + ZOffSetvector

    XYZgrid = (
        np.vstack(np.meshgrid(Xvector, Yvector, Zvector_FULLrange)).reshape(3, -1).T
    )
    # PixelArray = rd.pixel_array
    DoseScale = rd.DoseGridScaling
    # DoseGrid_3D_TPS = np.array(DoseScale * PixelArray)

    XYZgrid = XYZgrid[-XYZgrid[:, -1].argsort()]

    new_ordered_XYZ = []
    XYZ_per_slice = XYZgrid.reshape(len(Zvector_FULLrange), -1, 3)

    DoseGrid_3D_Optimised = []
    for sli in XYZ_per_slice:
        temp = []
        for s in sli[sli[:, 0].argsort()].reshape(len(Xvector), len(Yvector), 3):
            temp.append(s[s[:, 1].argsort()])

        new_ordered_XYZ.append(temp)

        dose_per_dwell_per_vol_pt = TG43calc_dose_per_dwell_per_vol_pt_cpu(
            np.array(np.array(temp).reshape(-1, 3)),
            dwell_times,
            plan_parameters,
            plan_parameters["dwell_coords_optimisation"],
            plan_parameters["dwell_pts_source_end_sup_optimisation"],
            plan_parameters["dwell_pts_source_end_inf_optimisation"],
        )

        dose_gpu_per_vol_pt = np.sum(dose_per_dwell_per_vol_pt, axis=0)

        DoseGrid_3D_Optimised.append(
            dose_gpu_per_vol_pt.reshape(len(Xvector), len(Yvector))
        )

    DoseGrid_3D_Optimised = np.array(DoseGrid_3D_Optimised)
    new_ordered_XYZ = np.array(new_ordered_XYZ)

    DoseGrid_3D_Optimised = np.swapaxes(DoseGrid_3D_Optimised, 2, 1)
    PixelArray_optimised = np.array(
        np.round(DoseGrid_3D_Optimised / DoseScale, 0), dtype=np.int32
    )
    PixelArray_optimised = PixelArray_optimised.tobytes()

    # update dose array in RT dose file
    rd_robust.PixelData = (
        PixelArray_optimised + b"\x00"
        if len(PixelArray_optimised) % 2
        else PixelArray_optimised
    )

    # update DVH code

    # make a change structure set with correct dwell pts, dwell pt ends, dwell times
    changed_structures = {
        "changed_prostate_contour_pts": np.array(
            plan_parameters["prostate_contour_pts"]
        ).astype(np.float32),
        "changed_urethra_contour_pts": np.array(
            plan_parameters["urethra_contour_pts"]
        ).astype(np.float32),
        "changed_rectum_contour_pts": np.array(
            plan_parameters["rectum_contour_pts"]
        ).astype(np.float32),
        "changed_dwell_coords": np.array(
            plan_parameters["dwell_coords_optimisation"]
        ).astype(np.float32),
        "changed_dwell_pts_source_end_inf": np.array(
            plan_parameters["dwell_pts_source_end_inf_optimisation"]
        ).astype(np.float32),
        "changed_dwell_pts_source_end_sup": np.array(
            plan_parameters["dwell_pts_source_end_sup_optimisation"]
        ).astype(np.float32),
        "changed_dwell_times": dwell_times,
    }

    # dose calc first
    all_DVHs = fast_TG43_cpu(
        changed_structures,
        plan_parameters,
        voxel_size=1,
    )

    all_DVHs = np.array(all_DVHs)
    bin_width = all_DVHs[0][0, 1] - all_DVHs[0][0, 0]
    all_bin_widths = np.repeat(bin_width, len(all_DVHs[0][0, :]))

    # DVH prostate
    volumes_optimised = all_DVHs[0][1, :] * float(plan_parameters["prostate_vol"]) / 100
    DVH = np.empty((volumes_optimised.size + all_bin_widths.size,), dtype="float64")
    DVH[0::2] = all_bin_widths
    DVH[1::2] = volumes_optimised

    DVH = np.array(DVH, dtype="float64")

    rd_robust[0x3004, 0x050][0][0x3004, 0x058].value = list(DVH.round(5))
    rd_robust[0x3004, 0x050][0].DVHNumberOfBins.value = str(len(volumes_optimised))

    # DVH Urthera
    volumes_optimised = all_DVHs[1][1, :] * float(plan_parameters["urethra_vol"]) / 100
    DVH = np.empty((volumes_optimised.size + all_bin_widths.size,), dtype="float64")
    DVH[0::2] = all_bin_widths
    DVH[1::2] = volumes_optimised
    DVH = np.array(DVH, dtype="float64")

    rd_robust.DVHSequence[1].DVHData = list(DVH.round(5))
    rd_robust[0x3004, 0x050][1].DVHNumberOfBins.value = str(len(volumes_optimised))
    # DVH rectum
    volumes_optimised = all_DVHs[2][1, :] * float(plan_parameters["rectum_vol"]) / 100
    DVH = np.empty((volumes_optimised.size + all_bin_widths.size,), dtype="float64")
    DVH[0::2] = all_bin_widths
    DVH[1::2] = volumes_optimised
    DVH = np.array(DVH, dtype="float64")

    rd_robust.DVHSequence[2].DVHData = list(DVH.round(5))
    rd_robust[0x3004, 0x050][2].DVHNumberOfBins.value = str(len(volumes_optimised))

    if len(rd_robust.DVHSequence[0].DVHReferencedROISequence) == 2:
        for roi in rd_robust.DVHSequence[0].DVHReferencedROISequence:
            if roi.ReferencedROINumber == 0:
                if roi.DVHROIContributionType == "INCLUDED":
                    continue
                else:
                    roi.DVHROIContributionType = "INCLUDED"

            if roi.ReferencedROINumber == 1:
                if roi.DVHROIContributionType == "EXCLUDED":
                    continue
                else:
                    roi.DVHROIContributionType = "EXCLUDED"

    if len(rd_robust.DVHSequence[0].DVHReferencedROISequence) == 1:
        rd_robust.DVHSequence[0].DVHReferencedROISequence.append(
            deepcopy(rd_robust.DVHSequence[0].DVHReferencedROISequence[0])
        )

        if (
            rd_robust.DVHSequence[0].DVHReferencedROISequence[0].ReferencedROINumber
            == 0
        ):
            if (
                rd_robust.DVHSequence[0]
                .DVHReferencedROISequence[0]
                .DVHROIContributionType
                != "INCLUDED"
            ):
                rd_robust.DVHSequence[0].DVHReferencedROISequence[
                    0
                ].DVHROIContributionType = "INCLUDED"

        rd_robust.DVHSequence[0].DVHReferencedROISequence[1].ReferencedROINumber = str(
            1
        )
        rd_robust.DVHSequence[0].DVHReferencedROISequence[
            1
        ].DVHROIContributionType = "EXCLUDED"

    return rp_robust, rd_robust
