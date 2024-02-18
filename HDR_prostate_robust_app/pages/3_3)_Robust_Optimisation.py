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

import streamlit as st
import numpy as np
import pandas as pd

try:
    import cupy as cp
    from robustbrachy.robustoptimisation.evaluation_gpu import *
    from robustbrachy.robustoptimisation.robust_evaluation_vectorised_gpu import *
except:
    print("no cupy")

from robustbrachy.robustoptimisation.robust_optimisation import *
from robustbrachy.robustoptimisation.evaluation_cpu import *
from robustbrachy.robustoptimisation.robust_evaluation_vectorised_cpu import *

# page text and instructions
st.set_page_config(page_title="Robust Optimisation", layout="wide")

st.markdown("# Robust Optimisation")
st.sidebar.header("Robust Optimisation")

st.write("##### Robust optimiser information")
st.markdown(
    """
    **Note that This robust optimiser has been trained to generate robust plans against the following uncertainty magnitudes:**
    - Parameter 1 SD (Dwells points move along the needle) = 1.5 mm
    - Parameter 2 SD (Prostate boundary expansion/contraction from geometric mean (prostate centre) in 3D) = 1.0 mm
    - Parameter 3a SD (Urethra boundary expanded/contracted in each transverse slice) = 0.25 mm
    - Parameter 3b SD (Rectum boundary expanded/contracted in each transverse slice) = 0.5 mm
    - Parameter 4 SD (Needle transverse plane movements) = 1.5 mm
    - Parameter 5a SD (Dwell times add constant value) = 0.06 s
    - Parameter 5b SD (Dwell times percentage change) = 4.4 %
    - Parameter 6a SD (Prostate and Urethra (+ needles) anterior-posterior movement) = 0.5 mm
    - Parameter 6b SD (Prostate and Urethra (+ needles) left-right movement) = 0.1 mm
    - Parameter 6c SD (Prostate and Urethra (+ needles) superior-inferior movement) = 0.0 mm
    """
)

st.markdown(
    """
        **Note also that this robust optimiser has also been trained to generate robust plans using the following DVH metrics:**
        - The minimum dose to the hottest 90 % of the prostate: D_90 > 100% (Note: D_90 pass-rate is a very close approximation for V_100 pass-rate)
        - The maximum dose to the urethra: D_0.01cc < 110%
        - The maximum dose to the Rectum: D_0.1cc < 13 Gy
        """
)
st.divider()
st.write("##### How to use this optimiser:")
st.markdown(
    """
         All of the following steps need to be completed before exploring the robust optimised plans:
         1) Set optimiser parameters
         2) Click the "Step 1: start Robust Optimisation" button on the sidebar to start the robust optimiser
         3) Click the "Step 2: Robust Evaluate All Solutions" button on the sidebar that will then appear after a successful optimisation process
         4) click the "Step 3: Generate Isodoses" button on the sidebar to generate the nominal isodoses of all plans on the pareto front
         5) click the "Step 4: Robust Evaluate TPS Plan" button on the sidebar to generate the corresponding TPS data
         6) Proceed to the next page "3 a) Select Best Plan"
         """
)

# state if GPU or CPU is being used
if st.session_state.GPU_CPU:
    str_gpu = "GPU use activated!"
    st.sidebar.write(f":green[{str_gpu}]")
else:
    str_cpu = "CPU use activated!"
    st.sidebar.write(f":red[{str_cpu}]")


st.divider()
st.write("##### Defining Optimiser Parameters")
st.sidebar.divider()
st.sidebar.write("## Summary of Parameters")

# defining optimiser parameters
col1, col2 = st.columns([1, 1])

with col1:
    num_of_itr = int(st.text_input("Number of iterations", "400"))
    pop_size = int(st.text_input("Population size", "200"))
    limit_on_dwells_outside_prostate_mm = float(
        st.text_input(
            "Limit the distance dwell points can be outside the prostate (units in mm)",
            "4.5",
        )
    )
    max_time_limit = float(
        st.text_input("Maximum dwell time limit (units in seconds)", "31.0")
    )
    show_progress = bool(
        int(
            st.text_input(
                "Show optimiser progress in the cmd window (True = 1 or False = 0)", "0"
            )
        )
    )
    offspring_size = int(
        st.text_input("Offspring size to generate each iteration", "200")
    )
    point_crossover_prob = float(st.text_input("Point crossover probability", "1.0"))

with col2:
    num_of_point_crossovers = int(st.text_input("Number of point crossover", "50"))
    mutation_prob = float(
        st.text_input("Probability of mutation of dwell times in offspring", "1.0")
    )
    mutation_spread = int(
        st.text_input(
            "Mutation probability function spread (smaller = less spread)", "50"
        )
    )
    margin_prostate = float(
        st.text_input(
            "D90 excess margin before constraint is satisfied (as a % of constraint)",
            "10.7",
        )
    )
    margin_urethra = float(
        st.text_input(
            "DmaxU excess margin before constraint is satisfied (as a % of constraint)",
            "9.0",
        )
    )
    margin_rectum = float(
        st.text_input(
            "DmaxR excess margin before constraint is satisfied (as a % of constraint)",
            "9.2",
        )
    )
    no_of_runs = int(
        st.text_input("Number of Uncertainty Scenarios in robust evaluation", "500")
    )
    no_of_runs_internal = int(
        st.text_input("Number of robust evaluation iterations to split into", "50")
    )

# making a dictionary of optimiser parameters
optimiser_parametres = {
    "num_of_itr": num_of_itr,
    "pop_size": pop_size,
    "limit_on_dwells_outside_prostate_mm": limit_on_dwells_outside_prostate_mm,
    "max_time_limit": max_time_limit,
    "show_progress": show_progress,
    "offspring_size": offspring_size,
    "point_crossover_prob": point_crossover_prob,
    "num_of_point_crossovers": num_of_point_crossovers,
    "mutation_prob": mutation_prob,
    "mutation_spread": mutation_spread,
    "margin_prostate": margin_prostate,
    "margin_urethra": margin_urethra,
    "margin_rectum": margin_rectum,
    "no_of_runs": no_of_runs,
    "no_of_runs_internal": no_of_runs_internal,
    "use_gpu_in_eval": st.session_state.GPU_CPU,
}

if "completed_RO" not in st.session_state:
    st.session_state.completed_RO = False

if "completed_RE" not in st.session_state:
    st.session_state.completed_RE = False

if "completed_DG" not in st.session_state:
    st.session_state.completed_DG = False

if "completed_RE_TPS" not in st.session_state:
    st.session_state.completed_RE_TPS = False


if st.sidebar.button("Step 1: Start Robust Optimisation"):
    st.divider()
    st.write("##### Robust Optimisation Progress")
    progress_bar_1 = st.progress(0, text="Intialising Population...")
    try:
        st.session_state.plan_parameters = to_cpu(st.session_state.plan_parameters)
    except:
        st.session_state.plan_parameters = arrays_to_numpy(st.session_state.plan_parameters)
        
    (
        results,
        full_time,
        dwell_structure_array,
        dwell_times,
        all_dose_per_dwell_per_vol_pt,
    ) = robust_optimisation(
        st.session_state.plan_parameters,
        optimiser_parametres,
        progress_bar_1,
    )

    # print(full_time)
    # print(results)
    if not isinstance(results.X, np.ndarray):
        st.error("No solutions found, try changing parameters", icon="🚨")
    else:
        dwell_times_pareto_front = np.array(results.X).round(1)
        dwell_times_pareto_front[dwell_times_pareto_front < 0.4] = 0.0
        solutions_pareto_front = np.array(100 - results.F)
        st.session_state.dwell_times_pareto_front = dwell_times_pareto_front
        st.session_state.solutions_pareto_front = solutions_pareto_front
        st.session_state.full_time = full_time
        st.session_state.dwell_times = dwell_times
        st.session_state.dwell_structure_array = dwell_structure_array
        st.session_state.all_dose_per_dwell_per_vol_pt = all_dose_per_dwell_per_vol_pt

        # nominal metrics, try using gpu version first, if fails use cpu/numpy version
        try:
            (
                nominal_metrics_pareto_front,
                all_nominal_dvhs_pareto_front,
            ) = get_nominal_metrics_array_gpu(
                cp.array(st.session_state.dwell_times_pareto_front),
                cp.array(st.session_state.all_dose_per_dwell_per_vol_pt[0]),
                cp.array(st.session_state.all_dose_per_dwell_per_vol_pt[1]),
                cp.array(st.session_state.all_dose_per_dwell_per_vol_pt[2]),
                st.session_state.plan_parameters["prescribed_dose"],
                st.session_state.plan_parameters["urethra_vol"],
                st.session_state.plan_parameters["rectum_vol"],
            )
        except:
            (
                nominal_metrics_pareto_front,
                all_nominal_dvhs_pareto_front,
            ) = get_nominal_metrics_array_cpu(
                np.array(st.session_state.dwell_times_pareto_front),
                np.array(st.session_state.all_dose_per_dwell_per_vol_pt[0]),
                np.array(st.session_state.all_dose_per_dwell_per_vol_pt[1]),
                np.array(st.session_state.all_dose_per_dwell_per_vol_pt[2]),
                st.session_state.plan_parameters["prescribed_dose"],
                st.session_state.plan_parameters["urethra_vol"],
                st.session_state.plan_parameters["rectum_vol"],
            )

        st.session_state.nominal_metrics_pareto_front = nominal_metrics_pareto_front
        st.session_state.all_nominal_dvhs_pareto_front = all_nominal_dvhs_pareto_front

        # print table of approximate passrates and nominal metrics
        data_to_print = np.concatenate(
            (
                st.session_state.solutions_pareto_front,
                st.session_state.nominal_metrics_pareto_front.T,
            ),
            axis=1,
        )
        df_pareto_front_data = pd.DataFrame(
            data_to_print,
            columns=[
                "Approx. Passrate D90",
                "Approx. Passrate DmaxU",
                "Approx. Passrate DmaxR",
                "Prostate nominal D90",
                "Prostate nominal V100",
                "Prostate nominal V150",
                "Prostate nominal V200",
                "Urethra nominal D10",
                "Urethra nominal DmaxU",
                "Rectum nominal V75",
                "Rectum nominal DmaxR",
            ],
        )
        df_pareto_front_data = df_pareto_front_data.rename_axis("Plan").reset_index()
        st.session_state.all_metrics_approx_pass_rates = df_pareto_front_data

    st.session_state.completed_RO = True

if st.session_state.completed_RO:
    st.dataframe(
        st.session_state.all_metrics_approx_pass_rates.round(2),
        # height = height_to_use,
        use_container_width=True,
        hide_index=True,
    )

    if st.sidebar.button("Step 2: Robust Evaluate All Solutions"):
        st.divider()
        st.write("##### Robust Evaluating All Solutions Progress")
        progress_bar_2 = st.progress(0, text="Starting Robust Evaluating...")

        if st.session_state.GPU_CPU:
            st.session_state.plan_parameters = to_gpu(st.session_state.plan_parameters)
            (
                all_passrate,
                D90_passrate,
                DmaxU_passrate,
                DmaxR_passrate,
                all_robust_dvh_summary,
            ) = robust_measure_array_gpu(
                optimiser_parametres,
                st.session_state.plan_parameters,
                cp.array(st.session_state.dwell_times_pareto_front),
                cp.array(st.session_state.all_nominal_dvhs_pareto_front),
                progress_bar_2,
            )
        else:
            (
                all_passrate,
                D90_passrate,
                DmaxU_passrate,
                DmaxR_passrate,
                all_robust_dvh_summary,
            ) = robust_measure_array_cpu(
                optimiser_parametres,
                st.session_state.plan_parameters,
                st.session_state.dwell_times_pareto_front,
                st.session_state.all_nominal_dvhs_pareto_front,
                progress_bar_2,
            )

        st.session_state.all_robust_dvhs = all_robust_dvh_summary
        pass_rates = np.array(
            [all_passrate, D90_passrate, DmaxU_passrate, DmaxR_passrate]
        )
        st.session_state.pass_rates = pass_rates
        data_to_print = np.concatenate(
            (
                st.session_state.pass_rates.T,
                st.session_state.nominal_metrics_pareto_front.T,
            ),
            axis=1,
        )
        df_pareto_front_data_RE = pd.DataFrame(
            data_to_print,
            columns=[
                "All Pass-rate (%)",
                "D90 Pass-rate (%)",
                "DmaxU Pass-rate (%)",
                "DmaxR Pass-rate (%)",
                "Prostate: D90 (Gy)",
                "Prostate: V100 (%)",
                "Prostate: V150 (%)",
                "Prostate: V200 (%)",
                "Urethra: D10 (Gy)",
                "Urethra: DmaxU (Gy)",
                "Rectum: V75 (cc)",
                "Rectum: DmaxR (Gy)",
            ],
        )
        df_pareto_front_data_RE = df_pareto_front_data_RE.rename_axis(
            "Plan"
        ).reset_index()
        st.session_state.df_pareto_front_data_RE = df_pareto_front_data_RE

        st.session_state.all_metrics = df_pareto_front_data_RE.round(2)

        st.session_state.completed_RE = True

    if st.session_state.completed_RE:
        if st.sidebar.button("Step 3: Generate Isodoses"):
            if st.session_state.GPU_CPU:
                dose_per_dwell_per_vol_pt, dose_calc_pts = get_dose_grid_gpu(
                    st.session_state.plan_parameters,
                    st.session_state.plan_parameters["dwell_coords_optimisation"],
                    st.session_state.plan_parameters[
                        "dwell_pts_source_end_sup_optimisation"
                    ],
                    st.session_state.plan_parameters[
                        "dwell_pts_source_end_inf_optimisation"
                    ],
                    voxel_size=1,
                )
            else:
                dose_per_dwell_per_vol_pt, dose_calc_pts = get_dose_grid_cpu(
                    st.session_state.plan_parameters,
                    st.session_state.plan_parameters["dwell_coords_optimisation"],
                    st.session_state.plan_parameters[
                        "dwell_pts_source_end_sup_optimisation"
                    ],
                    st.session_state.plan_parameters[
                        "dwell_pts_source_end_inf_optimisation"
                    ],
                    voxel_size=1,
                )

            st.session_state.dose_per_dwell_per_vol_pt = dose_per_dwell_per_vol_pt
            st.session_state.dose_calc_pts = dose_calc_pts

            st.session_state.completed_DG = True

    if st.session_state.completed_DG:
        if st.sidebar.button("Step 4: Robust Evaluate TPS Plan"):
            st.divider()
            st.write("##### Robust Evaluating TPS Plan Progress")
            progress_bar_3 = st.progress(0, text="Operation in progress. Please wait.")
            df_dose_metrics_to_include = pd.DataFrame(
                [
                    {
                        "Structure": 0,
                        "D": 90,
                        "% / cc": "%",
                        "Direction": ">",
                        "constraint": 100,
                        "% / Gy": "%",
                    },
                    {
                        "Structure": 1,
                        "D": 0.01,
                        "% / cc": "cc",
                        "Direction": "<",
                        "constraint": 110,
                        "% / Gy": "%",
                    },
                    {
                        "Structure": 2,
                        "D": 0.1,
                        "% / cc": "cc",
                        "Direction": "<",
                        "constraint": 13,
                        "% / Gy": "Gy",
                    },
                ]
            )

            # no volume metrics in the optimiser algorithm
            df_volume_metrics_to_include = pd.DataFrame([])

            # creating dictionary of uncertainties
            uncertainties_SD = {
                "dwells_shift_along_needle": 1.5,
                "mm_change_shrink_enlarge_prostate": 1.0,
                "mm_change_shrink_enlarge_urethra": 0.25,
                "mm_change_shrink_enlarge_rectum": 0.5,
                "mm_change_dwells_random_2D": 1.5,
                "dwell_time_increase_decrease": 0.06,
                "dwell_time_change_percentage": 4.4,
                "mm_change_Y_rigid": 0.5,
                "mm_change_X_rigid": 0.1,
                "mm_change_Z_rigid": 0.0,
            }

            if st.session_state.GPU_CPU:
                # TPS dose grid
                ## calculate dose rate array for dwells allocated by robust optimiser
                dwell_coords_TPS_every_2nd = st.session_state.plan_parameters[
                    "dwell_coords_TPS"
                ][
                    :, 1::2
                ]  # TPS codes dwell times and points as position to stop and start moving again, so doubles all values
                dwell_pts_source_end_sup_TPS_every_2nd = (
                    st.session_state.plan_parameters["dwell_pts_source_end_sup_TPS"][
                        :, 1::2
                    ]
                )
                dwell_pts_source_end_inf_TPS_every_2nd = (
                    st.session_state.plan_parameters["dwell_pts_source_end_inf_TPS"][
                        :, 1::2
                    ]
                )

                dose_per_dwell_per_vol_pt_TPS, dose_calc_pts = get_dose_grid_gpu(
                    st.session_state.plan_parameters,
                    cp.array(dwell_coords_TPS_every_2nd),
                    cp.array(dwell_pts_source_end_sup_TPS_every_2nd),
                    cp.array(dwell_pts_source_end_inf_TPS_every_2nd),
                    voxel_size=1
                )

                dwell_times_TPS_every_2nd = st.session_state.plan_parameters[
                    "dwell_times_TPS"
                ][:, 1::2]
                dwell_times_TPS_every_2nd[dwell_times_TPS_every_2nd == -100] = -100
                dwell_times_TPS_every_2nd_flat = cp.array(
                    [dwell_times_TPS_every_2nd.flatten()]
                )
                dwell_times_TPS_every_2nd_flat = dwell_times_TPS_every_2nd_flat[
                    dwell_times_TPS_every_2nd_flat != -100
                ]

                dwell_times_unity = cp.empty(
                    shape=(
                        dwell_coords_TPS_every_2nd.shape[0],
                        dwell_coords_TPS_every_2nd.shape[1],
                    ),
                    dtype=cp.float32,
                )
                dwell_times_unity.fill(1.0)
                dwell_times_unity[dwell_coords_TPS_every_2nd[:, :, 0] == -100] = -100

                # nominal metrics
                dose_calc_pts_prostate = get_dose_volume_pts_gpu(
                    cp.array(st.session_state.plan_parameters["prostate_contour_pts"]),
                    cp.array(st.session_state.plan_parameters["urethra_contour_pts"]),
                )
                dose_calc_pts_urethra = get_dose_volume_pts_gpu(
                    cp.array(st.session_state.plan_parameters["urethra_contour_pts"])
                )
                dose_calc_pts_rectum = get_dose_volume_pts_gpu(
                    cp.array(st.session_state.plan_parameters["rectum_contour_pts"])
                )

                dose_per_dwell_per_vol_pt_prostate_TPS = TG43calc_dose_per_dwell_per_vol_pt_gpu(
                    dose_calc_pts_prostate,
                    cp.array(dwell_times_unity),
                    st.session_state.plan_parameters,
                    cp.array(dwell_coords_TPS_every_2nd),
                    cp.array(dwell_pts_source_end_sup_TPS_every_2nd),
                    cp.array(dwell_pts_source_end_inf_TPS_every_2nd),
                )

                dose_per_dwell_per_vol_pt_urethra_TPS = TG43calc_dose_per_dwell_per_vol_pt_gpu(
                    dose_calc_pts_urethra,
                    cp.array(dwell_times_unity),
                    st.session_state.plan_parameters,
                    cp.array(dwell_coords_TPS_every_2nd),
                    cp.array(dwell_pts_source_end_sup_TPS_every_2nd),
                    cp.array(dwell_pts_source_end_inf_TPS_every_2nd),
                )

                dose_per_dwell_per_vol_pt_rectum_TPS = TG43calc_dose_per_dwell_per_vol_pt_gpu(
                    dose_calc_pts_rectum,
                    cp.array(dwell_times_unity),
                    st.session_state.plan_parameters,
                    cp.array(dwell_coords_TPS_every_2nd),
                    cp.array(dwell_pts_source_end_sup_TPS_every_2nd),
                    cp.array(dwell_pts_source_end_inf_TPS_every_2nd),
                )

                nominal_metrics_TPS, nominal_dvhs_TPS = get_nominal_metrics_array_gpu(
                    cp.array([dwell_times_TPS_every_2nd_flat]),
                    cp.array(dose_per_dwell_per_vol_pt_prostate_TPS),
                    cp.array(dose_per_dwell_per_vol_pt_urethra_TPS),
                    cp.array(dose_per_dwell_per_vol_pt_rectum_TPS),
                    st.session_state.plan_parameters["prescribed_dose"],
                    st.session_state.plan_parameters["urethra_vol"],
                    st.session_state.plan_parameters["rectum_vol"],
                )

                # robust evaluation of TPS
                (
                    pass_rates,
                    overall_pass_rate,
                    robust_dvh_summary_TPS,
                    _,
                ) = probabilistic_robust_measure_gpu(
                    no_of_runs,
                    st.session_state.plan_parameters,
                    df_dose_metrics_to_include,
                    df_volume_metrics_to_include,
                    uncertainties_SD,
                    progress_bar_3,
                )

            else:
                # TPS dose grid
                dwell_coords_TPS_every_2nd = st.session_state.plan_parameters[
                    "dwell_coords_TPS"
                ][
                    :, 1::2
                ]  # TPS codes dwell times and points as position to stop and start moving again, so doubles all values
                dwell_pts_source_end_sup_TPS_every_2nd = (
                    st.session_state.plan_parameters["dwell_pts_source_end_sup_TPS"][
                        :, 1::2
                    ]
                )
                dwell_pts_source_end_inf_TPS_every_2nd = (
                    st.session_state.plan_parameters["dwell_pts_source_end_inf_TPS"][
                        :, 1::2
                    ]
                )

                dose_per_dwell_per_vol_pt_TPS, dose_calc_pts = get_dose_grid_cpu(
                    st.session_state.plan_parameters,
                    np.array(dwell_coords_TPS_every_2nd),
                    np.array(dwell_pts_source_end_sup_TPS_every_2nd),
                    np.array(dwell_pts_source_end_inf_TPS_every_2nd),
                    voxel_size=1
                )

                dwell_times_TPS_every_2nd = st.session_state.plan_parameters[
                    "dwell_times_TPS"
                ][:, 1::2]
                dwell_times_TPS_every_2nd[dwell_times_TPS_every_2nd == -100] = -100
                dwell_times_TPS_every_2nd_flat = np.array(
                    [dwell_times_TPS_every_2nd.flatten()]
                )
                dwell_times_TPS_every_2nd_flat = dwell_times_TPS_every_2nd_flat[
                    dwell_times_TPS_every_2nd_flat != -100
                ]
                dwell_times_unity = np.empty(
                    shape=(
                        dwell_coords_TPS_every_2nd.shape[0],
                        dwell_coords_TPS_every_2nd.shape[1],
                    ),
                    dtype=np.float32,
                )
                dwell_times_unity.fill(1.0)
                dwell_times_unity[dwell_coords_TPS_every_2nd[:, :, 0] == -100] = -100

                # nominal metrics
                dose_calc_pts_prostate = get_dose_volume_pts_cpu(
                    np.array(st.session_state.plan_parameters["prostate_contour_pts"]),
                    np.array(st.session_state.plan_parameters["urethra_contour_pts"]),
                )
                dose_calc_pts_urethra = get_dose_volume_pts_cpu(
                    np.array(st.session_state.plan_parameters["urethra_contour_pts"])
                )
                dose_calc_pts_rectum = get_dose_volume_pts_cpu(
                    np.array(st.session_state.plan_parameters["rectum_contour_pts"])
                )

                dose_per_dwell_per_vol_pt_prostate_TPS = TG43calc_dose_per_dwell_per_vol_pt_cpu(
                    dose_calc_pts_prostate,
                    np.array(dwell_times_unity),
                    st.session_state.plan_parameters,
                    np.array(dwell_coords_TPS_every_2nd),
                    np.array(dwell_pts_source_end_sup_TPS_every_2nd),
                    np.array(dwell_pts_source_end_inf_TPS_every_2nd),
                )

                dose_per_dwell_per_vol_pt_urethra_TPS = TG43calc_dose_per_dwell_per_vol_pt_cpu(
                    dose_calc_pts_urethra,
                    np.array(dwell_times_unity),
                    st.session_state.plan_parameters,
                    np.array(dwell_coords_TPS_every_2nd),
                    np.array(dwell_pts_source_end_sup_TPS_every_2nd),
                    np.array(dwell_pts_source_end_inf_TPS_every_2nd),
                )

                dose_per_dwell_per_vol_pt_rectum_TPS = TG43calc_dose_per_dwell_per_vol_pt_cpu(
                    dose_calc_pts_rectum,
                    np.array(dwell_times_unity),
                    st.session_state.plan_parameters,
                    np.array(dwell_coords_TPS_every_2nd),
                    np.array(dwell_pts_source_end_sup_TPS_every_2nd),
                    np.array(dwell_pts_source_end_inf_TPS_every_2nd),
                )

                nominal_metrics_TPS, nominal_dvhs_TPS = get_nominal_metrics_array_cpu(
                    np.array([dwell_times_TPS_every_2nd_flat]),
                    np.array(dose_per_dwell_per_vol_pt_prostate_TPS),
                    np.array(dose_per_dwell_per_vol_pt_urethra_TPS),
                    np.array(dose_per_dwell_per_vol_pt_rectum_TPS),
                    st.session_state.plan_parameters["prescribed_dose"],
                    st.session_state.plan_parameters["urethra_vol"],
                    st.session_state.plan_parameters["rectum_vol"],
                )

                # robust evaluation of TPS
                (
                    pass_rates,
                    overall_pass_rate,
                    robust_dvh_summary_TPS,
                    _,
                ) = probabilistic_robust_measure_cpu(
                    no_of_runs,
                    st.session_state.plan_parameters,
                    df_dose_metrics_to_include,
                    df_volume_metrics_to_include,
                    uncertainties_SD,
                    progress_bar_3,
                )

            st.session_state.robust_dvh_TPS = robust_dvh_summary_TPS
            st.session_state.nominal_dvhs_TPS = nominal_dvhs_TPS
            st.session_state.dose_per_dwell_per_vol_pt_TPS = (
                dose_per_dwell_per_vol_pt_TPS
            )
            pass_rates = np.array([[overall_pass_rate, *pass_rates]])

            data_to_print = np.concatenate((pass_rates, nominal_metrics_TPS.T), axis=1)

            all_metrics_TPS = pd.DataFrame(
                data_to_print,
                columns=[
                    "All Pass-rate (%)",
                    "D90 Pass-rate (%)",
                    "DmaxU Pass-rate (%)",
                    "DmaxR Pass-rate (%)",
                    "Prostate: D90 (Gy)",
                    "Prostate: V100 (%)",
                    "Prostate: V150 (%)",
                    "Prostate: V200 (%)",
                    "Urethra: D10 (Gy)",
                    "Urethra: DmaxU (Gy)",
                    "Rectum: V75 (cc)",
                    "Rectum: DmaxR (Gy)",
                ],
            )
            all_metrics_TPS.insert(0, "Plan", ["TPS"], True)
            all_metrics_TPS = all_metrics_TPS.round(2)

            st.session_state.all_metrics_TPS = all_metrics_TPS
            st.session_state.completed_RE_TPS = True

    if st.session_state.completed_RE:
        st.divider()
        st.write("##### Directly Robust Evaluated Treatment Plans Summary")

        st.dataframe(
            st.session_state.all_metrics.round(2),
            # height = height_to_use,
            use_container_width=True,
            hide_index=True,
        )

    if st.session_state.completed_DG:
        st.divider()
        st.write("##### 3D Isodose")
        st.write("Process Completed")

    if st.session_state.completed_RE_TPS:
        st.divider()
        st.write("##### Directly Robust Evaluated TPS Plan")

        st.dataframe(
            st.session_state.all_metrics_TPS.round(2),
            # height = height_to_use,
            use_container_width=True,
            hide_index=True,
        )

        st.sidebar.success(
            "Now select the page above '3 a) Select Best Plan' to start exploring the optimised set of solutions"
        )


old_selected_1 = None
if "old_selected_1" not in st.session_state:
    st.session_state.old_selected_1 = old_selected_1
else:
    st.session_state.old_selected_1 = old_selected_1

old_selected_2 = None
if "old_selected_2" not in st.session_state:
    st.session_state.old_selected_2 = old_selected_2
else:
    st.session_state.old_selected_2 = old_selected_2

old_selected_3 = None
if "old_selected_3" not in st.session_state:
    st.session_state.old_selected_3 = old_selected_3
else:
    st.session_state.old_selected_3 = old_selected_3
