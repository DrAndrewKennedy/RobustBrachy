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
import plotly.graph_objects as go

try:
    import cupy as cp
    from robustbrachy.robustevaluation.robust_evaluation_gpu import *
    from robustbrachy.robustevaluation.utils_gpu import *
except:
    print("no cupy")

from robustbrachy.robustevaluation.robust_evaluation_cpu import *
from robustbrachy.robustevaluation.utils_cpu import *

# title of page + other information
st.set_page_config(page_title="Single Plan Robust Evaluation", layout="wide")

st.markdown("# Single Plan Robust Evaluation")
st.sidebar.header("Single Plan Robust Evaluation")

st.write(
    """
    #### Input parameter variables below
    
    """
)

# if GPU is activated
if st.session_state.GPU_CPU:
    str_gpu = "GPU use activated!"
    st.sidebar.write(f":green[{str_gpu}]")

else:
    str_cpu = "CPU use activated!"
    st.sidebar.write(f":red[{str_cpu}]")
st.divider()

st.write("##### Defining Uncertainty Parameter Variables")


st.sidebar.divider()
st.sidebar.write("## Summary of Parameters")

# input all uncertainty magnitudes parameters
col1, col2, col3 = st.columns([1, 1, 1])
# uncertainties_SD = []
with col1:
    st.write(
        "###### Parameter 1 Standard Deviation: Dwells points move along the needle"
    )
    needle_recon = st.text_input(
        "Needle Reconstruction SI (Needle Tip, units in mm)", "1.1"
    )
    needle_rigid = st.text_input("Needle Rigid movement SI (units in mm)", "0.34")
    source_position = st.text_input("Source positioning (units in mm)", "1.0")
    dwells_shift_along_needle = np.round(
        np.sqrt(
            float(needle_recon) ** 2
            + float(needle_rigid) ** 2
            + float(source_position) ** 2
        ),
        1,
    )
    st.write("###### Total Parameter 1 SD: " + str(dwells_shift_along_needle) + " mm")
    st.sidebar.write("Total Parameter 1 SD: " + str(dwells_shift_along_needle) + " mm")
    # uncertainties_SD.append(dwells_shift_along_needle)

    st.divider()
    st.write(
        "###### Parameter 2 Standard Deviation: Prostate boundary expansion/contraction from geometric mean (prostate centre) in 3D"
    )
    prostate_contour = st.text_input(
        "Total Prostate Contouring Uncertainty (units in mm)", "1.0"
    )
    mm_change_shrink_enlarge_prostate = np.round(float(prostate_contour), 1)
    st.write(
        "###### Total Parameter 2 SD: " + str(mm_change_shrink_enlarge_prostate) + " mm"
    )
    st.sidebar.write(
        "Total Parameter 2 SD: " + str(mm_change_shrink_enlarge_prostate) + " mm"
    )

    st.divider()
    st.write(
        "###### Parameter 3a Standard Deviation: Urethra boundary expanded/contracted in each transverse slice"
    )
    urethra_contour = st.text_input(
        "Total Urethra Contouring Uncertainty (units in mm)", "0.25"
    )
    mm_change_shrink_enlarge_urethra = np.round(float(urethra_contour), 2)
    st.write(
        "###### Total Parameter 3a SD: " + str(mm_change_shrink_enlarge_urethra) + " mm"
    )
    st.sidebar.write(
        "Total Parameter 3a SD: " + str(mm_change_shrink_enlarge_urethra) + " mm"
    )

with col2:
    st.write(
        "###### Parameter 3b Standard Deviation: Rectum boundary expanded/contracted in each transverse slice"
    )
    rectum_contour = st.text_input(
        "Total Rectum Contouring Uncertainty (units in mm)", "0.5"
    )
    mm_change_shrink_enlarge_rectum = np.round(float(rectum_contour), 1)
    st.write(
        "###### Total Parameter 3b SD: " + str(mm_change_shrink_enlarge_rectum) + " mm"
    )
    st.sidebar.write(
        "Total Parameter 3b SD: " + str(mm_change_shrink_enlarge_rectum) + " mm"
    )
    # uncertainties_SD.append(mm_change_shrink_enlarge_rectum)
    st.divider()

    st.write("###### Parameter 4 Standard Deviation: Needle transverse plane movements")
    needle_recon_2D = st.text_input(
        "Needle Reconstruction transverse slice (units in mm)", "1.2"
    )
    needle_rigid_2D = st.text_input(
        "Needle rigid movement (intra-fractional) in transverse plane (units in mm)",
        "0.86",
    )
    mm_change_dwells_random_2D = np.round(
        np.sqrt(float(needle_recon_2D) ** 2 + float(needle_rigid_2D) ** 2), 1
    )
    st.write("###### Total Parameter 4 SD: " + str(mm_change_dwells_random_2D) + " mm")
    st.sidebar.write("Total Parameter 4 SD: " + str(mm_change_dwells_random_2D) + " mm")

    st.divider()
    st.write("###### Parameter 5a Standard Deviation: Dwell times add constant value")
    dwell_time_precision = st.text_input(
        "Dwell time precision (units in seconds)", "0.06"
    )
    dwell_time_increase_decrease = np.round(float(dwell_time_precision), 2)
    st.write(
        "###### Total Parameter 5a SD: " + str(dwell_time_increase_decrease) + " s"
    )
    st.sidebar.write(
        "Total Parameter 5a SD: " + str(dwell_time_increase_decrease) + " s"
    )

with col3:
    st.write("###### Parameter 5b Standard Deviation: Dwell times percentage change")
    dose_calc_med_percent = st.text_input(
        "Dose Calculation - Medium (units in %)", "1.0"
    )
    dose_calc_TPS_percent = st.text_input(
        "Dose Calculation - Treatment Planning (units in %)", "3.0"
    )
    source_activity = st.text_input("Source Activity (units in %)", "3.0")
    dwell_time_change_percentage = np.round(
        np.sqrt(
            float(dose_calc_med_percent) ** 2
            + float(dose_calc_TPS_percent) ** 2
            + float(source_activity) ** 2
        ),
        1,
    )
    st.write(
        "###### Total Parameter 5b SD: " + str(dwell_time_change_percentage) + " %"
    )
    st.sidebar.write(
        "Total Parameter 5b SD: " + str(dwell_time_change_percentage) + " %"
    )

    st.divider()

    st.write(
        "###### Parameter 6a Standard Deviation: Prostate and Urethra (+ needles) anterior-posterior movement"
    )
    AP_rigid = st.text_input(
        "Intra-observer rigid movement anterior-posterior direction(units in mm)", "0.5"
    )
    mm_change_Y_rigid = np.round(float(AP_rigid), 2)
    st.write("###### Total Parameter 6a SD: " + str(mm_change_Y_rigid) + " mm")
    st.sidebar.write("Total Parameter 6a SD: " + str(mm_change_Y_rigid) + " mm")

    st.write(
        "###### Parameter 6b Standard Deviation: Prostate and Urethra (+ needles) left-right movement"
    )
    LR_rigid = st.text_input(
        "Intra-observer rigid movement left-right direction (units in mm)", "0.1"
    )
    mm_change_X_rigid = np.round(float(LR_rigid), 2)
    st.write("###### Total Parameter 6b SD: " + str(mm_change_X_rigid) + " mm")
    st.sidebar.write("Total Parameter 6b SD: " + str(mm_change_X_rigid) + " mm")

    st.write(
        "###### Parameter 6c Standard Deviation: Prostate and Urethra (+ needles) superior-inferior movement"
    )
    SI_rigid = st.text_input(
        "Intra-observer rigid movement superior-inferior direction (units in mm)", "0.0"
    )
    mm_change_Z_rigid = np.round(float(SI_rigid), 2)
    st.write("###### Total Parameter 6c SD: " + str(mm_change_Z_rigid) + " mm")
    st.sidebar.write("Total Parameter 6c SD: " + str(mm_change_Z_rigid) + " mm")

    # creating dictionary of uncertainties
    uncertainty_magnitudes = {
        "P1_needle_recon": needle_recon,
        "P1_needle_rigid": needle_rigid,
        "P1_source_position": source_position,
        "P2_prostate_contour_expand_shrink": prostate_contour,
        "P3a_urethra_contour_expand_shrink": urethra_contour,
        "P3b_rectum_contour_expand_shrink": rectum_contour,
        "P4_needle_recon_2D": needle_recon_2D,
        "P4_needle_rigid_2D": needle_rigid_2D,
        "P5a_dwell_time_precision": dwell_time_precision,
        "P5b_dose_calc_med_percent": dose_calc_med_percent,
        "P5b_dose_calc_TPS_percent": dose_calc_TPS_percent,
        "P5b_source_activity": source_activity,
        "P6_AP_rigid": AP_rigid,
        "P6_LR_rigid": LR_rigid,
        "P6_SI_rigid": SI_rigid,
    }
    # ** calculate parameter variable SD **
    uncertainties_SD = calculate_parameter_variable_SDs(uncertainty_magnitudes)

st.divider()

# define parameters for the robust evaluation algorithm
st.write("##### Robust Evaluation Algorithm Parameters")

col1_2, col2_2 = st.columns([1, 1])

with col1_2:
    st.write("###### Number of Ucertainty Scenarios")
    no_of_runs = st.text_input("Number of Uncertainty Scenarios", "500")
    no_of_runs = int(no_of_runs)
    st.write("There will be " + str(no_of_runs) + " scenarios.")
    st.sidebar.write("There will be " + str(no_of_runs) + " scenarios.")

# collects which DVH metrics the user wants pass-rates for
with col2_2:
    st.write("###### Dose metrics to include in the robust evaluation")
    select_all = st.checkbox(label="Select all")

    if select_all:
        metric_states_D = []
        metric_states_V = []
        for i, metric in enumerate(st.session_state.metric_labels_D):
            metric_states_D.append(st.checkbox(label=f"{metric}", value=True))

        for i, metric in enumerate(st.session_state.metric_labels_V):
            metric_states_V.append(st.checkbox(label=f"{metric}", value=True))

        st.sidebar.write("#### Metrics that will be included:")
        metric_indexs_D = []
        for i, m in enumerate(metric_states_D):
            if m == True:
                st.sidebar.write(
                    st.session_state.metric_labels_to_print_D[i], unsafe_allow_html=True
                )
                metric_indexs_D.append(i)

        metric_indexs_V = []
        for i, m in enumerate(metric_states_V):
            if m == True:
                st.sidebar.write(
                    st.session_state.metric_labels_to_print_V[i], unsafe_allow_html=True
                )
                metric_indexs_V.append(i)

    else:
        metric_states_D = []
        metric_states_V = []
        for i, metric in enumerate(st.session_state.metric_labels_D):
            metric_states_D.append(st.checkbox(label=f"{metric}"))

        for i, metric in enumerate(st.session_state.metric_labels_V):
            metric_states_V.append(st.checkbox(label=f"{metric}"))

        st.sidebar.write("#### Metrics that will be included:")
        metric_indexs_D = []
        for i, m in enumerate(metric_states_D):
            if m == True:
                st.sidebar.write(
                    st.session_state.metric_labels_to_print_D[i], unsafe_allow_html=True
                )
                metric_indexs_D.append(i)

        metric_indexs_V = []
        for i, m in enumerate(metric_states_V):
            if m == True:
                st.sidebar.write(
                    st.session_state.metric_labels_to_print_V[i], unsafe_allow_html=True
                )
                metric_indexs_V.append(i)

    df_dose_metrics_to_include = st.session_state.df_dose_metrics.loc[metric_indexs_D]
    df_volume_metrics_to_include = st.session_state.df_volume_metrics.loc[
        metric_indexs_V
    ]
    dose_passrates_to_include = np.array(st.session_state.passrates_labels_D)[
        metric_indexs_D
    ]
    volume_passrates_to_include = np.array(st.session_state.passrates_labels_V)[
        metric_indexs_V
    ]

st.sidebar.divider()

# performs the robust evaluation
if st.sidebar.button("Start Robust Evaluation"):
    st.sidebar.success("Started")

    st.divider()
    st.write("##### Robust Evaluation Progress")

    progress_bar = st.progress(0, text="Operation in progress. Please wait.")

    if st.session_state.GPU_CPU:
        (
            pass_rates,
            overall_pass_rate,
            all_robust_dvh_summary,
            all_nominal_dvhs,
        ) = probabilistic_robust_measure_gpu(
            no_of_runs,
            st.session_state.plan_parameters,
            df_dose_metrics_to_include,
            df_volume_metrics_to_include,
            uncertainties_SD,
            progress_bar,
        )

    else:
        (
            pass_rates,
            overall_pass_rate,
            all_robust_dvh_summary,
            all_nominal_dvhs,
        ) = probabilistic_robust_measure_cpu(
            no_of_runs,
            st.session_state.plan_parameters,
            df_dose_metrics_to_include,
            df_volume_metrics_to_include,
            uncertainties_SD,
            progress_bar,
        )

    st.session_state.pass_rates = pass_rates
    st.session_state.overall_pass_rate = overall_pass_rate
    st.session_state.all_robust_dvh_summary = all_robust_dvh_summary
    st.session_state.all_nominal_dvhs = all_nominal_dvhs

    st.write("###### Completed!")
    st.session_state.robust_eval_completed = True


if st.session_state.robust_eval_completed == True:
    st.divider()
    st.write("##### Robust Evaluation Results")

    df_pass_rates = pd.DataFrame(
        data=[[st.session_state.overall_pass_rate, *st.session_state.pass_rates]],
        columns=[
            "Overall Pass-rate",
            *dose_passrates_to_include,
            *volume_passrates_to_include,
        ],
    )

    st.write("###### DVH pass-rates")
    st.dataframe(df_pass_rates, use_container_width=True, hide_index=True)

    st.sidebar.divider()
    height_to_use_for_graphs = int(
        st.sidebar.text_input(
            "Change height of graph (pixels):", str(int(((20 + 1) * 35 + 3)))
        )
    )

    if st.sidebar.button("Increase Axis Font size"):
        st.session_state.axis_font_size += 1

    if st.sidebar.button("Decrease Axis Font size"):
        st.session_state.axis_font_size -= 1
        if st.session_state.axis_font_size < 0:
            st.session_state.axis_font_size = 1

    if st.sidebar.button("Increase Legend Font size"):
        st.session_state.legend_font_size += 1

    if st.sidebar.button("Decrease Legend Font size"):
        st.session_state.legend_font_size -= 1
        if st.session_state.legend_font_size < 0:
            st.session_state.legend_font_size = 1

    if st.sidebar.button("line width: Increase"):
        st.session_state.line_size += 0.5

    if st.sidebar.button("line width: Decrease"):
        st.session_state.line_size -= 0.5
        if st.session_state.line_size < 0:
            st.session_state.line_size = 0.5

    if st.sidebar.button("line width area edge: Increase"):
        st.session_state.line_size_areas += 0.5

    if st.sidebar.button("line width area edge: Decrease"):
        st.session_state.line_size_areas -= 0.5
        if st.session_state.line_size_areas < 0:
            st.session_state.line_size_areas = 0.5
    st.write(" ")
    st.write("###### Dose-Volume Histograms with Robustness")
    fig_dvh = go.Figure()

    prostate_color = [
        "rgba(6,47,95,1.0)",
        "rgba(18,97,160,0.5)",
        "rgba(56,149,211,0.4)",
        "rgba(88,204,237,0.3)",
    ]
    urethra_color = [
        "rgba(4,129,83,1.0)",
        "rgba(39,171,123,0.5)",
        "rgba(73,191,145,0.4)",
        "rgba(146,215,195,0.3)",
    ]
    rectum_color = [
        "rgba(167,0,23,1.0)",
        "rgba(255,0,41,0.5)",
        "rgba(255,123,123,0.4)",
        "rgba(255,186,186,0.3)",
    ]

    colours = [prostate_color, urethra_color, rectum_color]
    # nominal dvhs
    for i, name in enumerate(["Prostate Nominal", "Urethra Nominal", "Rectum Nominal"]):
        fig_dvh.add_trace(
            go.Scattergl(
                x=st.session_state.all_nominal_dvhs[0][i][0, :],
                y=st.session_state.all_nominal_dvhs[0][i][1, :],
                mode="lines",
                showlegend=True,
                name=name,
                line=dict(
                    color=colours[i][0],
                    width=st.session_state.line_size,
                ),
            )
        )

    # robust dvhs
    robust_dvh_names = [
        [
            ["Prostate 68% CI", "Prostate 68% CI"],
            ["Prostate 95% CI", "Prostate 95% CI"],
            ["Prostate max-min", "Prostate max-min"],
        ],
        [
            ["Urethra 68% CI", "Urethra 68% CI"],
            ["Urethra 95% CI", "Urethra 95% CI"],
            ["Urethra max-min", "Urethra max-min"],
        ],
        [
            ["Rectum 68% CI", "Rectum 68% CI"],
            ["Rectum 95% CI", "Rectum 95% CI"],
            ["Rectum max-min", "Rectum max-min"],
        ],
    ]
    robust_idx = [[1, 2], [3, 4], [5, 6]]

    # shape of all_robust_dvh_summary = [ CI = mu +/- n x SD ][ plan index = 0 for single RE ][ structure ][ dose/vol ][ arry values ]

    for i, names in enumerate(robust_dvh_names):
        for j, names_2 in enumerate(names):
            fig_dvh.add_trace(
                go.Scatter(
                    x=(
                        st.session_state.all_robust_dvh_summary[robust_idx[j][1]][0][i][
                            0, :
                        ]
                    )[
                        ~np.isnan(
                            st.session_state.all_robust_dvh_summary[robust_idx[j][1]][
                                0
                            ][i][0, :]
                        )
                    ],
                    y=(
                        st.session_state.all_robust_dvh_summary[robust_idx[j][1]][0][i][
                            1, :
                        ]
                    )[
                        ~np.isnan(
                            st.session_state.all_robust_dvh_summary[robust_idx[j][1]][
                                0
                            ][i][0, :]
                        )
                    ],
                    mode="lines",
                    showlegend=False,
                    legendgroup=names_2[0],
                    name=names_2[1],
                    line=dict(
                        color=colours[i][j + 1],
                        width=st.session_state.line_size,
                    ),
                )
            )
            fig_dvh.add_trace(
                go.Scatter(
                    x=(
                        st.session_state.all_robust_dvh_summary[robust_idx[j][0]][0][i][
                            0, :
                        ]
                    )[
                        ~np.isnan(
                            st.session_state.all_robust_dvh_summary[robust_idx[j][0]][
                                0
                            ][i][0, :]
                        )
                    ],
                    y=(
                        st.session_state.all_robust_dvh_summary[robust_idx[j][0]][0][i][
                            1, :
                        ]
                    )[
                        ~np.isnan(
                            st.session_state.all_robust_dvh_summary[robust_idx[j][0]][
                                0
                            ][i][0, :]
                        )
                    ],
                    mode="lines",
                    showlegend=True,
                    name=names_2[0],
                    legendgroup=names_2[0],
                    line=dict(
                        color=colours[i][j + 1],
                        width=st.session_state.line_size,
                    ),
                    fill="tonexty",
                    fillcolor=colours[i][j + 1],
                )
            )

    fig_dvh.update_xaxes(
        title_text="Dose (Gy)",  # <Br>(a)",
        title_font=dict(size=st.session_state.axis_font_size),
        minor=dict(dtick=1, showgrid=True),
        range=[0, 36],
        tick0=0,
        dtick=5,
    )

    fig_dvh.update_yaxes(
        title_text="Relative Volume (%)",
        title_font=dict(size=st.session_state.axis_font_size),
        range=[0, 101],
        minor=dict(dtick=2.5, showgrid=True),
        tick0=0,
        dtick=10,
    )

    fig_dvh.update_layout(
        height=height_to_use_for_graphs,  # width = 800,
        legend=dict(
            font=dict(  # family = "Courier",
                size=st.session_state.legend_font_size,
                # color = "black"
            )
        ),
    )

    st.plotly_chart(fig_dvh, height=height_to_use_for_graphs, use_container_width=True)
