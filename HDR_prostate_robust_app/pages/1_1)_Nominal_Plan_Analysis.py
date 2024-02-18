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
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go

try:
    import cupy as cp
    from robustbrachy.robustevaluation.utils_gpu import *
    from robustbrachy.robustoptimisation.evaluation_gpu import *
    from robustbrachy.robustevaluation.fast_TG43_gpu import *
except:
    print("no cupy")


from robustbrachy.robustevaluation.utils_cpu import *
from robustbrachy.robustevaluation.fast_TG43_cpu import *
from robustbrachy.robustoptimisation.evaluation_cpu import *

reload_page = False
height_to_use_for_graphs = int(((30 + 1) * 35 + 3))

st.set_page_config(page_title="Nominal Plan Analysis", layout="wide")

st.markdown("# Nominal Plan Analysis")
st.sidebar.header("Nominal Plan Analysis")

if st.session_state.GPU_CPU:
    str_gpu = "GPU use activated!"
    st.sidebar.write(f":green[{str_gpu}]")

else:
    str_cpu = "CPU use activated!"
    st.sidebar.write(f":red[{str_cpu}]")

# make sure calculations are not rerun each time
if "calcs_done" not in st.session_state:
    calcs_done = False
    st.session_state.calcs_done = calcs_done


# calculating the nominal DVHs using zero values for shifts
if st.session_state.GPU_CPU:
    if st.session_state.calcs_done != True:
        start = time.time()
        # make a copy of each structure array.
        # This is needed so the structures and plan_parameters are separated,
        # which is needed in the robust evaluation use of this function.
        copied_structures = copy_structures_gpu(st.session_state.plan_parameters)

        # calculate the nominal DVHs for each structure (dose calculation)
        all_nominal_dvhs = fast_TG43_gpu(
            copied_structures,
            st.session_state.plan_parameters,
            1,
        )
        run_time = time.time() - start

        # make the DVH plots
        plot_titles = ["Prostate", "Urethra", "Rectum"]
        fig_dvh = go.Figure()
        fig_dvh.add_trace(
            go.Scattergl(
                x=all_nominal_dvhs[0][0, :].get(),
                y=all_nominal_dvhs[0][1, :].get(),
                mode="lines",
                showlegend=True,
                name=plot_titles[0],
                # line=dict(color=colours_1[i][0],
                #             width=st.session_state.line_size,
                #             dash = line_style[i]
                #             )
            )
        )

        fig_dvh.add_trace(
            go.Scattergl(
                x=all_nominal_dvhs[1][0, :].get(),
                y=all_nominal_dvhs[1][1, :].get(),
                mode="lines",
                showlegend=True,
                name=plot_titles[1],
                # line=dict(color=colours_TPS[i][0],
                #             width=st.session_state.line_size,
                #             dash = line_style[i]
                #             )
            )
        )

        fig_dvh.add_trace(
            go.Scattergl(
                x=all_nominal_dvhs[2][0, :].get(),
                y=all_nominal_dvhs[2][1, :].get(),
                mode="lines",
                showlegend=True,
                name=plot_titles[2],
                # line=dict(color=colours_2[i][0],
                #             width=st.session_state.line_size,
                #             dash = line_style[i]
                #             )
            )
        )

        fig_dvh.update_xaxes(
            title_text="Dose (Gy)",  # <Br>(a)",
            # title_font=dict(size=st.session_state.axis_font_size),
            minor=dict(dtick=1, showgrid=True),
            range=[0, 36],
            tick0=0,
            dtick=5,
            # title_standoff=0
        )

        fig_dvh.update_yaxes(
            title_text="Relative Volume (%)",
            # title_font=dict(size=st.session_state.axis_font_size),
            range=[0, 101],
            minor=dict(dtick=2.5, showgrid=True),
            tick0=0,
            dtick=10,
            # title_standoff=0
        )
        st.session_state.fig_dvh = fig_dvh

        df_nominal_metric_data = calculate_all_nominal_metrics_cpu(
            st.session_state.df_dose_metrics,
            st.session_state.df_volume_metrics,
            st.session_state.metric_labels_D,
            st.session_state.metric_labels_V,
            np.array([all_nominal_dvhs.get()]),
            st.session_state.plan_parameters,
        )
        st.session_state.df_nominal_metric_data = df_nominal_metric_data

        # nominal plan information
        nominal_data = np.array(
            [
                round(st.session_state.plan_parameters["prostate_vol"], 1),
                round(st.session_state.plan_parameters["urethra_vol"], 1),
                round(st.session_state.plan_parameters["rectum_vol"], 1),
                round(st.session_state.plan_parameters["prescribed_dose"], 1),
                round(run_time, 1),
            ]
        )
        row_headings = [
            "Prostate Volume (cc)",
            "Urethra Volume (cc)",
            "Rectum Volume (cc)",
            "Prescribed Dose (Gy)",
            "Run Time (s)",
        ]
        df_nominal_data = pd.DataFrame(data=[nominal_data], columns=row_headings)
        st.session_state.df_nominal_data = df_nominal_data


else:
    if st.session_state.calcs_done != True:
            start = time.time()
            # make a copy of each structure array.
            # This is needed so the structures and plan_parameters are separated,
            # which is needed in the robust evaluation use of this function.
            copied_structures = copy_structures_cpu(st.session_state.plan_parameters)

            # calculate the nominal DVHs for each structure (dose calculation)
            all_nominal_dvhs = fast_TG43_cpu(
                copied_structures,
                st.session_state.plan_parameters,
                1,
            )
            run_time = time.time() - start

            # make the DVH plots
            plot_titles = ["Prostate", "Urethra", "Rectum"]
            fig_dvh = go.Figure()
            fig_dvh.add_trace(
                go.Scattergl(
                    x=all_nominal_dvhs[0][0, :],
                    y=all_nominal_dvhs[0][1, :],
                    mode="lines",
                    showlegend=True,
                    name=plot_titles[0],
                    # line=dict(color=colours_1[i][0],
                    #             width=st.session_state.line_size,
                    #             dash = line_style[i]
                    #             )
                )
            )

            fig_dvh.add_trace(
                go.Scattergl(
                    x=all_nominal_dvhs[1][0, :],
                    y=all_nominal_dvhs[1][1, :],
                    mode="lines",
                    showlegend=True,
                    name=plot_titles[1],
                    # line=dict(color=colours_TPS[i][0],
                    #             width=st.session_state.line_size,
                    #             dash = line_style[i]
                    #             )
                )
            )

            fig_dvh.add_trace(
                go.Scattergl(
                    x=all_nominal_dvhs[2][0, :],
                    y=all_nominal_dvhs[2][1, :],
                    mode="lines",
                    showlegend=True,
                    name=plot_titles[2],
                    # line=dict(color=colours_2[i][0],
                    #             width=st.session_state.line_size,
                    #             dash = line_style[i]
                    #             )
                )
            )

            fig_dvh.update_xaxes(
                title_text="Dose (Gy)",  # <Br>(a)",
                # title_font=dict(size=st.session_state.axis_font_size),
                minor=dict(dtick=1, showgrid=True),
                range=[0, 36],
                tick0=0,
                dtick=5,
                # title_standoff=0
            )

            fig_dvh.update_yaxes(
                title_text="Relative Volume (%)",
                # title_font=dict(size=st.session_state.axis_font_size),
                range=[0, 101],
                minor=dict(dtick=2.5, showgrid=True),
                tick0=0,
                dtick=10,
                # title_standoff=0
            )
            st.session_state.fig_dvh = fig_dvh

            df_nominal_metric_data = calculate_all_nominal_metrics_cpu(
                st.session_state.df_dose_metrics,
                st.session_state.df_volume_metrics,
                st.session_state.metric_labels_D,
                st.session_state.metric_labels_V,
                np.array([all_nominal_dvhs]),
                st.session_state.plan_parameters,
            )
            st.session_state.df_nominal_metric_data = df_nominal_metric_data

            # nominal plan information
            nominal_data = np.array(
                [
                    round(st.session_state.plan_parameters["prostate_vol"], 1),
                    round(st.session_state.plan_parameters["urethra_vol"], 1),
                    round(st.session_state.plan_parameters["rectum_vol"], 1),
                    round(st.session_state.plan_parameters["prescribed_dose"], 1),
                    round(run_time, 1),
                ]
            )
            row_headings = [
                "Prostate Volume (cc)",
                "Urethra Volume (cc)",
                "Rectum Volume (cc)",
                "Prescribed Dose (Gy)",
                "Run Time (s)",
            ]
            df_nominal_data = pd.DataFrame(data=[nominal_data], columns=row_headings)
            st.session_state.df_nominal_data = df_nominal_data
       


st.divider()
st.write("#### Nominal Plan Data")
st.dataframe(
    st.session_state.df_nominal_data, use_container_width=True, hide_index=True
)
st.divider()
st.write("#### Nominal Plan DVH metrics")
st.dataframe(
    st.session_state.df_nominal_metric_data, use_container_width=True, hide_index=True
)

st.divider()
st.write("#### Dose-Volume Plots")
st.plotly_chart(
    st.session_state.fig_dvh,
    # height = height_to_use_for_graphs,
    use_container_width=True,
)

# Display slices
if st.session_state.calcs_done != True:
    # calculating dose grid
    if st.session_state.GPU_CPU:
        ## calculate dose rate array for dwells allocated by robust optimiser

        dwell_coords_TPS_every_2nd = st.session_state.plan_parameters['dwell_coords_TPS'][
            :, 1::2
        ]  # TPS codes dwell times and points as position to stop and start moving again, so doubles all values
        dwell_pts_source_end_sup_TPS_every_2nd = (
            st.session_state.plan_parameters['dwell_pts_source_end_sup_TPS'][:, 1::2]
        )
        dwell_pts_source_end_inf_TPS_every_2nd = (
            st.session_state.plan_parameters['dwell_pts_source_end_inf_TPS'][:, 1::2]
        )

        dose_per_dwell_per_vol_pt_TPS, dose_calc_pts = get_dose_grid_gpu(
            st.session_state.plan_parameters,
            cp.array(dwell_coords_TPS_every_2nd),
            cp.array(dwell_pts_source_end_sup_TPS_every_2nd),
            cp.array(dwell_pts_source_end_inf_TPS_every_2nd),
            voxel_size=1,
        )

    else:
        ## calculate dose rate array for dwells allocated by robust optimiser
        dwell_coords_TPS_every_2nd = st.session_state.plan_parameters['dwell_coords_TPS'][
            :, 1::2
        ]  # TPS codes dwell times and points as position to stop and start moving again, so doubles all values
        dwell_pts_source_end_sup_TPS_every_2nd = (
            st.session_state.plan_parameters['dwell_pts_source_end_sup_TPS'][:, 1::2]
        )
        dwell_pts_source_end_inf_TPS_every_2nd = (
            st.session_state.plan_parameters['dwell_pts_source_end_inf_TPS'][:, 1::2]
        )

        dose_per_dwell_per_vol_pt_TPS, dose_calc_pts = get_dose_grid_cpu(
            st.session_state.plan_parameters,
            np.array(dwell_coords_TPS_every_2nd),
            np.array(dwell_pts_source_end_sup_TPS_every_2nd),
            np.array(dwell_pts_source_end_inf_TPS_every_2nd),
            voxel_size=1,
        )

    # send variables to the browser memory
    st.session_state.dose_per_dwell_per_vol_pt_TPS = dose_per_dwell_per_vol_pt_TPS
    st.session_state.dose_calc_pts = dose_calc_pts

    ## load dose rate array for dwells

    # flatten dwell times and remove place holder values "-100"
    dwell_times_TPS = st.session_state.plan_parameters['dwell_times_TPS'][:, 1::2].flatten()
    dwell_times_TPS = dwell_times_TPS[dwell_times_TPS >= 0]

    # calculate 3D dose array (this is for the contour plots)
    dose_per_vol_pt = np.einsum(
        "i,ij -> j", dwell_times_TPS, st.session_state.dose_per_dwell_per_vol_pt_TPS
    )

    # truncate values to 50% to 200% of precribed dose, what we are interested in
    dose_per_vol_pt[dose_per_vol_pt > st.session_state.plan_parameters['prescribed_dose'] * 2] = (
        st.session_state.plan_parameters['prescribed_dose'] * 2.01
    )
    dose_per_vol_pt[dose_per_vol_pt < st.session_state.plan_parameters['prescribed_dose'] * 0.50] = 0

    custom_color_scale = [
        # [  0  , 'rgba(0,     0,   0,   0)' ], # 0% of prescribed dose
        [50, "rgba(0,    25, 255, 0.1)"],  # 50% of prescribed dose
        [75, "rgba(0,   152, 255, 0.3)"],  # 75% of prescribed dose
        [90, "rgba(26,  214, 192, 0.3)"],  # 90% of prescribed dose
        [100, "rgba(44,  255, 150, 0.3)"],  # 100% of prescribed dose
        [110, "rgba(87,  255,  90, 0.3)"],  # 110% of prescribed dose
        [150, "rgba(255, 234,   0, 0.3)"],  # 150% of prescribed dose
        [200, "rgba(255,   0,   0, 0.3)"],  # 200% of prescribed dose
    ]

    coloring_type = ["fill", "heatmap", "lines", "none"]
    line_width = 4
    if "line_width" not in st.session_state:
        st.session_state.line_width = line_width
    else:
        st.session_state.line_width = line_width

    # get x, y, and z vectors for axis
    x_vector = np.unique(dose_calc_pts[:, 0])
    y_vector = np.unique(dose_calc_pts[:, 1])
    z_vector = np.flip(np.unique(dose_calc_pts[:, 2]))
    dose_per_vol_pt = (
        dose_per_vol_pt.reshape(len(y_vector), len(x_vector), len(z_vector))
        .swapaxes(2, 0)
        .swapaxes(1, 2)
    )

    # moves arrays back to cpu if they are in the gpu so isodose plots can be generated
    try:
        dose_per_vol_pt = dose_per_vol_pt.get()
    except:
        pass
    try:
        x_vector = x_vector.get()
    except:
        pass
    try:
        y_vector = y_vector.get()
    except:
        pass
    try:
        z_vector = z_vector.get()
    except:
        pass
    try:
        prostate_contour_pts = st.session_state.plan_parameters['prostate_contour_pts'].get()
    except:
        prostate_contour_pts = st.session_state.plan_parameters['prostate_contour_pts']
    try:
        urethra_contour_pts = st.session_state.plan_parameters['urethra_contour_pts'].get()
    except:
        urethra_contour_pts = st.session_state.plan_parameters['urethra_contour_pts']
    try:
        rectum_contour_pts = st.session_state.plan_parameters['rectum_contour_pts'].get()
    except:
        rectum_contour_pts = st.session_state.plan_parameters['rectum_contour_pts']

    # generating the isodose plots
    data_fig_isodoses_TPS = []
    for slice_to_view in range(len(z_vector)):
        slice_to_view_z = z_vector[slice_to_view]
        data_fig_temp = []
        for i in range(len(custom_color_scale)):
            data_fig_temp.append(
                go.Contour(
                    z=dose_per_vol_pt[slice_to_view]
                    / st.session_state.plan_parameters['prescribed_dose']
                    * 100,
                    x=x_vector,
                    y=y_vector,
                    name=str(custom_color_scale[i][0]) + "%",
                    # colorscale = custom_color_scale,
                    fillcolor=custom_color_scale[i][1],
                    contours=dict(
                        coloring=coloring_type[2],
                        type="constraint",
                        operation="<",
                        value=[custom_color_scale[i][0]],
                        showlabels=True,  # show labels on contours
                        labelfont=dict(  # label font properties
                            size=14,
                            color="black",
                        ),
                    ),
                )
            )

        strut_names = ["Prostate", "Urethra", "Rectum"]
        strut_color = ["#0D2A63", "#FA0087", "rgb(217,95,2)"]

        for s, strut in enumerate(
            [prostate_contour_pts, urethra_contour_pts, rectum_contour_pts]
        ):
            if (
                slice_to_view_z > strut[:, 2, 0].min()
                and slice_to_view_z < strut[:, 2, 0].max()
            ):
                data_fig_temp.append(
                    go.Scattergl(
                        x=np.array(
                            [
                                *np.array(strut)[
                                    strut[:, 2, 0] == slice_to_view_z, 0, :
                                ].flatten(),
                                np.array(strut)[
                                    strut[:, 2, 0] == slice_to_view_z, 0, :
                                ].flatten()[0],
                            ]
                        ),
                        y=np.array(
                            [
                                *np.array(strut)[
                                    strut[:, 2, 0] == slice_to_view_z, 1, :
                                ].flatten(),
                                np.array(strut)[
                                    strut[:, 2, 0] == slice_to_view_z, 1, :
                                ].flatten()[0],
                            ]
                        ),
                        mode="lines",
                        showlegend=True,
                        line=dict(width=line_width, color=strut_color[s]),
                        name=strut_names[s],
                    )
                )

        data_fig_isodoses_TPS.append(data_fig_temp)

    slice_to_view = int(len(data_fig_isodoses_TPS) / 2)

    st.session_state.slice_to_view = slice_to_view

    st.session_state.data_fig_isodoses_TPS = data_fig_isodoses_TPS
    titles_fig_isodoses_TPS = []
    for i in range(len(data_fig_isodoses_TPS)):
        titles_fig_isodoses_TPS.append(
            "TPS Plan: Superior-Inferior slice (z = " + str(z_vector[i]) + " mm)"
        )  # layout attribute

    st.session_state.titles_fig_isodoses_TPS = titles_fig_isodoses_TPS
    st.session_state.x_vector = x_vector
    st.session_state.y_vector = y_vector
    st.session_state.z_vector = z_vector
    st.session_state.calcs_done = True


st.divider()
st.write("#### Isodose of Nominal Treatment Plan")
# displaying the isodose plot and the controls
quick_slice_view = np.percentile(
    np.arange(len(st.session_state.z_vector)),
    [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    method="closest_observation",
)
(
    col_1,
    col_2,
    col_3,
    col_4,
    col_5,
    col_6,
    col_7,
    col_8,
    col_9,
    col_10,
    col_11,
) = st.columns([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

with col_1:
    if st.button(str(st.session_state.z_vector[quick_slice_view[0]]) + " mm"):
        st.session_state.slice_to_view = quick_slice_view[0]

with col_2:
    if st.button(str(st.session_state.z_vector[quick_slice_view[1]]) + " mm"):
        st.session_state.slice_to_view = quick_slice_view[1]

with col_3:
    if st.button(str(st.session_state.z_vector[quick_slice_view[2]]) + " mm"):
        st.session_state.slice_to_view = quick_slice_view[2]

with col_4:
    if st.button(str(st.session_state.z_vector[quick_slice_view[3]]) + " mm"):
        st.session_state.slice_to_view = quick_slice_view[3]

with col_5:
    if st.button(str(st.session_state.z_vector[quick_slice_view[4]]) + " mm"):
        st.session_state.slice_to_view = quick_slice_view[4]

with col_6:
    if st.button(str(st.session_state.z_vector[quick_slice_view[5]]) + " mm"):
        st.session_state.slice_to_view = quick_slice_view[5]

with col_7:
    if st.button(str(st.session_state.z_vector[quick_slice_view[6]]) + " mm"):
        st.session_state.slice_to_view = quick_slice_view[6]

with col_8:
    if st.button(str(st.session_state.z_vector[quick_slice_view[7]]) + " mm"):
        st.session_state.slice_to_view = quick_slice_view[7]

with col_9:
    if st.button(str(st.session_state.z_vector[quick_slice_view[8]]) + " mm"):
        st.session_state.slice_to_view = quick_slice_view[8]

with col_10:
    if st.button(str(st.session_state.z_vector[quick_slice_view[9]]) + " mm"):
        st.session_state.slice_to_view = quick_slice_view[9]

with col_11:
    if st.button(str(st.session_state.z_vector[quick_slice_view[10]]) + " mm"):
        st.session_state.slice_to_view = quick_slice_view[10]

col1, col2, col3, col4, col5 = st.columns([2, 1, 2, 1, 2])
skip = False
with col2:
    if st.button("Previous Slice"):
        st.session_state.slice_to_view -= 1
        skip = True
        if st.session_state.slice_to_view < 0:
            st.session_state.slice_to_view = 0
            st.write("No more slices Superiorly")


with col4:
    if st.button("Next Slice"):
        st.session_state.slice_to_view += 1
        skip = True
        if st.session_state.slice_to_view >= len(
            st.session_state.titles_fig_isodoses_TPS
        ):
            st.session_state.slice_to_view = (
                len(st.session_state.titles_fig_isodoses_TPS) - 1
            )
            st.write("No more slices Inferiorly")

with col3:
    jump_to_slice = st.text_input(
        "Got to slice (mm):",
        str(st.session_state.z_vector[st.session_state.slice_to_view]),
    )

    if skip != True:
        try:
            st.session_state.slice_to_view = float(jump_to_slice)
            if st.session_state.slice_to_view <= np.min(st.session_state.z_vector):
                st.session_state.slice_to_view = (
                    len(st.session_state.titles_fig_isodoses_TPS) - 1
                )
                st.write("No more slices Inferiorly")

            elif st.session_state.slice_to_view > np.max(st.session_state.z_vector):
                st.session_state.slice_to_view = 0
                st.write("No more slices Superiorly")
            else:
                st.session_state.slice_to_view = np.abs(
                    np.array(
                        (st.session_state.z_vector - st.session_state.slice_to_view)
                    )
                ).argmin()

        except:
            st.write("Number needs to by a float. (did you include mm?)")
            st.session_state.slice_to_view = int(
                len(st.session_state.titles_fig_isodoses_TPS) / 2
            )


fig_isodoses_to_show_TPS = go.Figure(
    data=st.session_state.data_fig_isodoses_TPS[st.session_state.slice_to_view],
    layout=go.Layout(
        title=go.layout.Title(
            text=st.session_state.titles_fig_isodoses_TPS[
                st.session_state.slice_to_view
            ],
            font=dict(
                # family="Courier New, monospace",
                size=st.session_state.title_font_size,
                # color="RebeccaPurple"
            ),
        )
    ),
)

if st.session_state.dont_show_rectum:
    fig_isodoses_to_show_TPS.update_traces(
        visible="legendonly", selector=dict(name="Rectum")
    )

if st.session_state.dont_show_urethra:
    fig_isodoses_to_show_TPS.update_traces(
        visible="legendonly", selector=dict(name="Urethra")
    )

if st.session_state.dont_show_prostate:
    fig_isodoses_to_show_TPS.update_traces(
        visible="legendonly", selector=dict(name="Prostate")
    )

if st.session_state.dont_show_50:
    fig_isodoses_to_show_TPS.update_traces(
        visible="legendonly", selector=dict(name="50%")
    )

if st.session_state.dont_show_75:
    fig_isodoses_to_show_TPS.update_traces(
        visible="legendonly", selector=dict(name="75%")
    )

if st.session_state.dont_show_90:
    fig_isodoses_to_show_TPS.update_traces(
        visible="legendonly", selector=dict(name="90%")
    )

if st.session_state.dont_show_100:
    fig_isodoses_to_show_TPS.update_traces(
        visible="legendonly", selector=dict(name="100%")
    )

if st.session_state.dont_show_110:
    fig_isodoses_to_show_TPS.update_traces(
        visible="legendonly", selector=dict(name="110%")
    )

if st.session_state.dont_show_150:
    fig_isodoses_to_show_TPS.update_traces(
        visible="legendonly", selector=dict(name="150%")
    )

if st.session_state.dont_show_200:
    fig_isodoses_to_show_TPS.update_traces(
        visible="legendonly", selector=dict(name="200%")
    )

fig_isodoses_to_show_TPS.update_xaxes(
    title_text="Left-Right (x) (mm)",
    title_font=dict(size=st.session_state.axis_font_size),
    minor=dict(dtick=2.5, showgrid=True),
    range=[st.session_state.x_vector[0], st.session_state.x_vector[-1]],
    tick0=0,
    dtick=5,
    # title_standoff=0
)

fig_isodoses_to_show_TPS.update_yaxes(
    title_text="Posterior-Anterior (y) (mm)",
    title_font=dict(size=st.session_state.axis_font_size),
    range=[st.session_state.y_vector[-1], st.session_state.y_vector[0]],
    minor=dict(dtick=2.5, showgrid=True),
    tick0=0,
    dtick=5,
    # autorange="reversed"
)
fig_isodoses_to_show_TPS.update_traces(
    line=dict(width=st.session_state.line_width), selector=dict(type="scattergl")
)
fig_isodoses_to_show_TPS.update_traces(
    line=dict(width=st.session_state.line_width), selector=dict(type="contour")
)
fig_isodoses_to_show_TPS.update_traces(showscale=False, selector=dict(type="heatmap"))

fig_isodoses_to_show_TPS.update_layout(
    height=height_to_use_for_graphs,  # width = 800,
    legend=dict(
        font=dict(  # family = "Courier",
            size=st.session_state.legend_font_size,
            # color = "black"
        )
    ),
)
st.plotly_chart(
    fig_isodoses_to_show_TPS, height=height_to_use_for_graphs, use_container_width=True
)

col_1, col_2, col_3 = st.columns([2, 1, 5])
with col_1:
    st.write("Turn off the following in all plots:")

with col_2:
    if st.button("Reload Page"):
        reload_page = True

col_1, col_2, col_3, col_4, col_5, col_6, col_7, col_8, col_9, col_10 = st.columns(
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
)

with col_1:
    st.session_state.dont_show_50 = st.checkbox("50%")

with col_2:
    st.session_state.dont_show_75 = st.checkbox("75%")

with col_3:
    st.session_state.dont_show_90 = st.checkbox("90%")

with col_4:
    st.session_state.dont_show_100 = st.checkbox("100%")

with col_5:
    st.session_state.dont_show_110 = st.checkbox("110%")

with col_6:
    st.session_state.dont_show_150 = st.checkbox("150%")

with col_7:
    st.session_state.dont_show_200 = st.checkbox("200%")

with col_8:
    st.session_state.dont_show_prostate = st.checkbox("Prostate")

with col_9:
    st.session_state.dont_show_urethra = st.checkbox("Urethra")

with col_10:
    st.session_state.dont_show_rectum = st.checkbox("Rectum")

if reload_page == True:
    st.experimental_rerun()
