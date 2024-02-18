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
import math
import plotly.graph_objects as go

try:
    import cupy as cp
except:
    print("no cupy")

try:
    st.session_state.dwell_times_pareto_front = (
        st.session_state.dwell_times_pareto_front.get()
    )
except:
    pass
try:
    st.session_state.dose_calc_pts = st.session_state.dose_calc_pts.get()
except:
    pass
try:
    st.session_state.dose_per_dwell_per_vol_pt = (
        st.session_state.dose_per_dwell_per_vol_pt.get()
    )
except:
    pass
try:
    st.session_state.plan_parameters['prostate_contour_pts'] = st.session_state.plan_parameters['prostate_contour_pts'].get()
except:
    pass
try:
    st.session_state.plan_parameters['urethra_contour_pts'] = st.session_state.plan_parameters['urethra_contour_pts'].get()
except:
    pass
try:
    st.session_state.plan_parameters['rectum_contour_pts'] = st.session_state.plan_parameters['rectum_contour_pts'].get()
except:
    pass
try:
    st.session_state.plan_parameters['dwell_times_TPS'] = st.session_state.plan_parameters['dwell_times_TPS'].get()
except:
    pass
try:
    st.session_state.dose_per_dwell_per_vol_pt_TPS = (
        st.session_state.dose_per_dwell_per_vol_pt_TPS.get()
    )
except:
    pass

def color_coding(row, indexs_of_rows):
    return (
        ["background-color:yellow"] * len(row)
        if row["Plan"] in indexs_of_rows
        else ["background-color:white"] * len(row)
    )

st.set_page_config(page_title="Step 1: Select Best Plan", layout="wide")

st.markdown("# Step 3 a) Select Best Plan")
st.sidebar.header("Step 3 a) Select Best Plan")

st.write(
    """
    1) Use the sliders to limit the range of shown plans in the table. 
    
    2) The first column is the plan number, place the best plans' number you want to select into the text box. You can compare upto 3 plans at a time.
       
    3) Click "Load Selected Plan Data".
       
    4) Then click on the '3 b) Isodose Plots' page to check isodose plots or go to '3 c) DVH Plots' to compare DVH plots.
    """
)

col1, col2, col3, col4 = st.columns([1, 4, 1, 4])

height_to_use_for_graphs = int(((30 + 1) * 35 + 3))

show_3D_graph = st.sidebar.checkbox("Show 3D graph of Pareto front")

if st.sidebar.button("Reset all selected plans"):
    st.session_state.selected_index = []

    if "selected_plan" in st.session_state:
        st.session_state.selected_plan = "None"
        st.session_state.first_plan_loaded = False

    if "selected_plan_2" in st.session_state:
        st.session_state.selected_plan_2 = "None"
        st.session_state.second_plan_loaded = False

    if "selected_plan_3" in st.session_state:
        st.session_state.selected_plan_3 = "None"
        st.session_state.third_plan_loaded = False

selected_plan = st.sidebar.text_input(
    "Select a plan by typing the 'Plan' number:", "None"
)
if "selected_plan" not in st.session_state:
    st.session_state.selected_plan = selected_plan
else:
    st.session_state.selected_plan = selected_plan

st.sidebar.write("The current selected plan is ", st.session_state.selected_plan)


if st.sidebar.button("Load Selected Plan Data"):
    st.sidebar.warning("loading data may take a few minutes!")
    # load data into session state

    ### load isodose plot first ###
    ## select dwell times
    dwell_times_RO = st.session_state.dwell_times_pareto_front[
        int(st.session_state.selected_plan)
    ]

    ## load dose calc points
    dose_calc_pts = st.session_state.dose_calc_pts

    ## load dose rate array for dwells allocated by robust optimiser
    dose_per_dwell_per_vol_pt = st.session_state.dose_per_dwell_per_vol_pt

    # flatten dwell times and remove place holder values "-100"
    dwell_times_RO = dwell_times_RO.flatten()
    dwell_times_RO = dwell_times_RO[dwell_times_RO != -100]

    # calculate 3D dose array (this is for the contour plots)
    dose_per_vol_pt = np.einsum("i,ij -> j", dwell_times_RO, dose_per_dwell_per_vol_pt)

    # truncate values to 50% to 200% of precribed dose, what we are interested in
    dose_per_vol_pt[
        dose_per_vol_pt > st.session_state.plan_parameters["prescribed_dose"] * 2
    ] = (st.session_state.plan_parameters["prescribed_dose"] * 2.01)
    dose_per_vol_pt[
        dose_per_vol_pt < st.session_state.plan_parameters["prescribed_dose"] * 0.50
    ] = 0

    # get x, y, and z vectors for axis
    x_vector = np.unique(dose_calc_pts[:, 0])
    y_vector = np.unique(dose_calc_pts[:, 1])
    z_vector = np.flip(np.unique(dose_calc_pts[:, 2]))
    dose_per_vol_pt = (
        dose_per_vol_pt.reshape(len(y_vector), len(x_vector), len(z_vector))
        .swapaxes(2, 0)
        .swapaxes(1, 2)
    )

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

    data_fig_isodoses = []
    for slice_to_view in range(len(z_vector)):
        slice_to_view_z = z_vector[slice_to_view]
        data_fig_temp = []
        for i in range(len(custom_color_scale)):
            data_fig_temp.append(
                go.Contour(
                    z=dose_per_vol_pt[slice_to_view]
                    / st.session_state.plan_parameters["prescribed_dose"]
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
            # num_of_traces +=1

        strut_names = ["Prostate", "Urethra", "Rectum"]
        strut_color = ["#0D2A63", "#FA0087", "rgb(217,95,2)"]
        for s, strut in enumerate(
            [
                st.session_state.plan_parameters['prostate_contour_pts'],
                st.session_state.plan_parameters['urethra_contour_pts'],
                st.session_state.plan_parameters['rectum_contour_pts'],
            ]
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

            # num_of_traces +=1

        data_fig_isodoses.append(data_fig_temp)

    titles_fig_isodoses = []
    for i in range(len(data_fig_isodoses)):
        titles_fig_isodoses.append(
            "Robust Plan "
            + str(st.session_state.selected_plan)
            + ": Superior-Inferior slice (z = "
            + str(z_vector[i])
            + " mm)"
        )  # layout attribute

    slice_to_view = int(len(data_fig_isodoses) / 2)

    if "slice_to_view" not in st.session_state:
        st.session_state.slice_to_view = slice_to_view
    else:
        st.session_state.slice_to_view = slice_to_view

    if "data_fig_isodoses" not in st.session_state:
        st.session_state.data_fig_isodoses = data_fig_isodoses
    else:
        st.session_state.data_fig_isodoses = data_fig_isodoses

    if "titles_fig_isodoses" not in st.session_state:
        st.session_state.titles_fig_isodoses = titles_fig_isodoses
    else:
        st.session_state.titles_fig_isodoses = titles_fig_isodoses

    jump_to_slice_old = -1.0
    if "jump_to_slice_old" not in st.session_state:
        st.session_state.jump_to_slice_old = jump_to_slice_old
    else:
        st.session_state.jump_to_slice_old = jump_to_slice_old

    if "x_vector" not in st.session_state:
        st.session_state.x_vector = x_vector
    else:
        st.session_state.x_vector = x_vector

    if "y_vector" not in st.session_state:
        st.session_state.y_vector = y_vector
    else:
        st.session_state.y_vector = y_vector

    if "z_vector" not in st.session_state:
        st.session_state.z_vector = z_vector
    else:
        st.session_state.z_vector = z_vector

    #### loading TPS

    ## load dose rate array for dwells allocated by robust optimiser

    # flatten dwell times and remove place holder values "-100"
    dwell_times_TPS = st.session_state.plan_parameters['dwell_times_TPS'][:, 1::2].flatten()
    dwell_times_TPS = dwell_times_TPS[dwell_times_TPS >= 0]

    # calculate 3D dose array (this is for the contour plots)
    dose_per_vol_pt = np.einsum(
        "i,ij -> j", dwell_times_TPS, st.session_state.dose_per_dwell_per_vol_pt_TPS
    )

    # truncate values to 50% to 200% of precribed dose, what we are interested in
    dose_per_vol_pt[
        dose_per_vol_pt > st.session_state.plan_parameters["prescribed_dose"] * 2
    ] = (st.session_state.plan_parameters["prescribed_dose"] * 2.01)
    dose_per_vol_pt[
        dose_per_vol_pt < st.session_state.plan_parameters["prescribed_dose"] * 0.50
    ] = 0

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

    data_fig_isodoses_TPS = []
    for slice_to_view in range(len(z_vector)):
        slice_to_view_z = z_vector[slice_to_view]
        data_fig_temp = []
        for i in range(len(custom_color_scale)):
            data_fig_temp.append(
                go.Contour(
                    z=dose_per_vol_pt[slice_to_view]
                    / st.session_state.plan_parameters["prescribed_dose"]
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
            [
                st.session_state.plan_parameters['prostate_contour_pts'],
                st.session_state.plan_parameters['urethra_contour_pts'],
                st.session_state.plan_parameters['rectum_contour_pts'],
            ]
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

    titles_fig_isodoses_TPS = []
    for i in range(len(data_fig_isodoses)):
        titles_fig_isodoses_TPS.append(
            "TPS Plan: Superior-Inferior slice (z = " + str(z_vector[i]) + " mm)"
        )  # layout attribute

    if "data_fig_isodoses_TPS" not in st.session_state:
        st.session_state.data_fig_isodoses_TPS = data_fig_isodoses_TPS
    else:
        st.session_state.data_fig_isodoses_TPS = data_fig_isodoses_TPS

    if "titles_fig_isodoses_TPS" not in st.session_state:
        st.session_state.titles_fig_isodoses_TPS = titles_fig_isodoses_TPS
    else:
        st.session_state.titles_fig_isodoses_TPS = titles_fig_isodoses_TPS

    # turn on all plots to start with
    dont_show_rectum = (
        dont_show_urethra
    ) = (
        dont_show_prostate
    ) = (
        dont_show_200
    ) = (
        dont_show_150
    ) = (
        dont_show_110
    ) = dont_show_100 = dont_show_90 = dont_show_75 = dont_show_50 = False

    if "dont_show_rectum" not in st.session_state:
        st.session_state.dont_show_rectum = dont_show_rectum
    else:
        st.session_state.dont_show_rectum = dont_show_rectum

    if "dont_show_urethra" not in st.session_state:
        st.session_state.dont_show_urethra = dont_show_urethra
    else:
        st.session_state.dont_show_urethra = dont_show_urethra

    if "dont_show_prostate" not in st.session_state:
        st.session_state.dont_show_prostate = dont_show_prostate
    else:
        st.session_state.dont_show_prostate = dont_show_prostate

    if "dont_show_200" not in st.session_state:
        st.session_state.dont_show_200 = dont_show_200
    else:
        st.session_state.dont_show_200 = dont_show_200

    if "dont_show_150" not in st.session_state:
        st.session_state.dont_show_150 = dont_show_150
    else:
        st.session_state.dont_show_150 = dont_show_150

    if "dont_show_110" not in st.session_state:
        st.session_state.dont_show_110 = dont_show_110
    else:
        st.session_state.dont_show_110 = dont_show_110

    if "dont_show_100" not in st.session_state:
        st.session_state.dont_show_100 = dont_show_100
    else:
        st.session_state.dont_show_100 = dont_show_100

    if "dont_show_90" not in st.session_state:
        st.session_state.dont_show_90 = dont_show_90
    else:
        st.session_state.dont_show_90 = dont_show_90

    if "dont_show_75" not in st.session_state:
        st.session_state.dont_show_75 = dont_show_75
    else:
        st.session_state.dont_show_75 = dont_show_75

    if "dont_show_50" not in st.session_state:
        st.session_state.dont_show_50 = dont_show_50
    else:
        st.session_state.dont_show_50 = dont_show_50

    # write update
    st.sidebar.write(
        "The selected plan ", st.session_state.selected_plan, " has been loaded."
    )
    st.session_state.first_plan_loaded = True
    st.session_state.selected_index.append(int(st.session_state.selected_plan))

    if st.session_state.old_selected_1 in st.session_state.selected_index:
        st.session_state.selected_index.remove(st.session_state.old_selected_1)
    st.session_state.old_selected_1 = int(st.session_state.selected_plan)


else:
    if st.session_state.first_plan_loaded != True:
        st.sidebar.write("No Plan loaded yet!")
    else:
        st.sidebar.write(
            "The selected plan ", st.session_state.selected_plan, " has been loaded."
        )

if st.session_state.first_plan_loaded == True:
    st.session_state.compare_2nd_plan = st.sidebar.checkbox("Compare a 2nd Plan")

    if st.session_state.compare_2nd_plan:
        selected_plan_2 = st.sidebar.text_input(
            "Select a 2nd plan by typing the 'Plan' number :", "None"
        )

        if "selected_plan_2" not in st.session_state:
            st.session_state.selected_plan_2 = selected_plan_2
        else:
            st.session_state.selected_plan_2 = selected_plan_2

        st.sidebar.write(
            "The current 2nd selected plan is ", st.session_state.selected_plan_2
        )

        if st.sidebar.button("Load 2nd Selected Plan Data"):
            # load data into session state
            # st.sidebar.warning("loading data may take a few minutes!")
            ### load isodose plot first ###
            ## select dwell times
            dwell_times_RO = st.session_state.dwell_times_pareto_front[
                int(st.session_state.selected_plan_2)
            ]

            ## load dose calc points
            dose_calc_pts = st.session_state.dose_calc_pts

            ## load dose rate array for dwells allocated by robust optimiser
            dose_per_dwell_per_vol_pt = st.session_state.dose_per_dwell_per_vol_pt

            # flatten dwell times and remove place holder values "-100"
            dwell_times_RO = dwell_times_RO.flatten()
            dwell_times_RO = dwell_times_RO[dwell_times_RO != -100]

            # calculate 3D dose array (this is for the contour plots)
            dose_per_vol_pt = np.einsum(
                "i,ij -> j", dwell_times_RO, dose_per_dwell_per_vol_pt
            )

            # truncate values to 50% to 200% of precribed dose, what we are interested in
            dose_per_vol_pt[
                dose_per_vol_pt
                > st.session_state.plan_parameters["prescribed_dose"] * 2
            ] = (st.session_state.plan_parameters["prescribed_dose"] * 2.01)
            dose_per_vol_pt[
                dose_per_vol_pt
                < st.session_state.plan_parameters["prescribed_dose"] * 0.50
            ] = 0

            # get x, y, and z vectors for axis
            x_vector = np.unique(dose_calc_pts[:, 0])
            y_vector = np.unique(dose_calc_pts[:, 1])
            z_vector = np.flip(np.unique(dose_calc_pts[:, 2]))
            dose_per_vol_pt = (
                dose_per_vol_pt.reshape(len(y_vector), len(x_vector), len(z_vector))
                .swapaxes(2, 0)
                .swapaxes(1, 2)
            )
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
            # fig_isodose = go.Figure()
            data_fig_isodoses_2 = []
            for slice_to_view in range(len(z_vector)):
                slice_to_view_z = z_vector[slice_to_view]
                data_fig_temp = []
                for i in range(len(custom_color_scale)):
                    data_fig_temp.append(
                        go.Contour(
                            z=dose_per_vol_pt[slice_to_view]
                            / st.session_state.plan_parameters["prescribed_dose"]
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
                    # num_of_traces +=1

                strut_names = ["Prostate", "Urethra", "Rectum"]
                strut_color = ["#0D2A63", "#FA0087", "rgb(217,95,2)"]
                for s, strut in enumerate(
                    [
                        st.session_state.plan_parameters['prostate_contour_pts'],
                        st.session_state.plan_parameters['urethra_contour_pts'],
                        st.session_state.plan_parameters['rectum_contour_pts'],
                    ]
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



                data_fig_isodoses_2.append(data_fig_temp)

            titles_fig_isodoses_2 = []
            for i in range(len(data_fig_isodoses_2)):
                titles_fig_isodoses_2.append(
                    "Robust Plan "
                    + str(st.session_state.selected_plan_2)
                    + ": Superior-Inferior slice (z = "
                    + str(z_vector[i])
                    + " mm)"
                )  # layout attribute

            if "data_fig_isodoses_2" not in st.session_state:
                st.session_state.data_fig_isodoses_2 = data_fig_isodoses_2
            else:
                st.session_state.data_fig_isodoses_2 = data_fig_isodoses_2

            if "titles_fig_isodoses_2" not in st.session_state:
                st.session_state.titles_fig_isodoses_2 = titles_fig_isodoses_2
            else:
                st.session_state.titles_fig_isodoses_2 = titles_fig_isodoses_2

            # write update
            st.sidebar.write(
                "The 2nd selected plan ",
                st.session_state.selected_plan_2,
                " has been loaded.",
            )
            st.session_state.second_plan_loaded = True
            st.session_state.selected_index.append(
                int(st.session_state.selected_plan_2)
            )
            if st.session_state.old_selected_2 in st.session_state.selected_index:
                st.session_state.selected_index.remove(st.session_state.old_selected_2)

            st.session_state.old_selected_2 = int(st.session_state.selected_plan_2)

        else:
            if st.session_state.second_plan_loaded == False:
                st.sidebar.write("No Plan loaded yet!")
            else:
                st.sidebar.write(
                    "The 2nd selected plan ",
                    st.session_state.selected_plan_2,
                    " has been loaded.",
                )

if st.session_state.second_plan_loaded == True:
    st.session_state.compare_3rd_plan = st.sidebar.checkbox("Compare a 3rd Plan")

    if st.session_state.compare_3rd_plan:
        selected_plan_3 = st.sidebar.text_input(
            "Select a 3rd plan by typing the 'Plan' number :", "None"
        )
        if "selected_plan_3" not in st.session_state:
            st.session_state.selected_plan_3 = selected_plan_3
        else:
            st.session_state.selected_plan_3 = selected_plan_3

        st.sidebar.write(
            "The current 3rd selected plan ", st.session_state.selected_plan_3
        )

        if st.sidebar.button("Load 3rd Selected Plan Data"):
            st.sidebar.warning("loading data may take a few minutes!")
            # load data into session state

            ### load isodose plot first ###
            ## select dwell times
            dwell_times_RO = st.session_state.dwell_times_pareto_front[
                int(st.session_state.selected_plan_3)
            ]

            ## load dose calc points
            dose_calc_pts = st.session_state.dose_calc_pts

            ## load dose rate array for dwells allocated by robust optimiser
            dose_per_dwell_per_vol_pt = st.session_state.dose_per_dwell_per_vol_pt

            # flatten dwell times and remove place holder values "-100"
            dwell_times_RO = dwell_times_RO.flatten()
            dwell_times_RO = dwell_times_RO[dwell_times_RO != -100]

            # calculate 3D dose array (this is for the contour plots)
            dose_per_vol_pt = np.einsum(
                "i,ij -> j", dwell_times_RO, dose_per_dwell_per_vol_pt
            )

            # truncate values to 50% to 200% of precribed dose, what we are interested in
            dose_per_vol_pt[
                dose_per_vol_pt
                > st.session_state.plan_parameters["prescribed_dose"] * 2
            ] = (st.session_state.plan_parameters["prescribed_dose"] * 2.01)
            dose_per_vol_pt[
                dose_per_vol_pt
                < st.session_state.plan_parameters["prescribed_dose"] * 0.50
            ] = 0

            # get x, y, and z vectors for axis
            x_vector = np.unique(dose_calc_pts[:, 0])
            y_vector = np.unique(dose_calc_pts[:, 1])
            z_vector = np.flip(np.unique(dose_calc_pts[:, 2]))
            dose_per_vol_pt = (
                dose_per_vol_pt.reshape(len(y_vector), len(x_vector), len(z_vector))
                .swapaxes(2, 0)
                .swapaxes(1, 2)
            )
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

            data_fig_isodoses_3 = []
            for slice_to_view in range(len(z_vector)):
                slice_to_view_z = z_vector[slice_to_view]
                data_fig_temp = []
                for i in range(len(custom_color_scale)):
                    data_fig_temp.append(
                        go.Contour(
                            z=dose_per_vol_pt[slice_to_view]
                            / st.session_state.plan_parameters["prescribed_dose"]
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
                    [
                        st.session_state.plan_parameters['prostate_contour_pts'],
                        st.session_state.plan_parameters['urethra_contour_pts'],
                        st.session_state.plan_parameters['rectum_contour_pts'],
                    ]
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

                data_fig_isodoses_3.append(data_fig_temp)

            titles_fig_isodoses_3 = []
            for i in range(len(data_fig_isodoses_3)):
                titles_fig_isodoses_3.append(
                    "Robust Plan "
                    + str(st.session_state.selected_plan_3)
                    + ": Superior-Inferior slice (z = "
                    + str(z_vector[i])
                    + " mm)"
                )  # layout attribute

            if "data_fig_isodoses_3" not in st.session_state:
                st.session_state.data_fig_isodoses_3 = data_fig_isodoses_3
            else:
                st.session_state.data_fig_isodoses_3 = data_fig_isodoses_3

            if "titles_fig_isodoses_3" not in st.session_state:
                st.session_state.titles_fig_isodoses_3 = titles_fig_isodoses_3
            else:
                st.session_state.titles_fig_isodoses_3 = titles_fig_isodoses_3

            # write update
            st.sidebar.write(
                "The selected plan ",
                st.session_state.selected_plan_3,
                " has been loaded.",
            )
            st.session_state.third_plan_loaded = True
            st.session_state.selected_index.append(
                int(st.session_state.selected_plan_3)
            )
            if st.session_state.old_selected_3 in st.session_state.selected_index:
                st.session_state.selected_index.remove(st.session_state.old_selected_3)

            st.session_state.old_selected_3 = int(st.session_state.selected_plan_3)
            print(st.session_state.selected_index)
        else:
            if st.session_state.third_plan_loaded == False:
                st.sidebar.write("No Plan loaded yet!")
            else:
                st.sidebar.write(
                    "The selected plan ",
                    st.session_state.selected_plan_3,
                    " has been loaded.",
                )

if show_3D_graph:
    col1, col2, col3, col4 = st.columns([1, 4, 1, 4])
    with col3:
        # Overall Passrate
        min_ = float(
            math.floor(st.session_state.all_metrics["All Pass-rate (%)"].min())
        )
        max_ = float(math.ceil(st.session_state.all_metrics["All Pass-rate (%)"].max()))

        if min_ == max_:
            Model_RE_All = st.slider(
                "Overall Pass-rate (%):",
                min_value=min_,
                max_value=min_ + 1.0,
                value=(min_, min_ + 1.0),
            )
        else:
            Model_RE_All = st.slider(
                "Overall Pass-rate (%):",
                min_value=min_,
                max_value=max_,
                value=(min_, max_),
            )
        sel_min_RE_All, sel_max_RE_All = Model_RE_All

        # D90 Passrate
        min_ = float(
            math.floor(st.session_state.all_metrics["D90 Pass-rate (%)"].min())
        )
        max_ = float(math.ceil(st.session_state.all_metrics["D90 Pass-rate (%)"].max()))

        if min_ == max_:
            Model_RE_D90 = st.slider(
                "D90 Pass-rate (%):",
                min_value=min_,
                max_value=min_ + 1.0,
                value=(min_, min_ + 1.0),
            )
        else:
            Model_RE_D90 = st.slider(
                "D90 Pass-rate (%):", min_value=min_, max_value=max_, value=(min_, max_)
            )
        sel_min_RE_D90, sel_max_RE_D90 = Model_RE_D90

        # DmaxU Passrate
        min_ = float(
            math.floor(st.session_state.all_metrics["DmaxU Pass-rate (%)"].min())
        )
        max_ = float(
            math.ceil(st.session_state.all_metrics["DmaxU Pass-rate (%)"].max())
        )

        if min_ == max_:
            Model_RE_DmaxU = st.slider(
                "DmaxU Pass-rate (%):",
                min_value=min_,
                max_value=min_ + 1.0,
                value=(min_, min_ + 1.0),
            )
        else:
            Model_RE_DmaxU = st.slider(
                "DmaxU Pass-rate (%):",
                min_value=min_,
                max_value=max_,
                value=(min_, max_),
            )
        sel_min_RE_DmaxU, sel_max_RE_DmaxU = Model_RE_DmaxU

        # DmaxR Passrate
        min_ = float(
            math.floor(st.session_state.all_metrics["DmaxR Pass-rate (%)"].min())
        )
        max_ = float(
            math.ceil(st.session_state.all_metrics["DmaxR Pass-rate (%)"].max())
        )

        if min_ == max_:
            Model_RE_DmaxR = st.slider(
                "DmaxR Pass-rate (%):",
                min_value=min_,
                max_value=min_ + 1.0,
                value=(min_, min_ + 1.0),
            )
        else:
            Model_RE_DmaxR = st.slider(
                "DmaxR Pass-rate (%):",
                min_value=min_,
                max_value=max_,
                value=(min_, max_),
            )
        sel_min_RE_DmaxR, sel_max_RE_DmaxR = Model_RE_DmaxR

    with col1:
        st.write(("### **Nominal Metrics**"))
        # V100
        min_ = float(
            math.floor(st.session_state.all_metrics["Prostate: V100 (%)"].min())
        )
        max_ = float(
            math.ceil(st.session_state.all_metrics["Prostate: V100 (%)"].max())
        )

        if min_ == max_:
            Model_V100 = st.slider(
                "Prostate: V100 (%):",
                min_value=min_,
                max_value=min_ + 1.0,
                value=(min_, min_ + 1.0),
            )
        else:
            Model_V100 = st.slider(
                "Prostate: V100 (%):",
                min_value=min_,
                max_value=max_,
                value=(min_, max_),
            )
        sel_min_V100, sel_max_V100 = Model_V100

        # D90
        min_ = float(
            math.floor(st.session_state.all_metrics["Prostate: D90 (Gy)"].min())
        )
        max_ = float(
            math.ceil(st.session_state.all_metrics["Prostate: D90 (Gy)"].max())
        )

        if min_ == max_:
            Model_D90 = st.slider(
                "Prostate: D90 (Gy):",
                min_value=min_,
                max_value=min_ + 1.0,
                value=(min_, min_ + 1.0),
            )
        else:
            Model_D90 = st.slider(
                "Prostate: D90 (Gy):",
                min_value=min_,
                max_value=max_,
                value=(min_, max_),
            )
        sel_min_D90, sel_max_D90 = Model_D90

        # V150
        min_ = float(
            math.floor(st.session_state.all_metrics["Prostate: V150 (%)"].min())
        )
        max_ = float(
            math.ceil(st.session_state.all_metrics["Prostate: V150 (%)"].max())
        )

        if min_ == max_:
            Model_V150 = st.slider(
                "Prostate: V150 (%):",
                min_value=min_,
                max_value=min_ + 1.0,
                value=(min_, min_ + 1.0),
            )
        else:
            Model_V150 = st.slider(
                "Prostate: V150 (%):",
                min_value=min_,
                max_value=max_,
                value=(min_, max_),
            )
        sel_min_V150, sel_max_V150 = Model_V150

        # V200
        min_ = float(
            math.floor(st.session_state.all_metrics["Prostate: V200 (%)"].min())
        )
        max_ = float(
            math.ceil(st.session_state.all_metrics["Prostate: V200 (%)"].max())
        )

        if min_ == max_:
            Model_V200 = st.slider(
                "Prostate: V200 (%):",
                min_value=min_,
                max_value=min_ + 1.0,
                value=(min_, min_ + 1.0),
            )
        else:
            Model_V200 = st.slider(
                "Prostate: V200 (%):",
                min_value=min_,
                max_value=max_,
                value=(min_, max_),
            )
        sel_min_V200, sel_max_V200 = Model_V200

        # D10
        min_ = float(
            math.floor(st.session_state.all_metrics["Urethra: D10 (Gy)"].min())
        )
        max_ = float(math.ceil(st.session_state.all_metrics["Urethra: D10 (Gy)"].max()))

        if min_ == max_:
            Model_D10 = st.slider(
                "Urethra: D10 (Gy):",
                min_value=min_,
                max_value=min_ + 1.0,
                value=(min_, min_ + 1.0),
            )
        else:
            Model_D10 = st.slider(
                "Urethra: D10 (Gy):", min_value=min_, max_value=max_, value=(min_, max_)
            )
        sel_min_D10, sel_max_D10 = Model_D10

        # DmaxU
        min_ = float(
            math.floor(st.session_state.all_metrics["Urethra: DmaxU (Gy)"].min())
        )
        max_ = float(
            math.ceil(st.session_state.all_metrics["Urethra: DmaxU (Gy)"].max())
        )

        if min_ == max_:
            Model_DmaxU = st.slider(
                "Urethra: DmaxU (Gy):",
                min_value=min_,
                max_value=min_ + 1.0,
                value=(min_, min_ + 1.0),
            )
        else:
            Model_DmaxU = st.slider(
                "Urethra: DmaxU (Gy):",
                min_value=min_,
                max_value=max_,
                value=(min_, max_),
            )
        sel_min_DmaxU, sel_max_DmaxU = Model_DmaxU

        # V75
        min_ = float(math.floor(st.session_state.all_metrics["Rectum: V75 (cc)"].min()))
        max_ = float(math.ceil(st.session_state.all_metrics["Rectum: V75 (cc)"].max()))

        if min_ == max_:
            Model_V75 = st.slider(
                "Rectum: V75 (cc):",
                min_value=min_,
                max_value=min_ + 1.0,
                value=(min_, min_ + 1.0),
            )
        else:
            Model_V75 = st.slider(
                "Rectum: V75 (cc):", min_value=min_, max_value=max_, value=(min_, max_)
            )

        sel_min_V75, sel_max_V75 = Model_V75

        # DmaxR

        min_ = float(
            math.floor(st.session_state.all_metrics["Rectum: DmaxR (Gy)"].min())
        )
        max_ = float(
            math.ceil(st.session_state.all_metrics["Rectum: DmaxR (Gy)"].max())
        )

        if min_ == max_:
            Model_DmaxR = st.slider(
                "Rectum: DmaxR (Gy):",
                min_value=min_,
                max_value=min_ + 1.0,
                value=(min_, min_ + 1.0),
            )
        else:
            Model_DmaxR = st.slider(
                "Rectum: DmaxR (Gy):",
                min_value=min_,
                max_value=max_,
                value=(min_, max_),
            )
        sel_min_DmaxR, sel_max_DmaxR = Model_DmaxR

    df_results = st.session_state.all_metrics.loc[
        (st.session_state.all_metrics["All Pass-rate (%)"] >= sel_min_RE_All)
        & (st.session_state.all_metrics["All Pass-rate (%)"] <= sel_max_RE_All)
        & (st.session_state.all_metrics["D90 Pass-rate (%)"] >= sel_min_RE_D90)
        & (st.session_state.all_metrics["D90 Pass-rate (%)"] <= sel_max_RE_D90)
        & (st.session_state.all_metrics["DmaxU Pass-rate (%)"] >= sel_min_RE_DmaxU)
        & (st.session_state.all_metrics["DmaxU Pass-rate (%)"] <= sel_max_RE_DmaxU)
        & (st.session_state.all_metrics["DmaxR Pass-rate (%)"] >= sel_min_RE_DmaxR)
        & (st.session_state.all_metrics["DmaxR Pass-rate (%)"] <= sel_max_RE_DmaxR)
        & (st.session_state.all_metrics["Prostate: V100 (%)"] >= sel_min_V100)
        & (st.session_state.all_metrics["Prostate: V100 (%)"] <= sel_max_V100)
        & (st.session_state.all_metrics["Prostate: D90 (Gy)"] >= sel_min_D90)
        & (st.session_state.all_metrics["Prostate: D90 (Gy)"] <= sel_max_D90)
        & (st.session_state.all_metrics["Prostate: V150 (%)"] >= sel_min_V150)
        & (st.session_state.all_metrics["Prostate: V150 (%)"] <= sel_max_V150)
        & (st.session_state.all_metrics["Prostate: V200 (%)"] >= sel_min_V200)
        & (st.session_state.all_metrics["Prostate: V200 (%)"] <= sel_max_V200)
        & (st.session_state.all_metrics["Urethra: D10 (Gy)"] >= sel_min_D10)
        & (st.session_state.all_metrics["Urethra: D10 (Gy)"] <= sel_max_D10)
        & (st.session_state.all_metrics["Urethra: DmaxU (Gy)"] >= sel_min_DmaxU)
        & (st.session_state.all_metrics["Urethra: DmaxU (Gy)"] <= sel_max_DmaxU)
        & (st.session_state.all_metrics["Rectum: V75 (cc)"] >= sel_min_V75)
        & (st.session_state.all_metrics["Rectum: V75 (cc)"] <= sel_max_V75)
        & (st.session_state.all_metrics["Rectum: DmaxR (Gy)"] >= sel_min_DmaxR)
        & (st.session_state.all_metrics["Rectum: DmaxR (Gy)"] <= sel_max_DmaxR)
    ]

    st.session_state.all_metrics_approx_pass_rates_results = (
        st.session_state.all_metrics_approx_pass_rates.loc[
            (st.session_state.all_metrics["All Pass-rate (%)"] >= sel_min_RE_All)
            & (st.session_state.all_metrics["All Pass-rate (%)"] <= sel_max_RE_All)
            & (st.session_state.all_metrics["D90 Pass-rate (%)"] >= sel_min_RE_D90)
            & (st.session_state.all_metrics["D90 Pass-rate (%)"] <= sel_max_RE_D90)
            & (st.session_state.all_metrics["DmaxU Pass-rate (%)"] >= sel_min_RE_DmaxU)
            & (st.session_state.all_metrics["DmaxU Pass-rate (%)"] <= sel_max_RE_DmaxU)
            & (st.session_state.all_metrics["DmaxR Pass-rate (%)"] >= sel_min_RE_DmaxR)
            & (st.session_state.all_metrics["DmaxR Pass-rate (%)"] <= sel_max_RE_DmaxR)
            & (st.session_state.all_metrics["Prostate: V100 (%)"] >= sel_min_V100)
            & (st.session_state.all_metrics["Prostate: V100 (%)"] <= sel_max_V100)
            & (st.session_state.all_metrics["Prostate: D90 (Gy)"] >= sel_min_D90)
            & (st.session_state.all_metrics["Prostate: D90 (Gy)"] <= sel_max_D90)
            & (st.session_state.all_metrics["Prostate: V150 (%)"] >= sel_min_V150)
            & (st.session_state.all_metrics["Prostate: V150 (%)"] <= sel_max_V150)
            & (st.session_state.all_metrics["Prostate: V200 (%)"] >= sel_min_V200)
            & (st.session_state.all_metrics["Prostate: V200 (%)"] <= sel_max_V200)
            & (st.session_state.all_metrics["Urethra: D10 (Gy)"] >= sel_min_D10)
            & (st.session_state.all_metrics["Urethra: D10 (Gy)"] <= sel_max_D10)
            & (st.session_state.all_metrics["Urethra: DmaxU (Gy)"] >= sel_min_DmaxU)
            & (st.session_state.all_metrics["Urethra: DmaxU (Gy)"] <= sel_max_DmaxU)
            & (st.session_state.all_metrics["Rectum: V75 (cc)"] >= sel_min_V75)
            & (st.session_state.all_metrics["Rectum: V75 (cc)"] <= sel_max_V75)
            & (st.session_state.all_metrics["Rectum: DmaxR (Gy)"] >= sel_min_DmaxR)
            & (st.session_state.all_metrics["Rectum: DmaxR (Gy)"] <= sel_max_DmaxR)
        ]
    )

    with col2:
        st.write(("### **Data from selected ranges**"))
        if df_results.shape[0] > 30:
            height_to_use = int(((30 + 1) * 35 + 3))
        else:
            height_to_use = int(((df_results.shape[0] + 1) * 35 + 3))
        print(st.session_state.selected_index)
        if len(st.session_state.selected_index) > 0:
            st.dataframe(
                (
                    df_results.style.apply(
                        color_coding,
                        indexs_of_rows=st.session_state.selected_index,
                        axis=1,
                    )
                ).format(precision=1),
                height=height_to_use,
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.dataframe(
                df_results.round(2),
                height=height_to_use,
                use_container_width=True,
                hide_index=True,
            )

    with col4:
        st.write(("### **Pareto Front**"))
        X_title = "<b>D<sub>90</sub>  Pass-rate (%)</b>"
        Y_title = "<b>D<sub>max</sub><sup>U</sup>  Pass-rate (%)</b>"
        Z_title = "<b>D<sub>max</sub><sup>R</sup>  Pass-rate (%)</b>"

        fig = go.Figure()

        fig.add_trace(
            go.Scatter3d(
                x=st.session_state.all_metrics_approx_pass_rates_results[
                    "Approx. Passrate D90"
                ],
                y=st.session_state.all_metrics_approx_pass_rates_results[
                    "Approx. Passrate DmaxU"
                ],
                z=st.session_state.all_metrics_approx_pass_rates_results[
                    "Approx. Passrate DmaxR"
                ],
                marker=dict(
                    color=st.session_state.all_metrics["All Pass-rate (%)"], size=4
                ),
                # showlegend=False,
                mode="markers",
                name="Approx. Passrate",
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=df_results["D90 Pass-rate (%)"],
                y=df_results["DmaxU Pass-rate (%)"],
                z=df_results["DmaxR Pass-rate (%)"],
                marker=dict(
                    color=st.session_state.all_metrics[
                        "All Pass-rate (%)"
                    ],  #'rgba(205,35,83,1)',
                    size=4,
                ),
                # showlegend=False,
                mode="markers",
                name="Direct Robust Evaluation",
            )
        )

        fig.add_trace(
            go.Scatter3d(
                x=[100],
                y=[100],
                z=[100],
                marker=dict(color="rgba(205,35,83,0)", size=1),
                showlegend=False,
                mode="markers",
            )
        )

        fig.update_layout(  # title="Pareto Front",

            scene=dict(
                xaxis=dict(
                    title=X_title,
                    titlefont=dict(family="Arial"),  # ,size=font_size_title),
                    tickfont=dict(family="Arial"),  # ,size=font_size_tick),
                    # range=[0,100],
                ),
                yaxis=dict(
                    title=Y_title,
                    titlefont=dict(family="Arial"),  # ,size=font_size_title),
                    tickfont=dict(family="Arial"),  # ,size=font_size_tick),
                    # range=[0,100],
                ),
                zaxis=dict(
                    title=Z_title,
                    titlefont=dict(family="Arial"),  # ,size=font_size_title),
                    tickfont=dict(family="Arial"),  # ,size=font_size_tick),
                    # range=[0,100],
                ),
            ),

        )
        fig.update_layout(height=height_to_use_for_graphs, width=800)
        st.plotly_chart(fig, height=height_to_use_for_graphs, use_container_width=True)
else:
    col1, col2, col3 = st.columns([1, 4, 1])
    with col3:
        st.write(("##### **Robustness Pass-rates**"))
        # Overall Passrate
        min_ = float(
            math.floor(st.session_state.all_metrics["All Pass-rate (%)"].min())
        )
        max_ = float(math.ceil(st.session_state.all_metrics["All Pass-rate (%)"].max()))

        if min_ == max_:
            Model_RE_All = st.slider(
                "Overall Pass-rate (%):",
                min_value=min_,
                max_value=min_ + 1.0,
                value=(min_, min_ + 1.0),
            )
        else:
            Model_RE_All = st.slider(
                "Overall Pass-rate (%):",
                min_value=min_,
                max_value=max_,
                value=(min_, max_),
            )
        sel_min_RE_All, sel_max_RE_All = Model_RE_All

        # D90 Passrate
        min_ = float(
            math.floor(st.session_state.all_metrics["D90 Pass-rate (%)"].min())
        )
        max_ = float(math.ceil(st.session_state.all_metrics["D90 Pass-rate (%)"].max()))

        if min_ == max_:
            Model_RE_D90 = st.slider(
                "D90 Pass-rate (%):",
                min_value=min_,
                max_value=min_ + 1.0,
                value=(min_, min_ + 1.0),
            )
        else:
            Model_RE_D90 = st.slider(
                "D90 Pass-rate (%):", min_value=min_, max_value=max_, value=(min_, max_)
            )
        sel_min_RE_D90, sel_max_RE_D90 = Model_RE_D90

        # DmaxU Passrate
        min_ = float(
            math.floor(st.session_state.all_metrics["DmaxU Pass-rate (%)"].min())
        )
        max_ = float(
            math.ceil(st.session_state.all_metrics["DmaxU Pass-rate (%)"].max())
        )

        if min_ == max_:
            Model_RE_DmaxU = st.slider(
                "DmaxU Pass-rate (%):",
                min_value=min_,
                max_value=min_ + 1.0,
                value=(min_, min_ + 1.0),
            )
        else:
            Model_RE_DmaxU = st.slider(
                "DmaxU Pass-rate (%):",
                min_value=min_,
                max_value=max_,
                value=(min_, max_),
            )
        sel_min_RE_DmaxU, sel_max_RE_DmaxU = Model_RE_DmaxU

        # DmaxR Passrate
        min_ = float(
            math.floor(st.session_state.all_metrics["DmaxR Pass-rate (%)"].min())
        )
        max_ = float(
            math.ceil(st.session_state.all_metrics["DmaxR Pass-rate (%)"].max())
        )

        if min_ == max_:
            Model_RE_DmaxR = st.slider(
                "DmaxR Pass-rate (%):",
                min_value=min_,
                max_value=min_ + 1.0,
                value=(min_, min_ + 1.0),
            )
        else:
            Model_RE_DmaxR = st.slider(
                "DmaxR Pass-rate (%):",
                min_value=min_,
                max_value=max_,
                value=(min_, max_),
            )
        sel_min_RE_DmaxR, sel_max_RE_DmaxR = Model_RE_DmaxR

    with col1:
        st.write(("### **Nominal Metrics**"))
        # V100
        min_ = float(
            math.floor(st.session_state.all_metrics["Prostate: V100 (%)"].min())
        )
        max_ = float(
            math.ceil(st.session_state.all_metrics["Prostate: V100 (%)"].max())
        )

        if min_ == max_:
            Model_V100 = st.slider(
                "Prostate: V100 (%):",
                min_value=min_,
                max_value=min_ + 1.0,
                value=(min_, min_ + 1.0),
            )
        else:
            Model_V100 = st.slider(
                "Prostate: V100 (%):",
                min_value=min_,
                max_value=max_,
                value=(min_, max_),
            )
        sel_min_V100, sel_max_V100 = Model_V100

        # D90
        min_ = float(
            math.floor(st.session_state.all_metrics["Prostate: D90 (Gy)"].min())
        )
        max_ = float(
            math.ceil(st.session_state.all_metrics["Prostate: D90 (Gy)"].max())
        )

        if min_ == max_:
            Model_D90 = st.slider(
                "Prostate: D90 (Gy):",
                min_value=min_,
                max_value=min_ + 1.0,
                value=(min_, min_ + 1.0),
            )
        else:
            Model_D90 = st.slider(
                "Prostate: D90 (Gy):",
                min_value=min_,
                max_value=max_,
                value=(min_, max_),
            )
        sel_min_D90, sel_max_D90 = Model_D90

        # V150
        min_ = float(
            math.floor(st.session_state.all_metrics["Prostate: V150 (%)"].min())
        )
        max_ = float(
            math.ceil(st.session_state.all_metrics["Prostate: V150 (%)"].max())
        )

        if min_ == max_:
            Model_V150 = st.slider(
                "Prostate: V150 (%):",
                min_value=min_,
                max_value=min_ + 1.0,
                value=(min_, min_ + 1.0),
            )
        else:
            Model_V150 = st.slider(
                "Prostate: V150 (%):",
                min_value=min_,
                max_value=max_,
                value=(min_, max_),
            )
        sel_min_V150, sel_max_V150 = Model_V150

        # V200
        min_ = float(
            math.floor(st.session_state.all_metrics["Prostate: V200 (%)"].min())
        )
        max_ = float(
            math.ceil(st.session_state.all_metrics["Prostate: V200 (%)"].max())
        )

        if min_ == max_:
            Model_V200 = st.slider(
                "Prostate: V200 (%):",
                min_value=min_,
                max_value=min_ + 1.0,
                value=(min_, min_ + 1.0),
            )
        else:
            Model_V200 = st.slider(
                "Prostate: V200 (%):",
                min_value=min_,
                max_value=max_,
                value=(min_, max_),
            )
        sel_min_V200, sel_max_V200 = Model_V200

        # D10
        min_ = float(
            math.floor(st.session_state.all_metrics["Urethra: D10 (Gy)"].min())
        )
        max_ = float(math.ceil(st.session_state.all_metrics["Urethra: D10 (Gy)"].max()))

        if min_ == max_:
            Model_D10 = st.slider(
                "Urethra: D10 (Gy):",
                min_value=min_,
                max_value=min_ + 1.0,
                value=(min_, min_ + 1.0),
            )
        else:
            Model_D10 = st.slider(
                "Urethra: D10 (Gy):", min_value=min_, max_value=max_, value=(min_, max_)
            )
        sel_min_D10, sel_max_D10 = Model_D10

        # DmaxU
        min_ = float(
            math.floor(st.session_state.all_metrics["Urethra: DmaxU (Gy)"].min())
        )
        max_ = float(
            math.ceil(st.session_state.all_metrics["Urethra: DmaxU (Gy)"].max())
        )

        if min_ == max_:
            Model_DmaxU = st.slider(
                "Urethra: DmaxU (Gy):",
                min_value=min_,
                max_value=min_ + 1.0,
                value=(min_, min_ + 1.0),
            )
        else:
            Model_DmaxU = st.slider(
                "Urethra: DmaxU (Gy):",
                min_value=min_,
                max_value=max_,
                value=(min_, max_),
            )
        sel_min_DmaxU, sel_max_DmaxU = Model_DmaxU

        # V75
        min_ = float(math.floor(st.session_state.all_metrics["Rectum: V75 (cc)"].min()))
        max_ = float(math.ceil(st.session_state.all_metrics["Rectum: V75 (cc)"].max()))

        if min_ == max_:
            Model_V75 = st.slider(
                "Rectum: V75 (cc):",
                min_value=min_,
                max_value=min_ + 1.0,
                value=(min_, min_ + 1.0),
            )
        else:
            Model_V75 = st.slider(
                "Rectum: V75 (cc):", min_value=min_, max_value=max_, value=(min_, max_)
            )

        sel_min_V75, sel_max_V75 = Model_V75

        # DmaxR

        min_ = float(
            math.floor(st.session_state.all_metrics["Rectum: DmaxR (Gy)"].min())
        )
        max_ = float(
            math.ceil(st.session_state.all_metrics["Rectum: DmaxR (Gy)"].max())
        )

        if min_ == max_:
            Model_DmaxR = st.slider(
                "Rectum: DmaxR (Gy):",
                min_value=min_,
                max_value=min_ + 1.0,
                value=(min_, min_ + 1.0),
            )
        else:
            Model_DmaxR = st.slider(
                "Rectum: DmaxR (Gy):",
                min_value=min_,
                max_value=max_,
                value=(min_, max_),
            )
        sel_min_DmaxR, sel_max_DmaxR = Model_DmaxR

    df_results = st.session_state.all_metrics.loc[
        (st.session_state.all_metrics["All Pass-rate (%)"] >= sel_min_RE_All)
        & (st.session_state.all_metrics["All Pass-rate (%)"] <= sel_max_RE_All)
        & (st.session_state.all_metrics["D90 Pass-rate (%)"] >= sel_min_RE_D90)
        & (st.session_state.all_metrics["D90 Pass-rate (%)"] <= sel_max_RE_D90)
        & (st.session_state.all_metrics["DmaxU Pass-rate (%)"] >= sel_min_RE_DmaxU)
        & (st.session_state.all_metrics["DmaxU Pass-rate (%)"] <= sel_max_RE_DmaxU)
        & (st.session_state.all_metrics["DmaxR Pass-rate (%)"] >= sel_min_RE_DmaxR)
        & (st.session_state.all_metrics["DmaxR Pass-rate (%)"] <= sel_max_RE_DmaxR)
        & (st.session_state.all_metrics["Prostate: V100 (%)"] >= sel_min_V100)
        & (st.session_state.all_metrics["Prostate: V100 (%)"] <= sel_max_V100)
        & (st.session_state.all_metrics["Prostate: D90 (Gy)"] >= sel_min_D90)
        & (st.session_state.all_metrics["Prostate: D90 (Gy)"] <= sel_max_D90)
        & (st.session_state.all_metrics["Prostate: V150 (%)"] >= sel_min_V150)
        & (st.session_state.all_metrics["Prostate: V150 (%)"] <= sel_max_V150)
        & (st.session_state.all_metrics["Prostate: V200 (%)"] >= sel_min_V200)
        & (st.session_state.all_metrics["Prostate: V200 (%)"] <= sel_max_V200)
        & (st.session_state.all_metrics["Urethra: D10 (Gy)"] >= sel_min_D10)
        & (st.session_state.all_metrics["Urethra: D10 (Gy)"] <= sel_max_D10)
        & (st.session_state.all_metrics["Urethra: DmaxU (Gy)"] >= sel_min_DmaxU)
        & (st.session_state.all_metrics["Urethra: DmaxU (Gy)"] <= sel_max_DmaxU)
        & (st.session_state.all_metrics["Rectum: V75 (cc)"] >= sel_min_V75)
        & (st.session_state.all_metrics["Rectum: V75 (cc)"] <= sel_max_V75)
        & (st.session_state.all_metrics["Rectum: DmaxR (Gy)"] >= sel_min_DmaxR)
        & (st.session_state.all_metrics["Rectum: DmaxR (Gy)"] <= sel_max_DmaxR)
    ]

    st.session_state.all_metrics_approx_pass_rates_results = (
        st.session_state.all_metrics_approx_pass_rates.loc[
            (st.session_state.all_metrics["All Pass-rate (%)"] >= sel_min_RE_All)
            & (st.session_state.all_metrics["All Pass-rate (%)"] <= sel_max_RE_All)
            & (st.session_state.all_metrics["D90 Pass-rate (%)"] >= sel_min_RE_D90)
            & (st.session_state.all_metrics["D90 Pass-rate (%)"] <= sel_max_RE_D90)
            & (st.session_state.all_metrics["DmaxU Pass-rate (%)"] >= sel_min_RE_DmaxU)
            & (st.session_state.all_metrics["DmaxU Pass-rate (%)"] <= sel_max_RE_DmaxU)
            & (st.session_state.all_metrics["DmaxR Pass-rate (%)"] >= sel_min_RE_DmaxR)
            & (st.session_state.all_metrics["DmaxR Pass-rate (%)"] <= sel_max_RE_DmaxR)
            & (st.session_state.all_metrics["Prostate: V100 (%)"] >= sel_min_V100)
            & (st.session_state.all_metrics["Prostate: V100 (%)"] <= sel_max_V100)
            & (st.session_state.all_metrics["Prostate: D90 (Gy)"] >= sel_min_D90)
            & (st.session_state.all_metrics["Prostate: D90 (Gy)"] <= sel_max_D90)
            & (st.session_state.all_metrics["Prostate: V150 (%)"] >= sel_min_V150)
            & (st.session_state.all_metrics["Prostate: V150 (%)"] <= sel_max_V150)
            & (st.session_state.all_metrics["Prostate: V200 (%)"] >= sel_min_V200)
            & (st.session_state.all_metrics["Prostate: V200 (%)"] <= sel_max_V200)
            & (st.session_state.all_metrics["Urethra: D10 (Gy)"] >= sel_min_D10)
            & (st.session_state.all_metrics["Urethra: D10 (Gy)"] <= sel_max_D10)
            & (st.session_state.all_metrics["Urethra: DmaxU (Gy)"] >= sel_min_DmaxU)
            & (st.session_state.all_metrics["Urethra: DmaxU (Gy)"] <= sel_max_DmaxU)
            & (st.session_state.all_metrics["Rectum: V75 (cc)"] >= sel_min_V75)
            & (st.session_state.all_metrics["Rectum: V75 (cc)"] <= sel_max_V75)
            & (st.session_state.all_metrics["Rectum: DmaxR (Gy)"] >= sel_min_DmaxR)
            & (st.session_state.all_metrics["Rectum: DmaxR (Gy)"] <= sel_max_DmaxR)
        ]
    )

    with col2:
        st.write(("### **Data from selected ranges**"))
        if df_results.shape[0] > 30:
            height_to_use = int(((30 + 1) * 35 + 3))
        else:
            height_to_use = int(((df_results.shape[0] + 1) * 35 + 3))
        print(st.session_state.selected_index)
        if len(st.session_state.selected_index) > 0:
            st.dataframe(
                (
                    df_results.style.apply(
                        color_coding,
                        indexs_of_rows=st.session_state.selected_index,
                        axis=1,
                    )
                ).format(precision=1),
                height=height_to_use,
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.dataframe(
                df_results.round(2),
                height=height_to_use,
                use_container_width=True,
                hide_index=True,
            )


show_1 = False

if "show_1" not in st.session_state:
    st.session_state.show_1 = show_1
else:
    st.session_state.show_1 = show_1

show_1_only = False

if "show_1_only" not in st.session_state:
    st.session_state.show_1_only = show_1_only
else:
    st.session_state.show_1_only = show_1_only

show_2 = False

if "show_2" not in st.session_state:
    st.session_state.show_2 = show_2
else:
    st.session_state.show_2 = show_2

show_3 = False

if "show_3" not in st.session_state:
    st.session_state.show_3 = show_3
else:
    st.session_state.show_3 = show_3

line_size = 2
line_size_areas = 0.5
if "line_size" not in st.session_state:
    st.session_state.line_size = line_size
else:
    st.session_state.line_size = line_size

if "line_size_areas" not in st.session_state:
    st.session_state.line_size_areas = line_size_areas
else:
    st.session_state.line_size_areas = line_size_areas

title_font_size = 20
axis_font_size = 16
legend_font_size = 16

if "title_font_size" not in st.session_state:
    st.session_state.title_font_size = title_font_size
else:
    st.session_state.title_font_size = title_font_size

if "legend_font_size" not in st.session_state:
    st.session_state.legend_font_size = legend_font_size
else:
    st.session_state.legend_font_size = legend_font_size

if "axis_font_size" not in st.session_state:
    st.session_state.axis_font_size = axis_font_size
else:
    st.session_state.axis_font_size = axis_font_size
