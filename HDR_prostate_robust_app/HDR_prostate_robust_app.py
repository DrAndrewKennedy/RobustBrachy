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
import pandas as pd
import subprocess
from pathlib import Path

path_temp = str(Path(__file__).parent.absolute()) + "/temp/"

from robustbrachy.robustevaluation.data_extraction import *
from robustbrachy.robustevaluation.utils_cpu import *

try:
    import cupy as cp
    from robustbrachy.robustevaluation.utils_gpu import *
except:
    print("no cupy")


# function to send variables to session state (the browser memory)
def send_to_session_state(var):
    if str(var) not in st.session_state:
        st.session_state.var = var
    else:
        st.session_state.var = var

    return st.session_state.var


# send the path to save all temporary files to browser memory
st.session_state.path_temp = send_to_session_state(path_temp)

# page tile and information
st.set_page_config(
    page_title="Robustness Software for HDR Prostate Brachytherapy",
)

st.write(
    "# Welcome to Robust Evaluation and Robust Optimisation in HDR Prostate Brachytherapy"
)

st.markdown(
    """
    **With this software you can:**
    1) View a nominal treatment plans' summary DVH metrics and plots
    2) Conduct a robust evaluation of a patient treatment plan 
    3) Generate robust plans using the robust optimiser and export the treatment plan files
    
    **Some notes:**
    - The software is designed to work on a Nvida GPU. Be aware that all **Robust Evaluation** algorthim runtimes can be very long using CPU alone (~ x15 longer) and might be prohibitory.
    - The software works with radiotherapy treatment plan files generated in Vitesse v4.0.1 (Varian Medical Systems, Palo Alto, USA). Deviating from this may result in unexpected errors. 
    - A TPS optimised plan is expected with a treatment plan file, dose file, and structures file. Also, the source data file is needed as an excel file.
    - All programs are developed in python
    - Start by uploading patient treatment plan files below :point_down:
    - Then **Select from the page options in the sidebar** 👈.
    """
)


# test if the computer has a Nivida GPU and print buttons to allow its use
try:
    subprocess.check_output("nvidia-smi")
    st.sidebar.write("***Use CPU or an Nivida GPU within the algorithm:***")
    if st.sidebar.button("Use GPU"):
        GPU_CPU = True
        st.session_state.GPU_CPU = send_to_session_state(GPU_CPU)

    if st.sidebar.button("Use CPU"):
        GPU_CPU = False
        st.session_state.GPU_CPU = send_to_session_state(GPU_CPU)

    if "GPU_CPU" not in st.session_state:
        GPU_CPU = False
        st.session_state.GPU_CPU = send_to_session_state(GPU_CPU)

    if st.session_state.GPU_CPU:
        str_gpu = "GPU use activated!"
        st.sidebar.write(f":green[{str_gpu}]")

    else:
        str_cpu = "CPU use activated!"
        st.sidebar.write(f":red[{str_cpu}]")


except Exception:
    GPU_CPU = False
    str_no_gpu = "No Nvidia GPU found! App will use CPU instead."
    st.sidebar.write(f":red[{str_no_gpu}]")
    st.session_state.GPU_CPU = send_to_session_state(GPU_CPU)

st.divider()

# section to upload files
uploaded_file_source_data = st.file_uploader(
    "**Select the source data file.** The GammaMed HDR 192Ir Plus xls file can be found in the root directory of the app folder.",
    type=["xls"],
)
uploaded_file_rp = st.file_uploader("**Select the plan (rp) file**", type=["dcm"])
uploaded_file_rs = st.file_uploader("**Select the structures (rs) file**", type=["dcm"])
uploaded_file_rd = st.file_uploader("**Select the dose (rd) file**", type=["dcm"])

if "files_uploaded" not in st.session_state:
    files_uploaded = False
    st.session_state.files_uploaded = send_to_session_state(files_uploaded)

# loading all necessary plan data
if (
    (uploaded_file_rp is not None)
    and (uploaded_file_source_data is not None)
    and (uploaded_file_rs is not None)
    and (uploaded_file_rd is not None)
):
    # get dictionary of plan parameters (structures, dwell points, times, source dose arrays, etc.)
    plan_parameters, rp, rs, rd = extract_all_needed_data(
        uploaded_file_rp, uploaded_file_rs, uploaded_file_rd, uploaded_file_source_data
    )

    st.session_state.plan_parameters = send_to_session_state(plan_parameters)
    st.session_state.rp = send_to_session_state(rp)
    st.session_state.rs = send_to_session_state(rs)
    st.session_state.rd = send_to_session_state(rd)
    st.session_state.files_uploaded = True

if st.session_state.files_uploaded:
    # run some code below for only the first time
    if "first_run" not in st.session_state:
        first_run = True
        st.session_state.first_run = first_run
        print('"first_run" not in st.session_state')

    # send arrays to GPU.
    # float32 is used since it is more memory efficient, and GPU memory is generally limited.
    if st.session_state.GPU_CPU:
        st.session_state.plan_parameters = to_gpu(st.session_state.plan_parameters)

    else:
        # keep arrays in CPU.
        # float32 is used since it is more memory efficient, and GPU memory is generally limited.
        st.session_state.plan_parameters = arrays_to_numpy(st.session_state.plan_parameters)

    st.divider()

    # Define the dose and volume metrics of interest for the patient
    st.write("### Define Dose and Volume metrics in the tables")

    st.write(
        """
             - Use "0" for prostate, "1" for urethra, "2" for rectum'
             - Input "%" OR "cc/Gy" values for metrics and constraints and then leave the other blank (leave as 'None')
             """
    )

    st.write("**Dose metric tables**")

    # start with some common DVH metrics
    if "df_dose_metrics" not in st.session_state:
        st.session_state.df_dose_metrics = pd.DataFrame(
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
                    "D": 10,
                    "% / cc": "%",
                    "Direction": "<",
                    "constraint": 17,
                    "% / Gy": "Gy",
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

        st.session_state.edited_df_dose_metrics = (
            st.session_state.df_dose_metrics.copy()
        )

        st.session_state.df_volume_metrics = pd.DataFrame(
            [
                {
                    "Structure": 0,
                    "V": 100,
                    "% / Gy": "%",
                    "Direction": ">",
                    "constraint": 90,
                    "% / cc": "%",
                },
                {
                    "Structure": 0,
                    "V": 150,
                    "% / Gy": "%",
                    "Direction": "<",
                    "constraint": 35,
                    "% / cc": "%",
                },
                {
                    "Structure": 0,
                    "V": 200,
                    "% / Gy": "%",
                    "Direction": "<",
                    "constraint": 15,
                    "% / cc": "%",
                },
                {
                    "Structure": 2,
                    "V": 75,
                    "% / Gy": "%",
                    "Direction": "<",
                    "constraint": 0.6,
                    "% / cc": "cc",
                },
            ]
        )

        st.session_state.edited_df_volume_metrics = (
            st.session_state.df_volume_metrics.copy()
        )

    # Shows the metrics current in the browser memory
    st.session_state.edited_df_dose_metrics = st.data_editor(
        st.session_state.df_dose_metrics, num_rows="dynamic"
    )
    st.write(" ")
    st.write("**Volume metric tables**")

    st.session_state.edited_df_volume_metrics = st.data_editor(
        st.session_state.df_volume_metrics, num_rows="dynamic"
    )
    st.sidebar.divider()
    st.sidebar.write("*Dose-Volume metrics*")
    if st.sidebar.button("Update Metrics"):
        st.session_state.df_dose_metrics = (
            st.session_state.edited_df_dose_metrics.copy()
        )
        st.session_state.df_volume_metrics = (
            st.session_state.edited_df_volume_metrics.copy()
        )

    (
        st.session_state.passrates_labels_D,
        st.session_state.metric_labels_D,
        st.session_state.metric_labels_to_print_D,
        st.session_state.passrates_labels_V,
        st.session_state.metric_labels_V,
        st.session_state.metric_labels_to_print_V,
    ) = make_dvh_metric_labels(
        st.session_state.df_dose_metrics, st.session_state.df_volume_metrics
    )

    for dv_label in [
        *st.session_state.metric_labels_to_print_D,
        *st.session_state.metric_labels_to_print_V,
    ]:
        st.sidebar.write(dv_label, unsafe_allow_html=True)

    st.session_state.first_run = False


# stores starting values for the graphs and plots for the next pages
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

robust_eval_completed = False

if "robust_eval_completed" not in st.session_state:
    st.session_state.robust_eval_completed = robust_eval_completed

first_plan_loaded = False

if "first_plan_loaded" not in st.session_state:
    st.session_state.first_plan_loaded = first_plan_loaded
else:
    st.session_state.first_plan_loaded = first_plan_loaded

second_plan_loaded = False

if "second_plan_loaded" not in st.session_state:
    st.session_state.second_plan_loaded = second_plan_loaded
else:
    st.session_state.second_plan_loaded = second_plan_loaded

third_plan_loaded = False

if "third_plan_loaded" not in st.session_state:
    st.session_state.third_plan_loaded = third_plan_loaded
else:
    st.session_state.third_plan_loaded = third_plan_loaded

selected_index = []
if "selected_index" not in st.session_state:
    st.session_state.selected_index = selected_index
else:
    st.session_state.selected_index = selected_index

# turn on all plots to start with
dont_show_rectum = (
    dont_show_urethra
) = (
    dont_show_prostate
) = (
    dont_show_200
) = (
    dont_show_150
) = dont_show_110 = dont_show_100 = dont_show_90 = dont_show_75 = dont_show_50 = False

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
