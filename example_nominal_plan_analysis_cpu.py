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

import pandas as pd
import numpy as np
from pathlib import Path
import webbrowser


from robustbrachy.robustevaluation.data_extraction import *
from robustbrachy.robustevaluation.utils_cpu import *
from robustbrachy.robustevaluation.fast_TG43_cpu import *
from robustbrachy.robustevaluation.display_results import *

path = Path(__file__).parent.absolute()
path_temp = str(Path(__file__).parent.absolute()) + "/temp/"
path_example = str(Path(__file__).parent.absolute()) + "/Example Plan/"
path_source_data = str(Path(__file__).parent.absolute()) + "/source data/"

######################
## define variables ##
######################

# define location of plan and source files.
# Currently looking in the example folder using 'path_example' above.
file_rp = (
    str(path_example) + "/PL001.dcm"
)  # str(path_temp) + "/PL001_Robust_Optimised.dcm"#str(path_example) + "/PL001.dcm"
file_rs = str(path_example) + "/SS001.dcm"
file_rd = (
    str(path_example) + "/DO001.dcm"
)  # str(path_temp) + "/DO001_Robust_Optimised.dcm"#str(path_example) + "/DO001.dcm"
file_source_data = str(path_source_data) + "/192ir-hdr_gammamed_plus.xls"

# define dose metrics to get robustness values for
df_dose_metrics = pd.DataFrame(
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

# define volume metrics to get robustness values for
df_volume_metrics = pd.DataFrame(
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

################################
##      Running program       ##
################################

# *** extract data  ***

# get dictionary of plan parameters (structures, dwell points, times, source dose arrays, etc.)
plan_parameters, _, _, _ = extract_all_needed_data(
    file_rp, file_rs, file_rd, file_source_data
)


# *** preparing parameters  ***
# get dose and volume metric labels
(
    passrates_labels_D,
    metric_labels_D,
    _,
    passrates_labels_V,
    metric_labels_V,
    _,
) = make_dvh_metric_labels(df_dose_metrics, df_volume_metrics)

# make a copy of each structure array.
# This is needed so the structures and plan_parameters are separated,
# which is needed in the robust evaluation use of this function.
copied_structures = copy_structures_cpu(plan_parameters)

# calculate the nominal DVHs for each structure (dose calculation)
all_nominal_dvhs = fast_TG43_cpu(
    copied_structures,
    plan_parameters,
    1,
)

df_nominal_metric_data = calculate_all_nominal_metrics_cpu(
    df_dose_metrics,
    df_volume_metrics,
    metric_labels_D,
    metric_labels_V,
    np.array([all_nominal_dvhs]),
    plan_parameters,
)

# nominal plan information
nominal_data = np.array(
    [
        round(plan_parameters["prostate_vol"], 1),
        round(plan_parameters["urethra_vol"], 1),
        round(plan_parameters["rectum_vol"], 1),
        round(plan_parameters["prescribed_dose"], 1),
    ]
)
row_headings = [
    "Prostate Volume (cc)",
    "Urethra Volume (cc)",
    "Rectum Volume (cc)",
    "Prescribed Dose (Gy)",
]
df_nominal_data = pd.DataFrame(data=[nominal_data], columns=row_headings)

################################
##   Displaying information   ##
################################
print("Robust Evaluation Results")
print("see web browser popups for plot and data")

#  *** function to write all data to html and display it ***
write_to_html_file(
    [df_nominal_data, df_nominal_metric_data],
    ["Nominal Plan Data", "Nominal Plan DVH metrics"],
    "df_nominal_analysis.html",
)
webbrowser.open("df_nominal_analysis.html")

# *** plotly figure ***
## define figure parameters
# nominal dvh, then area of 68%CI, 95%CI, and min-max
prostate_color = [
    "rgba(6,47,95,1.0)",
]
urethra_color = [
    "rgba(4,129,83,1.0)",
]
rectum_color = [
    "rgba(167,0,23,1.0)",
]
line_size = 2
axis_font_size = 16
legend_font_size = 16
colours = [prostate_color, urethra_color, rectum_color]

# make graph
fig_dvh = construct_nominal_dvhs_graph(
    all_nominal_dvhs,
    colours,
    line_size,
    axis_font_size,
    legend_font_size,
)

# show roubst graph
fig_dvh.show()
