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
from pathlib import Path
import webbrowser

from robustbrachy.robustevaluation.run_robust_evaluation_cpu import *
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
file_rp = str(path_example) + "/PL001.dcm"
file_rs = str(path_example) + "/SS001.dcm"
file_rd = str(path_example) + "/DO001.dcm"
file_source_data = str(path_source_data) + "/192ir-hdr_gammamed_plus.xls"

# define number of uncertainty scenarios to conduct
no_of_runs = 100  # 500 or 1000 has low (better) statistical uncertainty

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

# define uncertainty standard deviations

## Parameter 1 Standard Deviation: Dwells points move along the needle
needle_recon = 1.1  # Needle Reconstruction SI (Needle Tip, units in mm)
needle_rigid = 0.34  # Needle Rigid movement SI (units in mm)
source_position = 1.0  # Source positioning (units in mm)

## Parameter 2 Standard Deviation: Prostate boundary expansion/contraction from geometric mean (prostate centre) in 3D
prostate_contour_expand_shrink = 1.0

## Parameter 3a Standard Deviation: Urethra boundary expanded/contracted in each transverse slice
urethra_contour_expand_shrink = 0.25

## Parameter 3b Standard Deviation: Rectum boundary expanded/contracted in each transverse slice
rectum_contour_expand_shrink = 0.5

## Parameter 4 Standard Deviation: Needle transverse plane movements
needle_recon_2D = 1.2  # Needle Reconstruction transverse slice (units in mm)
needle_rigid_2D = (
    0.86  # Needle rigid movement (intra-fractional) in transverse plane (units in mm)
)

## Parameter 5a Standard Deviation: Dwell times add constant value
dwell_time_precision = 0.06

## Parameter 5b Standard Deviation: Dwell times percentage change
dose_calc_med_percent = 1.0  # Dose Calculation - Medium (units in %)
dose_calc_TPS_percent = 3.0  # Dose Calculation - Treatment Planning (units in %)
source_activity = 3.0  # Source Activity (units in %)

## Parameter 6 Standard Deviation: Prostate and Urethra (+ needles) anterior-posterior movement
AP_rigid = (
    0.5  # Intra-observer rigid movement anterior-posterior direction(units in mm)
)
LR_rigid = 0.1  # Intra-observer rigid movement left-right direction (units in mm)
SI_rigid = (
    0.0  # Intra-observer rigid movement superior-inferior direction (units in mm)
)

# creating dictionary of uncertainties
uncertainty_magnitudes = {
    "P1_needle_recon": needle_recon,
    "P1_needle_rigid": needle_rigid,
    "P1_source_position": source_position,
    "P2_prostate_contour_expand_shrink": prostate_contour_expand_shrink,
    "P3a_urethra_contour_expand_shrink": urethra_contour_expand_shrink,
    "P3b_rectum_contour_expand_shrink": rectum_contour_expand_shrink,
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

################################
## Running Robust Evaluation  ##
################################

(
    df_nominal_data,
    df_nominal_metric_data,
    df_pass_rates,
    all_nominal_dvhs,
    all_robust_dvh_summary,
) = run_robust_evaluation_program_cpu(
    file_rp,
    file_rs,
    file_rd,
    file_source_data,
    no_of_runs,
    df_dose_metrics,
    df_volume_metrics,
    uncertainty_magnitudes,
)

################################
##   Displaying information   ##
################################
print("Robust Evaluation Results")
print("see web browser popups for plot and data")

#  *** function to write all data to html and display it ***
write_to_html_file(
    [df_nominal_data, df_nominal_metric_data, df_pass_rates],
    ["Nominal Plan Data", "Nominal Plan DVH metrics", "Robust Evaluation Results"],
    "df_pass_rates.html",
)
webbrowser.open("df_pass_rates.html")


# *** plotly figure ***
## define figure parameters
# nominal dvh, then area of 68%CI, 95%CI, and min-max
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
line_size = 2
axis_font_size = 16
legend_font_size = 16
colours = [prostate_color, urethra_color, rectum_color]

# make graph
fig_dvh = construct_robust_dvhs_graph(
    all_nominal_dvhs,
    all_robust_dvh_summary,
    colours,
    line_size,
    axis_font_size,
    legend_font_size,
)

# show roubst graph
fig_dvh.show()
