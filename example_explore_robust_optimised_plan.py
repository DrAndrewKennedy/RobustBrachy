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
from pathlib import Path
from pydicom import dcmread


from robustbrachy.robustevaluation.display_results import *
from robustbrachy.robustevaluation.data_extraction import *
from robustbrachy.robustoptimisation.generate_dicoms import *


######################
## define variables ##
######################

treatment_plan_number = 50  # the plan number in the "Plan+ column from the table that is outputted at the end of robust optimiser example

save_folder = str(Path(__file__).parent.absolute()) + "/temp/"
path_example = str(Path(__file__).parent.absolute()) + "/Example Plan/"
path_source_data = str(Path(__file__).parent.absolute()) + "/source data/"


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


###############
## load data ##
###############
all_robust_dvhs = np.load(save_folder + "all_robust_dvhs_mu_sd.npy", allow_pickle=True)
all_nominal_dvhs_pareto_front = np.load(
    save_folder + "all_nominal_dvhs.npy", allow_pickle=True
)
dwell_times_pareto_front = np.load(
    save_folder + "dwell_times_pareto_front.npy", allow_pickle=True
)
plan_parameters, _, _, _ = extract_all_needed_data(
    file_rp, file_rs, file_rd, file_source_data
)

# loading DICOM rt files
rp = dcmread(file_rp)
rs = dcmread(file_rs)
rd = dcmread(file_rd)

################################
##   Displaying information   ##
################################

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
    np.array([all_nominal_dvhs_pareto_front[treatment_plan_number, :]]),
    all_robust_dvhs[:, [treatment_plan_number], :, :, :],
    colours,
    line_size,
    axis_font_size,
    legend_font_size,
)

# show roubst graph
fig_dvh.show()

################################
##      Generate DICOMs       ##
################################

dwell_times_nominal_robust = dwell_times_pareto_front[treatment_plan_number]
rp_robust, rd_robust = generate_DICOM_plan_files(
    rp,
    rs,
    rd,
    plan_parameters,
    dwell_times_nominal_robust,
)

rp_robust.save_as(str(save_folder) + "/PL001_Robust_Optimised.dcm")
rd_robust.save_as(str(save_folder) + "/DO001_Robust_Optimised.dcm")
