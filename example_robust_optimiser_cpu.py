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
import webbrowser
import warnings

np.warnings = warnings
np.warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
np.seterr(divide="ignore")

from robustbrachy.robustoptimisation.run_robust_optimiser_cpu import *
from robustbrachy.robustevaluation.display_results import *

path = Path(__file__).parent.absolute()
path_temp = str(Path(__file__).parent.absolute()) + "/temp/"
path_example = str(Path(__file__).parent.absolute()) + "/Example Plan/"
path_source_data = str(Path(__file__).parent.absolute()) + "/source data/"

# make directory if it doesn't exist already
Path(path_temp).mkdir(parents=True, exist_ok=True)

######################
## define variables ##
######################

# define location of plan and source files.
# Currently looking in the example folder using 'path_example' above.
file_rp = str(path_example) + "/PL001.dcm"
file_rs = str(path_example) + "/SS001.dcm"
file_rd = str(path_example) + "/DO001.dcm"
file_source_data = str(path_source_data) + "/192ir-hdr_gammamed_plus.xls"

# define save location, DVH's saved as numpy data files (.npy)
save_folder = str(Path(__file__).parent.absolute()) + "/temp/"

# *** optimiser parameters ***
# Number of iterations
num_of_itr = int(160)  # int(400)

# Population size
pop_size = 200  # int(200)

# Limit the distance dwell points can be outside the prostate (units in mm)
limit_on_dwells_outside_prostate_mm = float(4.5)

# Maximum dwell time limit (units in seconds)
max_time_limit = float(31.0)

# Show optimiser progress in the cmd window (True = 1 or False = 0)
show_progress = bool(int(1))

# Offspring size to generate each iteration
offspring_size = 100  # int(200)

# Point crossover probability. see pymoo documentation.
point_crossover_prob = float(1.0)  # float(0.0)

# Number of point crossover. see pymoo documentation.
num_of_point_crossovers = int(50)

# Probability of mutation of dwell times in offspring. see pymoo documentation.
mutation_prob = float(1.0)

# Mutation probability function spread (smaller = less spread). see pymoo documentation.
mutation_spread = int(50)

# D90 excess margin before constraint is satisfied (as a % of constraint)
margin_prostate = float(10.7)

# DmaxU excess margin before constraint is satisfied (as a % of constraint)
margin_urethra = float(9.0)

# DmaxR excess margin before constraint is satisfied (as a % of constraint)
margin_rectum = float(9.2)

# Number of Uncertainty Scenarios in final robust evaluation of all plans on pareto front
no_of_runs = 10  # int(500)

# Number of robust evaluation iterations to split into. Reduce this if there are out of memory error.
no_of_runs_internal = int(50)

use_gpu_in_eval = False

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
    "use_gpu_in_eval": use_gpu_in_eval,
}

##################################
## Running Robust Optimisaiton  ##
##################################


(
    results,
    full_time,
    df_pareto_front_data,
    df_pareto_front_data_RE,
    all_robust_dvhs,
    all_nominal_dvhs_pareto_front,
    dwell_times_pareto_front
) = run_robust_optimiser_program_cpu(
    file_rp,
    file_rs,
    file_rd,
    file_source_data,
    optimiser_parametres,
)

################################
##   Displaying information   ##
################################

print(
    'check browser for tables of robust optimised results. excel spreadsheets are saved to the specified save location. Choose a plan to load in the "example_explore_robust_optimised_plan.py"'
)
#  *** function to write all data to html and display it ***
write_to_html_file(
    [df_pareto_front_data_RE, df_pareto_front_data],
    ["Robust Evaluated Plans from Pareto Front", "Output from Robust Optimiser"],
    "df_robust_optimiser.html",
)
webbrowser.open("df_robust_optimiser.html")
df_pareto_front_data_RE.to_excel(save_folder + "df_pareto_front_data_RE.xlsx")
df_pareto_front_data.to_excel(save_folder + "df_pareto_front_data.xlsx")


# save files to access and view in another script file
np.save(save_folder + "dwell_times_pareto_front.npy", dwell_times_pareto_front)
np.save(save_folder + "all_robust_dvhs_mu_sd.npy", all_robust_dvhs)
np.save(save_folder + "all_nominal_dvhs.npy", all_nominal_dvhs_pareto_front)
