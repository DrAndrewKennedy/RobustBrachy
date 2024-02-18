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
import pandas as pd
from pydicom import dcmread

from robustbrachy.robustevaluation.data_extraction import *
from robustbrachy.robustevaluation.robust_evaluation_cpu import *
from robustbrachy.robustevaluation.utils_cpu import *

################################
## Running Robust Evaluation  ##
################################


def run_robust_evaluation_program_cpu(
    file_rp,
    file_rs,
    file_rd,
    file_source_data,
    no_of_runs,
    df_dose_metrics,
    df_volume_metrics,
    uncertainty_magnitudes,
):
    # get dictionary of plan parameters (structures, dwell points, times, source dose arrays, etc.)
    plan_parameters, _, _, _ = extract_all_needed_data(
        file_rp, file_rs, file_rd, file_source_data
    )

    # store as arrays with float32 to be consistent with gpu version.
    # float32 is used since it is more memory efficient, and GPU memory is generally limited.
    plan_parameters = arrays_to_numpy(plan_parameters)

    # *** preparing parameters  ***

    # ** calculate parameter variable SD **
    uncertainties_SD = calculate_parameter_variable_SDs(uncertainty_magnitudes)

    # get dose and volume metric labels
    (
        passrates_labels_D,
        metric_labels_D,
        _,
        passrates_labels_V,
        metric_labels_V,
        _,
    ) = make_dvh_metric_labels(df_dose_metrics, df_volume_metrics)

    # *** conduct robust evaluation  ***

    # conducting the simulations, dose calculation, DVH metrics, then output robustness information
    (
        pass_rates,
        overall_pass_rate,
        all_robust_dvh_summary,
        all_nominal_dvhs,
    ) = probabilistic_robust_measure_cpu(
        no_of_runs,
        plan_parameters,
        df_dose_metrics,
        df_volume_metrics,
        uncertainties_SD,
    )

    #  *** nominal data ***
    # calculate nominal metrics
    df_nominal_metric_data = calculate_all_nominal_metrics_cpu(
        df_dose_metrics,
        df_volume_metrics,
        metric_labels_D,
        metric_labels_V,
        all_nominal_dvhs,
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

    # dataframe of pass-rates for all metrics
    df_pass_rates = pd.DataFrame(
        data=[[overall_pass_rate, *pass_rates]],
        columns=[
            "Overall Pass-rate",
            *passrates_labels_D,
            *passrates_labels_V,
        ],
    )

    return (
        df_nominal_data,
        df_nominal_metric_data,
        df_pass_rates,
        all_nominal_dvhs,
        all_robust_dvh_summary,
    )


__all__ = ["run_robust_evaluation_program_cpu"]
