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

from robustbrachy.robustevaluation.data_extraction import *
from robustbrachy.robustoptimisation.robust_optimisation import *
from robustbrachy.robustoptimisation.evaluation_cpu import *
from robustbrachy.robustoptimisation.robust_evaluation_vectorised_cpu import *


################################
## Running Robust Evaluation  ##
################################


def run_robust_optimiser_program_cpu(
    file_rp,
    file_rs,
    file_rd,
    file_source_data,
    optimiser_parametres,
):
    # get dictionary of plan parameters (structures, dwell points, times, source dose arrays, etc.)
    plan_parameters, _, _, _ = extract_all_needed_data(
        file_rp, file_rs, file_rd, file_source_data
    )

    # keep arrays in CPU.
    # float32 is used since it is more memory efficient, and GPU memory is generally limited.
    plan_parameters = arrays_to_numpy(plan_parameters)

    # run the robust optimisation
    (
        results,
        full_time,
        dwell_structure_array,
        dwell_times,
        all_dose_per_dwell_per_vol_pt,
    ) = robust_optimisation(plan_parameters, optimiser_parametres)

    if not isinstance(results.X, np.ndarray):
        print("No solutions found, try changing parameters")

    else:
        dwell_times_pareto_front = np.array(results.X).round(1)
        dwell_times_pareto_front[dwell_times_pareto_front < 0.4] = 0.0
        solutions_pareto_front = np.array(100 - results.F)

        # nominal metrics
        (
            nominal_metrics_pareto_front,
            all_nominal_dvhs_pareto_front,
        ) = get_nominal_metrics_array_cpu(
            dwell_times_pareto_front,
            all_dose_per_dwell_per_vol_pt[0],
            all_dose_per_dwell_per_vol_pt[1],
            all_dose_per_dwell_per_vol_pt[2],
            plan_parameters["prescribed_dose"],
            plan_parameters["urethra_vol"],
            plan_parameters["rectum_vol"],
        )

        # table of approximate pass-rates and nominal metrics
        data_to_print = np.concatenate(
            (solutions_pareto_front, nominal_metrics_pareto_front.T), axis=1
        ).astype(float)
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
        df_pareto_front_data = (
            df_pareto_front_data.rename_axis("Plan").reset_index().round(3)
        )

        # Robust Evaluate All Solutions in parallel (vectorised)

        # conduct robust evaluation of all solutions on pareto front
        (
            all_passrate,
            D90_passrate,
            DmaxU_passrate,
            DmaxR_passrate,
            all_robust_dvh_summary,
        ) = robust_measure_array_cpu(
            optimiser_parametres,
            plan_parameters,
            dwell_times_pareto_front,
            all_nominal_dvhs_pareto_front,
        )

        all_robust_dvhs = all_robust_dvh_summary
        pass_rates = np.array(
            [all_passrate, D90_passrate, DmaxU_passrate, DmaxR_passrate]
        )

        data_to_print = np.concatenate(
            (pass_rates.T, nominal_metrics_pareto_front.T), axis=1
        ).astype(float)
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
        df_pareto_front_data_RE = df_pareto_front_data_RE.round(3)

    return (
        results,
        full_time,
        df_pareto_front_data,
        df_pareto_front_data_RE,
        all_robust_dvhs,
        all_nominal_dvhs_pareto_front,
        dwell_times_pareto_front
    )


__all__ = ["run_robust_optimiser_program_cpu"]
