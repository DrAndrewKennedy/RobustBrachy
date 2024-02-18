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
import time

try:
    import cupy as cp
    from robustbrachy.robustevaluation.utils_gpu import *
    from robustbrachy.robustoptimisation.dose_per_dwell_gpu import *
    from robustbrachy.robustoptimisation.evaluation_gpu import *
except:
    print("no cupy")

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.operators.crossover.pntx import PointCrossover
from pymoo.operators.mutation.pm import PM

from robustbrachy.robustevaluation.utils_cpu import *
from robustbrachy.robustevaluation.fast_TG43_cpu import *
from robustbrachy.robustoptimisation.initialise_population import *
from robustbrachy.robustoptimisation.dose_per_dwell_cpu import *
from robustbrachy.robustoptimisation.evaluation_cpu import *


#####################################
###      Robust Optimisation      ###
#####################################


def robust_optimisation(plan_parameters, optimiser_parametres, progress_bar=False):
    start = time.time()
    # defining parameters

    # This defines a piece-wise linear equation that sets limits on dwell times based on distance from urethra. It is a deterministic process to intiate population.
    # x = distance from urethra, y = dwell time limit.
    # It is a sloped line from (0, 0) to (x = max_dist_range, y = max_time_limit), then a hoizontal line (y = max_time_limit).
    time_limit_lin_fn_urethra = [
        3,
        -3,
        11.0,
        0,
    ]  # slope, intercept, max_dist_range, urethra_2D_uncertainty_movement

    # ignored for rectum.
    time_limit_lin_fn_rectum = [
        0.0,
        0.0,
        0.0,
        0.0,
    ]  # slope, intercept, max_dist_range, rectum_2D_uncertainty_movement

    # this is for the original structure of the dwell coordinates
    dwell_structure_array = (
        plan_parameters["dwell_coords_optimisation"][:, :, 0] != -100
    ).astype(np.bool_)

    # Calculate dose (with dwell times of 1 s) at all points of each structure and construct an array

    # create array of dwell times initialised as 1.0 s at each point. this will be changed from 1.0 s in the optimisation method.
    dwell_times = np.empty(
        shape=(
            plan_parameters["dwell_coords_optimisation"].shape[0],
            plan_parameters["dwell_coords_optimisation"].shape[1],
        ),
        dtype=np.float32,
    )
    dwell_times.fill(1.0)
    dwell_times[plan_parameters["dwell_coords_optimisation"][:, :, 0] == -100] = -100

    # get the dose calculation points per structure
    dose_calc_pts_prostate = get_dose_volume_pts_cpu(
        plan_parameters["prostate_contour_pts"],
        plan_parameters["urethra_contour_pts"],
        voxel_size=1,
    )
    dose_calc_pts_urethra = get_dose_volume_pts_cpu(
        plan_parameters["urethra_contour_pts"], voxel_size=1
    )
    dose_calc_pts_rectum = get_dose_volume_pts_cpu(
        plan_parameters["rectum_contour_pts"], voxel_size=1
    )

    all_dose_calc_pts = list(
        [dose_calc_pts_prostate, dose_calc_pts_urethra, dose_calc_pts_rectum]
    )

    # get dose rate arrays, the dose-rate at each dose calculation point due to each dwell point (so a 2D array). Note: dose calc point = volume point.
    dose_per_dwell_per_vol_pt_prostate = TG43calc_dose_per_dwell_per_vol_pt_cpu(
        dose_calc_pts_prostate,
        dwell_times,
        plan_parameters,
        plan_parameters["dwell_coords_optimisation"],
        plan_parameters["dwell_pts_source_end_sup_optimisation"],
        plan_parameters["dwell_pts_source_end_inf_optimisation"],
    )

    dose_per_dwell_per_vol_pt_urethra = TG43calc_dose_per_dwell_per_vol_pt_cpu(
        dose_calc_pts_urethra,
        dwell_times,
        plan_parameters,
        plan_parameters["dwell_coords_optimisation"],
        plan_parameters["dwell_pts_source_end_sup_optimisation"],
        plan_parameters["dwell_pts_source_end_inf_optimisation"],
    )

    dose_per_dwell_per_vol_pt_rectum = TG43calc_dose_per_dwell_per_vol_pt_cpu(
        dose_calc_pts_rectum,
        dwell_times,
        plan_parameters,
        plan_parameters["dwell_coords_optimisation"],
        plan_parameters["dwell_pts_source_end_sup_optimisation"],
        plan_parameters["dwell_pts_source_end_inf_optimisation"],
    )

    all_dose_per_dwell_per_vol_pt = list(
        [
            dose_per_dwell_per_vol_pt_prostate,
            dose_per_dwell_per_vol_pt_urethra,
            dose_per_dwell_per_vol_pt_rectum,
        ]
    )

    # intialise the population. This runs few a number of determinitic and random methods to generate the intial population of solutions.
    # note: this method may produce more than the required pop size but hte algorithm will make a selection of the best solutions of required pop_size in the pymoo genetic algorithm
    (
        pop_dwell_times,
        mask_dwells_over_mm_from_prostate,
        upper_time_limit_array,
    ) = generate_dwells_for_pop_optimisation(
        time_limit_lin_fn_urethra,
        time_limit_lin_fn_rectum,
        dwell_times,
        all_dose_calc_pts,
        all_dose_per_dwell_per_vol_pt,
        plan_parameters,
        optimiser_parametres,
    )

    pop_dwell_times = np.array(pop_dwell_times, dtype=np.float32)

    # the array of upper dwell time limits for the optimiser (distance from urethra or rectum not considered here)
    upper_limit_arry = (~mask_dwells_over_mm_from_prostate) * optimiser_parametres[
        "max_time_limit"
    ]

    num_dwells = len(mask_dwells_over_mm_from_prostate)

    # optimiser parameters
    stop_criteria = ("n_gen", optimiser_parametres["num_of_itr"])

    algorithm = NSGA2(
        pop_size=optimiser_parametres["pop_size"],
        n_offsprings=optimiser_parametres["offspring_size"],
        sampling=pop_dwell_times,
        crossover=PointCrossover(
            prob=optimiser_parametres["point_crossover_prob"],
            n_points=optimiser_parametres["num_of_point_crossovers"],
        ),
        mutation=PM(
            prob=optimiser_parametres["mutation_prob"],
            eta=optimiser_parametres["mutation_spread"],
        ),
        eliminate_duplicates=True,
        save_history=False,
        verbose=optimiser_parametres["show_progress"],
    )

    # these are margins placed on the D90, DmaxU, and DmaxR sigmoid functions so that they would be 100% pass rate.
    # they are dynamic and lower when the optimiser doesn't get any closer to a solution.
    # these are commented out since they are set in the function call
    # margin_prostate = 10.7 # as a percentage
    # margin_urethra = 9 # as a percentage
    # margin_rectum = 9.2  # as a percentage
    set_margins_to_0_after_solns_start = True
    current_loop = 1

    # this is forming the pymoo optimiser problem
    problem = MyProblem(
        plan_parameters,
        all_dose_per_dwell_per_vol_pt,
        all_dose_calc_pts,
        current_loop,
        optimiser_parametres,
        set_margins_to_0_after_solns_start,
        num_dwells,
        upper_limit_arry,
        progress_bar,
    )

    results = minimize(
        problem,
        algorithm,
        termination=stop_criteria,
        seed=1,
        pf=problem.pareto_front(use_cache=False),
        save_history=False,
        verbose=optimiser_parametres["show_progress"],
    )

    full_time = time.time() - start

    return (
        results,
        full_time,
        dwell_structure_array,
        dwell_times,
        all_dose_per_dwell_per_vol_pt,
    )


# setting up the problem using the pymoo framework
class MyProblem(Problem):
    def __init__(
        self,
        plan_parameters,
        all_dose_per_dwell_per_vol_pt,
        all_dose_calc_pts,
        current_loop,
        optimiser_parametres,
        set_margins_to_0_after_solns_start,
        num_dwells,
        upper_limit_arry,
        progress_bar,
    ):
        super().__init__(
            n_var=num_dwells,
            n_obj=3,
            n_constr=8,
            xl=np.array([0 for _ in range(len(upper_limit_arry))]),
            xu=upper_limit_arry,
        )

        self.urethra_volume = plan_parameters["urethra_vol"]
        self.rectum_volume = plan_parameters["rectum_vol"]
        self.prescribed_dose = plan_parameters["prescribed_dose"]
        self.all_dose_per_dwell_per_vol_pt = all_dose_per_dwell_per_vol_pt
        self.all_dose_calc_pts = all_dose_calc_pts
        self.current_loop = current_loop
        self.margin_prostate = optimiser_parametres["margin_prostate"]
        self.margin_urethra = optimiser_parametres["margin_urethra"]
        self.margin_rectum = optimiser_parametres["margin_rectum"]
        self.use_gpu_in_eval = optimiser_parametres["use_gpu_in_eval"]
        self.progress_bar = progress_bar
        self.num_of_itr = optimiser_parametres["num_of_itr"]

        # set values to start with for all constraints (g variables)
        self.g1_old = 5
        self.g2_old = 5
        self.g3_old = 5
        self.time_gpu = 0
        self.time_cpu = 0
        self.Volume_percentage_u = 0.01 / plan_parameters["urethra_vol"] * 100
        self.Volume_percentage_r = 0.1 / plan_parameters["rectum_vol"] * 100
        self.set_margins_to_0_after_solns_start = set_margins_to_0_after_solns_start
        self.start_time = time.time()
        self.total_plans_evaluated = 0

    def _evaluate(self, x, out, *args, **kwargs):
        # start = time.time()
        runtime = time.time() - self.start_time
        self.start_time = time.time()
        pop_dwell_times = x.round(1)
        self.total_plans_evaluated = self.total_plans_evaluated + len(pop_dwell_times)
        # sets the margins initially
        if self.current_loop == 1:
            self.margin_prostate = self.margin_prostate / 100 * self.prescribed_dose
            self.margin_urethra = (
                self.margin_urethra / 100 * 1.10 * self.prescribed_dose
            )
            self.margin_rectum = self.margin_rectum / 100 * 13

        if self.use_gpu_in_eval == True:
            # tests which is faster, gpu or cpu and then runs it on the faster option
            if self.current_loop < 3:
                f, g, g1, g2, g3 = gpu_eval(self, pop_dwell_times)
                used_gpu = True

            elif self.current_loop < 9:  # current_loop starts at 1
                start_gpu = time.time()
                f, g, g1, g2, g3 = gpu_eval(self, pop_dwell_times)
                used_gpu = True
                self.time_gpu = self.time_gpu + (time.time() - start_gpu)

            elif self.current_loop < 14:
                start_cpu = time.time()
                f, g, g1, g2, g3 = cpu_eval(self, pop_dwell_times)
                self.time_cpu = self.time_cpu + (time.time() - start_cpu)
                used_gpu = False

            # remaining loops
            else:
                if self.time_gpu < self.time_cpu:
                    f, g, g1, g2, g3 = gpu_eval(self, pop_dwell_times)
                    used_gpu = True
                else:
                    f, g, g1, g2, g3 = cpu_eval(self, pop_dwell_times)
                    used_gpu = False

        # use cpu only
        else:
            f, g, g1, g2, g3 = cpu_eval(self, pop_dwell_times)
            used_gpu = False

        # calculate the minimum values of the three constraints of interest
        min_gs = np.column_stack([g1, g2, g3])
        min_gs[min_gs < 0.0] = 0.0
        min_gs_index = np.linalg.norm(
            min_gs - np.array([0.0, 0.0, 0.0])[np.newaxis], axis=1
        ).argmin()

        # set the old values to the prevous ones so change between iterations can be calculated
        if self.current_loop == 1:
            self.g1_old = g1[min_gs_index]
            self.g2_old = g2[min_gs_index]
            self.g3_old = g3[min_gs_index]

        # check if solutions have gotten closer to constaints over 5 iterations
        # if not, decrease margins by 0.1
        if self.current_loop != 1 and self.current_loop % 5 == 0:
            if g1[min_gs_index] > 0:
                if not g1[min_gs_index] < self.g1_old:
                    self.margin_prostate = self.margin_prostate - 0.1
                    if self.margin_prostate < 0:
                        self.margin_prostate = 0

            if g2[min_gs_index] > 0:
                if not g2[min_gs_index] < self.g2_old:
                    self.margin_urethra = self.margin_urethra - 0.1
                    if self.margin_urethra < 0:
                        self.margin_urethra = 0

            if g3[min_gs_index] > 0:
                if not g3[min_gs_index] < self.g3_old:
                    self.margin_rectum = self.margin_rectum - 0.1
                    if self.margin_rectum < 0:
                        self.margin_rectum = 0

            # if all constraints are now being met, drop the margins so as to increase diversity of solutions
            if (
                g1[min_gs_index] <= 0
                and g2[min_gs_index] <= 0
                and g3[min_gs_index] <= 0
                and self.current_loop % 50 == 0
                and self.set_margins_to_0_after_solns_start == True
            ):
                self.margin_rectum = 0
                self.margin_urethra = 0
                self.margin_prostate = 0

            self.g1_old = g1[min_gs_index]
            self.g2_old = g2[min_gs_index]
            self.g3_old = g3[min_gs_index]

        # send all objective values and constrain values to the next iteration
        out["F"] = f
        out["G"] = g

        # progress bar
        if self.progress_bar != False:
            dist_from_satisfying = 0
            if g1[min_gs_index] > 0:
                dist_from_satisfying = dist_from_satisfying + g1[min_gs_index] ** 2
            if g2[min_gs_index] > 0:
                dist_from_satisfying = dist_from_satisfying + g2[min_gs_index] ** 2
            if g3[min_gs_index] > 0:
                dist_from_satisfying = dist_from_satisfying + g3[min_gs_index] ** 2
            dist_from_satisfying = round(np.sqrt(dist_from_satisfying), 4)
            if used_gpu == True:
                gpu_text = "GPU"
            else:
                gpu_text = "CPU"

            self.progress_bar.progress(
                self.current_loop / self.num_of_itr,
                text="Iteration number "
                + str(int(self.current_loop))
                + " out of "
                + str(self.num_of_itr)
                + ", iteration run time of "
                + str(round(runtime, 4))
                + " seconds using the "
                + gpu_text
                + ". Solutions are "
                + str(dist_from_satisfying)
                + " Gy away from constraints. A total of "
                + str(self.total_plans_evaluated)
                + " treatment plans evaluated.",
            )

        # increase loop
        self.current_loop = self.current_loop + 1


def gpu_eval(self, pop_dwell_times):
    # send population of dwell times to gpu
    pop_dwell_times = cp.asarray(pop_dwell_times)

    # set lower dwell times to zero
    pop_dwell_times[pop_dwell_times < 0.4] = 0.0

    # get DVHs for entire population
    all_DVHs_pop = get_dvh_for_population_optimisation_gpu(
        pop_dwell_times, self.all_dose_per_dwell_per_vol_pt, self.prescribed_dose
    )

    #### calculate the objective functions
    # get the D90 metric
    volume_percentage = cp.array([90.0], dtype=cp.float32)
    D90_pop = get_population_dose_metrics_gpu(volume_percentage, all_DVHs_pop[:, 0])

    # get the % difference between metric and the objective
    D90_pop_diff = (D90_pop - self.prescribed_dose) / (self.prescribed_dose) * 100

    # get approximate pass rates for each row using the difference above (this sigmoid has been modelled as part of building the optimiser)
    pass_rate_p = sigmoid(D90_pop_diff, 102, 2.8, 0.26, -2.1)

    # do the same as above but for DmaxU
    volume_percentage = cp.array([self.Volume_percentage_u], dtype=cp.float32)
    DmaxU_pop = get_population_dose_metrics_gpu(volume_percentage, all_DVHs_pop[:, 1])
    DmaxU_pop_diff = (
        (DmaxU_pop - 1.10 * self.prescribed_dose) / (1.10 * self.prescribed_dose) * 100
    )
    pass_rate_u = sigmoid(DmaxU_pop_diff, -102, -3.5, 0.31, 100)

    # do the same as above but for DmaxR
    volume_percentage = cp.array([self.Volume_percentage_r], dtype=cp.float32)
    DmaxR_pop = get_population_dose_metrics_gpu(volume_percentage, all_DVHs_pop[:, 2])
    DmaxR_pop_diff = (DmaxR_pop - 13) / (13) * 100
    pass_rate_r = sigmoid(DmaxR_pop_diff, -105, -2, 0.222, 100)

    # the three objective functions for the optimiser:
    # Want to maximise the pass rates but pymoo only minisers. so "100 - pass_rate" simply gets closer to zero as the pass rates in crease therefor turning a maximisation problem into a minimisation one.
    f1 = 100 - pass_rate_p
    f2 = 100 - pass_rate_u
    f3 = 100 - pass_rate_r

    #### calculate the constraints

    # place a margin on the constraints to push results further from the constaints and increase the pass rate.
    # Algorithm pymoo seems to be agressive to adhere to constraints first then look at minimising the objective function.
    # So it is quicker to come to an acceptable solutions.

    # g(x) <= 0 are how the constrants are defined in pymoo and h(x) = 0 is how equalities are defined
    g1 = (
        self.prescribed_dose - D90_pop + self.margin_prostate
    )  # prescribed_dose - D90_pop <= 0; i.e. want D90 to be larger than the prescribed dose
    g2 = (
        DmaxU_pop - 1.10 * self.prescribed_dose + self.margin_urethra
    )  # DmaxU_pop - 1.10 * prescribed_dose <= 0; i.e. want DmaxU to be less than 110% of the prescribed dose
    g3 = (
        DmaxR_pop - 13 + self.margin_rectum
    )  # DmaxR_pop - 13 <= 0; i.e. want DmaxR to be less than 13 Gy

    # the following constraints are the other clinical ones that should be adhered to in the clinical technqiue at the clinic this optimiser was built for
    # V100
    dose_of_interest = cp.array([self.prescribed_dose], dtype=cp.float32)
    V100_pop = get_population_volume_metrics_gpu(dose_of_interest, all_DVHs_pop[:, 0])

    # V150
    dose_of_interest = cp.array([self.prescribed_dose * 1.5], dtype=cp.float32)
    V150_pop = get_population_volume_metrics_gpu(dose_of_interest, all_DVHs_pop[:, 0])

    # V200
    dose_of_interest = cp.array([self.prescribed_dose * 2.0], dtype=cp.float32)
    V200_pop = get_population_volume_metrics_gpu(dose_of_interest, all_DVHs_pop[:, 0])

    # D10
    volume_percentage = cp.array([10], dtype=cp.float32)
    D10_pop = get_population_dose_metrics_gpu(volume_percentage, all_DVHs_pop[:, 1])

    # V75
    dose_of_interest = cp.array([self.prescribed_dose * 0.75], dtype=cp.float32)
    V75_pop = get_population_volume_metrics_gpu(dose_of_interest, all_DVHs_pop[:, 2])
    V75_pop = V75_pop / 100 * float(self.rectum_volume)
    g4 = (
        90 - V100_pop
    )  # 90 - V100_pop <= 0; i.e. want the prescribed dose to reach over 90% of the volume
    g5 = (
        V150_pop - 35
    )  # V150_pop - 40 <= 0; i.e.want less than 150 % of the prescribed dose to reach less than 35% of the volume
    g6 = (
        V200_pop - 15
    )  # V200_pop - 10 <= 0; i.e.want less than 200 % of the prescribed dose to reach less than 15% of the volume
    g7 = (
        D10_pop - 17
    )  # D10_pop - 17 <= 0; i.e. want the hotest 10% of the urthra volume to be less than  17 Gy ###note should be competed by DmaxU by default
    g8 = (
        V75_pop - 0.6
    )  # V75_pop - 1 ; i.e. want any 1 cc of the rectum to be less than 75% of the prescribe dose

    # put objective functions together and constraints
    f = cp.asnumpy(cp.column_stack([f1, f2, f3]))

    g = cp.asnumpy(cp.column_stack([g1, g2, g3, g4, g5, g6, g7, g8]))

    # pass metrics of interest back to cpu and return them to the called script
    g1 = cp.asnumpy(g1)
    g2 = cp.asnumpy(g2)
    g3 = cp.asnumpy(g3)

    return f, g, g1, g2, g3


def cpu_eval(self, pop_dwell_times):
    # see notes in gpu version

    pop_dwell_times[pop_dwell_times < 0.4] = 0.0

    all_DVHs_pop = get_dvh_for_population_optimisation_cpu(
        pop_dwell_times, self.all_dose_per_dwell_per_vol_pt, self.prescribed_dose
    )

    volume_percentage = np.array([90], dtype=np.float32)  # 90.4
    D90_pop = get_population_dose_metrics_cpu(volume_percentage, all_DVHs_pop[:, 0])
    D90_pop_diff = (D90_pop - self.prescribed_dose) / (self.prescribed_dose) * 100
    pass_rate_p = sigmoid(D90_pop_diff, 102, 2.8, 0.26, -2.1)

    volume_percentage = np.array([self.Volume_percentage_u], dtype=np.float32)
    DmaxU_pop = get_population_dose_metrics_cpu(volume_percentage, all_DVHs_pop[:, 1])
    DmaxU_pop_diff = (
        (DmaxU_pop - 1.10 * self.prescribed_dose) / (1.10 * self.prescribed_dose) * 100
    )
    pass_rate_u = sigmoid(DmaxU_pop_diff, -102, -3.5, 0.31, 100)

    volume_percentage = np.array([self.Volume_percentage_r], dtype=np.float32)
    DmaxR_pop = get_population_dose_metrics_cpu(volume_percentage, all_DVHs_pop[:, 2])

    DmaxR_pop_diff = (DmaxR_pop - 13) / (13) * 100
    pass_rate_r = sigmoid(DmaxR_pop_diff, -105, -2, 0.222, 100)

    f1 = 100 - pass_rate_p
    f2 = 100 - pass_rate_u
    f3 = 100 - pass_rate_r

    g1 = self.prescribed_dose - D90_pop + self.margin_prostate
    g2 = DmaxU_pop - 1.10 * self.prescribed_dose + self.margin_urethra
    g3 = DmaxR_pop - 13 + self.margin_rectum

    # V100
    dose_of_interest = np.array([self.prescribed_dose], dtype=np.float32)
    V100_pop = get_population_volume_metrics_cpu(dose_of_interest, all_DVHs_pop[:, 0])

    # V150
    dose_of_interest = np.array([self.prescribed_dose * 1.5], dtype=np.float32)
    V150_pop = get_population_volume_metrics_cpu(dose_of_interest, all_DVHs_pop[:, 0])

    # V200
    dose_of_interest = np.array([self.prescribed_dose * 2.0], dtype=np.float32)
    V200_pop = get_population_volume_metrics_cpu(dose_of_interest, all_DVHs_pop[:, 0])

    # D10
    volume_percentage = np.array([10], dtype=np.float32)
    D10_pop = get_population_dose_metrics_cpu(volume_percentage, all_DVHs_pop[:, 1])

    # V75
    dose_of_interest = np.array([self.prescribed_dose * 0.75], dtype=np.float32)
    V75_pop = get_population_volume_metrics_cpu(dose_of_interest, all_DVHs_pop[:, 2])
    V75_pop = V75_pop / 100 * self.rectum_volume

    g4 = 90 - V100_pop
    g5 = V150_pop - 35
    g6 = V200_pop - 15
    g7 = D10_pop - 17
    g8 = V75_pop - 0.6

    f = np.column_stack([f1, f2, f3])
    g = np.column_stack([g1, g2, g3, g4, g5, g6, g7, g8])

    return f, g, g1, g2, g3


def sigmoid(x, L, x0, k, b):
    # pass-rates from nominal metrics are modelled based on a sigmoid function
    y = L / (1 + np.exp(-k * (x - x0))) + b
    y[y > 100] = 100
    y[y < 0] = 0
    return y


__all__ = ["robust_optimisation"]
