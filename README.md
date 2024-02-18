# RobustBrachy
## Overview
A robust evaluation and robust optimisation algorithms for HDR prostate Brachytherapy, which includes 14 major uncertainty sources in each simulation and coded in Python.
Both algorithms have been incorporated into a streamlit web app with a graphical user interface.

### Published works detailing both algorithms
- <ins>For a comprehensive explanation of the robust evaluation algorithm, see:</ins>  
Kennedy, A.C., Douglass, M.J.J. & Santos, A.M.C. Being certain about uncertainties: a robust evaluation method for high-dose-rate prostate brachytherapy treatment plans including the combination of uncertainties. Phys Eng Sci Med 46, 1115–1130 (2023). https://doi.org/10.1007/s13246-023-01279-8

- <ins>For an evaluation of the robust evaluation algorithm on a 49-patient dataset, see:</ins>  
Kennedy, A.C., Douglass, M.J.J. & Santos, A.M.C. A robust evaluation of 49 high-dose-rate prostate brachytherapy treatment plans including all major uncertainties. J Appl Clin Med Phys. 2023;e14182. https://doi.org/10.1002/acm2.14182

- <ins>For a comprehensive explanation of the robust optimisation algorithm, see:</ins>  
Kennedy, A.C., Douglass, M.J.J. & Santos, A.M.C. Implementation of a robust optimisation genetic algorithm for HDR prostate brachytherapy including all major uncertainties, 31 December 2023, PREPRINT (Version 1) available at Research Square https://doi.org/10.21203/rs.3.rs-3820773/v1

- <ins>For an evaluation of the robust optimisation algorithm on a 49-patient dataset, see:</ins>  

> [!IMPORTANT]
> The software was designed for treatment plans generated by Vitesse v4.0.1 (Varian Medical Systems, Palo Alto, USA) and would most likely require modifications for other TPS-generated treatment plans.

> [!NOTE]
> This software is limited to research puposes only, no warranty or support will be given.

# File Descriptions
Below is a table that describes the purpose of each file:

| **Folder/File name**                                                       | **Description**                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| -------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **_Root folder_** |   |
| LICENSE | Licence information file. |
| README.md | RobustBrachy repository information |
| example_nominal_plan_analysis_cpu.py<br>example_nominal_plan_analysis_gpu.py | An example use script to perform a TG43 dose calculation and display interactive DVH plots and DVH metric data. |
| example_robust_evaluation_cpu.py<br>example_robust_evaluation_gpu.py         | An example use script to perform a robust evaluation of a treatment plan. Displaying interactive DVH plots with probabilistic robustness bands, DVH metric robustness pass-rates and nominal plan DVH metric data as output.|
| example_robust_optimiser_cpu.py<br>example_robust_optimiser_gpu.py           | An example use script to perform a robust optimisation for a patient. Input is a set of treatment DICOM files in which the set of contoured structures and reconstructed needles are imputed into the robust optimiser (dwell points are allocated automatically). Output is multiple robust treatment plans, each a set of dwell times that are the best trade-off for three robustness pass-rates (D90 for the prostate and the maximum doses to the urethra and rectum). The DVH metric robustness pass-rates and nominal plan DVH metrics are also outputted for each plan. |
| example_explore_robust_optimised_plan.py                                   | An example use script to display interactive DVH plots with probabilistic robustness bands for one chosen plan from the set of robust optimised plans. The three DICOM treatment files were also generated and saved within the script. |
| pyproject.toml                                                             | The file is used to install RobustBrachy and all its python dependencies. |
| environments_cpu.yml<br>environments_gpu.yml                                 | The file is used to install anaconda virtual environments for RobustBrachy. Included are the Python version and dependency Python package versions. GPU version requires a Nivida GPU with CUDA. |
| **_Example Plan/_**                                                        | The folder contains an example set of DICOM treatment plan files for a phantom. |
| DO001.dcm                                                                  | Example dose RT DICOM file.  |
| PL001.dcm                                                                  | Example plan RT DICOM file. |
| SS001.dcm                                                                  | Example structures RT DICOM file.  |
| **_Source Data/_**                                                         | Folder containing the brachy source data file  |
| 192ir-hdr_gammamed_plus.xls                                                | File Sourced from: [https://www.estro.org/ESTRO/media/ESTRO/About/Committees/GEC%20-ESTRO/BRAPHYQS/Ir-192%20HDR/192ir-hdr_gammamed_plus.xls](https://www.estro.org/ESTRO/media/ESTRO/About/Committees/GEC%20-ESTRO/BRAPHYQS/Ir-192%20HDR/192ir-hdr_gammamed_plus.xls) <br><br><ins>Article Reference:</ins> <br>Perez-Calatayud, J., Ballester, F., Das, R.K., DeWerd, L.A., Ibbott, G.S., Meigooni, A.S., Ouhib, Z., Rivard, M.J., Sloboda, R.S. and Williamson, J.F. (2012), Dose calculation for photon-emitting brachytherapy sources with average energy higher than 50 keV: Report of the AAPM and ESTRO. Med. Phys., 39: 2904-2929. [https://doi.org/10.1118/1.3703892](https://doi.org/10.1118/1.3703892) |
| **_RobustBrachy/_**                                                        | Source folder for RobustBrachy  |
| **_RobustBrachy/robustevaluation/_**                                       | The folder contains all robust evaluation related functions.|
| run_robust_evaluation_gpu.py run_robust_evaluation_cpu.py                 | The function that calls all other functions to run a robust evaluation. Input are DICOM file locations, clinical objectives defined as DVH metrics, and the 14 uncertainty magnitudes. Output is probabilistic robustness DVH bands, nominal DVHs, DVH metric robustness pass-rates, and nominal DVH metrics.  |
| data_extraction.py                                                         | Extracts all structures, dwell points, dwell times and other treatment plan parameters from the treatment DCIOM files and transforms them into suitable data structures.  |
| robust_evaluation_cpu.py<br>robust_evaluation_gpu.py                         | The function that calculates the robustness measures (probabilistic DVH pass-rates) by iterating through a function that:<br>\-  calculates the arrays for the six parameter variable changes<br>\-  conducts the changes<br>\-  calculates the dose distribution<br>\-  and the DVH metrics<br>The percentage of DVH metric values that pass for each DVH metric and the DVH bands for the DVH plots are then calculated.|
| simulations_cpu.py<br>simulations_gpu.py                                     | The script contains the functions that enact the six parameter variable changes to the nominal plan structures.|
| fast_TG43_cpu.py fast_TG43_gpu.py                                          | The script contains the functions to calculate the dose distributions. It is named fast due to being vectorised, calculating the dose at each voxel simultaneously per structure. |
| utils_cpu.py<br>utils_gpu.py                                                 | Utility functions that are used across the spectrum of robust evaluation and optimisation. Calculating DVH metric values from DVH arrays, transferring arrays between GPU and CPU, and copying structures are among the functions included in this script file.|
| display_results.py                                                         | The functions that construct dataframes to display results in HTML and graphs of DVH curves.|
| **_RobustBrachy/robustoptimisation_**                                      | The folder contains all robust optimisation related functions. |
| run_robust_optimiser_cpu.py<br>run_robust_optimiser_gpu.py                   | The function that calls all other functions to run a robust optimisation. Inputs are DICOM file locations and optimiser parameters. Outputs are all pareto front solutions (all robust treatment plans and their approximate DVH metric pass-rates) and the robust evaluation data for all Pareto front solutions (probabilistic DVH bands, nominal DVHs, DVH metric robustness pass-rates, and nominal DVH metrics). |
| robust_optimisation.py                                                     | The function that performs the robust optimisation using the pymoo Python package. Inputs are the dictionary of optimiser parameters and the dictionary of plan parameters obtained from the data extraction script in robust evaluation. Outputs are all Pareto front solutions (all robust treatment plans and their approximate DVH metric pass-rates).  |
| initialise_population.py                                                   | This function is called first in the robust optimiser to create the initial set of individuals fed into the genetic optimiser algorithm. Random and deterministic processes are employed to establish the population.  |
| dose_per_dwell_cpu.py<br>dose_per_dwell_gpu.py                               | The functions simultaneously calculate the dose for all treatment plans in the population. A 2D array is used, each row being a set of ordered dwell times. |
| evaluation_cpu.py<br>evaluation_gpu.py                                       | The optimiser calls this function to evaluate each solution's goodness (fitness). The dose is calculated, the DVH metrics are also calculated (including how far away these are from satisfying the constraints), and the three objective functions of the optimiser are calculated (these approximate the pass-rates for the prostate D90 and the maximum doses to the urethra and rectum from the values of their nominal DVH metrics).  |
| robust_evaluation_vectorised_gpu.py<br>robust_evaluation_vectorised_cpu.py   | The function to robust evaluate all robust optimised treatment plans simultaneously. This is called on the final set of solutions from the optimiser to obtain accurate pass-rates. The probabilistic DVH bands, nominal DVHs, DVH metric robustness pass-rates, and nominal DVH metrics are outputted.   |
| generate_dicoms.py                                                         | The function to create the three DICOM treatment plan files.  |
| **_HDR_prostate_robust_app/_**                                             | The folder that contains a user interface version of RobustBrachy as a web app using the Streamlit Python package.    |
| HDR_prostate_robust_app.py                                                 | The front page of the web app. DICOM treatment plan files and the brachy source data file are uploaded, and the clinical objectives as DVH metrics are defined.   |
| **_HDR_prostate_robust_app/pages/_**                                       | The folder contains subsequent pages.      |
| 1_1)_Nominal_Plan_Analysis.py                                              | This page performs a TG43 dose calculation and displays interactive DVH plots, DVH metric data in a table, and a 2D isodose plot that is interactive and can be scrolled through slice by slice.    |
| 2_2)_Single_Plan_Robust_Evaluation.py                                      | This page performs a robust evaluation of the nominal treatment plan. The DVH metrics that pass-rates are desired for can be selected, as well as how many uncertainty scenarios. Displayed after the evaluation are an interactive DVH plots with probabilistic robustness bands, DVH metric robustness pass-rates, and nominal plan DVH metric data in a table.      |
| 3_3)_Robust_Optimisation.py                                                | This page conducts a robust optimisation for the patient's treatment. Optimiser parameters can be selected, and the final set of solutions are robustly evaluated.     |
| 4_3_a)_Select_Best_Plan.py                                                 | This page displays the results of the robust optimisation in an interactive table with DVH pass-rates and nominal metric values displayed for all robust optimised plans. Sliders for the nominal DVH metrics and DVH pass-rates filter the solutions displayed in the table based on the selected ranges of each slider. Up to three plans can be selected for further analysis in the next two pages.      |
| 5_3_b)_Isodose_Plots.py                                                    | This page compares the 2D isodose plots side-by-side of the selected three robust optimised plans and the TPS optimised plan. Navigation through each slice is controlled by buttons or direct input of the slice number.    |
| 6_3_c)_DVH_Plots.py                                                        | This page compares the DVH plots with robustness bands. The DVHs for the selected three robust optimised plans and the TPS optimised plan can be overlayed on the same plot to aid comparisons.         |
| 7_3_d)_Export_Plan.py                                                      | The user can generate and export the three DICOM plan files from this page for the selected robust optimised treatment plan.                |


# How to Install

There are two versions of the RobustBrachy Python package: 
1. one that runs a majority of the code on Nivida **GPUs** (files end with '_gpu')
2. the other that runs on **CPU only** (files end with '_cpu').

It is recommended to create a virtual environment using Anaconda (https://www.anaconda.com/) to run RobustBrachy in, allowing RobustBrachy to work with the tested package dependencies and Python versions. You can create the Anaconda virtual environment that suits your purposes using either **'GPU'** or **'CPU only'** instructions below.

> [!NOTE]
> RobustBrachy has only been tested in Windows 11.

## Create a Virtual Environment

### GPU-enabled Virtual Environment
1. clone or download the RobustBrachy package zip from GitHub to your local machine and extract the repository if you haven't already.

2. start an Anaconda prompt

3. change the directory to where the RobustBrachy repository was saved. (the environment_gpu.yml should be in the directory)

4. install a new virtual environment by typing the following command:
```
conda env create --file "environment_gpu.yml"
```

5. Download and install Nvidia CUDA Toolkit version 10.2 with filename "cuda_10.2.89_441.22_win10.exe" by downloading it from "https://developer.nvidia.com/cuda-10.2-download-archive" and making the appropriate selections.

> [!NOTE]
> This is not the latest version, but it was found that the speed for RobustBachy decreased with more recent Nvidia CUDA Toolkits and CuPy versions.

### CPU-only Virtual Environment
1. clone or download the RobustBrachy package zip from GitHub to your local machine and extract the repository if you haven't already.

2. start an Anaconda prompt

3. change to the directory to where the RobustBrachy repository was saved. (the environment_cpu.yml should be in the directory)

4. install a new virtual environment by typing the following command:
```
conda env create --file "environment_cpu.yml"
```

## Install the RobustBrachy package
There are two methods to install RobustBrachy.  

- **Method 1:** Using git and installing RobustBrachy from the cloud (the RobustBrachy GitHub repository). 
- **Method 2:** Using a local copy of the RobustBrachy GitHub repository.

> [!CAUTION]
> If using the CUDA GPU capabilities of RobustBrachy, it is recommended to follow the "creating a GPU-enabled Virtual Environment" above.

### Method 1: Using git
1. download and install a git client if one is not already installed (https://git-scm.com/downloads)
   
2. start an Anaconda prompt and activate the virtual environment by typing the following command:
```
conda activate robust 
```
> [!NOTE]
> robust is the name given to the virtual environment from the "environments_cpu.yml" and "environments_gpu.yml" files

3. Update setuptools by typing the following commands:
```
pip install --upgrade setuptools
```
```
pip install --upgrade build
```

4. install RobustBrachy by typing the following command:
```
pip install git+https://github.com/DrHazza/RobustBrachy.git
```

### Method 2: Uisng a local copy
1. clone or download the RobustBrachy package zip from GitHub to your local machine and extract the repository if you haven't already.
2. start an Anaconda prompt
3. activate the virtual environment by typing the following command:
```
conda activate robust 
```
4. Update setuptools by typing the following commands:
```
pip install --upgrade setuptools
pip install --upgrade build
```
5. change to the directory of RobustBrachy where the "pyproject.toml" file is.
6. install by typing the following commands: 
```
python -m pip install .
```

## Running the HDR_prostate_robust_app streamlit webapp
1. start an Anaconda prompt
2. activate the virtual environment by typing the following command:
```
conda activate robust 
```
3. change to the directory of the 'HDR_prostate_robust_app' folder where the "HDR_prostate_robust_app.py" file is located.
4. run a streamlit instance by typing the following command:
```
streamlit run HDR_prostate_robust_app.py
```

> [!IMPORTANT]
> A browser window popup should appear with the user interface of the robust app running locally. If not, an instance of the app can be started by clicking the 'http://localhost:????' link in the Anaconda prompt window or copying and pasting the address into a browser window.

# Examples
## Example Treatment Plan DICOMS
The example treatment plan was created from a phantom with a small prostate volume. The small volume causes large variations in DVH metrics as uncertainty shifts are applied to needles and other structures. Although this plan provides an adaquate test plan to explore the features of RobustBrachy, the intention soon is to provide an example plan with a prostate volume that more examplifies the average patient undergoing HDR prostate brachytehrapy.
## Example Python Scripts
Python scripts demonstrating example uses of the RobustBrachy package are found in the root directory of this github repository, split into using cpu only and using gpu acceleration. 
- Nominal plan anaylsis example does a fast TG43 dose calculation and outputs the DVH plots and DVH metrics.
- Robust Evaluation of a single nominal treatment plan, outputing probablistic robustness pass-rates for each DVH metric and the DVH plot with robustness bands.
- Robust Optimiser that outputs a table containing probabilistic robustness pass-rates for all plans on the pareto front and nominal DVH metrics for each robust plan.
- Selecting one Robust optimisted treatment plan to explore further, DVH plot with robustness bands and generated treatment plan DICOM files.

## Example of the Streamlit HDR_prostate_robust_app webapp
### Initial process
Importing the source file and RT plan, structure, and dose DICOM files and defining the dose-volume-histogram metrics.
![Initial Process](https://github.com/DrHazza/RobustBrachy/assets/81962088/a6336d12-8659-4a64-95ff-eaf0e88fea18)

### Nominal plan analysis
![Nominal Plan Analysis](https://github.com/DrHazza/RobustBrachy/assets/81962088/a068a533-8a48-4b13-bd91-c936b1dc43bc)

### Robust Evaluation
Setting up the simulations:
![Robust Evaluation 1](https://github.com/DrHazza/RobustBrachy/assets/81962088/d31c7b8e-7844-4ef6-8411-db54181b2f2c)

Displaying the Results
![Robust Evaluation 2](https://github.com/DrHazza/RobustBrachy/assets/81962088/709ba301-f596-458e-a5a7-05bc06ec8deb)

### Robust Optimisation
setting up and completing the robust optimisation
![Robust Optimisation](https://github.com/DrHazza/RobustBrachy/assets/81962088/0c1e1eab-1383-42cf-aabe-bf60423283ce)

### Selecting robust plans to compare from the pareto front
selecting upto 3 plans to compare against the nominal TPS plan
![Selecting Robust Plans](https://github.com/DrHazza/RobustBrachy/assets/81962088/502cc337-ab02-4265-bd86-0be0bf703109)

### Comparing robust plans using isodose plots
![Isodose Comparisons](https://github.com/DrHazza/RobustBrachy/assets/81962088/6bf03eb4-4a72-4d15-8d0c-cfa3661b9091)

### Comparing robust plans using DVH plots with robust bands
![DVH Comparisons](https://github.com/DrHazza/RobustBrachy/assets/81962088/814d42fd-75fb-4468-8aed-0f165b3ebfe1)

### Exporting DICOM treatment files for robust plan
![Exporting DICOM Plan Files](https://github.com/DrHazza/RobustBrachy/assets/81962088/2262d095-b28a-4ac2-b42d-86b390fba964)


