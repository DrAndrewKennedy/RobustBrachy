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

from robustbrachy.robustoptimisation.generate_dicoms import *

try:
    from robustbrachy.robustevaluation.utils_gpu import *
    
    # change all arrays into numpy arrays if they are cupy
    st.session_state.plan_parameters = to_cpu(st.session_state.plan_parameters)

except:
    from robustbrachy.robustevaluation.utils_cpu import *
    
    # make sure all arrays are numpy arrays
    st.session_state.plan_parameters = arrays_to_numpy(st.session_state.plan_parameters)    


st.set_page_config(page_title="Step 3 d) Export Report", layout="wide")

st.markdown("# Step 3 d) Export Report")

st.write(
    """ 
    1) Select the final plan from the Robust optimised plans using the drop down list in the sidebar. The table will change to the one to save.
    
    2) Once the final plan is selected, the summary information will be able to be downloaded as a csv file.
    
    3) Also, once the final plan is selected, a button to generate the plan DICOM files will appear and then download buttons for each DICOM file will appear.

    """
)

if "completed_DICOMS" not in st.session_state:
    st.session_state.completed_DICOMS = False


plans_loaded = []

plans_loaded.append("None")

if st.session_state.first_plan_loaded == True:
    plans_loaded.append("Plan " + str(int(st.session_state.selected_plan)))

if st.session_state.second_plan_loaded == True:
    plans_loaded.append("Plan " + str(int(st.session_state.selected_plan_2)))

if st.session_state.third_plan_loaded == True:
    plans_loaded.append("Plan " + str(int(st.session_state.selected_plan_3)))

final_selection = st.sidebar.selectbox(
    "Select the Robust Optimised Plan:", plans_loaded
)

@st.cache_data
def convert_to_csv(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv(index=True).encode("utf-8")


if final_selection != "None":
    final_selection2 = [int(s) for s in final_selection.split() if s.isdigit()][0]
    st.markdown("""---""")

    st.write("##### Selected Robust Optimised Plans:")
    all_metrics_to_save = st.session_state.all_metrics.iloc[final_selection2]
    st.dataframe(
        all_metrics_to_save.to_frame().T, use_container_width=True, hide_index=True
    )

    st.write("##### TPS Plan:")
    st.dataframe(
        st.session_state.all_metrics_TPS.iloc[0].to_frame().T,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("""---""")

    csv = convert_to_csv(all_metrics_to_save)
    st.write("##### The following will be downloaded to CSV file: ")
    st.write(all_metrics_to_save)
    download = st.sidebar.download_button(
        label="Download Summary for " + final_selection + " as CSV",
        data=csv,
        file_name="robust_plan_summary_data.csv",
        mime="text/csv",
    )

    if st.sidebar.button("Generate Plan DICOM Files"):
        dwell_times_nominal_robust = st.session_state.dwell_times_pareto_front[
            final_selection2
        ]

        rp_robust, rd_robust = generate_DICOM_plan_files(
            st.session_state.rp,
            st.session_state.rs,
            st.session_state.rd,
            st.session_state.plan_parameters,
            dwell_times_nominal_robust,
        )

        rp_robust.save_as(
            str(st.session_state.path_temp) + "/PL001_Robust_Optimised.dcm"
        )
        rd_robust.save_as(
            str(st.session_state.path_temp) + "/DO001_Robust_Optimised.dcm"
        )
        st.session_state.rs.save_as(
            str(st.session_state.path_temp) + "/SS001_Robust_Optimised.dcm"
        )  # not needed to be changed

        st.session_state.completed_DICOMS = True


else:
    st.markdown("""---""")
    if (
        st.session_state.first_plan_loaded == True
        or st.session_state.second_plan_loaded == True
        or st.session_state.third_plan_loaded == True
    ):
        st.write("##### Selected Robust Optimised Plans:")
        if (
            sum(
                [
                    st.session_state.first_plan_loaded,
                    st.session_state.second_plan_loaded,
                    st.session_state.third_plan_loaded,
                ]
            )
            > 1
        ):
            st.dataframe(
                st.session_state.all_metrics.iloc[st.session_state.selected_index],
                use_container_width=True,
                hide_index=True,
            )

        else:
            st.dataframe(
                st.session_state.all_metrics.iloc[int(st.session_state.selected_plan)]
                .to_frame()
                .T,
                use_container_width=True,
                hide_index=True,
            )
    else:
        st.write("##### Selected Robust Optimised Plans:")
        st.write("No plan/s selected. Go back to Step 1.")

    st.write("##### TPS Plan:")
    st.dataframe(
        st.session_state.all_metrics_TPS.iloc[0].to_frame().T,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("""---""")

if st.session_state.completed_DICOMS:
    with open(
        str(st.session_state.path_temp) + "/PL001_Robust_Optimised.dcm", "rb"
    ) as f:
        st.sidebar.download_button(
            "Download RT plan file for " + final_selection,
            f,
            file_name="PL001_Robust_Optimised.dcm",
        )

    with open(
        str(st.session_state.path_temp) + "/DO001_Robust_Optimised.dcm", "rb"
    ) as f:
        st.sidebar.download_button(
            "Download RD plan file for " + final_selection,
            f,
            file_name="DO001_Robust_Optimised.dcm",
        )

    with open(
        str(st.session_state.path_temp) + "/SS001_Robust_Optimised.dcm", "rb"
    ) as f:
        st.sidebar.download_button(
            "Download RS plan file for " + final_selection,
            f,
            file_name="SS001_Robust_Optimised.dcm",
        )
