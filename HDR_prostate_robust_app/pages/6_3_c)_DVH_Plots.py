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
import numpy as np
import plotly.graph_objects as go
    
# change all arrays into numpy arrays if they are cupy
try:
    st.session_state.all_nominal_dvhs_pareto_front = st.session_state.all_nominal_dvhs_pareto_front.get()
except:
    pass

try:
    st.session_state.all_robust_dvhs = st.session_state.all_robust_dvhs.get()
except:
    pass

try:
    st.session_state.nominal_dvhs_TPS = st.session_state.nominal_dvhs_TPS.get()
except:
    pass

try:
    st.session_state.robust_dvh_TPS = st.session_state.robust_dvh_TPS.get()
except:
    pass

st.set_page_config(page_title="Step 3 c) DVH Plots", layout="wide")

st.markdown("# Step 3 c) Check DVH Plots")

st.write(
    """ 
    - You can remove all curves at once by double clicking on one name int he ledgend, and then add the items of interests
    - Single clicking a DVH curve in the ledgend removes it from the plot. 
    - You can zoom into specific areas and interact with the plot. You can also download a image of the current view.
    - The labeled names with CI stand for the **percentile** confidence interval. They are the closest DVH curves above and below the nominal DVH curve such that the area between the curve and the nominal DVH curve are the 1) 16th to 84th percentiles, and 2) 2.5-th and 97.5-th percentiles and then 3) is the highest areas above and below the nominal DVH curve. 
    """
)

st.markdown("""---""")

if st.session_state.first_plan_loaded == True or st.session_state.second_plan_loaded == True or st.session_state.third_plan_loaded == True:
    st.write("##### Selected Robust Optimised Plans:")
    if sum([st.session_state.first_plan_loaded, st.session_state.second_plan_loaded, st.session_state.third_plan_loaded]) > 1 :
        st.dataframe(st.session_state.all_metrics.iloc[st.session_state.selected_index],use_container_width = True, hide_index=True)
    
    else:
        st.dataframe(st.session_state.all_metrics.iloc[int(st.session_state.selected_plan)].to_frame().T,use_container_width = True, hide_index=True)
else:
    st.write("##### Selected Robust Optimised Plans:")
    st.write("No plan/s selected. Go back to Step 1.")

st.write("##### TPS Plan:")
st.dataframe(st.session_state.all_metrics_TPS.iloc[0].to_frame().T,use_container_width = True, hide_index=True)

st.markdown("""---""")

st.session_state.show_1_only = st.sidebar.checkbox('Show 1st Plan only', True)

st.sidebar.write("Uncheck to see TPS and other two selected robust plan options.")

    
st.sidebar.markdown("""---""")

if st.session_state.show_1_only == False:
    st.session_state.show_1 = st.sidebar.checkbox('Show 1st Plan')
    if st.session_state.second_plan_loaded:
        st.session_state.show_2 = st.sidebar.checkbox('Show 2nd Plan')
    if st.session_state.third_plan_loaded:
        st.session_state.show_3 = st.sidebar.checkbox('Show 3rd Plan')
    show_TPS_DVH = st.sidebar.checkbox('Show TPS Plan')
    
else: 
    st.session_state.show_1 = False
    if st.session_state.second_plan_loaded:
        st.session_state.show_2 = False
    if st.session_state.third_plan_loaded:
        st.session_state.show_3 = False
    show_TPS_DVH = False

height_to_use_for_graphs = int(st.sidebar.text_input('Change height of graph (pixels):', str(int(((20 + 1) * 35 + 3)))))
    
if st.sidebar.button('Increase Axis Font size'):
    st.session_state.axis_font_size += 1

if st.sidebar.button('Decrease Axis Font size'):
    st.session_state.axis_font_size -= 1
    if st.session_state.axis_font_size < 0:
        st.session_state.axis_font_size = 1
    
if st.sidebar.button('Increase Legend Font size'):
    st.session_state.legend_font_size += 1

if st.sidebar.button('Decrease Legend Font size'):
    st.session_state.legend_font_size -= 1  
    if st.session_state.legend_font_size < 0:
        st.session_state.legend_font_size = 1

if st.sidebar.button('line width: Increase'):
    st.session_state.line_size += 0.5 
    
if st.sidebar.button('line width: Decrease'):
    st.session_state.line_size -= 0.5   
    if st.session_state.line_size < 0:
        st.session_state.line_size = 0.5

if st.sidebar.button('line width area edge: Increase'):
    st.session_state.line_size_areas += 0.5 
    
if st.sidebar.button('line width area edge: Decrease'):
    st.session_state.line_size_areas -= 0.5  
    if st.session_state.line_size_areas < 0:
        st.session_state.line_size_areas = 0.5
    
line_colour = 'rgba(0,0,0,0.3)'
line_text_color = 'rgba(100,100,100,1)'


fig_dvh = go.Figure()

if st.session_state.show_1 and st.session_state.first_plan_loaded == True:
    prostate_color_1 = ["rgba(6,47,95,1.0)","rgba(6,47,95,0.4)","rgba(6,47,95,0.25)","rgba(6,47,95,0.2)"]
    urethra_color_1 = ["rgba(6,47,95,1.0)","rgba(6,47,95,0.4)","rgba(6,47,95,0.25)","rgba(6,47,95,0.2)"]
    rectum_color_1 = ["rgba(6,47,95,1.0)","rgba(6,47,95,0.4)","rgba(6,47,95,0.25)","rgba(6,47,95,0.2)"]
    colours_1 = [prostate_color_1, urethra_color_1, rectum_color_1]    
    plot_titles_1 = (["Prostate Nominal = "+str(int(st.session_state.selected_plan)),"Urethra Nominal = "+str(int(st.session_state.selected_plan)), "Rectum Nominal = "+str(int(st.session_state.selected_plan))])
    
if show_TPS_DVH:
    prostate_color_TPS = ["rgba(167,0,23,1.0)","rgba(167,0,23,0.4)","rgba(167,0,23,0.25)","rgba(167,0,23,0.2)"]
    urethra_color_TPS = ["rgba(167,0,23,1.0)","rgba(167,0,23,0.4)","rgba(167,0,23,0.25)","rgba(167,0,23,0.2)"]
    rectum_color_TPS = ["rgba(167,0,23,1.0)","rgba(167,0,23,0.4)","rgba(167,0,23,0.25)","rgba(167,0,23,0.2)"]
    colours_TPS = [prostate_color_TPS, urethra_color_TPS, rectum_color_TPS]
    plot_titles_TPS = (["Prostate Nominal = TPS","Urethra Nominal = TPS", "Rectum Nominal = TPS"])
    
    
if st.session_state.show_2 and st.session_state.second_plan_loaded == True:
    prostate_color_2 = ["rgba(12,100,0,1.0)","rgba(12,100,0,0.4)","rgba(12,100,0,0.25)","rgba(12,100,0,0.2)"]
    urethra_color_2 = ["rgba(12,100,0,1.0)","rgba(12,100,0,0.4)","rgba(12,100,0,0.25)","rgba(12,100,0,0.2)"]
    rectum_color_2 = ["rgba(12,100,0,1.0)","rgba(12,100,0,0.4)","rgba(12,100,0,0.25)","rgba(12,100,0,0.2)"]
    colours_2 = [prostate_color_2, urethra_color_2, rectum_color_2] 
    plot_titles_2 = (["Prostate Nominal = "+str(int(st.session_state.selected_plan_2)),"Urethra Nominal = "+str(int(st.session_state.selected_plan_2)), "Rectum Nominal = "+str(int(st.session_state.selected_plan_2))])
    
if st.session_state.show_3 and st.session_state.third_plan_loaded == True:
    prostate_color_3 = ["rgba(179, 179, 0,1.0)","rgba(179, 179, 0,0.4)","rgba(179, 179, 0,0.25)","rgba(179, 179, 0,0.2)"]
    urethra_color_3 = ["rgba(179, 179, 0,1.0)","rgba(179, 179, 0,0.4)","rgba(179, 179, 0,0.25)","rgba(179, 179, 0,0.2)"]
    rectum_color_3 = ["rgba(179, 179, 0,1.0)","rgba(179, 179, 0,0.4)","rgba(179, 179, 0,0.25)","rgba(179, 179, 0,0.2)"]
    colours_3 = [prostate_color_3, urethra_color_3, rectum_color_3] 
    plot_titles_3 = (["Prostate Nominal = "+str(int(st.session_state.selected_plan_3)),"Urethra Nominal = "+str(int(st.session_state.selected_plan_3)), "Rectum Nominal = "+str(int(st.session_state.selected_plan_3))])

line_style = ["solid", "dash", "dot"]
if st.session_state.show_1 or st.session_state.show_2 or st.session_state.show_3 or show_TPS_DVH:
    # nominal dvhs (Robust optimised)
    for i , name in enumerate(range(3)):
        if st.session_state.show_1 and st.session_state.first_plan_loaded == True:
            fig_dvh.add_trace(go.Scattergl(x = st.session_state.all_nominal_dvhs_pareto_front[int(st.session_state.selected_plan)][i][0,:],
                                        y = st.session_state.all_nominal_dvhs_pareto_front[int(st.session_state.selected_plan)][i][1,:],
                                        mode='lines',
                                        showlegend=True,
                                        name = plot_titles_1[i],
                                        line=dict(color=colours_1[i][0],
                                                    width=st.session_state.line_size,
                                                    dash = line_style[i]
                                                    )
                                        ))
            
        if show_TPS_DVH:
            fig_dvh.add_trace(go.Scattergl(x = st.session_state.nominal_dvhs_TPS[0][i][0,:],
                                        y = st.session_state.nominal_dvhs_TPS[0][i][1,:],
                                        mode='lines',
                                        showlegend=True,
                                        name = plot_titles_TPS[i],
                                        line=dict(color=colours_TPS[i][0],
                                                    width=st.session_state.line_size,
                                                    dash = line_style[i]
                                                    
                                                    )
                                        ))
        if st.session_state.show_2 and st.session_state.second_plan_loaded == True:
            fig_dvh.add_trace(go.Scattergl(x = st.session_state.all_nominal_dvhs_pareto_front[int(st.session_state.selected_plan_2)][i][0,:],
                                        y = st.session_state.all_nominal_dvhs_pareto_front[int(st.session_state.selected_plan_2)][i][1,:],
                                        mode='lines',
                                        showlegend=True,
                                        name = plot_titles_2[i],
                                        line=dict(color=colours_2[i][0],
                                                    width=st.session_state.line_size,
                                                    dash = line_style[i]
                                                    
                                                    )
                                        ))
        if st.session_state.show_3 and st.session_state.third_plan_loaded == True:
            fig_dvh.add_trace(go.Scattergl(x = st.session_state.all_nominal_dvhs_pareto_front[int(st.session_state.selected_plan_3)][i][0,:],
                                        y = st.session_state.all_nominal_dvhs_pareto_front[int(st.session_state.selected_plan_3)][i][1,:],
                                        mode='lines',
                                        showlegend=True,
                                        name = plot_titles_3[i],
                                        line=dict(color=colours_3[i][0],
                                                    width=st.session_state.line_size,
                                                    dash = line_style[i]
                                                    
                                                    )
                                        ))
    
    # robust dvhs (Robust Optimised)
    if st.session_state.show_1 and st.session_state.first_plan_loaded == True:
        robust_dvh_names_1 = [[["Prostate 68% CI = "+str(int(st.session_state.selected_plan)), "Prostate 68% CI = "+str(int(st.session_state.selected_plan))],
                            ["Prostate 95% CI = "+str(int(st.session_state.selected_plan)), "Prostate 95% CI = "+str(int(st.session_state.selected_plan))],
                            ["Prostate max-min = "+str(int(st.session_state.selected_plan)), "Prostate max-min = "+str(int(st.session_state.selected_plan))]],
                            
                            [["Urethra CI = "+str(int(st.session_state.selected_plan)), "Urethra 68% CI = "+str(int(st.session_state.selected_plan))],
                                                ["Urethra 95% CI  ="+str(int(st.session_state.selected_plan)), "Urethra 95% CI = "+str(int(st.session_state.selected_plan))],
                                                ["Urethra max-min  ="+str(int(st.session_state.selected_plan)), "Urethra max-min = "+str(int(st.session_state.selected_plan))]],
                            
                            [["Rectum 68% CI = "+str(int(st.session_state.selected_plan)), "Rectum 68% CI = "+str(int(st.session_state.selected_plan))],
                                                ["Rectum 95% CI =  "+str(int(st.session_state.selected_plan)), "Rectum 95% CI =  "+str(int(st.session_state.selected_plan))],
                                                ["Rectum max-min =  "+str(int(st.session_state.selected_plan)), "Rectum max-min =  "+str(int(st.session_state.selected_plan))]]]
    if show_TPS_DVH:  
        robust_dvh_names_TPS = [[["Prostate 68% CI = TPS", "Prostate 68% CI = TPS"],
                            ["Prostate 95% CI = TPS ", "Prostate 95% CI = TPS"],
                            ["Prostate max-min = TPS", "Prostate max-min = TPS"]],
                            
                            [["Urethra CI = TPS", "Urethra 68% CI = TPS"],
                                                ["Urethra 95% CI = TPS", "Urethra 95% CI = TPS"],
                                                ["Urethra max-min = TPS", "Urethra max-min = TPS"]],
                            
                            [["Rectum 68% = TPS", "Rectum 68% = TPS"],
                                                ["Rectum 95% CI = TPS", "Rectum 95% CI = TPS"],
                                                ["Rectum max-min = TPS", "Rectum max-min = TPS"]]]
    if st.session_state.show_2 and st.session_state.second_plan_loaded == True:
        robust_dvh_names_2 = [[["Prostate 68% CI = "+str(int(st.session_state.selected_plan_2)), "Prostate 68% CI = "+str(int(st.session_state.selected_plan_2))],
                            ["Prostate 95% CI = "+str(int(st.session_state.selected_plan_2)), "Prostate 95% CI = "+str(int(st.session_state.selected_plan_2))],
                            ["Prostate max-min = "+str(int(st.session_state.selected_plan_2)), "Prostate max-min = "+str(int(st.session_state.selected_plan_2))]],
                            
                            [["Urethra CI = "+str(int(st.session_state.selected_plan_2)), "Urethra 68% CI = "+str(int(st.session_state.selected_plan_2))],
                                                ["Urethra 95% CI  ="+str(int(st.session_state.selected_plan_2)), "Urethra 95% CI = "+str(int(st.session_state.selected_plan_2))],
                                                ["Urethra max-min  ="+str(int(st.session_state.selected_plan_2)), "Urethra max-min = "+str(int(st.session_state.selected_plan_2))]],
                            
                            [["Rectum 68% CI = "+str(int(st.session_state.selected_plan_2)), "Rectum 68% CI = "+str(int(st.session_state.selected_plan_2))],
                                                ["Rectum 95% CI =  "+str(int(st.session_state.selected_plan_2)), "Rectum 95% CI =  "+str(int(st.session_state.selected_plan_2))],
                                                ["Rectum max-min =  "+str(int(st.session_state.selected_plan_2)), "Rectum max-min =  "+str(int(st.session_state.selected_plan_2))]]]
        
    if st.session_state.show_3 and st.session_state.third_plan_loaded == True:
        robust_dvh_names_3 = [[["Prostate 68% CI = "+str(int(st.session_state.selected_plan_3)), "Prostate 68% CI = "+str(int(st.session_state.selected_plan_3))],
                            ["Prostate 95% CI = "+str(int(st.session_state.selected_plan_3)), "Prostate 95% CI = "+str(int(st.session_state.selected_plan_3))],
                            ["Prostate max-min = "+str(int(st.session_state.selected_plan_3)), "Prostate max-min = "+str(int(st.session_state.selected_plan_3))]],
                            
                            [["Urethra CI = "+str(int(st.session_state.selected_plan_3)), "Urethra 68% CI = "+str(int(st.session_state.selected_plan_3))],
                                                ["Urethra 95% CI  ="+str(int(st.session_state.selected_plan_3)), "Urethra 95% CI = "+str(int(st.session_state.selected_plan_3))],
                                                ["Urethra max-min  ="+str(int(st.session_state.selected_plan_3)), "Urethra max-min = "+str(int(st.session_state.selected_plan_3))]],
                            
                            [["Rectum 68% CI = "+str(int(st.session_state.selected_plan_3)), "Rectum 68% CI = "+str(int(st.session_state.selected_plan_3))],
                                                ["Rectum 95% CI =  "+str(int(st.session_state.selected_plan_3)), "Rectum 95% CI =  "+str(int(st.session_state.selected_plan_3))],
                                                ["Rectum max-min =  "+str(int(st.session_state.selected_plan_3)), "Rectum max-min =  "+str(int(st.session_state.selected_plan_3))]]]
    
    # shape of all_robust_dvhs = [ CI = mu +/- n x SD ][ plan index = 0 for single RE ][ structure ][ dose/vol ][ arry values ]
    robust_idx = [[1,2],[3,4],[5,6]]
    index_to_iterate = [[[0, 1],
                        [2, 3],
                        [4, 5]],
                        
                        [[6, 7],
                                            [8, 9],
                                            [10, 11]],
                        
                        [[12, 13],
                                            [14, 15],
                                            [16, 17]]]
    for i , names in enumerate(index_to_iterate):
        for j, names_2 in enumerate(names):
            if st.session_state.show_1 and st.session_state.first_plan_loaded == True:
                fig_dvh.add_trace(go.Scatter(x = (st.session_state.all_robust_dvhs[ robust_idx[j][1] ][int(st.session_state.selected_plan)][i][0,:])[ ~np.isnan(st.session_state.all_robust_dvhs[ robust_idx[j][1] ][int(st.session_state.selected_plan)][i][0,:]) ],
                                            y = (st.session_state.all_robust_dvhs[ robust_idx[j][1] ][int(st.session_state.selected_plan)][i][1,:])[ ~np.isnan(st.session_state.all_robust_dvhs[ robust_idx[j][1] ][int(st.session_state.selected_plan)][i][0,:]) ],
                                            mode='lines',
                                            showlegend=False,
                                            legendgroup = robust_dvh_names_1[i][j][0],
                                            name = robust_dvh_names_1[i][j][1],
                                            
                                            line=dict(color=colours_1[i][j+1],
                                                        width=st.session_state.line_size_areas,                                           
                                                        )
                                            ))
                fig_dvh.add_trace(go.Scatter(x = (st.session_state.all_robust_dvhs[ robust_idx[j][0] ][int(st.session_state.selected_plan)][i][0,:])[ ~np.isnan(st.session_state.all_robust_dvhs[ robust_idx[j][0] ][int(st.session_state.selected_plan)][i][0,:]) ],
                                            y = (st.session_state.all_robust_dvhs[ robust_idx[j][0] ][int(st.session_state.selected_plan)][i][1,:])[ ~np.isnan(st.session_state.all_robust_dvhs[ robust_idx[j][0] ][int(st.session_state.selected_plan)][i][0,:]) ],
                                            mode='lines',
                                            showlegend=True,
                                            legendgroup = robust_dvh_names_1[i][j][0],
                                            name = robust_dvh_names_1[i][j][0],
                                            
                                            line=dict(color=colours_1[i][j+1],
                                                        width=st.session_state.line_size_areas,
                                                        
                                                        ),
                                            
                                            fill='tonexty',
                                            fillcolor = colours_1[i][j+1],
                                            
                                            ))
            
            if show_TPS_DVH:  
                fig_dvh.add_trace(go.Scatter(x = (st.session_state.robust_dvh_TPS[ robust_idx[j][1] ][0][i][0,:])[ ~np.isnan(st.session_state.all_robust_dvhs[ robust_idx[j][1] ][0][i][0,:]) ],
                                            y = (st.session_state.robust_dvh_TPS[ robust_idx[j][1] ][0][i][1,:])[ ~np.isnan(st.session_state.all_robust_dvhs[ robust_idx[j][1] ][0][i][0,:]) ],
                                            mode='lines',
                                            showlegend=False,
                                            legendgroup = robust_dvh_names_TPS[i][j][0],
                                            name = robust_dvh_names_TPS[i][j][0],
                                            
                                            line=dict(color=colours_TPS[i][j+1],
                                                        width=st.session_state.line_size_areas,                                           
                                                        )
                                            ))
                fig_dvh.add_trace(go.Scatter(x = (st.session_state.robust_dvh_TPS[ robust_idx[j][0] ][0][i][0,:])[ ~np.isnan(st.session_state.all_robust_dvhs[ robust_idx[j][0] ][0][i][0,:]) ],
                                            y = (st.session_state.robust_dvh_TPS[ robust_idx[j][0] ][0][i][1,:])[ ~np.isnan(st.session_state.all_robust_dvhs[ robust_idx[j][0] ][0][i][0,:]) ],
                                            mode='lines',
                                            showlegend=True,
                             
                                            legendgroup = robust_dvh_names_TPS[i][j][0],
                                            name = robust_dvh_names_TPS[i][j][0],
                                            
                                            line=dict(color=colours_TPS[i][j+1],
                                                        width=st.session_state.line_size_areas,
                                                        
                                                        ),
                                            
                                            fill='tonexty',
                                            fillcolor = colours_TPS[i][j+1],
                                            
                                            ))
            if st.session_state.show_2 and st.session_state.second_plan_loaded == True:
                fig_dvh.add_trace(go.Scatter(x = (st.session_state.all_robust_dvhs[ robust_idx[j][1] ][int(st.session_state.selected_plan_2)][i][0,:])[ ~np.isnan(st.session_state.all_robust_dvhs[ robust_idx[j][1] ][int(st.session_state.selected_plan_2)][i][0,:]) ],
                                             y = (st.session_state.all_robust_dvhs[ robust_idx[j][1] ][int(st.session_state.selected_plan_2)][i][1,:])[ ~np.isnan(st.session_state.all_robust_dvhs[ robust_idx[j][1] ][int(st.session_state.selected_plan_2)][i][0,:]) ],
                                            mode='lines',
                                            showlegend=False,
                                            legendgroup = robust_dvh_names_2[i][j][0],
                                            name = robust_dvh_names_2[i][j][1],
                                            
                                            line=dict(color=colours_2[i][j+1],
                                                        width=st.session_state.line_size_areas,                                           
                                                        )
                                            ))
                fig_dvh.add_trace(go.Scatter(x = (st.session_state.all_robust_dvhs[ robust_idx[j][0] ][int(st.session_state.selected_plan_2)][i][0,:])[ ~np.isnan(st.session_state.all_robust_dvhs[ robust_idx[j][0] ][int(st.session_state.selected_plan_2)][i][0,:]) ],
                                            y = (st.session_state.all_robust_dvhs[ robust_idx[j][0] ][int(st.session_state.selected_plan_2)][i][1,:])[ ~np.isnan(st.session_state.all_robust_dvhs[ robust_idx[j][0] ][int(st.session_state.selected_plan_2)][i][0,:]) ],
                                            mode='lines',
                                            showlegend=True,
                                            legendgroup = robust_dvh_names_2[i][j][0],
                                            name = robust_dvh_names_2[i][j][0],
                                            
                                            line=dict(color=colours_2[i][j+1],
                                                        width=st.session_state.line_size_areas,
                                                        
                                                        ),
                                            
                                            fill='tonexty',
                                            fillcolor = colours_2[i][j+1],
                                            
                                            ))
            if st.session_state.show_3 and st.session_state.third_plan_loaded == True:
                fig_dvh.add_trace(go.Scatter(x = (st.session_state.all_robust_dvhs[ robust_idx[j][1] ][int(st.session_state.selected_plan_3)][i][0,:])[ ~np.isnan(st.session_state.all_robust_dvhs[ robust_idx[j][1] ][int(st.session_state.selected_plan_3)][i][0,:]) ],
                                            y = (st.session_state.all_robust_dvhs[ robust_idx[j][1] ][0][i][1,:])[ ~np.isnan(st.session_state.all_robust_dvhs[ robust_idx[j][1] ][0][i][0,:]) ],
                                            mode='lines',
                                            showlegend=False,
                                            legendgroup = robust_dvh_names_3[i][j][0],
                                            name = robust_dvh_names_3[i][j][1],
                                            
                                            line=dict(color=colours_3[i][j+1],
                                                        width=st.session_state.line_size_areas,                                           
                                                        )
                                            ))
                fig_dvh.add_trace(go.Scatter(x = (st.session_state.all_robust_dvhs[ robust_idx[j][0] ][int(st.session_state.selected_plan_3)][i][0,:])[ ~np.isnan(st.session_state.all_robust_dvhs[ robust_idx[j][0] ][int(st.session_state.selected_plan_3)][i][0,:]) ],
                                            y = (st.session_state.all_robust_dvhs[ robust_idx[j][0] ][int(st.session_state.selected_plan_3)][i][1,:])[ ~np.isnan(st.session_state.all_robust_dvhs[ robust_idx[j][0] ][int(st.session_state.selected_plan_3)][i][0,:]) ],
                                            mode='lines',
                                            showlegend=True,
                                            legendgroup = robust_dvh_names_3[i][j][0],
                                            name = robust_dvh_names_3[i][j][0],
                                            
                                            line=dict(color=colours_3[i][j+1],
                                                        width=st.session_state.line_size_areas,
                                                        
                                                        ),
                                            
                                            fill='tonexty',
                                            fillcolor = colours_3[i][j+1],
                                            
                                            ))
    fig_dvh.update_xaxes(title_text="Dose (Gy)",#<Br>(a)",
                         title_font=dict(size=st.session_state.axis_font_size),
                      minor=dict(dtick = 1,showgrid=True), 
                      range=[0, 36],
                      tick0=0, dtick = 5,
                      #title_standoff=0
                      )
    
    fig_dvh.update_yaxes(title_text="Relative Volume (%)", 
                         title_font=dict(size=st.session_state.axis_font_size),
                      range=[0, 101],
                      minor=dict(dtick = 2.5,showgrid=True),
                      tick0=0, dtick = 10, 
                      #title_standoff=0
                      )  
    
    
    
    fig_dvh.add_hline(y=90,line_width=3,line_dash="dot",annotation_text="<b>D<sub>90</sub> > 16 Gy</b>", 
                     annotation_font_color=line_text_color, annotation_position="top left",line=dict(color=line_colour))
    fig_dvh.add_vline(x=16,line_width=3,line_dash="dot",annotation_text="<b>V<sub>100</sub> > 90%</b>", 
                     annotation_font_color=line_text_color,annotation_position="top right",line=dict(color=line_colour))
    fig_dvh.add_vline(x=24,line_width=3,line_dash="dot",annotation_text="<b>V<sub>150</sub> < 35%</b>", 
                     annotation_font_color=line_text_color,annotation_position="bottom right",line=dict(color=line_colour))
    fig_dvh.add_vline(x=32,line_width=3,line_dash="dot",annotation_text="<b>V<sub>200</sub> < 15%</b>", 
                     annotation_font_color=line_text_color,annotation_position="bottom right",line=dict(color=line_colour))
    fig_dvh.add_hline(y=10,line_width=3,line_dash="dot",annotation_text="<b>D<sub>10</sub> < 17 Gy</b>", annotation_position="top left",
                     annotation_font_color=line_text_color,line=dict(color=line_colour))
    fig_dvh.add_vline(x=12,line_width=3,line_dash="dot",annotation_text="<b> V<sub>75</sub> < 0.6 cc</b>", annotation_position="bottom right",
                     annotation_font_color=line_text_color, line=dict(color=line_colour))
    
    fig_dvh.update_layout(height=height_to_use_for_graphs, #width = 800,
                                       legend = dict(font = dict(#family = "Courier", 
                                                                 size = st.session_state.legend_font_size, 
                                                                 #color = "black"
                                                                 )),
                                       )
    
    st.plotly_chart(fig_dvh, 
                    height = height_to_use_for_graphs,
                    use_container_width = True)
            
if st.session_state.show_1_only and st.session_state.first_plan_loaded == True:
    prostate_color = ["rgba(6,47,95,1.0)","rgba(18,97,160,0.5)","rgba(56,149,211,0.4)","rgba(88,204,237,0.3)"]
    urethra_color = ["rgba(4,129,83,1.0)","rgba(39,171,123,0.5)","rgba(73,191,145,0.4)","rgba(146,215,195,0.3)"]
    rectum_color = ["rgba(167,0,23,1.0)","rgba(255,0,41,0.5)","rgba(255,123,123,0.4)","rgba(255,186,186,0.3)"]
    
    colours = [prostate_color, urethra_color, rectum_color]
    # nominal dvhs
    for i , name in enumerate(["Prostate Nominal = "+str(int(st.session_state.selected_plan)),"Urethra Nominal = "+str(int(st.session_state.selected_plan)), "Rectum Nominal = "+str(int(st.session_state.selected_plan))]):
        fig_dvh.add_trace(go.Scattergl(x = st.session_state.all_nominal_dvhs_pareto_front[int(st.session_state.selected_plan)][i][0,:],
                                    y = st.session_state.all_nominal_dvhs_pareto_front[int(st.session_state.selected_plan)][i][1,:],
                                    mode='lines',
                                    showlegend=True,
                                    name = name,
                                    line=dict(color=colours[i][0],
                                                width=st.session_state.line_size,
                                                
                                                )
                                    ))
    
    # robust dvhs
    robust_dvh_names = [[["Prostate 68% CI = "+str(int(st.session_state.selected_plan)), "Prostate 68% CI = "+str(int(st.session_state.selected_plan))],
                        ["Prostate 95% CI = "+str(int(st.session_state.selected_plan)), "Prostate 95% CI = "+str(int(st.session_state.selected_plan))],
                        ["Prostate max-min = "+str(int(st.session_state.selected_plan)), "Prostate max-min = "+str(int(st.session_state.selected_plan))]],
                        
                        [["Urethra CI = "+str(int(st.session_state.selected_plan)), "Urethra 68% CI = "+str(int(st.session_state.selected_plan))],
                                            ["Urethra 95% CI  ="+str(int(st.session_state.selected_plan)), "Urethra 95% CI = "+str(int(st.session_state.selected_plan))],
                                            ["Urethra max-min  ="+str(int(st.session_state.selected_plan)), "Urethra max-min = "+str(int(st.session_state.selected_plan))]],
                        
                        [["Rectum 68% CI = "+str(int(st.session_state.selected_plan)), "Rectum 68% CI = "+str(int(st.session_state.selected_plan))],
                                            ["Rectum 95% CI =  "+str(int(st.session_state.selected_plan)), "Rectum 95% CI =  "+str(int(st.session_state.selected_plan))],
                                            ["Rectum max-min =  "+str(int(st.session_state.selected_plan)), "Rectum max-min =  "+str(int(st.session_state.selected_plan))]]]
    robust_idx = [[1,2],[3,4],[5,6]]
    
    for i , names in enumerate(robust_dvh_names):
        for j, names_2 in enumerate(names):
            
            fig_dvh.add_trace(go.Scatter(x = (st.session_state.all_robust_dvhs[ robust_idx[j][1] ][int(st.session_state.selected_plan)][i][0,:])[ ~np.isnan(st.session_state.all_robust_dvhs[ robust_idx[j][1] ][int(st.session_state.selected_plan)][i][0,:]) ],
                                        y = (st.session_state.all_robust_dvhs[ robust_idx[j][1] ][int(st.session_state.selected_plan)][i][1,:])[ ~np.isnan(st.session_state.all_robust_dvhs[ robust_idx[j][1] ][int(st.session_state.selected_plan)][i][0,:]) ],
                                        mode='lines',
                                        showlegend=False,
                                        legendgroup = names_2[0],
                                        name = names_2[1],
                                        
                                        line=dict(color=colours[i][j+1],
                                                    width=st.session_state.line_size,                                           
                                                    )
                                        ))
            fig_dvh.add_trace(go.Scatter(x = (st.session_state.all_robust_dvhs[ robust_idx[j][0]][int(st.session_state.selected_plan)][i][0,:])[ ~np.isnan(st.session_state.all_robust_dvhs[ robust_idx[j][0] ][int(st.session_state.selected_plan)][i][0,:]) ],
                                        y = (st.session_state.all_robust_dvhs[ robust_idx[j][0]][int(st.session_state.selected_plan)][i][1,:])[ ~np.isnan(st.session_state.all_robust_dvhs[ robust_idx[j][0] ][int(st.session_state.selected_plan)][i][0,:]) ],
                                        mode='lines',
                                        showlegend=True,
                                        name = names_2[0],
                                        legendgroup = names_2[0],
                                        
                                        line=dict(color=colours[i][j+1],
                                                    width=st.session_state.line_size,
                                                    
                                                    ),
                                        
                                        fill='tonexty',
                                        fillcolor = colours[i][j+1],
                                        
                                        ))


    fig_dvh.update_xaxes(title_text="Dose (Gy)",#<Br>(a)",
                         title_font=dict(size=st.session_state.axis_font_size),
                      minor=dict(dtick = 1,showgrid=True), 
                      range=[0, 36],
                      tick0=0, dtick = 5,
                      #title_standoff=0
                      )
    
    fig_dvh.update_yaxes(title_text="Relative Volume (%)", 
                         title_font=dict(size=st.session_state.axis_font_size),
                      range=[0, 101],
                      minor=dict(dtick = 2.5,showgrid=True),
                      tick0=0, dtick = 10, 
                      #title_standoff=0
                      )  
    
    
    
    fig_dvh.add_hline(y=90,line_width=3,line_dash="dot",annotation_text="<b>D<sub>90</sub> > 16 Gy</b>", 
                     annotation_font_color=line_text_color, annotation_position="top left",line=dict(color=line_colour))
    fig_dvh.add_vline(x=16,line_width=3,line_dash="dot",annotation_text="<b>V<sub>100</sub> > 90%</b>", 
                     annotation_font_color=line_text_color,annotation_position="top right",line=dict(color=line_colour))
    fig_dvh.add_vline(x=24,line_width=3,line_dash="dot",annotation_text="<b>V<sub>150</sub> < 35%</b>", 
                     annotation_font_color=line_text_color,annotation_position="bottom right",line=dict(color=line_colour))
    fig_dvh.add_vline(x=32,line_width=3,line_dash="dot",annotation_text="<b>V<sub>200</sub> < 15%</b>", 
                     annotation_font_color=line_text_color,annotation_position="bottom right",line=dict(color=line_colour))
    fig_dvh.add_hline(y=10,line_width=3,line_dash="dot",annotation_text="<b>D<sub>10</sub> < 17 Gy</b>", annotation_position="top left",
                     annotation_font_color=line_text_color,line=dict(color=line_colour))
    fig_dvh.add_vline(x=12,line_width=3,line_dash="dot",annotation_text="<b> V<sub>75</sub> < 0.6 cc</b>", annotation_position="bottom right",
                     annotation_font_color=line_text_color, line=dict(color=line_colour))
    
    fig_dvh.update_layout(height=height_to_use_for_graphs, #width = 800,
                                       legend = dict(font = dict(#family = "Courier", 
                                                                 size = st.session_state.legend_font_size, 
                                                                 #color = "black"
                                                                 )),
                                       )
    
    st.plotly_chart(fig_dvh, 
                    height = height_to_use_for_graphs,
                    use_container_width = True)