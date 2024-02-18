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
import pandas as pd
st.set_page_config(page_title="Step 3 b) Isodose Plots", layout="wide")

st.markdown("# Step 3 b) Check Isodoses")

st.write(
    """
    - Navigate through slices by using the "Previous Slice" and "Next Slice" buttons, or pressing the button to jump to a slice.
    
    - You can also navigate to specific slices by entering in the text box and pressing enter. Note: enter + or - decimal or whole numbers and it will show the nearest slice.
       
    - Remove isodoses or structures by pressing their name in the ledgend. You can zoom into specific areas and interact with the plot. You can also download a image of the current view.
    
    - To remove an isodose or structure in all plots, use the check box at the very bottom. Remember to hit the reload button to apply changes for this one.
    
    - Showing the TPS plan or the ultrasound image are also options in the sidebar.
    """
)


reload_page = False

show_TPS = st.sidebar.checkbox('Show TPS Plan')
if st.session_state.second_plan_loaded:
    st.session_state.show_2 = st.sidebar.checkbox('Show 2nd Plan', True)
if st.session_state.third_plan_loaded:
    st.session_state.show_3 = st.sidebar.checkbox('Show 3rd Plan', True)

# disabled show ultrasound code until the generating code is been added
#show_Ultrasound = st.sidebar.checkbox('Show Ultrasound')
show_Ultrasound = False
height_to_use_for_graphs = int(st.sidebar.text_input('Change Height of graph (pixels):', str(int(((20 + 1) * 35 + 3)))))


if st.sidebar.button('Title Font size: Increase'):
    st.session_state.title_font_size += 1

if st.sidebar.button('Title Font size: Decrease'):
    st.session_state.title_font_size -= 1
    if st.session_state.title_font_size < 0:
        st.session_state.title_font_size = 1
        
if st.sidebar.button('Axis Font size: Increase'):
    st.session_state.axis_font_size += 1

if st.sidebar.button('Axis Font size: Decrease'):
    st.session_state.axis_font_size -= 1 
    if st.session_state.axis_font_size < 0:
        st.session_state.axis_font_size = 1  
        
if st.sidebar.button('Legend Font size: Increase'):
    st.session_state.legend_font_size += 1

if st.sidebar.button('Legend Font size: Decrease'):
    st.session_state.legend_font_size -= 1   
    if st.session_state.legend_font_size < 0:
        st.session_state.legend_font_size = 1
        
if st.sidebar.button('line width: Increase'):
    st.session_state.line_width += 0.5
    
if st.sidebar.button('line width: Decrease'):
    st.session_state.line_width -= 0.5   
    if st.session_state.line_width < 0:
        st.session_state.line_width = 1
        
st.markdown("""---""")

st.write("##### Selected Robust Optimised Plans:")
if sum([st.session_state.first_plan_loaded, st.session_state.second_plan_loaded, st.session_state.third_plan_loaded]) > 1 :
    st.dataframe(st.session_state.all_metrics.iloc[st.session_state.selected_index],use_container_width = True, hide_index=True)

else:
    st.dataframe(st.session_state.all_metrics.iloc[int(st.session_state.selected_plan)].to_frame().T,use_container_width = True, hide_index=True)

st.write("##### TPS Plan:")
st.dataframe(st.session_state.all_metrics_TPS.iloc[0].to_frame().T,use_container_width = True, hide_index=True)
st.markdown("""---""")

quick_slice_view = np.percentile(np.arange(len(st.session_state.z_vector)), [0,10,20,30,40,50,60,70,80,90,100], method='closest_observation')
col_1, col_2, col_3, col_4, col_5, col_6, col_7, col_8, col_9, col_10, col_11 = st.columns([1,1,1,1,1,1,1,1,1,1,1])

with col_1:
    if st.button(str(st.session_state.z_vector[quick_slice_view[0]])+" mm"):
        st.session_state.slice_to_view = quick_slice_view[0]

with col_2:
    if st.button(str(st.session_state.z_vector[quick_slice_view[1]])+" mm"):
        st.session_state.slice_to_view = quick_slice_view[1]
        
with col_3:
    if st.button(str(st.session_state.z_vector[quick_slice_view[2]])+" mm"):
        st.session_state.slice_to_view = quick_slice_view[2]
    
with col_4:
    if st.button(str(st.session_state.z_vector[quick_slice_view[3]])+" mm"):
        st.session_state.slice_to_view = quick_slice_view[3]

with col_5:
    if st.button(str(st.session_state.z_vector[quick_slice_view[4]])+" mm"):
        st.session_state.slice_to_view = quick_slice_view[4]
        
with col_6:
    if st.button(str(st.session_state.z_vector[quick_slice_view[5]])+" mm"):
        st.session_state.slice_to_view = quick_slice_view[5]
        
with col_7:
    if st.button(str(st.session_state.z_vector[quick_slice_view[6]])+" mm"):
        st.session_state.slice_to_view = quick_slice_view[6]

with col_8:
    if st.button(str(st.session_state.z_vector[quick_slice_view[7]])+" mm"):
        st.session_state.slice_to_view = quick_slice_view[7]
        
with col_9:
    if st.button(str(st.session_state.z_vector[quick_slice_view[8]])+" mm"):
        st.session_state.slice_to_view = quick_slice_view[8]
        
with col_10:
    if st.button(str(st.session_state.z_vector[quick_slice_view[9]])+" mm"):
        st.session_state.slice_to_view = quick_slice_view[9]      

with col_11:
    if st.button(str(st.session_state.z_vector[quick_slice_view[10]])+" mm"):
        st.session_state.slice_to_view = quick_slice_view[10]    
    
col1, col2, col3, col4, col5 = st.columns([2,1,2,1,2])
skip = False
with col2:
    if st.button('Previous Slice'):
        
        st.session_state.slice_to_view -= 1
        skip = True
        if st.session_state.slice_to_view < 0:
            st.session_state.slice_to_view = 0
            st.write("No more slices Superiorly")

with col4:
    if st.button('Next Slice'):
        
        st.session_state.slice_to_view += 1  
        skip = True
        if st.session_state.slice_to_view >= len(st.session_state.titles_fig_isodoses):
            st.session_state.slice_to_view = len(st.session_state.titles_fig_isodoses) - 1
            st.write("No more slices Inferiorly")

with col3:   
    jump_to_slice = st.text_input('Got to slice (mm):', str(st.session_state.z_vector[st.session_state.slice_to_view]))
    
    if skip != True:
        try: 
            st.session_state.slice_to_view = float(jump_to_slice)
            if st.session_state.slice_to_view <= np.min(st.session_state.z_vector):
                st.session_state.slice_to_view = len(st.session_state.titles_fig_isodoses) - 1
                st.write("No more slices Inferiorly")
                
            elif st.session_state.slice_to_view > np.max(st.session_state.z_vector):
                st.session_state.slice_to_view = 0
                st.write("No more slices Superiorly")    
            else:    
                st.session_state.slice_to_view = np.abs(np.array((st.session_state.z_vector - st.session_state.slice_to_view))).argmin()
            
        except:
            st.write("Number needs to by a float. (did you include mm?)")    
            st.session_state.slice_to_view = int(len(st.session_state.titles_fig_isodoses) / 2 )
   
if sum([show_TPS,st.session_state.show_2,st.session_state.show_3]) > 0:
    
    col_robust, col_TPS = st.columns([1,1])
    st.session_state.jump_to_slice_old = jump_to_slice
    
    with col_robust:
        st.session_state.jump_to_slice_old = jump_to_slice
        
        # 1st robust plot
        if show_Ultrasound:
            data_w_US = (st.session_state.data_fig_isodoses[st.session_state.slice_to_view])[:]
            slice_index = np.where(st.session_state.US_z_axis_values == st.session_state.z_vector[st.session_state.slice_to_view]) 
            data_w_US.insert(0, go.Heatmap(z= st.session_state.US_img[:,:,int(slice_index[0])],
                                       x = st.session_state.US_x_axis_values,
                                       y = st.session_state.US_y_axis_values,
                                       colorscale='gray'))
            
            fig_isodoses_to_show = go.Figure(data=data_w_US,
                                             layout=go.Layout(title=go.layout.Title(text=st.session_state.titles_fig_isodoses[st.session_state.slice_to_view],                                   
                                                                                    font=dict(
                                                                                         #family="Courier New, monospace",
                                                                                         size=st.session_state.title_font_size,
                                                                                         #color="RebeccaPurple"
                                                                                     ))))
        else:
            fig_isodoses_to_show = go.Figure(data=st.session_state.data_fig_isodoses[st.session_state.slice_to_view],
                                             layout=go.Layout(title=go.layout.Title(text=st.session_state.titles_fig_isodoses[st.session_state.slice_to_view],                                   
                                                                                    font=dict(
                                                                                         #family="Courier New, monospace",
                                                                                         size=st.session_state.title_font_size,
                                                                                         #color="RebeccaPurple"
                                                                                     ))))
        if st.session_state.dont_show_rectum:
            fig_isodoses_to_show.update_traces(visible='legendonly',
                          selector=dict(name="Rectum"))
            
        if st.session_state.dont_show_urethra:
            fig_isodoses_to_show.update_traces(visible='legendonly',
                          selector=dict(name="Urethra"))
            
        if st.session_state.dont_show_prostate:
            fig_isodoses_to_show.update_traces(visible='legendonly',
                          selector=dict(name="Prostate"))
            
        if st.session_state.dont_show_50:
            fig_isodoses_to_show.update_traces(visible='legendonly',
                          selector=dict(name="50%"))
            
        if st.session_state.dont_show_75:
            fig_isodoses_to_show.update_traces(visible='legendonly',
                          selector=dict(name="75%"))
            
        if st.session_state.dont_show_90:
            fig_isodoses_to_show.update_traces(visible='legendonly',
                          selector=dict(name="90%"))
            
        if st.session_state.dont_show_100:
            fig_isodoses_to_show.update_traces(visible='legendonly',
                          selector=dict(name="100%"))
            
        if st.session_state.dont_show_110:
            fig_isodoses_to_show.update_traces(visible='legendonly',
                          selector=dict(name="110%"))
            
        if st.session_state.dont_show_150:
            fig_isodoses_to_show.update_traces(visible='legendonly',
                          selector=dict(name="150%"))
            
        if st.session_state.dont_show_200:
            fig_isodoses_to_show.update_traces(visible='legendonly',
                          selector=dict(name="200%"))
        fig_isodoses_to_show.update_xaxes(title_text="Left-Right (x) (mm)",
                                         title_font=dict(size=st.session_state.axis_font_size),
                                          minor=dict(dtick = 2.5,showgrid=True), 
                                          range=[st.session_state.x_vector[0], st.session_state.x_vector[-1]],
                                          tick0=0, dtick = 5,
                                          #title_standoff=0
                                          )
    
        fig_isodoses_to_show.update_yaxes(title_text="Posterior-Anterior (y) (mm)",
                                         title_font=dict(size=st.session_state.axis_font_size),
                          range=[st.session_state.y_vector[-1], st.session_state.y_vector[0]],
                          minor=dict(dtick = 2.5,showgrid=True),
                          tick0=0, dtick = 5, 
                          #autorange="reversed"
                          )  
        fig_isodoses_to_show.update_traces(line=dict(width=st.session_state.line_width),
                                            selector=dict(type="scattergl"))
        fig_isodoses_to_show.update_traces(line=dict(width=st.session_state.line_width),
                                            selector=dict(type="contour"))
        fig_isodoses_to_show.update_traces(showscale=False,
                                            selector=dict(type="heatmap"))  
        fig_isodoses_to_show.update_layout(height=height_to_use_for_graphs, #width = 800,
                                           legend = dict(font = dict(#family = "Courier", 
                                                                     size = st.session_state.legend_font_size, 
                                                                     #color = "black"
                                                                     ))
                                           )
        st.plotly_chart(fig_isodoses_to_show, 
                        height = height_to_use_for_graphs, 
                        use_container_width = True)
        
        # 3rd robust plot
        if st.session_state.show_3:
            if show_Ultrasound:
                data_w_US = (st.session_state.data_fig_isodoses_3[st.session_state.slice_to_view])[:]
                slice_index = np.where(st.session_state.US_z_axis_values == st.session_state.z_vector[st.session_state.slice_to_view]) 
                data_w_US.insert(0, go.Heatmap(z= st.session_state.US_img[:,:,int(slice_index[0])],
                                           x = st.session_state.US_x_axis_values,
                                           y = st.session_state.US_y_axis_values,
                                           colorscale='gray'))
                
                fig_isodoses_to_show_3 = go.Figure(data=data_w_US,
                                                 layout=go.Layout(title=go.layout.Title(text=st.session_state.titles_fig_isodoses_3[st.session_state.slice_to_view],                                   
                                                                                        font=dict(
                                                                                             #family="Courier New, monospace",
                                                                                             size=st.session_state.title_font_size,
                                                                                             #color="RebeccaPurple"
                                                                                         ))))
            else:
                fig_isodoses_to_show_3 = go.Figure(data=st.session_state.data_fig_isodoses_3[st.session_state.slice_to_view],
                                                 layout=go.Layout(title=go.layout.Title(text=st.session_state.titles_fig_isodoses_3[st.session_state.slice_to_view],                                   
                                                                                        font=dict(
                                                                                             #family="Courier New, monospace",
                                                                                             size=st.session_state.title_font_size,
                                                                                             #color="RebeccaPurple"
                                                                                         ))))
            if st.session_state.dont_show_rectum:
                fig_isodoses_to_show_3.update_traces(visible='legendonly',
                              selector=dict(name="Rectum"))
                
            if st.session_state.dont_show_urethra:
                fig_isodoses_to_show_3.update_traces(visible='legendonly',
                              selector=dict(name="Urethra"))
                
            if st.session_state.dont_show_prostate:
                fig_isodoses_to_show_3.update_traces(visible='legendonly',
                              selector=dict(name="Prostate"))
                
            if st.session_state.dont_show_50:
                fig_isodoses_to_show_3.update_traces(visible='legendonly',
                              selector=dict(name="50%"))
                
            if st.session_state.dont_show_75:
                fig_isodoses_to_show_3.update_traces(visible='legendonly',
                              selector=dict(name="75%"))
                
            if st.session_state.dont_show_90:
                fig_isodoses_to_show_3.update_traces(visible='legendonly',
                              selector=dict(name="90%"))
                
            if st.session_state.dont_show_100:
                fig_isodoses_to_show_3.update_traces(visible='legendonly',
                              selector=dict(name="100%"))
                
            if st.session_state.dont_show_110:
                fig_isodoses_to_show_3.update_traces(visible='legendonly',
                              selector=dict(name="110%"))
                
            if st.session_state.dont_show_150:
                fig_isodoses_to_show_3.update_traces(visible='legendonly',
                              selector=dict(name="150%"))
                
            if st.session_state.dont_show_200:
                fig_isodoses_to_show_3.update_traces(visible='legendonly',
                              selector=dict(name="200%"))
            fig_isodoses_to_show_3.update_xaxes(title_text="Left-Right (x) (mm)",
                                             title_font=dict(size=st.session_state.axis_font_size),
                                              minor=dict(dtick = 2.5,showgrid=True), 
                                              range=[st.session_state.x_vector[0], st.session_state.x_vector[-1]],
                                              tick0=0, dtick = 5,
                                              #title_standoff=0
                                              )
        
            fig_isodoses_to_show_3.update_yaxes(title_text="Posterior-Anterior (y) (mm)",
                                             title_font=dict(size=st.session_state.axis_font_size),
                              range=[st.session_state.y_vector[-1], st.session_state.y_vector[0]],
                              minor=dict(dtick = 2.5,showgrid=True),
                              tick0=0, dtick = 5, 
                              #autorange="reversed"
                              )  
            fig_isodoses_to_show_3.update_traces(line=dict(width=st.session_state.line_width),
                                                selector=dict(type="scattergl"))
            fig_isodoses_to_show_3.update_traces(line=dict(width=st.session_state.line_width),
                                                selector=dict(type="contour"))
            fig_isodoses_to_show_3.update_traces(showscale=False,
                                                selector=dict(type="heatmap"))  
            fig_isodoses_to_show_3.update_layout(height=height_to_use_for_graphs, #width = 800,
                                               legend = dict(font = dict(#family = "Courier", 
                                                                         size = st.session_state.legend_font_size, 
                                                                         #color = "black"
                                                                         ))
                                               )
            st.plotly_chart(fig_isodoses_to_show_3, 
                            height = height_to_use_for_graphs, 
                            use_container_width = True)
    
    with col_TPS:
        if show_TPS:
            if show_Ultrasound:
                data_w_US = (st.session_state.data_fig_isodoses_TPS[st.session_state.slice_to_view])[:]
                slice_index = np.where(st.session_state.US_z_axis_values == st.session_state.z_vector[st.session_state.slice_to_view]) 
                data_w_US.insert(0, go.Heatmap(z= st.session_state.US_img[:,:,int(slice_index[0])],
                                           x = st.session_state.US_x_axis_values,
                                           y = st.session_state.US_y_axis_values,
                                           colorscale='gray'))
    
                
                fig_isodoses_to_show_TPS = go.Figure(data=data_w_US,
                                                 layout=go.Layout(title=go.layout.Title(text=st.session_state.titles_fig_isodoses_TPS[st.session_state.slice_to_view],                                   
                                                                                        font=dict(
                                                                                             #family="Courier New, monospace",
                                                                                             size=st.session_state.title_font_size,
                                                                                             #color="RebeccaPurple"
                                                                                         ))))
            else:
                fig_isodoses_to_show_TPS = go.Figure(data=st.session_state.data_fig_isodoses_TPS[st.session_state.slice_to_view],
                                                 layout=go.Layout(title=go.layout.Title(text=st.session_state.titles_fig_isodoses_TPS[st.session_state.slice_to_view],                                   
                                                                                        font=dict(
                                                                                             #family="Courier New, monospace",
                                                                                             size=st.session_state.title_font_size,
                                                                                             #color="RebeccaPurple"
                                                                                         ))))
            if st.session_state.dont_show_rectum:
                fig_isodoses_to_show_TPS.update_traces(visible='legendonly',
                              selector=dict(name="Rectum"))
                
            if st.session_state.dont_show_urethra:
                fig_isodoses_to_show_TPS.update_traces(visible='legendonly',
                              selector=dict(name="Urethra"))
                
            if st.session_state.dont_show_prostate:
                fig_isodoses_to_show_TPS.update_traces(visible='legendonly',
                              selector=dict(name="Prostate"))
                
            if st.session_state.dont_show_50:
                fig_isodoses_to_show_TPS.update_traces(visible='legendonly',
                              selector=dict(name="50%"))
                
            if st.session_state.dont_show_75:
                fig_isodoses_to_show_TPS.update_traces(visible='legendonly',
                              selector=dict(name="75%"))
                
            if st.session_state.dont_show_90:
                fig_isodoses_to_show_TPS.update_traces(visible='legendonly',
                              selector=dict(name="90%"))
                
            if st.session_state.dont_show_100:
                fig_isodoses_to_show_TPS.update_traces(visible='legendonly',
                              selector=dict(name="100%"))
                
            if st.session_state.dont_show_110:
                fig_isodoses_to_show_TPS.update_traces(visible='legendonly',
                              selector=dict(name="110%"))
                
            if st.session_state.dont_show_150:
                fig_isodoses_to_show_TPS.update_traces(visible='legendonly',
                              selector=dict(name="150%"))
                
            if st.session_state.dont_show_200:
                fig_isodoses_to_show_TPS.update_traces(visible='legendonly',
                              selector=dict(name="200%"))
                
            fig_isodoses_to_show_TPS.update_xaxes(title_text="Left-Right (x) (mm)",
                                             title_font=dict(size=st.session_state.axis_font_size),
                                              minor=dict(dtick = 2.5,showgrid=True), 
                                              range=[st.session_state.x_vector[0], st.session_state.x_vector[-1]],
                                              tick0=0, dtick = 5,
                                              #title_standoff=0
                                              )
        
            fig_isodoses_to_show_TPS.update_yaxes(title_text="Posterior-Anterior (y) (mm)",
                                             title_font=dict(size=st.session_state.axis_font_size),
                              range=[st.session_state.y_vector[-1], st.session_state.y_vector[0]],
                              minor=dict(dtick = 2.5,showgrid=True),
                              tick0=0, dtick = 5, 
                              #autorange="reversed"
                              )  
            fig_isodoses_to_show_TPS.update_traces(line=dict(width=st.session_state.line_width),
                                                selector=dict(type="scattergl"))
            fig_isodoses_to_show_TPS.update_traces(line=dict(width=st.session_state.line_width),
                                                selector=dict(type="contour"))
            fig_isodoses_to_show_TPS.update_traces(showscale=False,
                                                selector=dict(type="heatmap"))    
    
            fig_isodoses_to_show_TPS.update_layout(height=height_to_use_for_graphs, #width = 800,
                                               legend = dict(font = dict(#family = "Courier", 
                                                                         size = st.session_state.legend_font_size, 
                                                                         #color = "black"
                                                                         ))
                                               )
            st.plotly_chart(fig_isodoses_to_show_TPS, 
                            height = height_to_use_for_graphs, 
                            use_container_width = True)
    
        # 3rd robust plot
        if st.session_state.show_2:
            if show_Ultrasound:
                data_w_US = (st.session_state.data_fig_isodoses_2[st.session_state.slice_to_view])[:]
                slice_index = np.where(st.session_state.US_z_axis_values == st.session_state.z_vector[st.session_state.slice_to_view]) 
                data_w_US.insert(0, go.Heatmap(z= st.session_state.US_img[:,:,int(slice_index[0])],
                                           x = st.session_state.US_x_axis_values,
                                           y = st.session_state.US_y_axis_values,
                                           colorscale='gray'))
                
                fig_isodoses_to_show_2 = go.Figure(data=data_w_US,
                                                 layout=go.Layout(title=go.layout.Title(text=st.session_state.titles_fig_isodoses_2[st.session_state.slice_to_view],                                   
                                                                                        font=dict(
                                                                                             #family="Courier New, monospace",
                                                                                             size=st.session_state.title_font_size,
                                                                                             #color="RebeccaPurple"
                                                                                         ))))
            else:
                fig_isodoses_to_show_2 = go.Figure(data=st.session_state.data_fig_isodoses_2[st.session_state.slice_to_view],
                                                 layout=go.Layout(title=go.layout.Title(text=st.session_state.titles_fig_isodoses_2[st.session_state.slice_to_view],                                   
                                                                                        font=dict(
                                                                                             #family="Courier New, monospace",
                                                                                             size=st.session_state.title_font_size,
                                                                                             #color="RebeccaPurple"
                                                                                         ))))
            if st.session_state.dont_show_rectum:
                fig_isodoses_to_show_2.update_traces(visible='legendonly',
                              selector=dict(name="Rectum"))
                
            if st.session_state.dont_show_urethra:
                fig_isodoses_to_show_2.update_traces(visible='legendonly',
                              selector=dict(name="Urethra"))
                
            if st.session_state.dont_show_prostate:
                fig_isodoses_to_show_2.update_traces(visible='legendonly',
                              selector=dict(name="Prostate"))
                
            if st.session_state.dont_show_50:
                fig_isodoses_to_show_2.update_traces(visible='legendonly',
                              selector=dict(name="50%"))
                
            if st.session_state.dont_show_75:
                fig_isodoses_to_show_2.update_traces(visible='legendonly',
                              selector=dict(name="75%"))
                
            if st.session_state.dont_show_90:
                fig_isodoses_to_show_2.update_traces(visible='legendonly',
                              selector=dict(name="90%"))
                
            if st.session_state.dont_show_100:
                fig_isodoses_to_show_2.update_traces(visible='legendonly',
                              selector=dict(name="100%"))
                
            if st.session_state.dont_show_110:
                fig_isodoses_to_show_2.update_traces(visible='legendonly',
                              selector=dict(name="110%"))
                
            if st.session_state.dont_show_150:
                fig_isodoses_to_show_2.update_traces(visible='legendonly',
                              selector=dict(name="150%"))
                
            if st.session_state.dont_show_200:
                fig_isodoses_to_show_2.update_traces(visible='legendonly',
                              selector=dict(name="200%"))
            fig_isodoses_to_show_2.update_xaxes(title_text="Left-Right (x) (mm)",
                                             title_font=dict(size=st.session_state.axis_font_size),
                                              minor=dict(dtick = 2.5,showgrid=True), 
                                              range=[st.session_state.x_vector[0], st.session_state.x_vector[-1]],
                                              tick0=0, dtick = 5,
                                              #title_standoff=0
                                              )
        
            fig_isodoses_to_show_2.update_yaxes(title_text="Posterior-Anterior (y) (mm)",
                                             title_font=dict(size=st.session_state.axis_font_size),
                              range=[st.session_state.y_vector[-1], st.session_state.y_vector[0]],
                              minor=dict(dtick = 2.5,showgrid=True),
                              tick0=0, dtick = 5, 
                              #autorange="reversed"
                              )  
            fig_isodoses_to_show_2.update_traces(line=dict(width=st.session_state.line_width),
                                                selector=dict(type="scattergl"))
            fig_isodoses_to_show_2.update_traces(line=dict(width=st.session_state.line_width),
                                                selector=dict(type="contour"))
            fig_isodoses_to_show_2.update_traces(showscale=False,
                                                selector=dict(type="heatmap"))  
            fig_isodoses_to_show_2.update_layout(height=height_to_use_for_graphs, #width = 800,
                                               legend = dict(font = dict(#family = "Courier", 
                                                                         size = st.session_state.legend_font_size, 
                                                                         #color = "black"
                                                                         ))
                                               )
            st.plotly_chart(fig_isodoses_to_show_2, 
                            height = height_to_use_for_graphs, 
                            use_container_width = True)

else:

    st.session_state.jump_to_slice_old = jump_to_slice
    if show_Ultrasound:

        data_w_US = (st.session_state.data_fig_isodoses[st.session_state.slice_to_view])[:]
        slice_index = np.where(st.session_state.US_z_axis_values == st.session_state.z_vector[st.session_state.slice_to_view]) 
        data_w_US.insert(0, go.Heatmap(z= st.session_state.US_img[:,:,int(slice_index[0])],
                                   x = st.session_state.US_x_axis_values,
                                   y = st.session_state.US_y_axis_values,
                                   colorscale='gray'))

        fig_isodoses_to_show = go.Figure(data=data_w_US,
                                         layout=go.Layout(title=go.layout.Title(text=st.session_state.titles_fig_isodoses[st.session_state.slice_to_view],                                   
                                                                                font=dict(
                                                                                     #family="Courier New, monospace",
                                                                                     size=st.session_state.title_font_size,
                                                                                     #color="RebeccaPurple"
                                                                                 ))))

    else:
        fig_isodoses_to_show = go.Figure(data=st.session_state.data_fig_isodoses[st.session_state.slice_to_view],
                                         layout=go.Layout(title=go.layout.Title(text=st.session_state.titles_fig_isodoses[st.session_state.slice_to_view],                                   
                                                                                font=dict(
                                                                                     #family="Courier New, monospace",
                                                                                     size=st.session_state.title_font_size,
                                                                                     #color="RebeccaPurple"
                                                                                 ))))

    if st.session_state.dont_show_rectum:
        fig_isodoses_to_show.update_traces(visible='legendonly',
                      selector=dict(name="Rectum"))
        
    if st.session_state.dont_show_urethra:
        fig_isodoses_to_show.update_traces(visible='legendonly',
                      selector=dict(name="Urethra"))
        
    if st.session_state.dont_show_prostate:
        fig_isodoses_to_show.update_traces(visible='legendonly',
                      selector=dict(name="Prostate"))
        
    if st.session_state.dont_show_50:
        fig_isodoses_to_show.update_traces(visible='legendonly',
                      selector=dict(name="50%"))
        
    if st.session_state.dont_show_75:
        fig_isodoses_to_show.update_traces(visible='legendonly',
                      selector=dict(name="75%"))
        
    if st.session_state.dont_show_90:
        fig_isodoses_to_show.update_traces(visible='legendonly',
                      selector=dict(name="90%"))
        
    if st.session_state.dont_show_100:
        fig_isodoses_to_show.update_traces(visible='legendonly',
                      selector=dict(name="100%"))
        
    if st.session_state.dont_show_110:
        fig_isodoses_to_show.update_traces(visible='legendonly',
                      selector=dict(name="110%"))
        
    if st.session_state.dont_show_150:
        fig_isodoses_to_show.update_traces(visible='legendonly',
                      selector=dict(name="150%"))
        
    if st.session_state.dont_show_200:
        fig_isodoses_to_show.update_traces(visible='legendonly',
                      selector=dict(name="200%"))
        
    fig_isodoses_to_show.update_xaxes(title_text="Left-Right (x) (mm)",
                                     title_font=dict(size=st.session_state.axis_font_size),
                                      minor=dict(dtick = 2.5,showgrid=True), 
                                      range=[st.session_state.x_vector[0], st.session_state.x_vector[-1]],
                                      tick0=0, dtick = 5,
                                      #title_standoff=0
                                      )

    fig_isodoses_to_show.update_yaxes(title_text="Posterior-Anterior (y) (mm)",
                                     title_font=dict(size=st.session_state.axis_font_size),
                      range=[st.session_state.y_vector[-1], st.session_state.y_vector[0]],
                      minor=dict(dtick = 2.5,showgrid=True),
                      tick0=0, dtick = 5, 
                      #autorange="reversed"
                      )  

    fig_isodoses_to_show.update_traces(line=dict(width=st.session_state.line_width),
                                        selector=dict(type="scattergl"))
    fig_isodoses_to_show.update_traces(line=dict(width=st.session_state.line_width),
                                        selector=dict(type="contour"))
    fig_isodoses_to_show.update_traces(showscale=False,
                                        selector=dict(type="heatmap"))    

    fig_isodoses_to_show.update_layout(height=height_to_use_for_graphs, #width = 800,
                                       legend = dict(font = dict(#family = "Courier", 
                                                                 size = st.session_state.legend_font_size, 
                                                                 #color = "black"
                                                                 ))
                                       )
    st.plotly_chart(fig_isodoses_to_show, 
                    height = height_to_use_for_graphs, 
                    use_container_width = True)

col_1, col_2, col_3 = st.columns([2,1,5])
with col_1:
    st.write("Turn off the following in all plots:")

with col_2:
    if st.button('Reload Page'):
        reload_page = True

col_1, col_2, col_3, col_4, col_5, col_6, col_7, col_8, col_9, col_10 = st.columns([1,1,1,1,1,1,1,1,1,1])

with col_1:
    st.session_state.dont_show_50 = st.checkbox('50%')
    
with col_2:
    st.session_state.dont_show_75 = st.checkbox('75%')

with col_3:
    st.session_state.dont_show_90 = st.checkbox('90%')
    
with col_4:
    st.session_state.dont_show_100 = st.checkbox('100%')
    
with col_5:
    st.session_state.dont_show_110 = st.checkbox('110%')

with col_6:
    st.session_state.dont_show_150 = st.checkbox('150%')

with col_7:
    st.session_state.dont_show_200 = st.checkbox('200%')

with col_8:
    st.session_state.dont_show_prostate = st.checkbox('Prostate')
    
with col_9:
    st.session_state.dont_show_urethra = st.checkbox('Urethra')

with col_10:
    st.session_state.dont_show_rectum = st.checkbox('Rectum')

if reload_page == True:
    st.experimental_rerun()


