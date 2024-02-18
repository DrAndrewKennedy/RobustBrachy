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
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import pandas.io.formats.style


def construct_robust_dvhs_graph(
    all_nominal_dvhs,
    all_robust_dvh_summary,
    colours,
    line_size,
    axis_font_size,
    legend_font_size,
):
    fig_dvh = go.Figure()

    # nominal dvhs
    for i, name in enumerate(["Prostate Nominal", "Urethra Nominal", "Rectum Nominal"]):
        fig_dvh.add_trace(
            go.Scattergl(
                x=all_nominal_dvhs[0][i][0, :],
                y=all_nominal_dvhs[0][i][1, :],
                mode="lines",
                showlegend=True,
                name=name,
                line=dict(
                    color=colours[i][0],
                    width=line_size,
                ),
            )
        )

    # robust dvhs
    robust_dvh_names = [
        [
            ["Prostate 68% CI", "Prostate 68% CI"],
            ["Prostate 95% CI", "Prostate 95% CI"],
            ["Prostate max-min", "Prostate max-min"],
        ],
        [
            ["Urethra 68% CI", "Urethra 68% CI"],
            ["Urethra 95% CI", "Urethra 95% CI"],
            ["Urethra max-min", "Urethra max-min"],
        ],
        [
            ["Rectum 68% CI", "Rectum 68% CI"],
            ["Rectum 95% CI", "Rectum 95% CI"],
            ["Rectum max-min", "Rectum max-min"],
        ],
    ]
    robust_idx = [[1, 2], [3, 4], [5, 6]]

    # shape of all_robust_dvh_summary = [ CI = mu +/- n x SD ][ plan index = 0 for single RE ][ structure ][ dose/vol ][ arry values ]

    for i, names in enumerate(robust_dvh_names):
        for j, names_2 in enumerate(names):
            fig_dvh.add_trace(
                go.Scatter(
                    x=(all_robust_dvh_summary[robust_idx[j][1]][0][i][0, :])[
                        ~np.isnan(all_robust_dvh_summary[robust_idx[j][1]][0][i][0, :])
                    ],
                    y=(all_robust_dvh_summary[robust_idx[j][1]][0][i][1, :])[
                        ~np.isnan(all_robust_dvh_summary[robust_idx[j][1]][0][i][0, :])
                    ],
                    mode="lines",
                    showlegend=False,
                    legendgroup=names_2[0],
                    name=names_2[1],
                    line=dict(
                        color=colours[i][j + 1],
                        width=line_size,
                    ),
                )
            )
            fig_dvh.add_trace(
                go.Scatter(
                    x=(all_robust_dvh_summary[robust_idx[j][0]][0][i][0, :])[
                        ~np.isnan(all_robust_dvh_summary[robust_idx[j][0]][0][i][0, :])
                    ],
                    y=(all_robust_dvh_summary[robust_idx[j][0]][0][i][1, :])[
                        ~np.isnan(all_robust_dvh_summary[robust_idx[j][0]][0][i][0, :])
                    ],
                    mode="lines",
                    showlegend=True,
                    name=names_2[0],
                    legendgroup=names_2[0],
                    line=dict(
                        color=colours[i][j + 1],
                        width=line_size,
                    ),
                    fill="tonexty",
                    fillcolor=colours[i][j + 1],
                )
            )

    fig_dvh.update_xaxes(
        title_text="Dose (Gy)",  # <Br>(a)",
        title_font=dict(size=axis_font_size),
        minor=dict(dtick=1, showgrid=True),
        range=[0, 36],
        tick0=0,
        dtick=5,
        # title_standoff=0
    )

    fig_dvh.update_yaxes(
        title_text="Relative Volume (%)",
        title_font=dict(size=axis_font_size),
        range=[0, 101],
        minor=dict(dtick=2.5, showgrid=True),
        tick0=0,
        dtick=10,
        # title_standoff=0
    )

    fig_dvh.update_layout(
        # height=height_to_use_for_graphs,  # width = 800,
        legend=dict(
            font=dict(  # family = "Courier",
                size=legend_font_size,
                # color = "black"
            )
        ),
    )

    return fig_dvh


def construct_nominal_dvhs_graph(
    all_nominal_dvhs,
    colours,
    line_size,
    axis_font_size,
    legend_font_size,
):
    fig_dvh = go.Figure()

    # nominal dvhs
    for i, name in enumerate(["Prostate Nominal", "Urethra Nominal", "Rectum Nominal"]):
        fig_dvh.add_trace(
            go.Scattergl(
                x=all_nominal_dvhs[i][0, :],
                y=all_nominal_dvhs[i][1, :],
                mode="lines",
                showlegend=True,
                name=name,
                line=dict(
                    color=colours[i][0],
                    width=line_size,
                ),
            )
        )

    fig_dvh.update_xaxes(
        title_text="Dose (Gy)",  # <Br>(a)",
        title_font=dict(size=axis_font_size),
        minor=dict(dtick=1, showgrid=True),
        range=[0, 36],
        tick0=0,
        dtick=5,
        # title_standoff=0
    )

    fig_dvh.update_yaxes(
        title_text="Relative Volume (%)",
        title_font=dict(size=axis_font_size),
        range=[0, 101],
        minor=dict(dtick=2.5, showgrid=True),
        tick0=0,
        dtick=10,
        # title_standoff=0
    )

    fig_dvh.update_layout(
        # height=height_to_use_for_graphs,  # width = 800,
        legend=dict(
            font=dict(  # family = "Courier",
                size=legend_font_size,
                # color = "black"
            )
        ),
    )

    return fig_dvh


#####################################################
###          Code from  stackoverflow.com         ###
###      https://stackoverflow.com/a/47723330     ###

###  **** Modifed to take mulitple inputs ****    ###

#####################################################


def write_to_html_file(
    arry_of_df, arry_of_titles=False, filename="out.html"
):  # df, title="", filename="out.html"):
    """
    Write an entire SET OF dataframeS to an HTML file with nice formatting.
    """
    if arry_of_titles == False:
        arry_of_titles = []
        for i in range(arry_of_df):
            arry_of_titles.append("")

    result = """
<html>
<head>
<style>

    h2 {
        text-align: center;
        font-family: Helvetica, Arial, sans-serif;
    }
    table { 
        margin-left: auto;
        margin-right: auto;
    }
    table, th, td {
        border: 1px solid black;
        border-collapse: collapse;
    }
    th, td {
        padding: 5px;
        text-align: center;
        font-family: Helvetica, Arial, sans-serif;
        font-size: 90%;
    }
    table tbody tr:hover {
        background-color: #dddddd;
    }
    .wide {
        width: 90%; 
    }

</style>
</head>
<body>
    """
    for i, df in enumerate(arry_of_df):
        result += "<h2> %s </h2>\n" % arry_of_titles[i]  # title
        if type(df) == pd.io.formats.style.Styler:
            result += df.render()
        else:
            result += df.to_html(classes="wide", escape=False)
        result += """
</body>
</html>
"""
    with open(filename, "w") as f:
        f.write(result)


__all__ = [
    "construct_robust_dvhs_graph",
    "construct_nominal_dvhs_graph",
    "write_to_html_file",
]
