import pandas as pd
import MonteCarloSimulation as mc
import RELRAD as rr
import copy
from concurrent.futures import ThreadPoolExecutor

'''
RELRAD-software, general software for reliability studies of radial power systems
    Copyright (C) 2025  Sondre Modalsli Aaberg

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''


# Main file, run either mc.MonteCarlo or rr.RELRAD here



#Examples:

#rr.RELRAD('RBMC p214.xlsx', 'RELRADResultsRBMCp214.xlsx')
#rr.RELRAD('Myhre 6Bus System.xlsx', 'RELRADResultsMyhre6Bus.xlsx', DSEBF = False, DERS = False)
#mc.MonteCarlo('Myhre 6Bus System.xlsx', 'MonteCarloResultsMyhre6Bus.xlsx', beta=0.02, DSEBF = False, LoadCurve=False, DERS=False)

#mc.MonteCarlo('BUS 2.xlsx', 'MonteCarloResultsBUS2.xlsx', beta=0.05, DSEBF = False)
#rr.RELRAD('BUS 2.xlsx', 'RELRADResultsBUS2.xlsx', DSEBF = False, createFIM=True, DERS=False)

'''
mc.MonteCarlo('BUS 4.xlsx', 'MonteCarloResultsBUS4.xlsx', beta=0.02, DSEBF = False)
rr.RELRAD('BUS 4.xlsx', 'RELRADResultsBUS4.xlsx', DSEBF = False)
mc.MonteCarlo('RBMC p214.xlsx', 'MonteCarloResultsRBMCp214.xlsx', beta=0.02, DSEBF = False)
rr.RELRAD('RBMC p214.xlsx', 'RELRADResultsRBMCp214.xlsx', DSEBF = False)
mc.MonteCarlo('BUS 2.xlsx', 'MonteCarloResultsBUS2.xlsx', beta=0.02, DSEBF = False)
rr.RELRAD('BUS 2.xlsx', 'RELRADResultsBUS2.xlsx', DSEBF = False)
mc.MonteCarlo('BUS 6.xlsx', 'MonteCarloResultsBUS6.xlsx', beta=0.02, DSEBF = False)
rr.RELRAD('BUS 6.xlsx', 'RELRADResultsBUS6.xlsx', DSEBF = False)
mc.MonteCarlo('SimpleTest.xlsx', 'MonteCarloResultsSimpleTest.xlsx', beta=0.02, DSEBF = False, DERS=False)
rr.RELRAD('SimpleTest.xlsx', 'RELRADResultsSimpleTest.xlsx', DSEBF = False, DERS=False)
'''

#mc.MonteCarlo('BUS 6 Sub2 mod.xlsx', 'MonteCarloResultsBUS6Sub2M.xlsx', beta=0.02, DSEBF = False, LoadCurve=False, DERS=False)
#mc.MonteCarlo('BUS 6 Sub2 mod.xlsx', 'MonteCarloResultsBUS6Sub2MLoadCurve.xlsx', beta=0.02, DSEBF = False, LoadCurve=True, DERS=False)
#mc.MonteCarlo('BUS 6 Sub2 mod.xlsx', 'MonteCarloResultsBUS6Sub2MDERS.xlsx', beta=0.02, DSEBF = False, LoadCurve=False, DERS=True)
#mc.MonteCarlo('BUS 6 Sub2 mod.xlsx', 'MonteCarloResultsBUS6Sub2MLoadCurveDERS.xlsx', beta=0.02, DSEBF = False, LoadCurve=True, DERS=True)
#mc.MonteCarlo('BUS 6 Sub2 mod.xlsx', 'MonteCarloResultsBUS6Sub2MDERSCurve.xlsx', beta=0.02, DSEBF = False, LoadCurve=True, DERScurve=True, DERS=True)


#rr.RELRAD('RBMC p214.xlsx', 'RELRADResultsRBMCp214DERS_BESS.xlsx', DSEBF = False, DERS = True)
#rr.RELRAD('RBMC p214.xlsx', 'RELRADResultsRBMCp214_DERSCase3.xlsx', DSEBF = False, DERS = True)


#mc.MonteCarlo('RBMC p214.xlsx', 'MonteCarloResults_p214_BaseCase.xlsx', beta=0.02, DSEBF = False, LoadCurve=False, DERS=False)
#mc.MonteCarlo('RBMC p214.xlsx', 'MonteCarloResults_p214_LoadCurve.xlsx', beta=0.02, DSEBF = False, LoadCurve=True, DERS=False)
#mc.MonteCarlo('RBMC p214.xlsx', 'MonteCarloResults_p214_DERS.xlsx', beta=0.02, DSEBF = False, LoadCurve=False, DERS=True)
#mc.MonteCarlo('RBMC p214.xlsx', 'MonteCarloResults_p214_LoadCurveDERS.xlsx', beta=0.02, DSEBF = False, LoadCurve=True, DERS=True)
#mc.MonteCarlo('RBMC p214.xlsx', 'MonteCarloResults_p214_DERSCurve.xlsx', beta=0.02, DSEBF = False, LoadCurve=True, DERScurve=True, DERS=True)

#rr.RELRAD('RBMC p214.xlsx', 'RELRADResultsRBMCp214.xlsx', DSEBF = False, DERS = False)
#rr.RELRAD('RBMC p214.xlsx', 'RELRADResultsRBMCp214DERS.xlsx', DSEBF = False, DERS = True)


#rr.RELRAD('BUS 6.xlsx', 'RELRADResultsBUS6.xlsx', DSEBF = False)



#rr.RELRAD('BUS 5 RELSAD MOD.xlsx', 'RELRADResultsBUS5.xlsx', DSEBF = False, DERS=False)
#mc.MonteCarlo('BUS 5 RELSAD MOD.xlsx', 'MonteCarloResultsBUS5.xlsx', beta=0.05, DSEBF = False, LoadCurve=False, DERS=False)

######## NYTT ##########

#bus 2 med loadcurve

#rr.RELRAD('Test_Systems_Verified/BUS 2 Case E.xlsx', 'Verified_Results/RELRAD_Results_Bus2_Case_E_SondreModalsliAaberg.xlsx', DSEBF = False, DERS = False, createFIM=False)
#mc.MonteCarlo('Test_Systems_Verified/BUS 2 Case E.xlsx', 'Verified_Results/MonteCarlo_Results_Bus2_Case_E_Load_Curve_SondreModalsliAaberg.xlsx', beta=0.02, DSEBF = False, LoadCurve=True, DERS=False)
#mc.MonteCarlo('Test_Systems_Verified/BUS 2 Case E.xlsx', 'Verified_Results/MonteCarlo_Results_Bus2_Case_E_SondreModalsliAaberg.xlsx', beta=0.02, DSEBF = False, LoadCurve=False, DERS=False)

# bus 4 alle tre
rr.RELRAD('Test_Systems_Verified/BUS 4 Case A.xlsx', 'Verified_Results/RELRAD_Results_Bus4_Case_A_SondreModalsliAaberg.xlsx', DSEBF = False, DERS = False, createFIM=False)
mc.MonteCarlo('Test_Systems_Verified/BUS 4 Case A.xlsx', 'Verified_Results/MonteCarlo_Results_Bus4_Case_A_Load_Curve_SondreModalsliAaberg.xlsx', beta=0.02, DSEBF = False, LoadCurve=True, DERS=False)
mc.MonteCarlo('Test_Systems_Verified/BUS 4 Case A.xlsx', 'Verified_Results/MonteCarlo_Results_Bus4_Case_A_SondreModalsliAaberg.xlsx', beta=0.02, DSEBF = False, LoadCurve=False, DERS=False)

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def compare_bus_case(bus=2, case='E',
                     reference_values=[0.248, 0.77, 3.08, 8.844],
                     results_folder='Verified_Results', save_fig=True):
    """
    Sammenligner SAIFI, SAIDI, CAIDI og EENS mellom RELRAD, MCS, MCS (med load curve) og referanse.
    Viser N og β direkte under hver metode i legenden nederst.
    """

    # Filnavn
    relrad_file = f"RELRAD_Results_Bus{bus}_Case_{case}_SondreModalsliAaberg.xlsx"
    mcs_file = f"MonteCarlo_Results_Bus{bus}_Case_{case}_SondreModalsliAaberg.xlsx"
    mcs_loadcurve_file = f"MonteCarlo_Results_Bus{bus}_Case_{case}_Load_Curve_SondreModalsliAaberg.xlsx"

    relrad_path = os.path.join(results_folder, relrad_file)
    mcs_path = os.path.join(results_folder, mcs_file)
    mcs_loadcurve_path = os.path.join(results_folder, mcs_loadcurve_file)

    # Les Excel
    relrad_df = pd.read_excel(relrad_path, sheet_name="Load Points", index_col=0)
    mcs_df = pd.read_excel(mcs_path, sheet_name="Load Points", index_col=0) if os.path.exists(mcs_path) else None
    mcs_loadcurve_df = pd.read_excel(mcs_loadcurve_path, sheet_name="Load Points", index_col=0) if os.path.exists(mcs_loadcurve_path) else None

    # Hent TOTAL-rader
    relrad_total = relrad_df.loc['TOTAL', ['SAIFI', 'SAIDI', 'CAIDI', 'EENS']].astype(float).values
    mcs_total = mcs_df.loc['TOTAL', ['SAIFI', 'SAIDI', 'CAIDI', 'EENS']].astype(float).values if mcs_df is not None else [np.nan]*4
    mcs_loadcurve_total = mcs_loadcurve_df.loc['TOTAL', ['SAIFI', 'SAIDI', 'CAIDI', 'EENS']].astype(float).values if mcs_loadcurve_df is not None else [np.nan]*4

    # Hent N og β
    def extract_info(df):
        if df is None:
            return None, None
        n_sim, beta = None, None
        for c in df.columns:
            lc = c.lower().strip()
            if "nr" in lc and "simul" in lc:
                val = df[c].dropna()
                if not val.empty:
                    n_sim = val.iloc[-1]
            elif "beta" in lc:
                val = df[c].dropna()
                if not val.empty:
                    beta = val.iloc[-1]
        return n_sim, beta

    mcs_n, mcs_beta = extract_info(mcs_df)
    mcs_lc_n, mcs_lc_beta = extract_info(mcs_loadcurve_df)

    # Beregn prosentavvik
    def diff(vals): 
        return [(v / ref - 1) * 100 if not np.isnan(v) else np.nan for v, ref in zip(vals, reference_values)]

    relrad_diff = diff(relrad_total)
    mcs_diff = diff(mcs_total)
    mcs_loadcurve_diff = diff(mcs_loadcurve_total)

    # Enheter
    units = {
        "SAIFI": r"$\frac{f.}{\mathrm{cust.}\cdot \mathrm{yr}}$",
        "SAIDI": r"$\frac{h}{\mathrm{cust.}\cdot \mathrm{yr}}$",
        "CAIDI": r"$\frac{h}{\mathrm{f.}}$",
        "EENS":  r"$\frac{\mathrm{MWh}}{\mathrm{yr}}$"
    }

    metrics = ['SAIFI', 'SAIDI', 'CAIDI', 'EENS']
    datasets = {
        'Reference': reference_values,
        'RELRAD': relrad_total,
        'MCS': mcs_total,
        'MCS (Load Curve)': mcs_loadcurve_total
    }
    colors = {
        'Reference': 'lightgray',
        'RELRAD': '#2E86C1',
        'MCS': '#28B463',
        'MCS (Load Curve)': '#F39C12'
    }

    # Lag 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    width = 0.2
    x = np.arange(len(datasets))

    for i, metric in enumerate(metrics):
        ax = axes[i]
        vals = [datasets[key][i] for key in datasets.keys()]
        bars = ax.bar(x, vals, color=[colors[k] for k in datasets.keys()],
                      edgecolor='black', width=width*2.5)

        # Verdier på stolpene
        for j, bar in enumerate(bars):
            h = bar.get_height()
            if not np.isnan(h):
                ax.annotate(f'{h:.4f}', xy=(bar.get_x()+bar.get_width()/2, h),
                            xytext=(0, 4), textcoords="offset points",
                            ha='center', va='bottom', fontsize=9)

        # Prosentavvik
        diffs = [0, relrad_diff[i], mcs_diff[i], mcs_loadcurve_diff[i]]
        for j, diff_val in enumerate(diffs):
            if not np.isnan(diff_val) and j > 0:
                ax.text(j, vals[j] + (max(vals) * 0.10),
                        f'{diff_val:+.3f}%',
                        ha='center', va='bottom',
                        fontsize=9, fontweight='bold',
                        color=colors[list(datasets.keys())[j]])

        ax.set_title(f"{metric}", fontsize=11)
        ax.set_ylabel(f"{units[metric]}", fontsize=10)
        ax.set_xticks([])
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        ax.set_ylim(0, max(vals)*1.25)

    # Felles legend nederst med N og β under
    legend_labels = list(datasets.keys())
    # legg til N og β på MCS og MCS (Load Curve) i legend labels
    legend_labels[2] += f"\nN={int(mcs_n):,}\nβ={mcs_beta:.5f}" if mcs_n and mcs_beta else legend_labels[2]
    legend_labels[3] += f"\nN={int(mcs_lc_n):,}\nβ={mcs_lc_beta:.5f}" if mcs_lc_n and mcs_lc_beta else legend_labels[3]
    legend_colors = [colors[k] for k in datasets.keys()]
    legend_patches = [plt.Rectangle((0, 0), 1, 1, color=clr, ec='black') for clr in legend_colors]

    # Tegn selve legend-boksene
    legend = fig.legend(legend_patches, legend_labels,
                        loc='lower center', bbox_to_anchor=(0.5, 0.01),
                        ncol=4, fontsize=11, frameon=True)

    # --- Legg N og β rett under hver boks i legenden ---
    text_y = 0.05  # vertikal plassering under boksen
    text_fontsize = 9
    spacing = np.linspace(0.125, 0.875, 4)  # 4 kolonner

    # Felles tittel
    fig.suptitle(f"Reliability Indices Comparison – RBTS Bus {bus} Case {case}",
                 fontsize=14, fontweight='bold', y=0.99)

    #plt.tight_layout(rect=[0, 0.05, 1, 0.93])

    # Lagre figur
    if save_fig:
        save_name = f"Comparison_Bus{bus}_Case_{case}_Subplots.png"
        plt.savefig(os.path.join(results_folder, save_name), dpi=300, bbox_inches='tight')
        print(f"Figur lagret som {save_name}")

    plt.show()

compare_bus_case(bus=4, case='A')
