# Exploratory Data Analysis and Visualization System for Biostatistical Research

## [`data/`](./data/)

_Please note: For the public repository, `data/` has been omitted to respect data privacy/licensing._

The [`data/`](./data/) directory contains the datasets used in the project. It includes two subdirectories:
<details>
<summary>
<a href="./data/original/">
<code>original/</code>
</a>
Original data files imported from outside this repository
</summary>

- [`adni/`](./data/original/adni/)
- [`aric/`](./data/original/aric/)
- [`calcium/`](./data/original/calcium/)
- [`other/`](./data/original/other/)
</details>

<details>
<summary>
<a href="./src/data/processed/">
<code>processed/</code>
</a>
Interim data files generated manually or by a script within this repository
</summary>

- [`adni/`](./data/processed/adni/)
- [`aric/`](./data/processed/aric/)
- [`other/`](./data/processed/other/)
</details>

## [`assets/`](./assets/)
The [`assets/`](./assets/) directory contains final, presentation-ready tables, figures, and slides

<details>
<summary>
<a href="./assets/tables/">
<code>tables/</code>
</a>
Demographic characteristics tables, summary statistics tables, etc.
</summary>

- [`adni/`](./assets/tables/adni/)
- [`aric/`](./assets/tables/aric/)
</details>

<details>
<summary>
<a href="./assets/figures/">
<code>figures/</code>
</a>
Saved figures in PNG (raster/pixel) and PDF (vector) formats by interactive Dash apps
</summary>

- [`adni/`](./assets/figures/adni/)
- [`aric/`](./assets/figures/aric/)
</details>

<details>
<summary>
<a href="./assets/slides/">
<code>slides/</code>
</a>
Summary slides
</summary>

</details>

## [`src/`](./src/)
The [`src/`](./src/) directory contains the source code for this project. It is organized into the following subdirectories:

<ul>

### [`src/lib/`](./src/lib/)

This directory contains the library code for the project. The utility functions are organized into the following categories:
<details>
<summary>
<a href="./src/lib/general.py">
<code>general.py</code>
</a>
General utility functions
</summary>

- `get_stage_list()` returns the list of names of stages stratified by biomarkers Ab42, amyloid PET, (p-Tau or t-Tau)
</details>

<details>
<summary>
<a href="./src/lib/stats.py">
<code>stats.py</code>
</a>
Statistical analysis
</summary>

- `demographics_characteristics()` computes a summary statistics table of the study population
- `multiple_linear_regression()` fits a multiple linear regression model and returns model statistics
- `cluster_corr_df()` hierarchially clusters a correlation matrix
    - `get_linkage_methods()` returns list of available linkage methods for hierarchical clustering
    - `get_cluster_criteria()` returns list of available cluster criteria for hierarchical clustering
- `remove_diagonal()` masks the diagonal of a square matrix with NaN
- `fill_mirror()` fills a triangular matrix to a square matrix with its transpose
- `mask_outlier()` returns a mask that removes outliers when applied. Outliers are determined by [Local Outlier Factor](https://scikit-learn.org/stable/auto_examples/neighbors/plot_lof_outlier_detection.html)
- `corr_remove_outliers()` computes the outlier-removed correlation coefficient for each pair of variables; returns a correlation matrix
</details>

<details>
<summary>
<a href="./src/lib/dialog.py">
<code>dialog.py</code>
</a>
Tkinter dialogs
</summary>

- `dialog_select_directory()` prompts the user to select a directory in a dialog selection window and returns its absolute path
- `dialog_select_file()` prompts the user to select a file in a dialog selection window and returns its absolute path
</details>
<details>
<summary>
<a href="./src/lib/plotly.py">
<code>plotly.py</code>
</a>
Modifications to Plotly figure objects
</summary>

- `standard_layout()` configures a standard layout for plot template, axes, and fonts
- `add_box()` draws a box plot on top of a strip plot
- `add_pairwise_comparison()` annotates pairwise comparison results on top of a strip plot or box plot
    - `annotation_t_test()` computes the p-value from independent-sample t-test
    - `annotation_cohens_d()` computes Cohen's d effect size
    - `annotation_tukey()` performs Tukey's multiple comparison post-hoc to obtain the p-value
</details>
<details>
<summary>
<a href="./src/lib/r_interface.py">
<code>r_interface.py</code>
</a>
Interface to R
</summary>

- `tukey()` conducts Tukey's multiple comparison post-hoc in R for a single dependent variable
- `tukey_multiple_dvs()` conducts Tukey's multiple comparison post-hoc in R sequentially for a list of dependent variables; returns a table of the resultant p-values
</details>

### [`src/processing/`](./src/processing/)

This directory contains scripts that process [_original files_](./data/original/) and/or [_processed files_](./data/processed/) to [_processed files_](./data/processed/) for downstream analyses.

<details>
<summary>
<a href="./src/processing/adni/">
<code>adni/</code>
</a>
</summary>

<blockquote>

<details>
<summary>
<a href="./src/processing/adni/bmi.ipynb">
<code>bmi.ipynb</code>
</a>
Body Mass Index (BMI)
</summary>

- Input: [`VITALS_14Jul2023.csv`](./data/original/adni/VITALS_14Jul2023.csv)
- Output: [`bmi.csv`](./data/processed/adni/bmi.csv)
</details>

<details>
<summary>
<a href="./src/processing/adni/strem2.ipynb">
<code>strem2.ipynb</code>
</a>
CSF soluble triggering receptor expressed on myeloid cells 2 (sTREM2)
</summary>

- Input: [`ADNI_HAASS_WASHU_LAB_13Jul2023.csv`](./data/original/adni/ADNI_HAASS_WASHU_LAB_13Jul2023.csv)
- Output: [`strem2.csv`](./data/processed/adni/strem2.csv)
</details>

<details>
<summary>
<a href="./src/processing/adni/demographics.ipynb">
<code>demographics.ipynb</code>
</a>
Basic demographics
</summary>

- Input: [`ADNIMERGE_14Jul2023.csv`](./data/original/adni/ADNIMERGE_14Jul2023.csv), [`bmi.csv`](./data/processed/adni/bmi.csv)
- Output: [`demographics.csv`](./data/processed/adni/demographics.csv)
</details>

<details>
<summary>
<a href="./src/processing/adni/demographics_tau.ipynb">
<code>demographics_tau.ipynb</code>
</a>
Demographics with tau biomarker data
</summary>

- Input: [`ADNIMERGE_14Jul2023.csv`](./data/original/adni/ADNIMERGE_14Jul2023.csv), [`bmi.csv`](./data/processed/adni/bmi.csv)
- Output: [`demographics_tau.csv`](./data/processed/adni/demographics_tau.csv)
</details>

<details>
<summary>
<a href="./src/processing/adni/demographics_biomarkers.ipynb">
<code>demographics_biomarkers.ipynb</code>
</a>
Demographics with amyloid and tau biomarker data and stage assignment
</summary>

- Input: [`ADNIMERGE_14Jul2023.csv`](./data/original/adni/ADNIMERGE_14Jul2023.csv), [`bmi.csv`](./data/processed/adni/bmi.csv), [`strem2.csv`](./data/processed/adni/strem2.csv)
- Output: [`demographics_biomarkers.csv`](./data/processed/adni/demographics_biomarkers.csv)
</details>

<details>
<summary>
<a href="./src/processing/adni/lipidomics.ipynb">
<code>lipidomics.ipynb</code>
</a>
Plasma lipidomics, Meikle lab, longitudinal
</summary>

- Input: [`ADMCLIPIDOMICSMEIKLELABLONG_13Jul2023.csv`](./data/original/adni/ADMCLIPIDOMICSMEIKLELABLONG_13Jul2023.csv), [`Lipid_Models_Final.xlsx`](./data/original/adni/Lipid_Models_Final.xlsx)
- Output: [`lipidomics.csv`](./data/processed/adni/lipidomics.csv), [`lipidomics_total.csv`](./data/processed/adni/lipidomics_total.csv), [`lipidomics_dictionary.csv`](./data/processed/adni/lipidomics_dictionary.csv)
</details>

<details>
<summary>
<a href="./src/processing/adni/lipoprotein.ipynb">
<code>lipoprotein.ipynb</code>
</a>
Nightingale NMR analysis of lipoproteins and metabolites
</summary>

- Input: [`ADNINIGHTINGALELONG_05_24_21_27Jul2023.csv`](./data/original/adni/ADNINIGHTINGALELONG_05_24_21_27Jul2023.csv)
- Output: [`lipoprotein.csv`](./data/processed/adni/lipoprotein.csv), [`lipoprotein_dict.csv`](./data/processed/adni/lipoprotein_dict.csv)
</details>

<details>
<summary>
<a href="./src/processing/adni/somascan.ipynb">
<code>somascan.ipynb</code>
</a>
CSF proteomics SOMAscan 7000+ proteins post-QC, Cruchaga lab
</summary>

- Input: [`CruchagaLab_CSF_SOMAscan7k_Protein_matrix_postQC_20230620.csv`](./data/original/adni/CruchagaLab_CSF_SOMAscan7k_Protein_matrix_postQC_20230620.csv), [`ADNI_Cruchaga_lab_CSF_SOMAscan7k_analyte_information_20_06_2023.csv`](./data/original/adni/ADNI_Cruchaga_lab_CSF_SOMAscan7k_analyte_information_20_06_2023.csv)
- Output: [`somascan.csv`](./data/processed/adni/somascan.csv), [`somascan_dict.csv`](./data/processed/adni/somascan_dict.csv)
</details>

<details>
<summary>
<a href="./src/processing/adni/converters.ipynb">
<code>converters.ipynb</code>
</a>
Longitudinal decline in cognitive status (CN to MCI, MCI to AD, or CN to AD), excluding participants diagnosed with AD at baseline
</summary>

- Input: [`ADNIMERGE_14Jul2023.csv`](./data/original/adni/ADNIMERGE_14Jul2023.csv)
- Output: [`converters.csv`](./data/processed/adni/converters.csv)
</details>

<details>
<summary>
<a href="./src/processing/adni/converters_to_ad.ipynb">
<code>converters_to_ad.ipynb</code>
</a>
Longitudinal decline in cognitive status from CN or MCI to AD, excluding participants diagnosed with AD at baseline
</summary>

- Input: [`ADNIMERGE_14Jul2023.csv`](./data/original/adni/ADNIMERGE_14Jul2023.csv)
- Output: [`converters_to_ad.csv`](./data/processed/adni/converters_to_ad.csv)
</details>

</blockquote>

</details>

<details>
<summary>
<a href="./src/processing/aric/">
<code>aric/</code>
</a>
</summary>

<blockquote>

<details>
<summary>
<a href="./src/processing/aric/pilot.ipynb">
<code>pilot.ipynb</code>
</a>
Demographics and brain MRI data from the ARIC server for participants included in the pilot study
</summary>

- Input from ARIC server: `ARIC_NP/DATA_NP/Visits/Visit 5/derive54_np.sas7bdat`, `DATA_NP/Visits/Visit 5/derive_ncs51_np.sas7bdat`, `DATA_NP/Visits/Visit 1/derive13_np.sas7bdat`
- Input: [`all_eleigible_samples_AS2021_25v3.xlsx`](./data/original/aric/sample_selection/all_eleigible_samples_AS2021_25v3.xlsx), [`lipoproteins_6_29_23.csv`](./data/original/aric/lipoproteins_6_29_23.csv), [`dictionary.csv`](./data/processed/aric/dictionary.csv)
- Output: [`lipoprotein_list.csv`](./data/processed/aric/lipoprotein_list.csv), [`pilot.csv`](./data/processed/aric/pilot.csv), [`demographic_characteristics.csv`](./downloads/tables/aric/demographic_characteristics.csv)
</details>

<details>
<summary>
<a href="./src/processing/aric/pilot_eligible.ipynb">
<code>pilot_eligible.ipynb</code>
</a>
Demographics and brain MRI data for ARIC participants eligible under the inclusion criteria
</summary>

- Input from ARIC server: `ARIC_NP/DATA_NP/Visits/Visit 5/derive54_np.sas7bdat`, `DATA_NP/Visits/Visit 5/derive_ncs51_np.sas7bdat`, `DATA_NP/Visits/Visit 1/derive13_np.sas7bdat`, `DATA_NP/Visits/MultiVisit/V5_V11 Longitudinal MRI data/v5_v11_mri_derv_np_240221.sas7bdat`
- Input: [`all_eleigible_samples_AS2021_25v3.xlsx`](./data/original/aric/sample_selection/all_eleigible_samples_AS2021_25v3.xlsx), [`ARIC_Pilot_Updated_06032022.csv`](./data/original/aric/ARIC_Pilot_Updated_06032022.csv), [`lipoproteins_6_29_23.csv`](./data/original/aric/lipoproteins_6_29_23.csv), [`dictionary.csv`](./data/processed/aric/dictionary.csv)
- Output: [`lipoprotein_list.csv`](./data/processed/aric/lipoprotein_list.csv), [`pilot.csv`](./data/processed/aric/pilot.csv), [`demographic_characteristics.csv`](./downloads/tables/aric/demographic_characteristics.csv)
</details>

</blockquote>

</details>

<details>
<summary>
<a href="./src/processing/other/">
<code>other/</code>
</a>
</summary>

<blockquote>

<details>
<summary>
<a href="./src/processing/other/davidson.ipynb">
<code>davidson.ipynb</code>
</a>
Sean Davidson HDL Proteome Watch 2023
</summary>

- Input: [`HDL Proteome Watch 2023 Final.xlsx`](./data/original/other/HDL%20Proteome%20Watch%202023%20Final.xlsx)
- Output: [`hdl_proteome_davidson.csv`](./data/processed/other/hdl_proteome_davidson.csv)
</details>

</blockquote>

</details>

### [`src/analysis/`](./src/analysis/)

This directory contains Jupyter notebook files that perform analyses.
<details>
<summary>
<a href="./src/analysis/adni/">
<code>adni/</code>
</a>
</summary>

- [`lipidomics_tukey.ipynb`](./src/analysis/adni/lipidomics_tukey.ipynb) ANCOVA followed by Tukey post-hoc to determine which plasma lipids or biomarkers differ significantly between stages
- [`lipidomics_boxplot.ipynb`](./src/analysis/adni/lipidomics_boxplot.ipynb) Distribution of plasma lipids or biomarkers across stages
- [`survival.ipynb`](./src/analysis/adni/survival.ipynb) Survival analysis (Kaplan-Meier survival curve, Cox's proportional hazard model) comparing risk of conversion to AD between biomarker groups.
- [`survival_hdl_ratio.ipynb`](./src/analysis/adni/survival_hdl_ratio.ipynb) Survival analysis comparing cognitive decline between tertiles of non-small HDL FC-to-CE ratio.
- [`somascan_pca.ipynb`](./src/analysis/adni/somascan_pca.ipynb) Clustering of CSF proteins by PCA, followed by linear regression with dependent variable pTau
- [`somascan_boxplot.ipynb`](./src/analysis/adni/lipidomics_tukey.ipynb) Distribution of CSF proteins across cognitive statuses
- [`strem2_lipidomics_regression.ipynb`](./src/analysis/adni/strem2_lipidomics_regression.ipynb) Linear regression of CSF sTREM2 on plasma lipids.
- [`strem2_lipoprotein_regression.ipynb`](./src/analysis/adni/strem2_lipoprotein_regression.ipynb) Linear regression of CSF sTREM2 on plasma lipoprotein subclasses.
</details>

<details>
<summary>
<a href="./src/analysis/aric/">
<code>aric/</code>
</a>
</summary>
</details>

<details>
<summary>
<a href="./src/analysis/calcium/">
<code>calcium/</code>
</a>
</summary>

- [`calcium_all_sites.ipynb`](./src/analysis/calcium/calcium_all_sites.ipynb) Distribution of calcium measurements compared between Vista and Roche, data from all sites combined
</details>

<details>
<summary>
<a href="./src/analysis/other/">
<code>other/</code>
</a>
</summary>

- [`imagej_particle_results_hdl.ipynb`](./src/processing/other/imagej_particle_results_hdl.ipynb) HDL1 and HDL2 particle analysis on EM images using results exported from ImageJ
</details>

</ul>

## Publications
The analysis in this repository contributed to the following publications:

- Li, D.; Mantyh, W. G.; Men, L.; Jain, I.; Glittenberg, M.; An, B.; Zhang, L.; Li, L.; for the Alzheimer’s Disease Neuroimaging Initiative. sTREM2 in Discordant CSF Aβ42 and P‐tau181. _Alz & Dem Diag Ass & Dis Mo_ **2025**, _17_ (1), e70072. https://doi.org/10.1002/dad2.70072.

- Li, D.; An, B.; Men, L.; Glittenberg, M.; Lutsey, P. L.; Mielke, M. M.; Yu, F.; Hoogeveen, R. C.; Gottesman, R.; Zhang, L.; Meyer, M.; Sullivan, K.; Zantek, N.; Alonso, A.; Walker, K. A. The Association of High-Density Lipoprotein Cargo Proteins with Brain Volume in Older Adults in the Atherosclerosis Risk in Communities (ARIC). _Journal of Alzheimer’s Disease_ **2025**, _103_ (3), 724–734. https://doi.org/10.1177/13872877241305806.

## Data Sources
- [Alzheimer's Disease Neuroimaging Initiative (ADNI)](https://adni.loni.usc.edu/)
- [The Atherosclerosis Risk in Communities Study (ARIC)](https://aric.cscc.unc.edu/aric9/)