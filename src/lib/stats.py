"""This module contains utility functions for statistical analysis."""

import pandas as pd
import numpy as np
from itertools import compress
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.tools.tools import add_constant
from statsmodels.regression.linear_model import OLS, RegressionResultsWrapper
from scipy.stats import fisher_exact, zscore, shapiro
from scipy.stats.contingency import crosstab, expected_freq, chi2_contingency
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.neighbors import LocalOutlierFactor


def demographic_characteristics(data: pd.DataFrame, stage_list: list[str]) -> pd.DataFrame:
    """
    Compute a DataFrame of summary statistics for demographics data.
    Continuous variables are reported with group means, std, and p-value from type II ANOVA.
    Category variables are reported with counts, percentages, and p-value from chi-squared
    contingency table test.

    Parameters
    ----------
    data : pd.DataFrame
        dataframe with demographics data
    stage_list : list[str]
        list of stages

    Returns
    -------
    pd.DataFrame
        demographic characteristics table
    """
    #################### Define continuous and categorical variables ####################
    # Note: Use -1 to represent missing value np.nan
    stats_dict: pd.DataFrame = pd.DataFrame(
        columns=[
            "variable_full_name",
            "variable",
            "unit",
            "value",
            "value_full_name",
        ],
        data=np.array(
            [
                ["Age", "age", "year", None, None],
                ["BMI", "bmi", "kg/m^2", None, None],
                ["Sex", "sex", None, [0, 1], ["Female", "Male"]],
                ["Cognitive Status", "cog", None, [0, 1, 2], ["CN", "MCI", "AD"]],
                ["APOE-\u03B54 Carrier Status", "apoe4", None, [0, 1], ["No", "Yes"]],
                ["Race", "race", None, [True, False], ["White", "Other"]],
                ["Diabetes", "diabetes", None, [False, True], ["No", "Yes"]],
                ["CSF A\u03B242", "ab42", "pg/mL", None, None],
                ["Amyloid PET", "av45", "SUVR", None, None],
                ["CSF pTau", "ptau", "ng/mL", None, None],
                ["CSF pTau/A\u03B242", "ptau_ab42", "1", None, None],
                ["CSF sTREM2", "strem2", "pg/mL", None, None],
                ["UPenn CSF A\u03B242", "ab42_upenn", "pg/mL", None, None],
                ["UPenn CSF A\u03B240", "ab40_upenn", "ng/mL", None, None],
                ["UPenn CSF pTau", "ptau_upenn", "pg/mL", None, None],
                ["UPenn CSF A\u03B242/40", "ab4240", "e-6", None, None],
                ["Age", "V5AGE52", "year", None, None],
                ["BMI", "BMI51", "kg/m^2", None, None],
                ["\u03B2-Amyloid Load on PET", "globalcortex", "SUVR", None, None],
                ["Gender", "GENDER51", None, ["F", "M"], ["Female", "Male"]],
                ["Race", "RACEGRP51", None, ["A", "B", "I", "W"], ["Asian", "Black", "Native American", "White"]],
                ["Cognitive Status Diagnosis", "COGDIAG51", None, ["N", "M", "D", "U"], ["Normal", "MCI", "Dementia", "Unknown"]],
                ["APOE-\u03B54 Carrier Status", "FINAL_APOE", None, [0, 1], ["No", "Yes"]],
                ["Education Level", "ELEVEL02", None, [1, 2, 3, -1], ["Grade 0-11", "Grade 12-16", "Grade 17 or Higher", "Missing"]],
                ["Pro-rated MMSE Score", "PRORATEDMMS51", "1", None, None],
                ["Systolic Blood Pressure", "SYSTOLIC51", "mmHg", None, None],
                ["Diastolic Blood Pressure", "DIASTOLIC51", "mmHg", None, None],
                ["HDL Cholesterol", "HDL51", "mg/dL", None, None],
                ["LDL Cholesterol", "LDL51", "mg/dL", None, None],
                ["Total Cholesterol", "TCH51", "mg/dL", None, None],
                ["Triglycerides", "TRG51", "mg/dL", None, None],
                ["NCS Estimated Total Intracranial Volume", "ETIV51", "mm^3", None, None],
                ["NCS Total AD Signature Region Volume", "ADSIGREGVOL51", "mm^3", None, None],
                ["Estimated Total Intracranial Volume", "EstimatedTotalIntraCranialVol", "mm^3", None, None],
                ["Temporal Parietal Meta-ROI Volume", "temporal_parietal_metaroi_vol", "mm^3", None, None],
                [
                    "ARIC Field Center",
                    "CENTER",
                    None,
                    ["F", "J", "M", "W"],
                    ["Forsyth County, NC", "Jackson City, MS", "Minneapolis Townships, MN", "Washington County, MD"],
                ],
                ["Current Cigarette Smoking Status", "CURSMK52", None, [0, 1, -1], ["Non-smoker", "Smoker", "Missing"]],
                ["Diabetes Prevalence", "DIABTS54", None, [0, 1, -1], ["No", "Yes", "Missing"]],
                ["Hypertension Prevalence", "HYPERT55", None, [0, 1, -1], ["No", "Yes", "Missing"]],
                ["Coronary Heart Disease Prevalence", "PRVCHD51", None, [0, 1, -1], ["No", "Yes", "Missing"]],
                ["Stroke Prevalence", "PRVSTR51", None, [0, 1, -1], ["No", "Yes", "Missing"]],
                ["Definite or Possible Heart Failure Prevalence", "PREVDEFPOSSHF51", None, [0, 1, -1], ["No", "Yes", "Missing"]],
                ["Cholesterol Lowering Medication Use", "CHOLMDCODE53", None, [0, 1, -1], ["No", "Yes", "Missing"]],
                ["Antihypertensive Medication Use", "HYPTMDCODE52", None, [0, 1, -1], ["No", "Yes", "Missing"]],
                ["Statin Use", "STATINCODE52", None, [0, 1, -1], ["No", "Yes", "Missing"]],
            ],
            dtype=object,
        ),
    ).convert_dtypes()
    stats_dict["continuous"] = stats_dict["value"].isnull().astype(bool)

    ##### Filter the statistics DataFrame to include only columns present in the data #####
    stats_dict: pd.DataFrame = stats_dict.loc[stats_dict["variable"].isin(data.columns)].reset_index(drop=True)

    ############### Compute and store p-values for each variable in a list ###############
    p_value_list: list[str] = []

    # Iterate over each variable
    for index, row in stats_dict.iterrows():

        # Extract variable name
        var_str: str = row["variable"]

        if row["continuous"]:
            # If continuous, perform two-way ANOVA
            data_cleaned: pd.DataFrame = data.dropna(subset=[var_str, "stage"])
            ols_model: RegressionResultsWrapper = ols(f"{var_str} ~ C(stage)", data=data_cleaned).fit()
            p_value: float = anova_lm(ols_model, typ=3)["PR(>F)"].loc["C(stage)"]
        else:
            # If categorical, compute the contingency table
            elements, observed = crosstab(data[var_str], data["stage"])

            # Update categorical values to include only the observed elements
            values_observed: np.ndarray = np.nan_to_num(elements[0], copy=True, nan=-1, posinf=None, neginf=None)
            membership: np.ndarray = np.isin(row["value"], values_observed)
            stats_dict.at[index, "value"] = list(compress(row["value"], membership))
            stats_dict.at[index, "value_full_name"] = list(compress(row["value_full_name"], membership))

            # Check whether
            # 1. the size of contingency table is 2-by-2, and
            # 2. the number of expected frequencies <= 5 is more than 20%
            expected: np.ndarray = expected_freq(observed)
            if expected.size == 4 and (expected < 5).any():
                # If so, perform Fisher's exact test for small sample sizes
                p_value: float = fisher_exact(observed, alternative="two-sided")[1]
            else:
                # Otherwise, perform chi-squared contingency table test
                p_value: float = chi2_contingency(observed, correction=True, lambda_=None)[1]

        # Append p-value to the list
        if p_value < 0.001:
            p_value_list.append("< 0.001")
        else:
            p_value_list.append(f"{p_value:.4f}")

    ############### Create variable name indices for the statistics DataFrame ###############
    index1: list[str] = ["N"]
    variable_index: list[str] = []

    # Iterate over each variable
    for _, row in stats_dict.iterrows():

        # Extract variable name and unit
        var_full_name_str: str = row["variable_full_name"]

        if row["continuous"]:
            # If continuous, append variable name to the index
            unit_str: str = row["unit"]
            index1.append(f"{var_full_name_str} ({unit_str})")
            variable_index.append(f"{var_full_name_str} ({unit_str})")
        else:
            # If categorical, append variable name and all possible values to the index
            variable_index.append(var_full_name_str)
            index1.extend([var_full_name_str] + [f"{value} (n)" for value in row["value_full_name"]])

    # Instantiate a DataFrame to store summary statistics
    stats_data: pd.DataFrame = pd.DataFrame(index=index1, columns=["Total"] + stage_list)

    # Store group subsets in a list
    stage_subset: list[pd.DataFrame] = [data.loc[data["stage"] == stage] for stage in stage_list]

    #################### Compute summary statistics by group/column ####################
    for ii, group in enumerate([data] + stage_subset):
        # Initialize column with total count
        group_stats: list[int | str] = [len(group)]

        # Iterate over each variable
        for _, row in stats_dict.iterrows():

            # Extract variable name
            var_str: str = row["variable"]

            if row["continuous"]:
                # If continuous, compute mean and std
                group_stats.append(f"{group[var_str].mean():.1f} ({group[var_str].std():.1f})")
            else:
                # If categorical, compute counts and percentages for each possible value
                group_stats.append("")
                for value in row["value"]:
                    if value == -1:
                        count: int = group[var_str].isna().sum()
                        group_stats.append(f"{count} ({(100 * count / len(group)):.0f}%)")
                    else:
                        count: int = (group[var_str] == value).sum()
                        group_stats.append(f"{count} ({(100 * count / len(group)):.0f}%)")

        # Append column to the DataFrame
        stats_data[stats_data.columns[ii]] = group_stats

    #################### Insert p-values into the DataFrame ####################
    stats_data["p_value"] = ""
    for ii, variable_name in enumerate(variable_index):
        if variable_name in stats_dict["variable_full_name"].values:
            # If categorical, insert p-value in the following row
            ind: int = index1.index(variable_name) + 1
            stats_data.iat[ind, len(stage_list) + 1] = p_value_list[ii]
        else:
            # If continuous, insert p-value in the same row
            stats_data.at[variable_name, "p_value"] = p_value_list[ii]

    #################### Rename the index ####################
    stats_data.rename_axis(index=[""], inplace=True)

    return stats_data


def multiple_linear_regression(
    data: pd.DataFrame,
    list_of_predictors: list[str],
    dv: str,
    not_adjust: None | list[str] = None,
) -> pd.DataFrame:
    """
    Fit a multiple linear regression model with formula `response ~ Const + predictor + age + sex + bmi + cog + apoe4`
    for each predictor in `list_of_predictors`.
    Return a DataFrame with coefficients, p-values, and more.

    Parameters
    ----------
    data : pd.DataFrame
        input data
    list_of_predictors : list[str]
        list of predictors
    dv : str
        name of the response variable
    not_adjust : list[str], optional
        list of variables not to adjust for, by default None

    Returns
    -------
    pd.DataFrame
        regression results
    """
    cols: list[str] = [
        "predictor",
        "coef",
        "std_err",
        "conf_int_low",
        "conf_int_high",
        "rsq",
        "rsq_adj",
        "p_value",
        "normality",
    ]

    df: pd.DataFrame = pd.DataFrame(columns=cols)

    # Define variables to adjust for
    adjust_for: list[str] = ["age", "sex", "bmi", "cog", "apoe4"]
    if not_adjust is None:
        adjust_var_list: list[str] = adjust_for
    else:
        adjust_var_list: list[str] = [var for var in adjust_for if var not in not_adjust]

    # Iterate over each predictor
    for predictor in list_of_predictors:
        # Compute p-value from multiple predictor model
        x: pd.DataFrame = add_constant(data[[predictor] + adjust_var_list])
        model_multi: RegressionResultsWrapper = OLS(data[dv], exog=x.astype(float)).fit()
        coef, std_err, conf_int, p_value = (
            model_multi.params[predictor],
            model_multi.bse[predictor],
            model_multi.conf_int().loc[predictor].values,
            model_multi.pvalues[predictor],
        )

        # Verify normality of residuals using Shapiro-Wilk test
        normality: bool = shapiro(model_multi.resid)[1] > 0.05

        # Compute R-squared from single predictor model
        x: pd.DataFrame = add_constant(data[[predictor]])
        model_single: RegressionResultsWrapper = OLS(data[dv], exog=x, missing="drop").fit()

        # Append results to the DataFrame
        df.loc[len(df)] = [
            predictor,
            coef,
            std_err,
            conf_int[0],
            conf_int[1],
            model_single.rsquared,
            model_single.rsquared_adj,
            p_value,
            normality,
        ]

    return df


def get_linkage_methods() -> list[str]:
    """
    Get the list of linkage methods for hierarchical clustering.

    Returns
    -------
    list[str]
        linkage methods list
    """
    linkage_methods: list[str] = ["single", "complete", "average", "weighted", "centroid", "median", "ward"]
    return linkage_methods


def get_cluster_criteria() -> list[str]:
    """
    Get the list of cluster criteria for hierarchical clustering.

    Returns
    -------
    list[str]
        cluster criteria list
    """
    cluster_criteria: list[str] = ["inconsistent", "distance", "maxclust", "monocrit", "maxclust_monocrit"]
    return cluster_criteria


def cluster_corr_df(
    corr_df: pd.DataFrame,
    x: bool = True,
    y: bool = True,
    linkage_method: str = "single",
    cluster_criterion: str = "inconsistent",
) -> pd.DataFrame:
    """
    Cluster a correlation matrix using hierarchical clustering.

    Parameters
    ----------
    corr_df : pd.DataFrame
        correlation matrix
    x : bool, optional
        cluster columns, by default True
    y : bool, optional
        cluster rows, by default True
    linkage_method : str, optional
        linkage method, by default 'single'
    cluster_criterion : str, optional
        cluster criterion, by default 'inconsistent'

    Returns
    -------
    pd.DataFrame
        clustered correlation matrix
    """
    df: pd.DataFrame = corr_df.copy()
    if y:  # Cluster in the y direction (reorder rows)
        pairwise_dists_y: np.ndarray = pdist(df)
        linkage_y: np.ndarray = linkage(pairwise_dists_y, method=linkage_method)
        idy: np.ndarray = fcluster(linkage_y, pairwise_dists_y.max() / 10, criterion=cluster_criterion)
        df: pd.DataFrame = df.iloc[np.argsort(idy), :]
    if x:  # Cluster in the x direction (reorder columns)
        pairwise_dists_x: np.ndarray = pdist(df.transpose())
        linkage_x: np.ndarray = linkage(pairwise_dists_x, method=linkage_method)
        idx: np.ndarray = fcluster(linkage_x, pairwise_dists_x.max() / 10, criterion=cluster_criterion)
        df: pd.DataFrame = df.transpose().iloc[np.argsort(idx), :].transpose()
    return df


def remove_diagonal(corr_df: pd.DataFrame) -> pd.DataFrame:
    """
    Mask the diagonal of a square matrix with NaN.

    Parameters
    ----------
    corr_df : pd.DataFrame
        square matrix

    Returns
    -------
    pd.DataFrame
        square matrix with NaN diagonal
    """
    return corr_df.mask(np.identity(n=len(corr_df), dtype=bool))


def fill_mirror(corr_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill the lower/upper triangle of a correlation matrix with the upper/lower triangle.

    Parameters
    ----------
    corr_df : pd.DataFrame
        correlation matrix

    Returns
    -------
    pd.DataFrame
        filled correlation matrix
    """
    df_t: pd.DataFrame = corr_df.transpose().infer_objects().fillna(0)
    return remove_diagonal(corr_df.infer_objects().fillna(0) + df_t)


def mask_outlier(data: pd.DataFrame) -> np.ndarray:
    """
    Return a mask that indicates outliers with False using Local Outlier Factor.

    Parameters
    ----------
    data : pd.DataFrame
        input data

    Returns
    -------
    np.ndarray
        mask that indicates outliers with False
    """
    olf: LocalOutlierFactor = LocalOutlierFactor(n_neighbors=20)
    olf.fit(zscore(data, axis=0))
    return (olf.negative_outlier_factor_ > -1.96) & (olf.negative_outlier_factor_ < 1.96)


def corr_remove_outliers(data: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the correlation matrix after removing outliers.

    Parameters
    ----------
    data : pd.DataFrame
        input data

    Returns
    -------
    pd.DataFrame
        correlation matrix
    """
    # Initialize a DataFrame to store the correlation matrix
    corr_df: pd.DataFrame = pd.DataFrame(index=data.columns, columns=data.columns, dtype=float)

    # Iterate over each pair of variables
    for ind1, var1 in enumerate(data.columns):
        for ind2 in range(ind1):
            var2: str = data.columns[ind2]

            # Compute the correlation after removing outliers
            mask: np.ndarray = mask_outlier(data[[var1, var2]])
            corr_df.at[var1, var2] = data.loc[mask, [var1, var2]].corr(method="pearson").iat[0, 1]

    return fill_mirror(corr_df)
