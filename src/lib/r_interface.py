"""This module provides functions to run R code in Python."""

import os
from itertools import combinations
import pandas as pd
import numpy as np

os.environ["R_HOME"] = "C:\\Program Files\\R\\R-4.5.1"
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter


def tukey(df: pd.DataFrame, dv: str, stage_list: list[str]) -> pd.DataFrame:
    """
    Conduct Tukey post-hoc for a dependent variable in R.

    Parameters
    ----------
    df : pd.DataFrame
        input data
    dv : str
        dependent variable
    stage_list : list[str]
        list of stages in order

    Returns
    -------
    pd.DataFrame
        Tukey post-hoc test results with columns: coef, lower, upper, se, p_adj
    """

    # Convert df: pd.DataFrame to R dataframe
    with localconverter(ro.default_converter + pandas2ri.converter):
        ro.globalenv["df"] = ro.conversion.py2rpy(df)

    # Convert Python lists to R vectors
    ro.globalenv["dv"] = dv
    ro.globalenv["stage_list"] = ro.StrVector(stage_list)

    # Create R formula string
    covars: list[str] = ["age", "bmi", "sex", "cog", "apoe4"]
    ro.globalenv["rhs"] = f" ~ stage + {' + '.join(covars)}"

    # Run R code
    ro.r(
        """
        # Load required libraries
        library(mvtnorm)
        library(survival)
        library(TH.data)
        library(MASS)
        library(multcomp)
        
        # Convert variables to factors
        df$stage <- factor(df$stage, levels = stage_list)
        df$sex <- as.factor(df$sex)
        df$cog <- as.factor(df$cog)
        df$apoe4 <- as.factor(df$apoe4)
        
        # Conduct Tukey post-hoc
        formula_str <- as.formula(paste(dv, rhs))
        ancova_model <- aov(formula_str, data = df)
        glht_object <- glht(ancova_model, linfct = mcp(stage = "Tukey"))
        
        # Get coefficients and confidence intervals from confint() on glht object
        glht_confint <- confint(glht_object)$confint
        
        # Get standard errors and p-values from summary() on glht object
        glht_summary <- summary(glht_object)$test
        glht_pvalues <- cbind(glht_summary$sigma, glht_summary$pvalues)
        colnames(glht_pvalues) <- c("Std. Error", "Pr(>|t|)")
        
        # Combine into a single dataframe
        r_output <- cbind(glht_confint, glht_pvalues)
        """
    )

    # Convert R output to np.ndarray
    r_arr: ro.vectors.FloatVector = ro.globalenv["r_output"]
    with localconverter(ro.default_converter + pandas2ri.converter):
        np_arr: np.ndarray = ro.conversion.rpy2py(r_arr)

    # Convert np.ndarray to pd.DataFrame
    df_return: pd.DataFrame = pd.DataFrame(
        np_arr,
        index=list(combinations(stage_list, 2)),
        columns=["coef", "lower", "upper", "std_err", "p_adj"],
    )

    return df_return


def tukey_multiple_dvs(
    df: pd.DataFrame, dvs: list[str], stage_list: list[str]
) -> pd.DataFrame:
    """
    Conduct Tukey post-hoc for multiple dependent variables in R. Output is a pd.DataFrame of Tukey adjusted
    p-values for each pair of stages for each dependent variable.

    Parameters
    ----------
    df : pd.DataFrame
        input data
    dvs : list[str]
        list of dependent variables
    stage_list : list[str]
        list of stages in order

    Returns
    -------
    pd.DataFrame
        dataframe of Tukey adjusted p-values for each pair of stages for each dependent variable
    """

    # Convert df: pd.DataFrame to R dataframe
    with localconverter(ro.default_converter + pandas2ri.converter):
        ro.globalenv["df"] = ro.conversion.py2rpy(df)

    # Convert Python lists to R vectors
    ro.globalenv["dvs"] = ro.StrVector(dvs)
    ro.globalenv["stage_list"] = ro.StrVector(stage_list)

    # Create R formula string
    covar: list[str] = ["age", "bmi", "sex", "cog", "apoe4"]
    ro.globalenv["rhs"] = f" ~ stage + {' + '.join(covar)}"

    # Run R code
    ro.r(
        """
        # Load required libraries
        library(mvtnorm)
        library(survival)
        library(TH.data)
        library(MASS)
        library(multcomp)
        
        # Convert variables to factors
        df$stage <- factor(df$stage, levels = stage_list)
        df$sex <- as.factor(df$sex)
        df$cog <- as.factor(df$cog)
        df$apoe4 <- as.factor(df$apoe4)
        
        # Initialize list to store p-values
        p_list <- list()

        # For each dependent variable, conduct Tukey post-hoc
        for (i in seq_along(dvs)) {
            formula_str <- as.formula(paste(dvs[i], rhs))
            ancova_model <- aov(formula_str, data = df)
            glht_object <- glht(ancova_model, linfct = mcp(stage = "Tukey"))
            
            # Store p-values in a list
            glht_summary <- summary(glht_object)
            glht_results <- glht_summary$test
            p_list[[i]] <- glht_results$pvalues
        }
        
        # Concatenate p-values into a single dataframe
        r_output <- do.call(rbind, p_list)
        """
    )

    # Convert R output to np.ndarray
    r_arr: ro.vectors.FloatVector = ro.globalenv["r_output"]
    with localconverter(ro.default_converter + pandas2ri.converter):
        np_arr: np.ndarray = ro.conversion.rpy2py(r_arr)

    # Convert np.ndarray to pd.DataFrame
    df_return: pd.DataFrame = pd.DataFrame(
        np_arr, columns=list(combinations(stage_list, 2)), index=dvs
    )

    return df_return
