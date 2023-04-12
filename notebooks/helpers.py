import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels
from scipy import stats
from statsmodels.compat import lzip
from statsmodels.graphics.regressionplots import plot_partregress_grid, plot_ccpr_grid, influence_plot, \
    plot_leverage_resid2, abline_plot
from statsmodels.tools.tools import maybe_unwrap_results
from statsmodels.graphics.gofplots import ProbPlot
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
from typing import Type
import statsmodels.stats.api as sms
import statsmodels.api as sm

style_talk = 'seaborn-talk'  # refer to plt.style.available


def import_data():
    data = pd.read_excel('../data/data.xlsx', index_col=False)
    # cmax = data['Cmax']
    # tmax = data['Tmax']
    data = data.drop(['Tmax', 'Cmax'], axis=1)
    return data


def import_raw_data():
    data = pd.read_excel('../data/raw_data.xlsx', index_col=False)
    return data


def radar_graph(true, pred):
    import plotly.graph_objects as go
    import plotly.io as pio

    pio.renderers.default = 'png'

    categories = ['PLGA_L_IN', 'PLGA_CS_IN', 'PLGA_L_IV', 'PLGA_CS_IV']
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=pred,
        theta=categories,
        fill='toself',
        name='Pred'
    ))
    fig.add_trace(go.Scatterpolar(
        r=true,
        theta=categories,
        fill='toself',
        name='True'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[min(min(true), min(pred)) - 1, max(max(true), max(pred)) + 1]
            )),
        showlegend=True
    )

    return fig


def predict(model):
    return model.predict(exog=pd.DataFrame(
        {'Weight': [252.268, 252.268, 252.268, 252.268], 'logP': [2.260, 2.260, 2.260, 2.260],
         'Solubility': [0.071, 0.071, 0.071, 0.071], 'Pgp': [1, 1, 1, 1],
         'DrugCarrierRatio': [0.480, 0.480, 0.480, 0.480], 'Position': [0, 0, 0, 0],
         'Comp1': [1.239028, 1.239028, 1.239028, 1.239028], 'Comp2': [1.116500, 1.106221, 1.116500, 1.106221],
         'Size': [170.630, 453.100, 170.630, 453.100], 'Zeta': [-37.700, 33.400, -37.700, 33.400],
         'Release': [2.26, 2.95, 2.26, 2.95], 'Route': [1, 1, 0, 0]}))


def predict_lmem(model):
    return model.predict(exog=pd.DataFrame(
        {'Weight': [252.268, 252.268, 252.268, 252.268], 'logP': [2.260, 2.260, 2.260, 2.260],
         'Solubility': [0.071, 0.071, 0.071, 0.071], 'Pgp': [1, 1, 1, 1],
         'DrugCarrierRatio': [0.480, 0.480, 0.480, 0.480], 'Position': [0, 0, 0, 0],
         'Comp1': [2.34761, 2.34761, 2.34761, 2.34761], 'Comp2': [2.383959, 2.286618, 2.383959, 2.286618],
         'Size': [170.630, 453.100, 170.630, 453.100], 'Zeta': [-37.700, 33.400, -37.700, 33.400],
         'Release': [2.26, 2.95, 2.26, 2.95], 'Route': [2.65484, 2.65484, 1.532931, 1.532931]}))


def lmem_diagnostic(lmemf, y_true, data):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import scipy.stats as stats

    # influence_plot(lmemf)

    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111)
    sns.scatterplot(x=lmemf.fittedvalues, y=y_true, ax=ax)
    line_fit = sm.OLS(y_true, sm.add_constant(lmemf.fittedvalues, prepend=True)).fit()
    abline_plot(model_results=line_fit, ax=ax)
    ax.set_title('Model Fit Plot')
    ax.set_ylabel('Observed values')
    ax.set_xlabel('Fitted values')
    plt.show()

    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111)
    residual_norm_abs_sqrt = np.sqrt(np.abs(lmemf.resid))
    ax.scatter(lmemf.fittedvalues, residual_norm_abs_sqrt, alpha=0.5);
    sns.regplot(
        x=lmemf.fittedvalues,
        y=residual_norm_abs_sqrt,
        scatter=False, ci=False,
        lowess=True,
        line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8},
        ax=ax)
    # annotations
    abs_sq_norm_resid = np.flip(np.argsort(residual_norm_abs_sqrt), 0)
    abs_sq_norm_resid_top_3 = abs_sq_norm_resid[:3]
    for i in abs_sq_norm_resid_top_3:
        ax.annotate(
            i,
            xy=(lmemf.fittedvalues[i], residual_norm_abs_sqrt[i]),
            color='C3')
    ax.set_title('Scale-Location', fontweight="bold")
    ax.set_xlabel('Fitted values')
    ax.set_ylabel(r'$\sqrt{|\mathrm{Standardized\ Residuals}|}$');
    plt.show()

    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111)
    sm.qqplot(lmemf.resid, dist=stats.norm, line='s', ax=ax)
    ax.set_title("Q-Q Plot")
    plt.show()

    fig = plt.figure(figsize=(16, 9))
    ax = sns.distplot(lmemf.resid, hist=False, kde_kws={"shade": True, "lw": 1}, fit=stats.norm)
    ax.set_title("KDE Plot of Model Residuals (Blue) and Normal Distribution (Black)")
    ax.set_xlabel("Residuals")
    plt.show()

    labels = ["Statistic", "p-value"]
    norm_res = stats.shapiro(lmemf.resid)
    for key, val in dict(zip(labels, norm_res)).items():
        print(key, val)

    fig = plt.figure(figsize=(16, 9))
    ax = sns.scatterplot(y=lmemf.resid, x=lmemf.fittedvalues)
    ax.set_title("RVF Plot")
    ax.set_xlabel("Fitted Values")
    ax.set_ylabel("Residuals")
    plt.show()

    fig = plt.figure(figsize=(16, 9))
    ax = sns.boxplot(x=lmemf.model.groups, y=lmemf.resid)
    ax.set_title("Distribution of Residuals for Weight by Litter")
    ax.set_ylabel("Residuals")
    ax.set_xlabel("AUC")
    plt.show()

    from statsmodels.stats.diagnostic import het_white
    try:
        het_white_res = het_white(lmemf.resid, lmemf.model.exog)
        labels = ["LM Statistic", "LM-Test p-value", "F-Statistic", "F-Test p-value"]
        for key, val in dict(zip(labels, het_white_res)).items():
            print(key, val)
    except:
        print("White's heterogeneity test failed")

    print(lmemf.aic)

    var_resid = lmemf.scale
    var_random_effect = float(lmemf.cov_re.iloc[0])
    var_fixed_effect = lmemf.predict(data).var()

    total_var = var_fixed_effect + var_random_effect + var_resid
    marginal_r2 = var_fixed_effect / total_var
    conditional_r2 = (var_fixed_effect + var_random_effect) / total_var
    print(f"marginal r2 = {marginal_r2}")
    print(f"conditional r2 = {conditional_r2}")


class LinearRegDiagnostic:
    """
    Diagnostic plots to identify potential problems in a linear regression fit.
    Mainly,
        a. non-linearity of data
        b. Correlation of error terms
        c. non-constant variance
        d. outliers
        e. high-leverage points
        f. collinearity

    Author:
        Prajwal Kafle (p33ajkafle@gmail.com, where 3 = r)
        Does not come with any sort of warranty.
        Please test the code one your end before using.
    """

    def __init__(self,
                 results: Type[statsmodels.regression.linear_model.RegressionResultsWrapper]) -> None:
        """
        For a linear regression model, generates following diagnostic plots:

        a. residual
        b. qq
        c. scale location and
        d. leverage

        and a table

        e. vif

        Args:
            results (Type[statsmodels.regression.linear_model.RegressionResultsWrapper]):
                must be instance of statsmodels.regression.linear_model object

        Raises:
            TypeError: if instance does not belong to above object

        Example:
        >>> import numpy as np
        >>> import pandas as pd
        >>> import statsmodels.formula.api as smf
        >>> x = np.linspace(-np.pi, np.pi, 100)
        >>> y = 3*x + 8 + np.random.normal(0,1, 100)
        >>> df = pd.DataFrame({'x':x, 'y':y})
        >>> res = smf.ols(formula= "y ~ x", data=df).fit()
        >>> cls = Linear_Reg_Diagnostic(res)
        >>> cls(plot_context="seaborn-paper")

        In case you do not need all plots you can also independently make an individual plot/table
        in following ways

        >>> cls = Linear_Reg_Diagnostic(res)
        >>> cls.residual_plot()
        >>> cls.qq_plot()
        >>> cls.scale_location_plot()
        >>> cls.leverage_plot()
        >>> cls.vif_table()
        """

        if isinstance(results, statsmodels.regression.linear_model.RegressionResultsWrapper) is False:
            raise TypeError(
                "result must be instance of statsmodels.regression.linear_model.RegressionResultsWrapper object")

        self.results = maybe_unwrap_results(results)

        self.y_true = self.results.model.endog
        self.y_predict = self.results.fittedvalues
        self.xvar = self.results.model.exog
        self.xvar_names = self.results.model.exog_names

        try:
            self.residual = np.array(self.results.resid)

        except:
            self.residual = np.array(self.results.resid_response)

        try:
            influence = self.results.get_influence()
            try:
                self.residual_norm = influence.resid_studentized_internal
            except:
                self.residual_norm = influence.resid_studentized
            self.leverage = influence.hat_matrix_diag
            self.cooks_distance = influence.cooks_distance[0]
        except:
            print("Failed to calculate influence")

        self.nparams = len(self.results.params)

    def __call__(self, plot_context='seaborn-paper'):
        # print(plt.style.available)
        name = ["Lagrange multiplier statistic", "p-value", "f-value", "f p-value"]
        test = sms.het_breuschpagan(self.residual, self.results.model.exog)
        print("\nHeteroscedasticity test 1 (Breusch-Pagan):\n", lzip(name, test))
        name = ["F statistic", "p-value"]
        test = sms.het_goldfeldquandt(self.residual, self.results.model.exog)
        lzip(name, test)
        print("\nHeteroscedasticity test 2 (Goldfeld-Quandt):\n", lzip(name, test))

        try:
            name = ["t value", "p value"]
            test = sms.linear_harvey_collier(self.results, skip=12)
            print("\nLinearity:\n", lzip(name, test))
        except:
            print("\nLinearity test (Harvey-Collier):\n", "Failed")
        # fig = plt.figure(figsize=(10, 20))
        # plot_partregress_grid(self.results, fig=fig)
        # fig = plt.figure(figsize=(10, 20))
        # plot_ccpr_grid(self.results, fig=fig)
        try:
            influence_plot(self.results)
        except:
            try:
                self.results.get_influence().plot_influence()
                plot_leverage_resid2(self.results)
            except:
                print("influence unavailable")

        with plt.style.context(plot_context):
            fig, ax = plt.subplots(nrows=4, ncols=2, figsize=(10, 20))
            try:
                self.residual_plot(ax=ax[0, 0])
                self.qq_plot(ax=ax[0, 1])
                self.scale_location_plot(ax=ax[1, 0])
                self.leverage_plot(ax=ax[1, 1])
            except:
                print("residual/qqplot/scale-location plot unavailable")

            self.model_fit(ax=ax[2, 0])
            self.residual_dependence_plot(ax=ax[2, 1])
            self.hist_std_deviance_res(ax=ax[3, 0])
            plt.show()

        print(self.vif_table())
        return fig, ax

    def model_fit(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        sns.scatterplot(x=self.y_predict, y=self.y_true, ax=ax)
        line_fit = sm.OLS(self.y_true, sm.add_constant(self.y_predict, prepend=True)).fit()
        abline_plot(model_results=line_fit, ax=ax)
        ax.set_title('Model Fit Plot')
        ax.set_ylabel('Observed values')
        ax.set_xlabel('Fitted values')
        return ax

    def residual_dependence_plot(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        sns.scatterplot(x=self.y_predict, y=self.results.resid_pearson, ax=ax)
        ax.hlines(0, 0, 1)
        ax.set_xlim(0, 1)
        ax.set_title('Residual Dependence Plot')
        ax.set_ylabel('Pearson Residuals')
        ax.set_xlabel('Fitted values')
        return ax

    def hist_std_deviance_res(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        try:
            resid = self.results.resid_deviance.copy()
            resid_std = stats.zscore(resid)
            sns.histplot(resid_std, bins=25, ax=ax)
        except:
            sns.histplot(ax=ax)
        ax.set_title('Histogram of standardized deviance residuals')
        return ax

    def residual_plot(self, ax=None):
        """
        Residual vs Fitted Plot

        Graphical tool to identify non-linearity.
        (Roughly) Horizontal red line is an indicator that the residual has a linear pattern
        """
        if ax is None:
            fig, ax = plt.subplots()

        sns.residplot(
            x=self.y_predict,
            y=self.residual,
            lowess=True,
            scatter_kws={'alpha': 0.5},
            line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8},
            ax=ax)

        # annotations
        residual_abs = np.abs(self.residual)
        abs_resid = np.flip(np.sort(residual_abs))
        abs_resid_top_3 = abs_resid[:3]
        for i, _ in enumerate(abs_resid_top_3):
            ax.annotate(
                i,
                xy=(self.y_predict[i], self.residual[i]),
                color='C3')

        ax.set_title('Residuals vs Fitted', fontweight="bold")
        ax.set_xlabel('Fitted values')
        ax.set_ylabel('Residuals')
        return ax

    def qq_plot(self, ax=None):
        """
        Standarized Residual vs Theoretical Quantile plot

        Used to visually check if residuals are normally distributed.
        Points spread along the diagonal line will suggest so.
        """
        if ax is None:
            fig, ax = plt.subplots()

        QQ = ProbPlot(self.residual_norm)
        QQ.qqplot(line='45', alpha=0.5, lw=1, ax=ax)

        # annotations
        abs_norm_resid = np.flip(np.argsort(np.abs(self.residual_norm)), 0)
        abs_norm_resid_top_3 = abs_norm_resid[:3]
        for r, i in enumerate(abs_norm_resid_top_3):
            ax.annotate(
                i,
                xy=(np.flip(QQ.theoretical_quantiles, 0)[r], self.residual_norm[i]),
                ha='right', color='C3')

        ax.set_title('Normal Q-Q', fontweight="bold")
        ax.set_xlabel('Theoretical Quantiles')
        ax.set_ylabel('Standardized Residuals')
        return ax

    def scale_location_plot(self, ax=None):
        """
        Sqrt(Standarized Residual) vs Fitted values plot

        Used to check homoscedasticity of the residuals.
        Horizontal line will suggest so.
        """
        if ax is None:
            fig, ax = plt.subplots()

        residual_norm_abs_sqrt = np.sqrt(np.abs(self.residual_norm))

        ax.scatter(self.y_predict, residual_norm_abs_sqrt, alpha=0.5);
        sns.regplot(
            x=self.y_predict,
            y=residual_norm_abs_sqrt,
            scatter=False, ci=False,
            lowess=True,
            line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8},
            ax=ax)

        # annotations
        abs_sq_norm_resid = np.flip(np.argsort(residual_norm_abs_sqrt), 0)
        abs_sq_norm_resid_top_3 = abs_sq_norm_resid[:3]
        for i in abs_sq_norm_resid_top_3:
            ax.annotate(
                i,
                xy=(self.y_predict[i], residual_norm_abs_sqrt[i]),
                color='C3')
        ax.set_title('Scale-Location', fontweight="bold")
        ax.set_xlabel('Fitted values')
        ax.set_ylabel(r'$\sqrt{|\mathrm{Standardized\ Residuals}|}$');
        return ax

    def leverage_plot(self, ax=None):
        """
        Residual vs Leverage plot

        Points falling outside Cook's distance curves are considered observation that can sway the fit
        aka are influential.
        Good to have none outside the curves.
        """
        if ax is None:
            fig, ax = plt.subplots()

        ax.scatter(
            self.leverage,
            self.residual_norm,
            alpha=0.5);

        sns.regplot(
            x=self.leverage,
            y=self.residual_norm,
            scatter=False,
            ci=False,
            lowess=True,
            line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8},
            ax=ax)

        # annotations
        leverage_top_3 = np.flip(np.argsort(self.cooks_distance), 0)[:3]
        for i in leverage_top_3:
            ax.annotate(
                i,
                xy=(self.leverage[i], self.residual_norm[i]),
                color='C3')

        xtemp, ytemp = self.__cooks_dist_line(0.5)  # 0.5 line
        ax.plot(xtemp, ytemp, label="Cook's distance", lw=1, ls='--', color='red')
        xtemp, ytemp = self.__cooks_dist_line(1)  # 1 line
        ax.plot(xtemp, ytemp, lw=1, ls='--', color='red')

        ax.set_xlim(0, max(self.leverage) + 0.01)
        ax.set_title('Residuals vs Leverage', fontweight="bold")
        ax.set_xlabel('Leverage')
        ax.set_ylabel('Standardized Residuals')
        ax.legend(loc='upper right')
        return ax

    def vif_table(self):
        """
        VIF table

        VIF, the variance inflation factor, is a measure of multicollinearity.
        VIF > 5 for a variable indicates that it is highly collinear with the
        other input variables.
        """
        vif_df = pd.DataFrame()
        vif_df["Features"] = self.xvar_names
        vif_df["VIF Factor"] = [variance_inflation_factor(self.xvar, i) for i in range(self.xvar.shape[1])]

        print(vif_df
              .sort_values("VIF Factor")
              .round(2))

    def __cooks_dist_line(self, factor):
        """
        Helper function for plotting Cook's distance curves
        """
        p = self.nparams
        formula = lambda x: np.sqrt((factor * p * (1 - x)) / x)
        x = np.linspace(0.001, max(self.leverage), 50)
        y = formula(x)
        return x, y
