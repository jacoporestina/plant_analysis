import os
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

# === CONFIG ===
data_files = [
    "data/plant_measurements.csv",
    "data/fruit_measurements.csv",
    "data/fruit_quality.csv",
    "data/firmness.csv",
    "data/plant_yield.csv",
]

output_base_dir = "statistics_output"
anova_dir = os.path.join(output_base_dir, "anova_results")
qqplot_dir = os.path.join(output_base_dir, "qq_plots")

os.makedirs(anova_dir, exist_ok=True)
os.makedirs(qqplot_dir, exist_ok=True)

# === PROCESS EACH FILE ===
for file_path in data_files:
    df = pd.read_csv(file_path)

    # Drop 'DAH' and 'DAT' columns if they exist
    df = df.drop(columns=[col for col in ['DAH', 'DAT'] if col in df.columns])

    # Prepare file identifier
    file_name = os.path.splitext(os.path.basename(file_path))[0]

    # Prepare date columns
    df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
    df['date_str'] = df['date'].dt.strftime('%Y-%m-%d')

    # Clean up any unnamed columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # Exclude metadata columns
    excluded_cols = ['date', 'date_str', 'treatment', 'tunnel', 'plant ID']
    variables = [col for col in df.columns if col not in excluded_cols]

    for variable in variables:
        # Output file setup
        safe_var = variable.replace(' ', '_').replace('/', '_')
        result_path = os.path.join(anova_dir, f"{file_name}__{safe_var}_anova.txt")

        with open(result_path, 'w') as summary_file:
            summary_file.write(f"ANOVA and Assumption Tests for {variable} ({file_name})\n\n")

            for date in sorted(df['date_str'].dropna().unique()):
                date_data = df[df['date_str'] == date]

                if date_data[variable].dropna().empty:
                    continue

                try:
                    model = ols(f'Q("{variable}") ~ C(treatment)', data=date_data).fit()
                    anova_table = sm.stats.anova_lm(model, typ=2)

                    ss = anova_table["sum_sq"]
                    df_ = anova_table["df"]
                    ms = ss / df_
                    f_val = anova_table["F"][0]
                    p_val = anova_table["PR(>F)"][0]

                    means = date_data.groupby("treatment")[variable].mean()

                    summary_file.write(f"Date: {date}\n")
                    summary_file.write("Means:\n")
                    for treatment, mean_val in means.items():
                        summary_file.write(f"  {treatment.capitalize()}: {mean_val:.3f}\n")
                    summary_file.write("\nANOVA Table:\n")
                    summary_file.write(f"{'Source':<15}{'DF':<6}{'SS':<12}{'MS':<12}{'F':<10}{'P-value':<10}\n")
                    summary_file.write(f"{'Treatment':<15}{int(df_[0]):<6}{ss[0]:<12.3f}{ms[0]:<12.3f}{f_val:<10.3f}{'<0.001' if p_val < 0.001 else round(p_val, 3):<10}\n")
                    summary_file.write(f"{'Residual':<15}{int(df_[1]):<6}{ss[1]:<12.3f}{ms[1]:<12.3f}\n")

                    # Assumption checks
                    control = date_data[date_data['treatment'] == 'control'][variable].dropna()
                    shaded = date_data[date_data['treatment'] == 'shaded'][variable].dropna()

                    if len(control) > 2 and len(shaded) > 2:
                        shapiro_control = stats.shapiro(control)
                        shapiro_shaded = stats.shapiro(shaded)
                        levene_test = stats.levene(control, shaded)

                        summary_file.write(f"\nShapiro-Wilk (Control): W={shapiro_control.statistic:.3f}, p={'<0.001' if shapiro_control.pvalue < 0.001 else round(shapiro_control.pvalue, 3)}\n")
                        summary_file.write(f"Shapiro-Wilk (Shaded): W={shapiro_shaded.statistic:.3f}, p={'<0.001' if shapiro_shaded.pvalue < 0.001 else round(shapiro_shaded.pvalue, 3)}\n")
                        summary_file.write(f"Levene's Test:         W={levene_test.statistic:.3f}, p={'<0.001' if levene_test.pvalue < 0.001 else round(levene_test.pvalue, 3)}\n")

                    summary_file.write("\n")

                    # QQ Plot
                    plot_folder = os.path.join(qqplot_dir, file_name, safe_var)
                    os.makedirs(plot_folder, exist_ok=True)
                    plt.figure()
                    stats.probplot(date_data[variable].dropna(), dist="norm", plot=plt)
                    plt.title(f"QQ Plot for {variable} on {date}")
                    plt.tight_layout()
                    plt.savefig(os.path.join(plot_folder, f"{date}_qq.png"))
                    plt.close()

                except Exception as e:
                    summary_file.write(f"Date: {date} - Error: {str(e)}\n\n")
