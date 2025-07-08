import pandas as pd
import numpy as np
import os
from scipy.stats import f_oneway, shapiro, bartlett
import matplotlib.pyplot as plt
import seaborn as sns

# ========== Configuration ==========

input_configs = {
    'data/plant_measurements.csv': [
        'crown no.', 'leaf no.', 'inflorescence no.', 'flower no.',
        'fruit no.', 'petiole length mean (cm)', 'plant_FW (g)', 'plant_DW (g)',
        'shoots_DW (g)', 'leaf_DW (g)', 'leaf_area (cm2)', 'DW area-1',
    ],
    'data/fruit_quality.csv': ['acidity (meq 100gr -1)', 'brix (Bx%)', 'Bx acidity-1'],
    'data/fruit_measurements.csv': ['fruit_weight(g)', 'oidio (g)'],
    'data/firmness.csv': ['firmness (kg)'],
    'data/plant_yield.csv': ['yield_plant (g)'],
    'data/data per crown.csv': ['leaf no.','inflorescence no.',
        'fruit no.', 'plant_FW (g)', 'shoots_DW (g)', 'leaf_DW (g)', 'leaf_area (cm2)', 'DW area-1'],
    'data/cumulative_yield.csv': ['plant_yield (g)']
}

output_dir = 'anova_results'
os.makedirs(output_dir, exist_ok=True)

boxplot_dir = os.path.join(output_dir, 'boxplots')
os.makedirs(boxplot_dir, exist_ok=True)


# ========== Functions ==========

def run_anova(df, variable, date):
    date_df = df[df['date'] == date]
    values_all = date_df[variable].dropna()

    if len(values_all) < 3:
        return None, None

    grouped = date_df.groupby('treatment')[variable]
    groups_data = [group.dropna().values for _, group in grouped if len(group.dropna()) > 0]
    if len(groups_data) < 2:
        return None, None

    group_stats = []
    for name, values in grouped:
        vals = values.dropna().values
        if len(vals) > 0:
            group_stats.append({
                'treatment': name,
                'mean': np.mean(vals),
                'std': np.std(vals, ddof=1),
                'n': len(vals)
            })

    # Shapiro-Wilk test on overall values
    shapiro_p = shapiro(values_all)[1]
    normality_check = "Pass" if shapiro_p > 0.05 else "Fail"

    # Bartlett test for equal variances
    bartlett_p = bartlett(*groups_data)[1]
    bartlett_check = "Pass" if bartlett_p > 0.05 else "Fail"

    # ANOVA: sum of squares and degrees of freedom
    grand_mean = np.mean(values_all)
    ss_treatment = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups_data)
    ss_residuals = sum(((g - np.mean(g)) ** 2).sum() for g in groups_data)
    df_treatment = len(groups_data) - 1
    df_residuals = len(values_all) - len(groups_data)

    ms_treatment = ss_treatment / df_treatment if df_treatment > 0 else np.nan
    ms_residuals = ss_residuals / df_residuals if df_residuals > 0 else np.nan
    ss_total = ss_treatment + ss_residuals

    f_stat, p_value = f_oneway(*groups_data)

    anova_result = {
        'date': date,
        'Variable': variable,
        'Groups': len(groups_data),
        'F': f_stat,
        'p-value': p_value,
        'SS_treatment': ss_treatment,
        'DF_treatment': df_treatment,
        'MS_treatment': ms_treatment,
        'SS_residuals': ss_residuals,
        'DF_residuals': df_residuals,
        'MS_residuals': ms_residuals,
        'SS_total': ss_total,
        'Normality (Shapiro)': normality_check,
        'Shapiro p-value': shapiro_p,
        'Equal Variance (Bartlett)': bartlett_check,
        'Bartlett p-value': bartlett_p
    }

    return anova_result, group_stats

def save_results_to_txt(filename, variable, results):
    """Save ANOVA and assumptions results to .txt."""
    with open(filename, 'w') as f:
        f.write(f"ANOVA Results for: {variable}\n")
        f.write("=" * 80 + "\n\n")
        for result, stats in results:
            if result is None:
                continue

            f.write(f"Date: {result['date']}\n")
            f.write("-" * 80 + "\n")

            # Summary statistics
            f.write("Summary Statistics (Mean ± Std Dev):\n")
            f.write("{:<15} {:>10} {:>10} {:>6}\n".format("Treatment", "Mean", "Std Dev", "N"))
            for stat in stats:
                f.write("{:<15} {:>10.4f} {:>10.4f} {:>6}\n".format(
                    stat['treatment'], stat['mean'], stat['std'], stat['n']))
            f.write("\n")

            # ANOVA table
            f.write("ANOVA Table:\n")
            f.write("{:<12} {:>12} {:>12} {:>12} {:>12} {:>12}\n".format(
                "Source", "SS", "DF", "MS", "F", "p-value"))
            f.write("{:<12} {:>12.4f} {:>12} {:>12.4f} {:>12.4f} {:>12.4f}\n".format(
                "Treatment",
                result['SS_treatment'], result['DF_treatment'],
                result['MS_treatment'], result['F'], result['p-value']))
            f.write("{:<12} {:>12.4f} {:>12} {:>12.4f}\n".format(
                "Residuals",
                result['SS_residuals'], result['DF_residuals'], result['MS_residuals']))
            f.write("{:<12} {:>12.4f}\n".format(
                "Total", result['SS_total']))
            f.write("\n")

            # Assumption tests
            f.write("Assumption Checks:\n")
            f.write(f"  Normality (Shapiro-Wilk): {result['Normality (Shapiro)']} (p = {result['Shapiro p-value']:.4f})\n")
            f.write(f"  Equal Variance (Bartlett): {result['Equal Variance (Bartlett)']} (p = {result['Bartlett p-value']:.4f})\n")
            f.write("=" * 80 + "\n\n")


def create_boxplot(df, variable, file_prefix):
    # Filter to valid dates where at least 1 non-NaN exists
    valid_dates = df.groupby('date')[variable].apply(lambda x: not x.isna().all())
    valid_dates = valid_dates[valid_dates].index.tolist()
    filtered_df = df[df['date'].isin(valid_dates)]

    if len(valid_dates) == 0:
        print(f"  Skipping boxplot for {variable}: No valid data (all NaN).")
        return

    plt.figure(figsize=(12, 6))
    sns.boxplot(
        x="date",
        y=variable,
        hue="treatment",
        data=filtered_df,
        palette="viridis"
    )
    plt.title(f'Boxplot of {variable} by Treatment')
    plt.xlabel("Date")
    plt.ylabel(variable)
    plt.legend(title="Treatment", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    safe_var = variable.replace(' ', '_').replace('/', '_')
    plot_path = os.path.join(boxplot_dir, f"{file_prefix}_{safe_var}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"  Saved boxplot: {plot_path}")


# ========== Main Loop ==========

for file_path, variable_list in input_configs.items():
    df = pd.read_csv(file_path)
    print(f"\nProcessing file: {file_path}")
    print("Available columns:", df.columns.tolist())

    for variable in variable_list:
        print(f"  Analyzing variable: {variable}")
        variable_results = []
        valid_dates = df[df[variable].notna()]['date'].unique()

        for date in valid_dates:
            result, stats = run_anova(df, variable, date)
            if result:
                variable_results.append((result, stats))

        if variable_results:
            safe_var = variable.replace(' ', '_').replace('/', '_')
            filename = os.path.join(
                output_dir,
                f"{os.path.basename(file_path).replace('.csv','')}_{safe_var}.txt"
            )
            save_results_to_txt(filename, variable, variable_results)

            # Generate boxplot
            base_name = os.path.basename(file_path).replace('.csv', '')
            create_boxplot(df, variable, base_name)
