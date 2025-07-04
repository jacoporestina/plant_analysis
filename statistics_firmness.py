import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway, shapiro, levene
import os
import numpy as np

# Load the dataset
file_path = 'data/firmness.csv'

df = pd.read_csv(file_path)

# Let's look at the first few rows to understand the structure
print(df.head())

# Check column names
print(df.columns)

# Check unique treatments
print(df['treatment'].unique())

# Create a directory for results if it doesn't exist
os.makedirs('anova_results', exist_ok=True)

# Specify the variables to analyze
variables = ['firmness (kg)']


def run_anova(df, variable, DAH):
    """Run ANOVA with assumption checks and return formatted results."""
    DAH_df = df[df['DAH'] == DAH]
    groups_data = []
    group_stats = []

    for name, group in DAH_df.groupby('treatment'):
        values = group[variable].dropna().values
        groups_data.append(values)
        if len(values) > 0:
            group_stats.append({
                'treatment': name,
                'mean': np.mean(values),
                'std': np.std(values, ddof=1),  # Sample standard deviation
                'n': len(values)
            })

    if len(groups_data) < 2:
        return None, None

    # Normality check (Shapiro-Wilk)
    normality = [shapiro(g)[1] for g in groups_data]
    normality_check = all(p > 0.05 for p in normality)

    # Equal variance check (Levene)
    levene_p = levene(*groups_data)[1] if len(groups_data) > 1 else 1.0

    # ANOVA
    f_stat, p_value = f_oneway(*groups_data)

    anova_result = {
        'DAH': DAH,
        'Variable': variable,
        'Groups': len(groups_data),
        'F': f"{f_stat:.4f}",
        'p-value': f"{p_value:.4f}",
        'Normality (Shapiro)': "Pass" if normality_check else "Fail",
        'Normality p-values': ", ".join([f"{p:.4f}" for p in normality]),
        'Equal Variance (Levene)': "Pass" if levene_p > 0.05 else "Fail",
        'Levene p-value': f"{levene_p:.4f}"
    }

    return anova_result, group_stats

def save_as_txt(variable, results):
    """Save ANOVA results with summary statistics to a formatted .txt file."""
    filename = f"anova_results/{variable.replace(' ', '_').replace('/', '_')}.txt"
    with open(filename, 'w') as f:
        f.write(f"ANOVA Results for: {variable}\n")
        f.write("="*50 + "\n\n")

        for result, stats in results:
            if result is None:
                continue

            f.write(f"DAH: {result['DAH']}\n")
            f.write("-"*50 + "\n")

            # Write summary statistics
            f.write("Summary Statistics:\n")
            f.write("{:<15} {:<10} {:<10} {:<10}\n".format(
                "Treatment", "Mean", "Std Dev", "N"))
            for stat in stats:
                f.write("{:<15} {:<10.4f} {:<10.4f} {:<10}\n".format(
                    stat['treatment'], stat['mean'], stat['std'], stat['n']))
            f.write("\n")

            # Write ANOVA results
            f.write("ANOVA Results:\n")
            f.write(f"Groups (treatments): {result['Groups']}\n")
            f.write(f"F-statistic: {result['F']}\n")
            f.write(f"p-value: {result['p-value']}\n")
            f.write(f"Normality Check (Shapiro-Wilk): {result['Normality (Shapiro)']}\n")
            f.write(f"Normality p-values: {result['Normality p-values']}\n")
            f.write(f"Equal Variance Check (Levene): {result['Equal Variance (Levene)']}\n")
            f.write(f"Levene p-value: {result['Levene p-value']}\n\n")
            f.write("="*50 + "\n\n")


for variable in variables:
    variable_results = []
    valid_dates = df[df[variable].notna()]['DAH'].unique()

    for date in valid_dates:
        anova_result, group_stats = run_anova(df, variable, date)
        if anova_result:
            variable_results.append((anova_result, group_stats))

    if variable_results:
        save_as_txt(variable, variable_results)



'''Create boxplots for the variables'''
# Plot to check distribution of a variable
for variable in variables:
    # Filter the DataFrame to exclude dates where ALL values are NaN for this variable
    valid_dates = df.groupby('DAH')[variable].apply(lambda x: not x.isna().all())
    valid_dates = valid_dates[valid_dates].index.tolist()  # Get dates with at least 1 non-NaN
    filtered_df = df[df['DAH'].isin(valid_dates)]

    if len(valid_dates) == 0:
        print(f"Skipping {variable}: No valid data (all NaN).")
        continue  # Skip to next variable if no dates remain

    plt.figure(figsize=(12, 6))
    sns.boxplot(
        x="DAH",
        y=variable,
        hue="treatment",
        data=filtered_df,
        palette="viridis"  # Optional: Use a color palette
    )
    plt.title(f'Boxplot of {variable} by Treatment')
    plt.tight_layout()
    plt.savefig(f'boxplots/boxplot_{variable}.png')
    plt.close()
