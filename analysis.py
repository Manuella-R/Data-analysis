import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

# Load dataset
# Replace 'file_path' with your actual dataset path
data = pd.read_csv('ai_job_market_insights.csv')
data = data.dropna() 
df = pd.DataFrame(data)

# Data Overview
print(df.head())
print(df.describe())
print(df.info())

# Convert relevant columns to categorical if not already
df['Automation_Risk'] = pd.Categorical(df['Automation_Risk'], categories=['Low', 'Medium', 'High'], ordered=True)
df['Company_Size'] = pd.Categorical(df['Company_Size'], categories=['Small', 'Medium', 'Large'], ordered=True)
df['AI_Adoption_Level'] = pd.Categorical(df['AI_Adoption_Level'], categories=['Low', 'Medium', 'High'], ordered=True)

# EDA: Pivot Table
pivot_table = pd.crosstab(index=df['Automation_Risk'], columns=df['Company_Size'], values=df['AI_Adoption_Level'], aggfunc='count', normalize='columns')
print(pivot_table)

# Visualization: Stacked Bar Chart
pivot_table.plot(kind='bar', stacked=True, figsize=(10, 6), color=['skyblue', 'lightgreen', 'lightcoral'])
plt.title('AI Adoption Levels by Automation Risk and Company Size')
plt.xlabel('Automation Risk')
plt.ylabel('Proportion of AI Adoption Levels')
plt.xticks(rotation=0)
plt.legend(title="Company Size")
plt.show()

# Heatmap
sns.heatmap(pivot_table, annot=True, cmap='Blues', cbar_kws={'label': 'Proportion'})
plt.title('AI Adoption Heatmap by Automation Risk and Company Size')
plt.xlabel('Company Size')
plt.ylabel('Automation Risk')
plt.show()

# Chi-Square Test
contingency_table = pd.crosstab(df['Automation_Risk'], df['AI_Adoption_Level'])
chi2, p, dof, expected = chi2_contingency(contingency_table)
print(f"Chi-Square Test Results: Chi2 = {chi2}, p-value = {p}")

# Insights and Recommendations
if p < 0.05:
    print("There is a significant association between Automation Risk and AI Adoption Levels.")
else:
    print("No significant association between Automation Risk and AI Adoption Levels.")
