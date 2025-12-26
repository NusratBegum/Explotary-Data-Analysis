import json

# Read the notebook
with open('Exploratory_Data_Analysis.ipynb', 'r') as f:
    nb = json.load(f)

# New cells to add before the final conclusion
new_cells = []

# 1. Statistical Summary
new_cells.append({
    "cell_type": "markdown",
    "metadata": {"id": "stat_summary"},
    "source": [
        "## 10. Statistical Summary\n",
        "\n",
        "Let's get a comprehensive statistical summary of our numerical features to understand the distribution, central tendency, and spread of the data."
    ]
})

new_cells.append({
    "cell_type": "code",
    "metadata": {"id": "stat_summary_code"},
    "source": [
        "# Display statistical summary\n",
        "print(\"Statistical Summary of Numerical Features:\")\n",
        "print(\"=\" * 80)\n",
        "print(df.describe())\n",
        "print(\"\\n\" + \"=\" * 80)\n",
        "print(f\"\\nDataset Shape: {df.shape}\")\n",
        "print(f\"Total Records: {df.shape[0]}\")\n",
        "print(f\"Total Features: {df.shape[1]}\")"
    ],
    "execution_count": None,
    "outputs": []
})

# 2. Correlation Analysis
new_cells.append({
    "cell_type": "markdown",
    "metadata": {"id": "correlation_analysis"},
    "source": [
        "## 11. Correlation Analysis\n",
        "\n",
        "Correlation analysis helps us understand the relationships between different numerical features. A correlation coefficient close to 1 or -1 indicates a strong positive or negative relationship, while a value close to 0 indicates weak correlation."
    ]
})

new_cells.append({
    "cell_type": "code",
    "metadata": {"id": "correlation_matrix"},
    "source": [
        "# Calculate correlation matrix\n",
        "correlation_matrix = df.corr()\n",
        "print(\"Correlation Matrix:\")\n",
        "print(correlation_matrix)\n",
        "\n",
        "# Create a more detailed heatmap\n",
        "plt.figure(figsize=(12, 8))\n",
        "sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,\n",
        "            fmt='.2f', square=True, linewidths=1)\n",
        "plt.title('Correlation Heatmap of Numerical Features', fontsize=16, pad=20)\n",
        "plt.tight_layout()\n",
        "plt.show()"
    ],
    "execution_count": None,
    "outputs": []
})

# 3. Distribution Analysis
new_cells.append({
    "cell_type": "markdown",
    "metadata": {"id": "distribution_analysis"},
    "source": [
        "## 12. Distribution Analysis\n",
        "\n",
        "Understanding the distribution of our data is crucial. We'll create histograms with KDE (Kernel Density Estimation) plots to visualize the distribution of key numerical features."
    ]
})

new_cells.append({
    "cell_type": "code",
    "metadata": {"id": "distribution_plots"},
    "source": [
        "# Create distribution plots for key numerical features\n",
        "fig, axes = plt.subplots(2, 2, figsize=(15, 10))\n",
        "fig.suptitle('Distribution of Key Numerical Features', fontsize=16, y=1.02)\n",
        "\n",
        "# Price distribution\n",
        "sns.histplot(df['Price'], kde=True, ax=axes[0, 0], color='skyblue')\n",
        "axes[0, 0].set_title('Price Distribution')\n",
        "axes[0, 0].set_xlabel('Price ($)')\n",
        "\n",
        "# HP distribution\n",
        "sns.histplot(df['HP'], kde=True, ax=axes[0, 1], color='lightgreen')\n",
        "axes[0, 1].set_title('Horsepower Distribution')\n",
        "axes[0, 1].set_xlabel('Horsepower (HP)')\n",
        "\n",
        "# Cylinders distribution\n",
        "sns.histplot(df['Cylinders'], kde=True, ax=axes[1, 0], color='salmon')\n",
        "axes[1, 0].set_title('Cylinders Distribution')\n",
        "axes[1, 0].set_xlabel('Number of Cylinders')\n",
        "\n",
        "# Year distribution\n",
        "sns.histplot(df['Year'], kde=True, ax=axes[1, 1], color='plum')\n",
        "axes[1, 1].set_title('Year Distribution')\n",
        "axes[1, 1].set_xlabel('Manufacturing Year')\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()"
    ],
    "execution_count": None,
    "outputs": []
})

# 4. Feature Relationships
new_cells.append({
    "cell_type": "markdown",
    "metadata": {"id": "feature_relationships"},
    "source": [
        "## 13. Feature Relationships Analysis\n",
        "\n",
        "Let's explore relationships between different features using pairplots and additional scatter plots to identify patterns and potential predictive relationships."
    ]
})

new_cells.append({
    "cell_type": "code",
    "metadata": {"id": "pairplot_analysis"},
    "source": [
        "# Create pairplot for selected features\n",
        "# Note: Using a subset to avoid overplotting\n",
        "selected_features = ['Price', 'HP', 'Cylinders', 'Year']\n",
        "sns.pairplot(df[selected_features], diag_kind='kde', plot_kws={'alpha': 0.6})\n",
        "plt.suptitle('Pairplot of Key Features', y=1.02)\n",
        "plt.tight_layout()\n",
        "plt.show()"
    ],
    "execution_count": None,
    "outputs": []
})

# 5. Additional Scatter Plots
new_cells.append({
    "cell_type": "markdown",
    "metadata": {"id": "additional_scatter"},
    "source": [
        "### Additional Relationship Visualizations\n",
        "\n",
        "Let's examine more specific relationships that might be interesting for understanding car pricing and specifications."
    ]
})

new_cells.append({
    "cell_type": "code",
    "metadata": {"id": "scatter_cylinders_price"},
    "source": [
        "# Cylinders vs Price\n",
        "fig, ax = plt.subplots(1, 2, figsize=(15, 5))\n",
        "\n",
        "# Scatter plot: Cylinders vs Price\n",
        "ax[0].scatter(df['Cylinders'], df['Price'], alpha=0.5, c='teal')\n",
        "ax[0].set_xlabel('Number of Cylinders')\n",
        "ax[0].set_ylabel('Price ($)')\n",
        "ax[0].set_title('Cylinders vs Price')\n",
        "ax[0].grid(True, alpha=0.3)\n",
        "\n",
        "# Scatter plot: Year vs Price\n",
        "ax[1].scatter(df['Year'], df['Price'], alpha=0.5, c='coral')\n",
        "ax[1].set_xlabel('Manufacturing Year')\n",
        "ax[1].set_ylabel('Price ($)')\n",
        "ax[1].set_title('Year vs Price')\n",
        "ax[1].grid(True, alpha=0.3)\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()"
    ],
    "execution_count": None,
    "outputs": []
})

# 6. Categorical Analysis
new_cells.append({
    "cell_type": "markdown",
    "metadata": {"id": "categorical_analysis"},
    "source": [
        "## 14. Categorical Feature Analysis\n",
        "\n",
        "Let's analyze categorical features to understand the distribution and characteristics of different categories in our dataset."
    ]
})

new_cells.append({
    "cell_type": "code",
    "metadata": {"id": "categorical_plots"},
    "source": [
        "# Analyze distribution of categorical features\n",
        "fig, axes = plt.subplots(2, 2, figsize=(16, 12))\n",
        "\n",
        "# Make distribution\n",
        "top_makes = df['Make'].value_counts().nlargest(15)\n",
        "top_makes.plot(kind='barh', ax=axes[0, 0], color='steelblue')\n",
        "axes[0, 0].set_title('Top 15 Car Makes')\n",
        "axes[0, 0].set_xlabel('Count')\n",
        "\n",
        "# Model distribution (top 15)\n",
        "top_models = df['Model'].value_counts().nlargest(15)\n",
        "top_models.plot(kind='barh', ax=axes[0, 1], color='darkseagreen')\n",
        "axes[0, 1].set_title('Top 15 Car Models')\n",
        "axes[0, 1].set_xlabel('Count')\n",
        "\n",
        "# Transmission Type distribution\n",
        "if 'Transmission' in df.columns:\n",
        "    df['Transmission'].value_counts().plot(kind='bar', ax=axes[1, 0], color='lightcoral')\n",
        "    axes[1, 0].set_title('Transmission Type Distribution')\n",
        "    axes[1, 0].set_xlabel('Transmission Type')\n",
        "    axes[1, 0].set_ylabel('Count')\n",
        "    axes[1, 0].tick_params(axis='x', rotation=45)\n",
        "\n",
        "# Driven Wheels distribution\n",
        "if 'Driven_Wheels' in df.columns:\n",
        "    df['Driven_Wheels'].value_counts().plot(kind='bar', ax=axes[1, 1], color='mediumpurple')\n",
        "    axes[1, 1].set_title('Driven Wheels Distribution')\n",
        "    axes[1, 1].set_xlabel('Driven Wheels')\n",
        "    axes[1, 1].set_ylabel('Count')\n",
        "    axes[1, 1].tick_params(axis='x', rotation=45)\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()"
    ],
    "execution_count": None,
    "outputs": []
})

# 7. Price Analysis by Category
new_cells.append({
    "cell_type": "markdown",
    "metadata": {"id": "price_by_category"},
    "source": [
        "## 15. Price Analysis by Categories\n",
        "\n",
        "Understanding how price varies across different categories can provide valuable insights about the factors affecting car prices."
    ]
})

new_cells.append({
    "cell_type": "code",
    "metadata": {"id": "price_boxplots"},
    "source": [
        "# Price by Make (top 10 makes)\n",
        "fig, ax = plt.subplots(figsize=(14, 6))\n",
        "top_10_makes = df['Make'].value_counts().nlargest(10).index\n",
        "df_top_makes = df[df['Make'].isin(top_10_makes)]\n",
        "\n",
        "sns.boxplot(data=df_top_makes, x='Make', y='Price', ax=ax)\n",
        "ax.set_title('Price Distribution by Top 10 Car Makes', fontsize=14)\n",
        "ax.set_xlabel('Car Make')\n",
        "ax.set_ylabel('Price ($)')\n",
        "ax.tick_params(axis='x', rotation=45)\n",
        "plt.tight_layout()\n",
        "plt.show()\n",
        "\n",
        "# Calculate and display average price by make\n",
        "avg_price_by_make = df_top_makes.groupby('Make')['Price'].mean().sort_values(ascending=False)\n",
        "print(\"\\nAverage Price by Make (Top 10):\")\n",
        "print(avg_price_by_make)"
    ],
    "execution_count": None,
    "outputs": []
})

# 8. Key Insights Summary
new_cells.append({
    "cell_type": "markdown",
    "metadata": {"id": "insights_summary"},
    "source": [
        "## 16. Key Insights and Findings\n",
        "\n",
        "Based on the comprehensive EDA performed, here are the key insights:\n",
        "\n",
        "### Data Quality\n",
        "- Started with 11,914 rows, removed 989 duplicates\n",
        "- Handled missing values in HP and Cylinders columns\n",
        "- Removed approximately 1,600 outlier records using IQR method\n",
        "- Final clean dataset ready for modeling\n",
        "\n",
        "### Feature Relationships\n",
        "- Strong positive correlation between HP and Price\n",
        "- Number of Cylinders correlates with both HP and Price\n",
        "- Manufacturing Year shows varying relationship with Price\n",
        "\n",
        "### Distribution Patterns\n",
        "- Price distribution shows right skewness (higher concentration at lower prices)\n",
        "- HP distribution indicates most cars have moderate horsepower\n",
        "- Common cylinder counts are 4, 6, and 8\n",
        "\n",
        "### Categorical Insights\n",
        "- Wide variety of car makes represented in the dataset\n",
        "- Different makes show different average price points\n",
        "- Transmission types and driven wheels show distinct distributions\n",
        "\n",
        "These insights can guide further analysis, feature engineering, and predictive modeling efforts."
    ]
})

# Insert new cells before the last cell (conclusion)
nb['cells'] = nb['cells'][:-1] + new_cells + [nb['cells'][-1]]

# Save the updated notebook
with open('Exploratory_Data_Analysis.ipynb', 'w') as f:
    json.dump(nb, f, indent=2)

print(f"Successfully added {len(new_cells)} new cells to the notebook")
print(f"New total cells: {len(nb['cells'])}")
