# Exploratory Data Analysis

A comprehensive guide to Exploratory Data Analysis (EDA) in Python, demonstrating key techniques and visualization methods using a cars dataset.

## Overview

This repository contains a detailed Jupyter notebook that walks through the process of Exploratory Data Analysis (EDA) - the crucial step of understanding datasets by summarizing their main characteristics and visualizing them effectively.

## What is Exploratory Data Analysis?

Exploratory Data Analysis (EDA) is the process of understanding datasets by:
- Summarizing main characteristics of the data
- Creating visual representations through plots and charts
- Identifying patterns, trends, and anomalies
- Formulating hypotheses and questions about the data
- Preparing data for machine learning models

## Dataset

This project uses a **Cars Dataset** from Kaggle, which provides a rich collection of automobile specifications and characteristics perfect for demonstrating EDA techniques.

Dataset source: [Kaggle Cars Dataset](https://www.kaggle.com/CooperUnion/cardataset)

## Technologies & Libraries

The notebook utilizes essential Python libraries for data analysis and visualization:
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Matplotlib** - Data visualization
- **Seaborn** - Statistical data visualization

## EDA Work Completed

This notebook demonstrates a complete EDA workflow including:

### 1. Data Loading and Initial Exploration
- Loading the cars dataset using pandas
- Displaying sample rows with head() and tail()
- Understanding data structure and dimensions

### 2. Data Type Analysis
- Checking data types for each column
- Identifying numerical vs. categorical features
- Ensuring proper data type assignments

### 3. Data Cleaning
- **Column Selection**: Removing irrelevant columns (Engine Fuel Type, Market Category, Vehicle Style, Popularity, Number of Doors, Vehicle Size)
- **Column Renaming**: Simplifying column names for better readability (e.g., "Engine HP" to "HP", "Engine Cylinders" to "Cylinders")
- **Duplicate Removal**: Identifying and removing 989 duplicate rows from 11,914 total rows
- **Missing Value Treatment**: Detecting and dropping rows with null values in Cylinders and HP columns

### 4. Outlier Detection and Treatment
- Using box plots to visualize outliers in Price, HP, and Cylinders
- Applying IQR (Interquartile Range) method to identify outliers
- Removing approximately 1,600 outlier rows while preserving data integrity

### 5. Data Visualization
- **Bar Charts**: Analyzing the distribution of car makes (top 40 manufacturers)
- **Heat Maps**: Visualizing correlations between numerical features
- **Scatter Plots**: Examining relationships between Horsepower (HP) and Price

### 6. Statistical Analysis
- Computing quantiles (Q1, Q3) for outlier detection
- Calculating IQR for numerical features
- Counting and analyzing data distribution across features

## Key Insights from Analysis

The EDA revealed:
- Dataset initially contained 11,914 rows with 989 duplicates
- After cleaning: 10,925 unique records
- Outliers present in Price, HP, and Cylinders features
- Strong representation of certain car manufacturers in the dataset
- Relationships between vehicle specifications (HP, Cylinders) and Price

## Notebook Contents

The complete EDA workflow includes:

1. **Importing Required Libraries** - Setting up pandas, numpy, matplotlib, and seaborn
2. **Loading the Dataset** - Reading CSV data into pandas DataFrame
3. **Data Type Analysis** - Checking and validating column data types
4. **Data Cleaning** - Dropping irrelevant columns and renaming for clarity
5. **Duplicate Detection and Removal** - Identifying and removing 989 duplicate entries
6. **Missing Value Treatment** - Handling null values in Cylinders and HP columns
7. **Outlier Detection** - Using box plots and IQR method to identify outliers
8. **Outlier Treatment** - Removing approximately 1,600 outlier records
9. **Data Visualization** - Creating bar charts, heat maps, and scatter plots

## How to Use

### Option 1: Google Colab (Recommended)
Click the "Open in Colab" badge at the top of the notebook to run it directly in your browser without any setup.

### Option 2: Local Environment
1. Clone this repository:
   ```bash
   git clone https://github.com/NusratBegum/Explotary-Data-Analysis.git
   cd Explotary-Data-Analysis
   ```

2. Install required dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn jupyter
   ```

3. Launch Jupyter Notebook:
   ```bash
   jupyter notebook Exploratory_Data_Analysis.ipynb
   ```

## Repository Structure

```
.
├── Exploratory_Data_Analysis.ipynb    # Main EDA tutorial notebook
├── course/                             # Additional course materials
│   └── videos/                        # Video tutorials
│       └── save_load_dataset.ipynb   # Dataset handling guide
└── README.md                          # This file
```

## Learning Outcomes

After going through this notebook, you will be able to:
- Understand the importance of EDA in the data science workflow
- Load and inspect datasets effectively
- Clean data by handling duplicates, missing values, and outliers
- Perform statistical analysis using quantiles and IQR
- Create meaningful visualizations to understand data patterns
- Identify data quality issues and anomalies
- Apply systematic EDA methodology to any dataset
- Prepare data for further analysis or machine learning

## Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page or submit a pull request.

## License

This project is open source and available for educational purposes.

## Author

**NusratBegum**

---

If you find this repository helpful, please consider giving it a star!