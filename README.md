
# Power Consumption in India (2019-20) Analytics

## Project Overview

This project, **"Power Consumption in India (2019-20) Analytics"**, involves analyzing electricity consumption data across different states in India for the period from January 2, 2019, to May 23, 2020. The dataset provides a comprehensive view of power consumption trends and examines the impact of the COVID-19 lockdown on energy consumption at the state and regional levels.

## Table of Contents

1. [Introduction](#introduction)
2. [Data Description](#data-description)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Data Analysis](#data-analysis)
6. [Acknowledgements](#acknowledgements)
7. [License](#license)

## Introduction

India is the world's third-largest producer and consumer of electricity, with a national electric grid having an installed capacity of 370.106 GW as of March 31, 2020. The dataset used in this project reflects energy consumption in Mega Units (MU) on a state-wise basis. The study aims to understand how the COVID-19 lockdown has impacted electricity consumption across various states.

## Data Description

- **Time Period**: January 2, 2019, to May 23, 2020
- **Format**: Time series data
- **Index**: Dates
- **Columns**: States of India
- **Data Points**: Power consumed in Mega Units (MU) by each state on each date

## Installation

To run this analysis, you need to have the following Python libraries installed:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`

You can install these libraries using pip:

```bash
pip install pandas numpy matplotlib seaborn
```

## Usage

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/yourusername/power-consumption-india.git
   cd power-consumption-india
   ```

2. **Load the Dataset**:

   The dataset is included in the repository under the `data` directory. You can load it using pandas:

   ```python
   import pandas as pd

   data = pd.read_csv('data/power_consumption_india.csv', index_col='Date', parse_dates=True)
   ```

3. **Run the Analysis**:

   Use the Jupyter notebooks or Python scripts in the `analysis` directory to perform exploratory data analysis (EDA), visualize trends, and analyze the impact of the COVID-19 lockdown.

## Data Analysis

In this project, the following steps were performed:

- **Exploratory Data Analysis (EDA)**: Utilized Pandas, NumPy, Matplotlib, and Seaborn for data cleaning, manipulation, and visualization.
- **Impact Analysis**: Studied the effects of the COVID-19 lockdown on energy consumption, focusing on changes in consumption patterns across states.

## Acknowledgements

- **Power System Operation Corporation Limited (POSOCO)**: Data was sourced from weekly energy reports provided by POSOCO, a government enterprise under the Ministry of Power.
- **Dataset**: Scraped from POSOCO’s reports, providing an exhaustive view of energy consumption data across states.

## License

This project is licensed under the [MIT License](LICENSE).

---

