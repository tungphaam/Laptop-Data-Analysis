"""
visualization.py

Contains visualization and exploratory data analysis (EDA) functions.
"""

import matplotlib.pyplot as plt
import seaborn as sns


def plot_price_distribution(df):
    """Plot the distribution of laptop prices."""
    plt.figure(figsize=(8, 5))
    sns.histplot(df["Price"], bins=40, kde=True)
    plt.title("Distribution of Laptop Prices")
    plt.xlabel("Price")
    plt.ylabel("Count")
    plt.show()


def plot_brand_vs_price(df):
    """Plot average price by laptop brand."""
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Brand", y="Price", data=df, estimator="mean", errorbar=None)
    plt.title("Average Laptop Price by Brand")
    plt.xticks(rotation=45)
    plt.show()


def plot_correlation_heatmap(df):
    """Plot correlation matrix of numerical features."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), cmap="coolwarm", annot=True, fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.show()

