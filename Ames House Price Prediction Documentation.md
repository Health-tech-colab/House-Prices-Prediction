# AUTHOR: BRUNO NWAGBO

# TITLE: AMES HOUSING 


# Abstract

This project aimed to develop a predictive model for estimating house prices in Ames, Iowa, by employing various machine learning techniques. The study focused on understanding the key features influencing property values, including construction quality (`OverallQual`), living area size (`GrLivArea`), and neighbourhood (`Neighborhood`). After extensive data preprocessing and exploratory data analysis, a Lasso regression model with polynomial features was identified as the most effective for prediction, balancing accuracy with the need to avoid overfitting.

The model's performance was evaluated across several iterations, each refining the approach by scaling features, handling outliers, and incorporating non-linear relationships. The final model demonstrated a strong predictive capability, with a mean absolute error (MAE) of 14,756.95 and an R² score of 0.85, indicating its robustness in capturing the underlying patterns in the dataset.

While the study confirmed the initial hypothesis that higher-quality construction, larger living spaces, and desirable locations significantly impact house prices, it also highlighted limitations such as the small dataset size and its geographical focus. The findings offer valuable insights for real estate professionals, homeowners, and policymakers, suggesting that a data-driven approach can significantly enhance property valuation practices. Future work could expand on these results by applying the model to different regions and considering additional variables to improve accuracy


# Table of Contents
```
1. Introduction
2. Literature Review
3. Methodology
4. Results
5. Discussion
6. Conclusion
7. References
8. Appendices
```

# Introduction

## Background

The real estate market is a complex and dynamic industry where numerous factors influence the prices of properties. Accurate prediction of house prices is crucial for various stakeholders, including buyers, sellers, real estate agents, and policymakers. In the age of big data, machine learning techniques offer robust tools to analyse large datasets and uncover hidden patterns that can significantly enhance the accuracy of house price predictions. This project leverages advanced regression techniques to develop a predictive model for house prices, utilising a rich dataset from Kaggle.

## Objective

The primary objective of this project is to develop a machine learning model that can predict house prices based on various features related to the properties. By analysing and understanding the factors that most significantly affect house prices, the model aims to provide accurate and reliable predictions that can assist stakeholders in making informed decisions.

## Problem Statement

Predicting house prices is a challenging task due to the multifaceted nature of real estate valuation. Factors such as the quality of construction, location, size, age of the property, and additional amenities all play a role in determining the final sale price. Traditional methods of property valuation often fall short in capturing the intricate relationships between these variables. This project seeks to address this problem by developing a comprehensive predictive model that incorporates a wide range of features to improve the accuracy of house price predictions.

## Scope

This project will focus on the "House Prices - Advanced Regression Techniques" dataset from Kaggle, which includes detailed information on various house-related features. The scope includes:

* Data cleaning and preparation to handle missing values and outliers.
* Exploratory Data Analysis (EDA) to understand the distribution of features and their relationships with house prices.
* Feature engineering to create new variables and preprocess existing ones.
* Model training using multiple machine learning algorithms and evaluation of their performance.
* Interpretation of model results to identify the most significant predictors of house prices.

Limitations of this project include the size of the data available to train the model. Also reliance on the provided dataset, which may not capture all possible factors influencing house prices, and the inherent assumptions and constraints of the chosen machine learning algorithms.

## Hypothesis

After researching and speaking with professionals, these hypotheses will be tested in this project:


1. **H1**: Houses with higher overall quality and condition ratings will have higher sale prices.
2. **H2**: Greater above-grade living area will positively influence house prices.
3. **H3**: Location within more desirable neighbourhoods will result in higher house prices.
4. **H4**: Newer houses, or those with recent renovations, will be priced higher than older properties.
5. **H5**: Additional features such as more bathrooms, bedrooms, and higher kitchen quality will significantly increase house prices.
6. **H6**: Better exterior quality and condition, along with a solid foundation, will contribute to higher house prices.
7. **H7**: The presence and size of a garage, as well as a paved driveway, will positively affect house prices.

# Methodology

This project aimed to develop a predictive model for house prices using a comprehensive data science pipeline. The methodology encompassed a series of systematic steps, including data collection, preparation, exploratory data analysis (EDA), handling outliers, and the application of machine learning techniques, culminating in a model that was both robust and interpretable.

## Importing Libraries and Data Collection

The project commenced with the importation of essential libraries such as `pandas`, `numpy`, `scikit-learn`, and `plotly`. These libraries provided the necessary tools for data manipulation, visualisation, and model building. The dataset used in this project was collected from a housing database, which contained various features related to residential properties, such as their physical characteristics, location, and condition. The target variable was `SalePrice`, representing the price of the houses.

## Data Preparation

The raw data underwent a thorough cleaning process to address missing values, outliers, and inconsistencies. This step was critical to ensuring that the dataset was ready for analysis and model training. Missing values in categorical variables, such as `BsmtCond` (Basement Condition) and `GarageCond` (Garage Condition), were imputed with appropriate placeholders like "No Basement" or "No Garage." For numerical features, missing values were replaced with zeros where relevant, such as in `GarageArea` and `TotalBsmtSF`.

Categorical features were encoded to transform them into numerical formats suitable for machine learning algorithms. Ordinal features, which have a meaningful order, were encoded using custom ordinal encoding schemes. For instance, `KitchenQual` (Kitchen Quality) was encoded based on the increasing order of quality, from "Fa" (Fair) to "Ex" (Excellent). Nominal features, like `Neighborhood` and `Foundation`, were one-hot encoded to create binary columns, each representing a distinct category.

## Exploratory Data Analysis (EDA)

Exploratory Data Analysis was conducted to uncover patterns, relationships, and insights within the dataset. This phase included visualisations and statistical summaries to understand the distribution of features and their relationships with the target variable, `SalePrice`.

* **House Age**: The age of the house, determined by `YearBuilt` and `YearRemodAdd`, was analysed to assess its impact on price. It was observed that newer houses and those that had undergone recent renovations generally commanded higher prices.
* **Location**: The influence of location on house prices was examined through features like `Neighborhood`, `Street`, and `PavedDrive`. Houses in certain neighborhoods, such as `Northridge` and `Stone Brook`, were found to have significantly higher prices, indicating the importance of location in property valuation.
* **House Condition and Quality**: The overall condition (`OverallCond`) and quality (`OverallQual`) of the houses were analysed. As expected, houses in better condition and with higher-quality finishes tended to have higher prices. This relationship was visualised using box plots and scatter plots, showing clear trends.
* **House Size**: The size of the house, represented by features like `GrLivArea` (Above Ground Living Area), `BedroomAbvGr` (Number of Bedrooms), and the number of bathrooms (`FullBath`, `HalfBath`), was another critical factor in determining house prices. Larger houses with more bedrooms and bathrooms generally had higher prices.
* **Kitchen and Exterior Quality**: The quality of the kitchen (`KitchenQual`) and the exterior materials (`ExterQual`, `ExterCond`) were also significant predictors of house prices. High-quality kitchens and well-maintained exteriors contributed positively to house value.
* **Basement and Garage**: The basement area (`TotalBsmtSF`) and the condition and size of the garage (`GarageCond`, `GarageArea`) were analysed. Larger and better-maintained basements and garages were associated with higher property prices.

The EDA revealed key features that influenced house prices, guiding the subsequent steps in the modelling process. These insights were essential for feature selection and engineering.

## Handling Outliers

Outliers in the dataset were identified and handled to prevent them from skewing the model's predictions. Box plots and z-score analysis were employed to detect outliers, particularly in the `SalePrice` and `GrLivArea` features. These outliers were either removed or treated depending on their impact on the overall dataset. This step was crucial in improving the model's robustness and generalisability.

## Machine Learning

### Model Preprocessing

Before training the models, the dataset was preprocessed to ensure compatibility with machine learning algorithms. This preprocessing included scaling numerical features using the `RobustScaler`, which is less sensitive to outliers compared to other scaling methods. The dataset was then split into training and test sets to evaluate the model's performance on unseen data.

### Model Training

Multiple iterations of model training were conducted to find the most accurate and reliable predictive model:

* **Iteration 1**: The baseline model was developed using the raw, unscaled features. This model served as a reference point for evaluating the impact of various preprocessing steps.
* **Iteration 2**: In this iteration, numerical features were scaled using `RobustScaler` to improve model performance. Scaling helped standardise the range of features, reducing bias in the model's predictions.
* **Iteration 3**: A square root transformation was applied to the target variable, `SalePrice`, to address its skewed distribution. This transformation helped in normalising the target variable, leading to more accurate predictions.
* **Iteration 4**: Both feature scaling and the square root transformation of `SalePrice` were applied in this iteration. The combined effect of these preprocessing steps led to significant improvements in model performance.
* **Iteration 5**: Outliers were removed from the dataset to assess their impact on the model's accuracy. This iteration used unscaled features to compare the results with previous iterations.
* **Iteration 6**: The final iteration involved removing outliers and applying feature scaling. This model, which combined all the best practices from previous iterations, was chosen for further tuning.

### Model Tuning

The final model was fine-tuned to optimise its hyperparameters. Given the overfitting issues encountered with tree-based models, the Lasso model was selected for its ability to perform well on small datasets by penalising overly complex models. Polynomial features were also generated to capture non-linear relationships identified during the EDA. The tuned model demonstrated improved generalisability and accuracy.

### Model Testing and Saving

The tuned model was tested on the test set to evaluate its performance. Metrics such as Root Mean Squared Error (RMSE) and R-squared were used to assess the model's accuracy. After confirming its robustness, the final model was saved for future use. This step ensured that the model could be easily deployed for predicting house prices on new data.

# Results

The results section summarises the outcomes of each phase of the project, including data exploration, model training, and the final model’s performance.

## Data Exploration Results

During the exploratory data analysis (EDA), several key insights were derived:

* **House Age and Renovation**: Houses that were either newly built or recently renovated (`YearRemodAdd`) showed a significant positive correlation with `SalePrice`. Newer homes and those with recent improvements tended to fetch higher prices in the market.
* **Location**: The analysis of `Neighborhood` revealed that location played a crucial role in determining house prices. For example, properties in neighborhoods like `Northridge` and `Stone Brook` were notably more expensive compared to those in areas like `Edwards` and `BrkSide`.
* **Condition and Quality**: Houses with higher overall quality (`OverallQual`) and better overall condition (`OverallCond`) were associated with higher sale prices. The influence of these factors was visually confirmed through box plots and scatter plots, which showed a clear upward trend in prices with increasing quality and condition ratings.
* **House Size**: The size of the house, particularly the `GrLivArea` (Above Ground Living Area) and the number of bathrooms (`FullBath`), emerged as strong predictors of `SalePrice`. Larger living areas and more bathrooms were directly linked to higher house prices.
* **Kitchen and Exterior Quality**: Features like `KitchenQual` (Kitchen Quality) and `ExterQual` (Exterior Material Quality) were found to be significant determinants of house prices. Houses with high-quality kitchens and well-maintained exteriors commanded premium prices.
* **Basement and Garage**: The analysis indicated that a larger `TotalBsmtSF` (Total Basement Square Feet) and a well-maintained garage (`GarageCond`) were positively correlated with `SalePrice`. Houses with better basement and garage conditions were valued higher.

These insights guided the selection of features for model training, ensuring that the most relevant variables were used to predict house prices.

## Model Training Results

The project involved several iterations of model training, each building on the previous one to enhance performance. The key results from these iterations are as follows:

* **Iteration 1 (Baseline Model)**: The baseline model, which used raw, unscaled features, provided a starting point for evaluating the effectiveness of subsequent preprocessing steps. The model achieved a basic level of accuracy but was hindered by the variability in feature scales.
* **Iteration 2 (Scaled Features)**: Scaling the numerical features with `RobustScaler` improved the model's performance by standardising the range of inputs. This iteration reduced bias in predictions, particularly for features with a broad range of values.
* **Iteration 3 (Sqrt Transformed Target)**: Applying a square root transformation to `SalePrice` helped normalise the target variable’s distribution. This transformation improved the model's ability to make more accurate predictions, especially for houses with extremely high or low prices.
* **Iteration 4 (Scaled Features and Sqrt Transformed Target)**: Combining scaled features with the transformed target led to significant improvements in the model’s accuracy. This iteration outperformed the previous models, indicating the value of addressing both feature scaling and target normalisation.
* **Iteration 5 (No Outliers and Unscaled Features)**: Removing outliers from the dataset reduced the model’s tendency to overfit. However, using unscaled features in this iteration highlighted the importance of scaling, as the model's performance was not as strong as when scaling was applied.
* **Iteration 6 (No Outliers and Scaled Features)**: The final iteration, which combined outlier removal with feature scaling, produced the best results. The model demonstrated strong predictive power and generalised well to unseen data, making it the optimal choice for further tuning and testing.

## Model Tuning Results

Given the overfitting observed with tree-based models, the Lasso regression model was selected for its simplicity and effectiveness on smaller datasets. The final model, enhanced with polynomial features to capture non-linear relationships, showed excellent performance:

* **Mean Absolute Error (MAE)**: The MAE of the final model on the test set was low, indicating that the model's predictions were close to the actual sale prices. This metric confirmed the model's accuracy and reliability.
* **R-squared**: The final model achieved a high R-squared value, reflecting its ability to explain a significant portion of the variance in house prices. This result indicated that the selected features and preprocessing steps effectively captured the key drivers of house prices.

## Final Model and Predictions

The final predictive model, after tuning and testing, demonstrated robust performance. It successfully predicted house prices with a high degree of accuracy, making it a valuable tool for real estate analysis. The inclusion of polynomial features, combined with the Lasso regression model, allowed the model to account for complex interactions between features, further enhancing its predictive capabilities.

# Discussion

The discussion section delves into the interpretation of the results, explores the broader implications of the findings, and acknowledges the limitations of the study.

## Interpretation of Results
The results from the modelling process reveal several important insights into the factors that drive house prices:

1. **Feature Importance**:
   * **Overall Quality (**`OverallQual`) and **Living Area (**`GrLivArea`) were consistently strong predictors of `SalePrice`. This indicates that buyers place significant value on the overall build quality and size of the living space when determining the value of a home.
   * **Location** (captured through `Neighborhood`) was another crucial factor, underscoring the classic real estate principle of "location, location, location." Properties in more desirable neighbourhoods like `Northridge Heights` and `Stone Brook` commanded higher prices, reflecting the premium associated with these areas.
   * **House Condition (**`OverallCond`) and **Year of Remodel (**`YearRemodAdd`) also played significant roles, suggesting that both the upkeep of the property and recent improvements positively influence buyer perceptions and, consequently, the market value.
2. **Effect of Preprocessing**:
   * **Scaling**: Applying scaling to the numerical features improved model performance by reducing the bias introduced by varying feature ranges. This was especially evident when comparing the baseline model with iterations that included scaling.
   * **Target Transformation**: The square root transformation of the `SalePrice` target variable effectively normalised the distribution, making the model more robust, particularly for predicting prices at the extremes (very high or very low).
   * **Handling Outliers**: The removal of outliers led to a more generalised model, reducing the risk of overfitting and improving the model’s ability to predict prices for new, unseen data.
3. **Model Selection**:
   * The decision to use Lasso regression, coupled with polynomial feature expansion, proved to be effective given the dataset's size and characteristics. Lasso’s ability to perform feature selection by penalising less important variables helped create a more parsimonious and interpretable model without sacrificing predictive power.

## Implications

The findings from this study have several implications:

1. **For Real Estate Professionals**:
   * The strong predictive power of features such as `OverallQual`, `GrLivArea`, and `Neighborhood` suggests that real estate agents and appraisers should focus on these attributes when assessing property values. Emphasising these factors in marketing strategies could also help attract potential buyers.
2. **For Homeowners and Buyers**:
   * Homeowners considering renovations or improvements should prioritise upgrades that enhance the overall quality of the property, such as kitchen and exterior improvements, to maximise the return on investment. The significant influence of `YearRemodAdd` on price highlights the value of modernising homes.
   * Buyers should be aware of how much location impacts price, even within a single city. Investing in homes within more sought-after neighbourhoods could offer better long-term appreciation potential.
3. **For Future Research**:
   * The methodology and findings can serve as a reference for future studies on house price prediction. Researchers might explore how integrating additional variables, such as economic indicators or more granular neighbourhood data, could further improve predictive accuracy.

## Limitations

Despite the strengths of the study, several limitations should be acknowledged:

1. **Small Dataset**:
   * The relatively small size of the dataset posed challenges, particularly in preventing overfitting. Although techniques like Lasso regression and outlier removal were employed to mitigate these issues, the results might differ with a larger and more diverse dataset.
2. **Limited Geographic Scope**:
   * The dataset was restricted to properties within Ames, Iowa. While this provided a focused analysis, it limits the generalisability of the findings to other regions or markets with different economic conditions and housing market dynamics.
3. **Feature Representation**:
   * Although the model included a comprehensive set of features, some aspects of house pricing, such as proximity to amenities, school quality, or crime rates, were not explicitly captured. These factors could play a significant role in real-world pricing decisions and should be considered in future models.
4. **Model Complexity**:
   * The final model, which included polynomial features, increased in complexity, potentially making it harder to interpret and apply in a real-world setting. While this complexity improved predictive accuracy, it could be a trade-off when deploying the model in practice.

# Conclusion

This study set out to develop a predictive model for house prices in Ames, Iowa, using various machine learning techniques. The analysis confirmed the critical importance of factors such as `OverallQual`, `GrLivArea`, and `Neighborhood` in determining property values. These findings align with the initial hypotheses that higher construction quality, larger living areas, and desirable locations lead to higher house prices.

The Lasso regression model with polynomial features emerged as the most effective in predicting house prices, showcasing the value of advanced modelling techniques in real estate valuation. Data preprocessing, including feature scaling and outlier handling, was essential in improving model performance and ensuring accurate predictions.

While the results are promising, the study's limitations, such as the small dataset size and the focus on a specific geographic area, should be considered. Future research could extend this work to other regions and explore additional factors that may influence house prices.

In summary, the findings provide a solid foundation for real estate valuation, offering valuable insights for professionals and homeowners. However, ongoing analysis and model refinement are necessary to adapt to the ever-evolving real estate market.

# Appendices

## Appendix A: Code Snippets
To view the codes written for this project, click [here!](https://colab.research.google.com/drive/1po6bBfvNqXjXi_6A7AzKmpCUR_CG4_Qh?usp=sharing)

## Appendix B: Dataset
The dataset used for this project is the Kaggle House Prices - Advanced Regression Techniques (Ames Housing Dataset). To download the dataset, click [here!](https://github.com/Health-tech-colab/House-Prices-Prediction/tree/main/House%20Price%20Data)