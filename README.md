# Semiconductor Stock Return Prediction

Multi-Horizon Machine Learning

Prediction Horizons: 5, 10, and 21 Trading Days

Models: Linear Regression, Ridge, Random Forest, XGBoost, LightGBM, and CatBoost 
 
Authors: Stella (Yanan) Zhang, David Cho, Melody (Yilin) Liu

Github link: https://github.com/cipher9/Semiconductor-Stock-Return-Prediction-ML-Project.git

## Installation and Setup Instructions 
First run semiconductor_dataset_cleaning_and_preprocessing_with_5_10_21_day_targets.ipynb to get preprocessed data then run combined_semiconductor_models_5_10_21_day_targets.ipynb for modeling.

1. Open the notebook in Google Colab. 
2. Mount Google Drive and confirm PROJECT_DIR points to /content/drive/MyDrive/Semiconductor-Stock-Return-Prediction-ML-Project. 
3. Install required libraries when prompted by the notebook, including scikit-learn, xgboost, lightgbm, and catboost. 
4. Confirm that processed_data/semiconductor_model_data_43_features_cleaned.csv exists. 
5. Run the notebook from top to bottom. The notebook will create horizon-specific output folders and save result CSVs, plots, predictions, model rankings, coefficients, and feature-importance files. 

Input/output directory paths should be edited if this repository is cloned/forked.
 
## Introduction 
Semiconductor companies are central to artificial intelligence, cloud infrastructure, consumer electronics, automobiles, and industrial automation. Their stock prices often react strongly to macroeconomic conditions, product cycles, capital spending, and investor expectations about AI and data-center demand. This makes semiconductor return prediction an important but difficult machine learning problem, where even small improvements over a simple baseline can be meaningful.
The primary task is to predict future percentage returns for 12 U.S.-listed semiconductor companies: ADI, AMD, AVGO, INTC, MCHP, MPWR, MRVL, MU, NVDA, ON, QCOM, and TXN. The final Jupyter notebook evaluates three horizons: 5 trading days, 10 trading days, and 21 trading days. This allows us to compare whether the same features are more useful for short-term or monthly return prediction.
The main challenges are data alignment, volatility, and leakage prevention. Prices and technical indicators are daily, fundamentals are quarterly, and macroeconomic variables may have different reporting frequencies, so all features must be aligned using only information available on the prediction date. Returns are also noisy and contain outliers, and overlapping future-return labels can create leakage. We address these issues with date-based splitting, horizon-specific cross-validation gaps, and an untouched chronological holdout set.


## Methods 
Our proposed method is a pooled, time-aware, multi-horizon regression framework that compares linear and nonlinear models under identical chronological evaluation rules. The strongest final model is Random Forest because it achieves the lowest holdout RMSE for all three horizons and slightly improves over the training-mean baseline. We believe this method is better than the other models tried because Random Forest captures nonlinear relationships and feature interactions while remaining relatively robust to noisy tabular data. Linear and Ridge models are interpretable but too restrictive for this problem; boosting models were competitive but did not outperform Random Forest on the final holdout sets. 


We tested six regression models: Linear Regression, Ridge Regression, Random Forest, XGBoost, LightGBM, and CatBoost. Linear Regression was used as the simplest interpretable benchmark, modeling future return as a weighted sum of the input features. Ridge Regression adds L2 regularization, which helps stabilize coefficients when features such as moving averages, exponential moving averages, and price variables are highly correlated.

## Results 
The final holdout results show that Random Forest is the best model for all three horizons. It slightly beats the baseline by RMSE on the 5-day, 10-day, and 21-day targets. The improvement is small but consistent with RMSE of 0.000078 for 5-day returns, 0.000176 for 10-day returns, and 0.000380 for 21-day returns, indicating that the Random Forest learned a weak but measurable signal beyond the average-return benchmark. 

The 5-day target is the easiest to predict in absolute error scale, with RMSE of 0.078128 and MAE of 0.056483 in Random Forest. The 10-day target has higher error, with RMSE of 0.113710 and MAE of 0.082099. The 21-day target is hardest in absolute terms, with RMSE of 0.181443 and MAE of 0.126192. This pattern is expected as longer horizons produce higher volatility. 

The R-squared remains close to zero, even for the best models. The 5-day Random Forest has 0.000175, while the 10-day and 21-day Random Forest models have -0.000317 and -0.003745, respectively, indicating that the models are only slightly better than mean prediction under squared-error evaluation. This is also consistent with the difficulty of stock return prediction. 

Directional accuracy is also modest. For all three horizons, the holdout positive return rate is about 54%, with the Random Forest directional accuracy being almost identical to that rate. This indicates the Random Forest mainly improves the magnitude prediction slightly instead of producing a strong directional trading signal. 

The comparison among models is informative. Linear Regression is consistently the weakest model because the relationship between the engineered features and future returns is not well represented by a simple linear function. Ridge improves the linear model by reducing coefficient instability, but it still performs worse than tree-based models. XGBoost, LightGBM, and CatBoost are competitive in some horizons but do not beat Random Forest on final holdout RMSE. This collectively suggests that a moderately regularized bagged-tree model was better suited to this dataset than the boosting configurations. 

The plots demonstrate key notebook functionality. Figure 1 compares Random Forest with the baseline across horizons. Figure 2 shows the 21-day model ranking by holdout RMSE. Figure 3 shows Random Forest feature importance for the 21-day target. The most important Random Forest features include Treasury yield, 20-day volatility, copper, VIX, moving averages, ATR, and lagged returns, indicating that the model uses a mix of macroeconomic, volatility, and technical signals. 

## Discussion
The strongest insight is that disciplined evaluation matters more than model complexity. Several advanced models, including XGBoost, LightGBM, and CatBoost, were tested, but none of them outperformed Random Forest on the final holdout set. The improvement over the training-mean baseline is extremely small, suggesting that most of the predictable component of these returns is weak. 


Random Forest likely worked best because it balances flexibility and regularization. It can capture nonlinear relationships without assuming a fixed linear form, but averaging many trees prevents any single tree from dominating. In earlier single-horizon experiments, shallower trees and larger leaf sizes performed better than deeper trees, which supports the idea that financial returns are noisy and that aggressive model flexibility can overfit. 


The feature-importance results suggest that similar financial datasets may benefit from combining market, macroeconomic, volatility, and technical features. The best Random Forest model does not rely on one feature category alone. Instead, important variables include Treasury yields, VIX, copper, volatility, ATR, and lagged returns. Similar designs could be useful for other sector-level stock universes, especially where firms share common macro and industry drivers. 


Our investigation also shows why target selection matters. The 5-day horizon has the lowest absolute error because short-horizon returns have lower variance, but the 21-day horizon gives more economically meaningful monthly predictions. The R-squared remains close to zero across all horizons, so the best horizon is not obvious from it alone. A business user might prefer the 5-day target for lower error or the 21-day target for portfolio rebalancing relevance. 
What made the solution successful was not a large R-squared, but the implementation of a fair comparison. Since every model used identical features, date partitions, folds, and baseline and outputs aggregate metrics, per-ticker metrics, predictions, residuals, coefficients, feature importance, and saved result files, this makes the work reproducible and helps avoid unsupported claims. 


The main challenge was avoiding leakage in a dataset with mixed reporting frequencies and overlapping forward-return targets. Our study addresses most of this by using filing-date-aware fundamentals, chronological splitting, date-based cross-validation, and horizon-specific validation gaps. One future improvement is to add an embargo before the final holdout period so that the last training labels cannot overlap with prices inside the holdout period. 


A second challenge was that the models improved only slightly over the baseline. This could happen because the current features do not contain enough stable predictive signal for future return magnitude. Future work could add earnings surprises, options-implied volatility, supply-chain information, and sector-level order data. These sources may contain forward-looking information not captured by prices and simple fundamentals. 


We could also test a monthly observation design. The current dataset uses daily rows with 5-, 10-, and 21-day forward returns. For the 21-day target, adjacent observations have overlapping future windows, which increases dependence among rows. A monthly dataset with one observation per ticker per month would be smaller but could align more naturally with monthly return prediction. 
