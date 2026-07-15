# ML Midterm

You are allowed to use Scikit documentaion, and your work in this class assignments

🎓 ML Midterm Exercise (2 Hours)

"Heart Disease Prediction — ML Pipeline"

Dataset: Heart Disease UCI (303 rows, 14 features) — provided.

----------------------------------------------------------------------------------------------------------------------------------------

Part 1: Data Exploration & Visualization (25 min) — 20 pts

 1. Load dataset, display shape, dtypes, check for missing values
 2. Show descriptive statistics for numerical columns
 3. Create a correlation heatmap
 4. Plot distribution of target variable (disease vs. no disease)
 5. Which features appear most correlated with the target? Would you drop any features before modeling? Why? How might this impact your model?

----------------------------------------------------------------------------------------------------------------------------------------

Part 2: Linear Regression (20 min) — 15 pts

 6. Build a Linear Regression predicting max heart rate from age
 7. Plot the regression line over the scatter plot
 8. Report R² score
 9. Is this model overfitting or underfitting? What evidence supports your conclusion? What are the limitations of using only one feature?

----------------------------------------------------------------------------------------------------------------------------------------

Part 3: Classification (40 min) — 40 pts

 10. Split data 80/20 for predicting heart disease (target)
 11. Train a Logistic Regression classifier
 12. Print the confusion matrix
 13. Report Accuracy, Precision, Recall, F1-score
 14. Interpret your confusion matrix: how many patients would be told they're healthy but actually have heart disease? What's the real-world impact?
 15. In a medical diagnosis scenario, would you optimize for precision or recall? Justify your choice.

----------------------------------------------------------------------------------------------------------------------------------------

Part 4: Analysis & Reflection (15 min) — 25 pts

 16. Compare your regression R² with your classification performance (Accuracy and F1-score). Which task appears more predictable from this dataset, and what does this suggest about the underlying relationships in the data?
 17. If you had to add one more feature to improve your classifier, which would you pick from the correlation analysis (Part 1) and why?
 18. Name two things you would do differently if you had more time and a larger dataset.

----------------------------------------------------------------------------------------------------------------------------------------

🌟 Bonus: Clustering (+15 pts)

 19. Pick 3 numerical features, standardize them (StandardScaler)
 20. Plot the Elbow curve (inertia vs. k for k=1 to 8)
 21. Choose optimal k from the elbow, apply K-Means
 22. Visualize clusters on a 2D scatter plot (use any 2 features)
 23. Compare cluster assignments with the target variable. Do clusters align with disease presence?

----------------------------------------------------------------------------------------------------------------------------------------
