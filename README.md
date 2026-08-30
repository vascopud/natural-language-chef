# Natural Language Chef Classification

A simple NLP classification project using a provided recipe dataset to predict the chef associated with each recipe.

The project explores TF-IDF representations over recipe descriptions and tags, compares LinearSVC and Logistic Regression, tests word and character-level features and evaluates models using stratified 5-fold cross-validation. The best model combines word and character-level TF-IDF features with LinearSVC, achieving approximately 91.9% mean cross-validation accuracy.