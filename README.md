# ALL-Image-Classification
CNN Modelling using modified VGGNet for ALL blood smear classification

This project was conducted as part of a 3rd year degree module in artificial intelligence. The resultant model and writeup are not currently shared. Data used were
https://www.kaggle.com/datasets/mehradaria/leukemia and similar results were achieved with a much simpler model (F1 0.94 & 0.97 in x-val and test set vs 0.99  in x-val for the
original authors). Feel free to take the code to learn from, modify, and continue my work in accordance with the license conditions.

Note: There is a known, minor, data leakage risk in the cross validation implementation due to how pre-processing was implemented. It is possible for the internal validation fold to influence the distribution of the internal training folds as the data are globally normalised outside of the cross-validation routine. This was a choice made to save gpu time and memory due to extremely limited computational resources. It is mitigated by the use of separate validation and test folds.
