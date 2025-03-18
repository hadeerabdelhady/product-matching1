Product Matching Model Overview : This model matches product names from a seller's dataset to a master product list using TF-IDF and cosine similarity. It assigns SKUs based on the highest similarity score and uses a Random Forest model to evaluate the confidence of the match.

to run the code: open a folder contain dataset of seller, master sheet and finalproject.py file run the code by using code editor as visual studio code to test the code: After execution, the script generates new_df.csv, containing:

The best matched SKU for each seller product. The similarity score and confidence level. Verify Accuracy Performance Evaluation
