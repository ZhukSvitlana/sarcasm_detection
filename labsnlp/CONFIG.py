import os

#define dlobal variables paths to data
TRAIN_DATA_PATH = os.path.join(os.path.dirname(__file__), '../data/train.csv')
TEST_DATA_PATH =  os.path.join(os.path.dirname(__file__), '../data/test.csv')

SUBMISSION_DATA_TEST_PATH =  os.path.join(os.path.dirname(__file__), '../data/test.csv')

#define key name in data table
OUTPUT_KEY = "is_sarcastic"

TFIDF_OPTIONS = {
    "max_df": 0.8,
    "max_features": 5000
}