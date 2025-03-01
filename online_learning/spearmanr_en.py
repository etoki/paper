import pandas as pd
from scipy.stats import spearmanr

data = pd.read_csv("csv/0723/raw_studysapuri_2.csv")

data_all = data[data['category'] == 'all'].drop('category', axis=1)
data_ja  = data[data['category'] ==  'ja'].drop('category', axis=1)
data_mt  = data[data['category'] ==  'mt'].drop('category', axis=1)
data_en  = data[data['category'] ==  'en'].drop('category', axis=1)
data_sc  = data[data['category'] ==  'sc'].drop('category', axis=1)

traits = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism',
          # facets
          'IntellectualCuriosity', 'AestheticSensitivity', 'CreativeImagination', 
          'Organization', 'Productiveness', 'Responsibility',
          'Sociability', 'Assertiveness', 'EnergyLevel',
          'Compassion', 'Respectfulness', 'Trust',
          'Anxiety', 'Depression', 'EmotionalVolatility']

usage_items = ['NumberOfLecturesWatched', 'ViewingTime', 'NumberOfConfirmationTestsCompleted', 'NumberOfConfirmationTestsMastered', 'AverageFirstAttemptCorrectAnswerRate']

def save_correlation_to_csv(traits, usage_items, data, filename):
    results = []
    for trait in traits:
        for usage in usage_items:
            correlation, p_value = spearmanr(data[trait], data[usage])
            results.append([trait, usage, correlation, p_value])
    
    df = pd.DataFrame(results, columns=['personality', 'item', 'correlation', 'p_value'])
    df.to_csv(filename, index=False)

save_correlation_to_csv(traits, usage_items, data_all, 'csv/0723/correlation_all.csv')
save_correlation_to_csv(traits, usage_items, data_ja,  'csv/0723/correlation_ja.csv')
save_correlation_to_csv(traits, usage_items, data_mt,  'csv/0723/correlation_mt.csv')
save_correlation_to_csv(traits, usage_items, data_en,  'csv/0723/correlation_en.csv')
save_correlation_to_csv(traits, usage_items, data_sc,  'csv/0723/correlation_sc.csv')
