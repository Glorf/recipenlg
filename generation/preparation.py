import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer 

nltk.download("punkt")
nltk.download("wordnet")

lemmatizer = WordNetLemmatizer()

recipes = pd.read_json('layer1.json') # This is the original file from unpacked recipe1m
recipes.drop(['id', 'partition','url'], axis=1, inplace=True)

#Extract dictionaries
r_title = recipes.title
r_recipes = recipes.instructions.apply(lambda row: [[value for _, value in item.items()][0] for item in row])
r_ingredients = recipes.ingredients.apply(lambda row: [[value for _, value in item.items()][0] for item in row])

#Create ingredients list copy to work on ingredient extraction
r_ingredients_work = r_ingredients.copy(deep=True)
r_ingredients_work.str.replace("(\([^\)]*\))|


#Extract curated ingredients list to pandas series
c_ingredients = pd.read_csv("curated_ingredients.csv", header=None)[0]

r_ingredients_work = r_ingredients.copy(deep=True)
r_ingredients_work = r_ingredients_work.str.join(" ")
r_ingredients_work = r_ingredients_work.str.replace("\([^\)]*\)", " ") #remove quotes
r_ingredients_work = r_ingredients_work.str.lower() #lowercase
r_ingredients_work = r_ingredients_work.str.replace("[^a-z\s]+", " ") #remove all special characters
r_ingredients_work = r_ingredients_work.str.replace("(\s+|$\s+|\s+^)", " ") #remove all additional white characters
r_ingredients_lemmas = r_ingredients_work.apply(lambda row: ' '.join([lemmatizer.lemmatize(word) for word in word_tokenize(row)])) #lemmatize

matches = r_ingredients_lemmas.apply(lambda row: [ingredient for ingredient in c_ingredients if ingredient in row]) #list lemmas that match the curated dataset


df = "<RECIPE_START> <INPUT_START> " + matches.str.join(" ") + " <INPUT_END> <INGR_START> " + \
  r_ingredients.str.join(" <NEXT_INGR> ") + " <INGR_END> <INSTR_START> " + \
  r_recipes.str.join(" <NEXT_INSTR> ") + " <INSTR_END> <TITLE_START> " + r_title + " <TITLE_END> <RECIPE_END>"

train, test = train_test_split(df, test_size=0.05) #use 5% for test set
np.savetxt(r'unsupervised_train.txt', train, fmt='%s')
np.savetxt(r'unsupervised_test.txt', test, fmt='%s')
