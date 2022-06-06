import csv
import os
import pandas as pd

def main() :
    index = []
    labels = []

    if not os.path.isdir("./result"):
        print("### Error: Missing dictionary <result>! ###")
    else:
        with open('./result/resultPerPatient_covid.csv', newline='') as csvfile:

            rows = csv.DictReader(csvfile)

            for row in rows:
                index.append(row['Image Index'])
                labels.append(row['Predict label'])

        with open('./result/resultPerPatient_14.csv', newline='') as csvfile:

            rows = csv.DictReader(csvfile)

            for row in rows:
                index.append(row['Image Index'])
                print(type(row['Predict label']))
                labels.append(row['Predict label'])

        dict = {"Image Index": index,  
            "Predict label": labels
        }

        df = pd.DataFrame(dict)
        sort_df = df.sort_values(by=['Image Index'], ascending=False)

        sort_df.to_csv('./result/prediction_of_all_patients.csv',index=False,header=True)