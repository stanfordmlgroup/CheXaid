import os
import sys

import pandas as pd
import numpy as np

ORDERS = [
	[1, 2, 7, 8, 5, 6, 3, 4],
	[3, 4, 1, 2, 7, 8, 5, 6],
	[5, 6, 3, 4, 1, 2, 7, 8],
	[7, 8, 5, 6, 3, 4, 1, 2]
]

def get_groups(df, column_name, num_in_set):
	""" Return a list of groups of subjectIDs."""
	subjID_list = df[column_name].tolist()
	curr_group, all_groups = [], []
	for i, subjectID in enumerate(subjID_list):
		curr_group.append(subjectID)
		if len(curr_group) == num_in_set:
			all_groups.append(curr_group)
			curr_group = []
	return all_groups

def split_to_groups(df, subjectID_groups, path, column_name='subjectID', assisted_col='assisted'):
	""" Reorder csv according to groups and add assisted/unassisted."""
	if not os.path.exists(path):
		os.makedirs(path)

	df[column_name] = df['subjectID'].astype("category")
	group_len = len(subjectID_groups[0])
	for i, order in enumerate(ORDERS):
		subjectID_orders = np.array([subjectID_groups[group_num - 1] for group_num in order]).flatten()
		df[column_name].cat.set_categories(subjectID_orders, inplace=True)
		df = df.sort_values([column_name])
		df[assisted_col] = 0
		for index, row in df.iterrows():
			assisted = ((int(index / group_len) % 2) == 0)
			df.at[index, assisted_col] = int(assisted)

		curr_csv_name_assisted_first = f'group_{i}_assisted_first.csv'
		df.to_csv(os.path.join(path, curr_csv_name_assisted_first))

		curr_csv_name_assisted_first = f'group_{i}_unassisted_first.csv'
		df[assisted_col] = df[assisted_col].apply(lambda x: np.abs(1 - x))
		df.to_csv(os.path.join(path, curr_csv_name_assisted_first))


if __name__ == '__main__':
	probabilities_path = sys.argv[1] # Path to probabilities directory e.g. experiment_final_data/probabilities
	file_name = sys.argv[2] # CSV file name e.g. probabilities_test_calibrated_ensemble.csv

	df = pd.read_csv(os.path.join(probabilities_path, file_name))
	# Shuffle the df with a seed
	df = df.sample(frac=1, random_state=1).reset_index(drop=True)
	subjectID_groups = get_groups(df, column_name='subjectID', num_in_set=15)
	split_to_groups(df, subjectID_groups, os.path.join(probabilities_path, 'assisted'))

