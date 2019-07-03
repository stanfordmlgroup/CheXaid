import numpy as np
import json
import os
import re
import pickle
import matplotlib.pyplot as plt
import shutil

base_dir = '/deep/u/amits/cxr/aihc-fall18-2dimaging/'
ckpts_dir = base_dir + 'ckpts/'
logs_dir = base_dir + 'logs/'

start_epoch = r'start of epoch [0-9]+'
valid_auroc = r'valid.*?tbAUROC: [01](.[0-9]+)?'
auroc = r'AUROC: [01](.[0-9]+)?'
valid_loss = r'valid_loss: [01](.[0-9]+)?'

class ModelData(object):
	"""Store model data - args, performance, optimal performance conditions."""
	def __init__(self, model_name):
		self.model_name = model_name
		self.args = {}
		self.performances = []

	def get_best_valid_auroc(self):
		return np.amax(self.performances[:, 1])

	def get_best_valid_loss(self):
		return np.amin(self.performances[:, 2])
	
	def calculate_performances(self, lines):
		cur_epoch, cur_valid_auroc, cur_valid_loss, cur_train_loss = 0, 0.0, 0.0, 0.0
		for line in lines:
			content = detect_line_intent(line)                    
			if 'start_epoch' in content:
				if cur_epoch != 0:
					self.performances.append([cur_epoch, cur_valid_auroc, cur_valid_loss])
				cur_epoch = content['start_epoch']
			elif 'valid_auroc' in content:
				cur_valid_auroc = content['valid_auroc']
			elif 'valid_loss' in content:
				cur_valid_loss = content['valid_loss']
		self.performances = np.array(self.performances)

def detect_line_intent(line):
	"""Detect intent of the given line.
	
	Here, we check if the current line starts an epoch, specified the validation 
	auroc or the validation loss of the current epoch.
	"""

	content = {}
	if re.search(start_epoch, line) is not None:
		match = re.search(start_epoch, line)
		content = {'start_epoch': int(match.group()[15:])}
	elif re.search(valid_auroc, line) is not None:
		match = re.search(auroc, line)
		content = {'valid_auroc': float(match.group()[7:])}
	elif re.search(valid_loss, line) is not None:
		match = re.search(valid_loss, line)
		content = {'valid_loss': float(match.group()[12:])}
	return content


def find_bad_models(models, threshold=0.65, maximize=True):
	bad_models = []
	for model in models:
		curr_models_perf = models[model].get_best_valid_auroc()
		if curr_models_perf < threshold:
			bad_models.append(model)
			print(curr_models_perf)
			shutil.rmtree(ckpts_dir + models[model].model_name)
			shutil.rmtree(logs_dir + models[model].model_name)

if __name__ == '__main__':
	models = {}
	for model_dir in os.listdir(ckpts_dir):
		model_name = str(model_dir)
		model_class = ModelData(model_name)
		model_full_path = ckpts_dir + model_dir

		log_file_name = None
		if 'sgd.log' in os.listdir(model_full_path):
			log_file_name = model_full_path + '/sgd.log'
		elif 'adam.log' in os.listdir(model_full_path):
			log_file_name = model_full_path + '/adam.log'

		if log_file_name:
			with open(log_file_name, 'r') as perf_file:
				lines = perf_file.readlines()
				model_class.calculate_performances(lines)
				if len(model_class.performances) > 0:
					models[model_name] = model_class

	find_bad_models(models)
