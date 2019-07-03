# CXR_PULM_TB_MEAN = [.5020, .5020, .5020]
# CXR_PULM_TB_STD = [.3, .3, .3]
CXR_PULM_TB_MEAN = [0.5196193, 0.5196193, 0.5196193]
CXR_PULM_TB_STD = [0.26072893, 0.26072893, 0.26072893]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CXR_COV_MEAN = {
	'sex': 0.064,
	'age': 37.274,
	'prev_tb': 0.5335,
	'art_status': 0.4822,
	'temp': 37.0432,
	'oxy_sat': 95.692,
	'hgb': 9.66,
	'cd4': 165.1655,
	'wbc': 10.4448,
	'cough':0.8622,
}

CXR_COV_STD = {
	'sex': 0.4899,
	'age': 9.7035,
	'prev_tb': 0.4997,
	'art_status': 0.5005,
	'temp': 1.2196,
	'oxy_sat': 4.4725,
	'hgb': 2.6323,
	'cd4': 243.6240,
	'wbc': 10.1818,
	'cough': 0.3453,
}

COL_PATH = 'Path'
COL_STUDY = 'Study'
COL_SPLIT = 'DataSplit'
COL_PATIENT = 'Patient'

CFG_TASK2MODELS = 'task2models'
CFG_AGG_METHOD = 'aggregation_method'
CFG_CKPT_PATH = 'ckpt_path'
CFG_IS_3CLASS = 'is_3class'

COL_TCGA_SLIDE_ID = 'slide_id'
COL_TCGA_FILE_ID = 'file_id'
COL_TCGA_FILE_NAME = 'file_name'
COL_TCGA_CASE_ID = 'case_id'
COL_TCGA_LABEL = 'label'
COL_TCGA_PATCH_ID = 'patch_num'
COL_TCGA_NUM_PATCHES = 'num_patches'
COL_TCGA_INDICES = 'indices'
COL_TCGA_PATH = 'path_to_slide'
COL_TCGA_ENTITIES = 'associated_entities'

SLIDE_METADATA_FILE = 'slide_metadata.csv'
SLIDE_PKL_FILE = 'slide_list.pkl'

DEFAULT_PATCH_SIZE = 512
