import os

# Directories for datasets and logs
DATA_DIR = '/Data'
LOG_DIR = os.path.join(DATA_DIR, 'Logs/SpotArtifacts')

# Directories for converted datasets
STL10_TF_DATADIR = os.path.join(DATA_DIR, 'TF_Records/STL10_TFRecords/')

# Source directories for datasets
STL10_DATADIR = os.path.join(DATA_DIR, 'Datasets/STL10/')