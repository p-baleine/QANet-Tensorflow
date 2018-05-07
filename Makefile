# Variables for preprocessing.
TRAIN_DATA ?= data/train-v1.1.json
DEV_DATA ?= data/dev-v1.1.json
GLOVE ?= word2vec/glove.6B.300d.txt

DATA_SAVE_DIR = data/preprocessed_$(shell date +'%Y%m%d%H%M')
PREPROCESSED_TRAIN_DATA = $(DATA_SAVE_DIR)/train.json
PREPROCESSED_DEV_DATA = $(DATA_SAVE_DIR)/dev.json
GLOVE_WORD2VEC = word2vec/glove.6B.300d.word2vec.bin

# Variables for training.
# Use latest directory as default.
TRAIN_DATA_DIR = $(shell ls -d data/preprocessed_*  | sort | tail -n 1)
HPARAMS_PATH = hparams/default.json
LOG_DIR = /tmp/qanet/$(shell date +'%Y%m%d%H%M')

# Variables for evaluation.
EVAL_DATA_DIR = $(shell ls -d data/preprocessed_* | sort | tail -n 1)
EVAL_LOG_DIR = $(shell ls -d /tmp/qanet/* | sort | tail -n 1)
EVAL_WEIGHTS_FILE = $(shell ls $(EVAL_LOG_DIR)/weights.* | sort -V | tail -n 1)

all: $(PREPROCESSED_TRAIN_DATA) $(PREPROCESSED_DEV_DATA)

$(PREPROCESSED_TRAIN_DATA): $(TRAIN_DATA) $(GLOVE_WORD2VEC) # $(DEV_DATA) 
	mkdir -p $(DATA_SAVE_DIR) && \
		python -u -m scripts.preprocess_data \
			--train-data $(TRAIN_DATA) \
			--dev-data $(DEV_DATA) \
			--glove $(GLOVE_WORD2VEC) \
			--out $(DATA_SAVE_DIR) 2>&1 \
		| tee $(DATA_SAVE_DIR)/preprocess.log

$(GLOVE_WORD2VEC): $(GLOVE)
	python -m scripts.convert_glove2word2vec \
		$(GLOVE) $(GLOVE_WORD2VEC)

train:
	mkdir -p $(LOG_DIR) && \
		python -m scripts.train \
			--data $(TRAIN_DATA_DIR) \
			--hparams $(HPARAMS_PATH) \
			--save-path $(LOG_DIR) 2>&1 \
		| tee $(LOG_DIR)/train.log

evaluate:
	python -m scripts.evaluate \
		--data $(EVAL_DATA_DIR) \
		--raw-data-file $(DEV_DATA) \
		--save-path $(EVAL_LOG_DIR) \
		--weights-file $(EVAL_WEIGHTS_FILE)

test:
	nosetests

.PHONY: test
