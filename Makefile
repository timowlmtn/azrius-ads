black:
	black src/python

create-iceberg:
	python src/python/gz_to_iceberg.py \
		--train data/train.txt.gz --test data/test.txt.gz --catalog local

counts:
	python src/python/analyze_ad.py --counts

describe:
	python src/python/analyze_ad.py --describe

explore:
	python src/python/analyze_ad.py \
		--sql "SELECT Label, COUNT(*) FROM criteo_ad GROUP BY Label ORDER BY Label"

RUN_VAR := $(shell date +'%Y-%m-%d-%H%M%S')
train:
	python src/python/train_xgb_baseline.py \
	  --sample-frac 1.0 \
	  --max-rows 200000 \
	  --submission-path data/$(RUN_VAR).csv
