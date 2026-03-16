.PHONY: install ingest index train evaluate search serve test

install:
	pip install -r requirements.txt

ingest:
	python -m aura.data.ingest

index:
	python -m aura.indexing.build_index

train:
	python -m aura.ranking.train_classifier

evaluate:
	python -m aura.evaluation.evaluate

serve:
	uvicorn aura.api.app:app --host 0.0.0.0 --port 8000 --reload

test:
	pytest tests/ -v --tb=short
