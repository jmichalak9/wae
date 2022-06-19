install:
	poetry install

clean:
	rm -rf exdata ppdata *.pydict

run:
	python benchmark.py cmaes
	python benchmark.py ipop-maes

postprocessing:
	python post_process_merge.py

shell:
	poetry shell

.PHONY: install run postprocessing shell