.PHONY: run test index search stats clean

# One-command build/run entry point, per hackathon rules ("one command builds").
# `make run` with no args just shows CLI help; use the subtargets below for
# actual usage during development, and document real invocations in README.md.

run:
	python3 -m src.cli --help

test:
	python3 -m unittest discover -s tests -v

clean:
	find . -name '__pycache__' -exec rm -rf {} +
