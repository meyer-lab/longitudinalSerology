flist = 1

all: $(patsubst %, output/figure%.svg, $(flist))

output/figure%.svg: genFigure.py syserol/figures/figure%.py
	mkdir -p output
	poetry run fbuild $*

test:
	poetry run pytest -s -v -x

testcover:
	poetry run pytest --cov=syserol --cov-report=xml --cov-config=.github/workflows/coveragerc

clean:
	rm -rf output
