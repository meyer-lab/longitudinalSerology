flist = 1 2 3

all: $(patsubst %, output/figure%.svg, $(flist))

output/figure%.svg: syserol/figures/figure%.py
	mkdir -p output
	poetry run fbuild $*

test:
	poetry run pytest -s -v -x

testprofile:
	poetry run python3 -m cProfile -o profile -m pytest -s -v -x

testcover:
	poetry run pytest --cov=syserol --cov-report=xml --cov-config=.github/workflows/coveragerc

clean:
	rm -rf output
