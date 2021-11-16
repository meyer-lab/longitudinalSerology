flist = 1

all: $(patsubst %, output/figure%.svg, $(flist))

output/figure%.svg: genFigure.py syserol/figures/figure%.py
	mkdir -p output
	poetry run genFigure.py $*

test:
	poetry run pytest -s -v -x

testcover:
	poetry run pytest --cov=syserol --cov-report=xml --cov-config=.github/workflows/coveragerc

output/manuscript.md: manuscript/*.md
	poetry run manubot process --content-directory=manuscript --output-directory=output --cache-directory=cache --skip-citations --log-level=INFO
	cp -r manuscript/images output/
	git remote rm rootstock

output/manuscript.html: output/manuscript.md $(patsubst %, output/figure%.svg, $(flist))
	pandoc --verbose \
		--defaults=./common/templates/manubot/pandoc/common.yaml \
		--defaults=./common/templates/manubot/pandoc/html.yaml \
		--csl=./manuscript/molecular-systems-biology.csl \
		output/manuscript.md

output/manuscript.docx: output/manuscript.md $(patsubst %, output/figure%.svg, $(flist))
	pandoc --verbose \
		--defaults=./common/templates/manubot/pandoc/common.yaml \
		--defaults=./common/templates/manubot/pandoc/docx.yaml \
		--csl=./manuscript/molecular-systems-biology.csl \
		output/manuscript.md

clean:
	rm -rf output
