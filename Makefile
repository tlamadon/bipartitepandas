# get the remote push url
GIT_REMOTE:=$(shell git remote get-url --push origin)

test:
	poetry run python -m pytest

doc:
	cp README.rst docs/source/README.rst
	rm -rf docs/build
	$(MAKE) -C docs html copy
	cd docs/build/html && \
	git init && \
	git add . && \
	git commit -m "Update documentation using Makefile [ci skip]" && \
	git remote add origin $(GIT_REMOTE) && \
	git push --force origin master:gh-pages
	rm -rf docs/build

conda:
	conda env create -f conda-env-file.yml 