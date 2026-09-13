# Offline gate. No network, no writes outside the repo, no credentials.
#
#   make check      unit suite, then the curated mutation gate
#   make test       unit suite only
#   make mutants    mutation gate only (needs mutt_check installed)
#
# `check` is safe to run reflexively by design: it reads nothing outside this
# directory and mutates only a temp copy of it.

PYTHON ?= python3

.PHONY: check test mutants clean

check: test mutants

test:
	$(PYTHON) -m unittest discover -s tests -t . -v

mutants:
	@command -v mutt_check >/dev/null 2>&1 || { \
	  echo "mutt_check not installed; skipping the mutation gate."; \
	  echo "  pip install 'mutt_check @ git+https://github.com/nlmundis/mutt_check'"; \
	  exit 1; }
	mutt_check

clean:
	rm -rf __pycache__ tests/__pycache__
