# Offline gate. No network, no writes outside the repo, no credentials.
#
#   make test       unit suite only; standard library only
#   make check      unit suite, then the curated mutation gate
#   make mutants    mutation gate only (needs mutt_check)
#
# `check` is safe to run reflexively by design: it reads nothing outside this
# directory and mutates only a temp copy of it.
#
# The gate is a separate tool. It stops with the install line when absent
# rather than skipping, because a skipped gate reads as a passed one.

PYTHON ?= python3
MUTT_CHECK ?= mutt_check

.PHONY: check test mutants clean

check: test mutants

test:
	$(PYTHON) -B -m unittest discover -s tests -t .

# One shell line on purpose: the availability test and the run share a shell,
# so the exit status of the test IS the exit status of the target.
mutants:
	@if ! $(MUTT_CHECK) --version >/dev/null 2>&1; then \
	  echo "mutt_check is not installed, so the mutation gate cannot run:"; \
	  echo "  pip install 'mutt_check @ git+https://github.com/nlmundis/mutt_check@v0.0.2'"; \
	  echo "or point at a checkout: make check MUTT_CHECK=\"python3 path/to/mutt_check.py\""; \
	  exit 2; \
	fi; \
	$(MUTT_CHECK)

clean:
	rm -rf __pycache__ tests/__pycache__
