# nelisp-system-one -- typed, calibrated, single-pass decisions.
#
# Consumes the sibling repositories rather than vendoring them; override these
# when they live elsewhere.
EMACS  ?= emacs
PHOTON ?= ../nelisp-photon/lisp
LLM    ?= ../nelisp-llm/lisp
NELISP ?= ../nelisp/target/nelisp

LOAD = -L lisp -L $(PHOTON) -L $(LLM)

.PHONY: test compile clean

# P0 lands the evaluation harness and its negative controls before any model
# exists.  Until then this target has nothing to run, and says so rather than
# reporting success over an empty set.
test:
	@if [ -z "$$(ls test/*-test.el 2>/dev/null)" ]; then \
	  echo "no suites yet -- see docs/design/01-system-one.org, phase P0"; \
	  exit 1; \
	fi
	@for t in test/*-test.el; do \
	  echo "== $$t"; \
	  $(EMACS) -Q --batch $(LOAD) -l $$t || exit 1; \
	done

compile:
	$(EMACS) -Q --batch $(LOAD) -f batch-byte-compile lisp/*.el

clean:
	rm -f lisp/*.elc test/*.elc
