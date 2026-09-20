# nelisp-system-one -- typed, calibrated, single-pass decisions.
#
# Consumes the sibling repositories rather than vendoring them; override these
# when they live elsewhere.
EMACS  ?= emacs
PHOTON ?= ../nelisp-photon/lisp
LLM    ?= ../nelisp-llm/lisp
NELISP ?= ../nelisp/target/nelisp

LOAD = -L lisp -L $(PHOTON) -L $(LLM)

.PHONY: test compile clean p1 score

# Everything runs byte-compiled, and that is a performance decision rather
# than a tidiness one.  The numeric loops in lisp/ are interpreted when Emacs
# loads the .el directly, and interpreted they manage about 3.3M float
# multiply-adds a second on this machine: one attention configuration in the
# P1 probe took 54 minutes.  Byte-compiled, the same work is 10 minutes -- 5.3x
# for a build step that costs a second.
#
# The targets below always recompile before they run, so there is no window in
# which a stale .elc shadows an edited .el.  `make clean' removes them.

# P0 lands the evaluation harness and its negative controls before any model
# exists.  Until then this target has nothing to run, and says so rather than
# reporting success over an empty set.
test: compile
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

# The P1 encode-and-probe run.  Goes through compile for the same reason the
# suites do; NSO_P1_STAGE selects tokenize / encode / probe / all.
p1: compile
	$(EMACS) -Q --batch $(LOAD) -l tools/p1-encode-probe.el

# The Score encode-and-probe run.  Goes through compile for the same reason
# p1 does; NSO_SCORE_STAGE selects tokenize / encode / probe / all.  The probe
# path is exercised without a GPU by tools/score-fake-states.el, in both the
# signal and the noise mode, which is what says it can report a failure.
score: compile
	$(EMACS) -Q --batch -L build/elc $(LOAD) -l tools/score-encode-probe.el

clean:
	rm -f lisp/*.elc test/*.elc
