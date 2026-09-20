# nelisp-system-one -- typed, calibrated, single-pass decisions.
#
# Consumes the sibling repositories rather than vendoring them; override these
# when they live elsewhere.
EMACS  ?= emacs
PHOTON ?= ../nelisp-photon/lisp
LLM    ?= ../nelisp-llm/lisp
NELISP ?= ../nelisp/target/nelisp

ELC  = build/elc
LOAD = -L lisp -L $(PHOTON) -L $(LLM)
# The same load path with the byte-compiled sibling cache in FRONT.  Only the
# targets that depend on `deps' may use it: build/elc shadows the sibling
# sources, so a cache nothing rebuilt is a cache that silently wins.
FAST = -L $(ELC) $(LOAD)

.PHONY: test compile deps clean p1 score

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

# The sibling lisp, byte-compiled into build/elc.  A dependency of every
# target that puts $(ELC) on the load path, and it is one because leaving it
# manual cost an encode run: the cache was fourteen hours older than
# nelisp-llm's weight loader, the stale .elc shadowed the edited source, and
# the failure surfaced as a wrong-type-argument naming a tensor inside the
# first block rather than as anything to do with loading.  byte-compile-file
# rebuilds unconditionally, so depending on this is enough.
deps:
	$(EMACS) -Q --batch -l tools/compile-deps.el

# The P1 encode-and-probe run.  Goes through compile for the same reason the
# suites do; NSO_P1_STAGE selects tokenize / encode / probe / all.
# Through $(FAST), which is how P1's reported 10.0s per example was actually
# measured -- by hand, with -L build/elc, which `make p1' did not reproduce.
# A number in the design document that the committed command cannot produce is
# a number nobody can check.
p1: compile deps
	$(EMACS) -Q --batch $(FAST) -l tools/p1-encode-probe.el

# The Score encode-and-probe run.  Goes through compile for the same reason
# p1 does; NSO_SCORE_STAGE selects tokenize / encode / probe / all.  The probe
# path is exercised without a GPU by tools/score-fake-states.el, in both the
# signal and the noise mode, which is what says it can report a failure.
score: compile deps
	$(EMACS) -Q --batch $(FAST) -l tools/score-encode-probe.el

clean:
	rm -f lisp/*.elc test/*.elc
	rm -rf $(ELC)
