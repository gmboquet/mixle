# Minimal Sphinx makefile for local documentation builds.

SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SPHINXAPIDOC  ?= sphinx-apidoc
SOURCEDIR     = .
BUILDDIR      = _build

.PHONY: help clean html linkcheck apidoc

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

clean:
	rm -rf "$(BUILDDIR)"

# `-f` OVERWRITES every api/*.rst, including hand-added directives. One is load-bearing today:
# api/mixle.doe.amplify.rst carries `:exclude-members: permutation_test`, because that name is
# re-exported from scipy and scipy decorates it with `@_transition_to_rng`, which injects a
# keyword-only parameter that Sphinx's autodoc_preserve_defaults cannot reconcile with the AST --
# it raises IndexError and fails the -W docs gate. Re-add that line after regenerating, or the
# gate goes red. (The combinator toctree fix IS reproduced by regeneration; only this one is not.)
apidoc:
	$(SPHINXAPIDOC) -f -e -M -P -o api ../mixle ../mixle/tests '*/__pycache__/*'
	@echo "NOTE: re-add ':exclude-members: permutation_test' to api/mixle.doe.amplify.rst (see comment above)"

html:
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

linkcheck:
	@$(SPHINXBUILD) -M linkcheck "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)
