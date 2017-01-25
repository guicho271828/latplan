
all: problem.sasp
	touch domain.pddl

problem.sas: all_actions.csv problem.csv
	../../lisp/sas.bin -t all_actions.csv $(shell cat problem.csv) > $@

problem.sasp: problem.sas
	../../downward/fast-downward.py --preprocess $<
	mv output $@
