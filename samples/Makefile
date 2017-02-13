
all: problem.sasp
	touch domain.pddl

%_all.sas: %.csv all_actions.csv
	../../lisp/sas.bin -t all_actions.csv $(shell cat $<) > $@
%_raw.sas: %.csv actions.csv
	../../lisp/sas.bin -t actions.csv $(shell cat $<) > $@
%_aug.sas: %.csv augmented.csv
	../../lisp/sas.bin -t augmented.csv $(shell cat $<) > $@

%.sasp: %.sas
	../../downward/fast-downward.py --preprocess $<
	mv output $@
