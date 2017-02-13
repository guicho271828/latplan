
all: problem.sasp
	touch domain.pddl

%_all.sas: %.csv all_actions.csv
	../../lisp/sas.bin -t all_actions.csv <($<) > $@
%_raw.sas: %.csv actions.csv
	../../lisp/sas.bin -t actions.csv <($<) > $@
%_aug.sas: %.csv augmented.csv
	../../lisp/sas.bin -t augmented.csv <($<) > $@

%.sasp: %.sas
	../../downward/fast-downward.py --preprocess $<
	mv output $@
