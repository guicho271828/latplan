
.PRECIOUS: %.sasp
.SECONDARY: %.sas

all: table

table: table.ros
	ros build table.ros


domain.pddl:
	touch domain.pddl

%_all.sas: %.csv all_actions.csv
	../../lisp/sas.bin -t all_actions.csv $(shell cat $<) > $@
%_raw.sas: %.csv actions.csv
	../../lisp/sas.bin -t actions.csv $(shell cat $<) > $@
%_aug.sas: %.csv augmented.csv
	../../lisp/sas.bin -t augmented.csv $(shell cat $<) > $@
%_iter.sas: %.csv iter.csv
	../../lisp/sas.bin -t iter.csv $(shell cat $<) > $@

%.sasp: %.sas
	../../downward/builds/release64/bin/preprocess < $< > $@
