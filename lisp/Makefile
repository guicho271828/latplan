
all: domain.bin problem.bin parse-plan.bin sas.bin domain-actionlearner.bin domain-fol.bin problem-fol.bin parse-plan-fol.ros

%.bin: %.ros
	ros build $<
	mv $* $@

clean:
	-rm *.bin *.pddl

test: all
	python3 test.py
	ros ./msdd.ros -t -g 256 actions2-m.csv
