

%.bin: %.ros
	ros build $<
	mv $* $@

all: domain.bin problem.bin parse-plan.bin

clean:
	-rm *.bin *.pddl

test: all
	python3 test.py
	ros ./msdd.ros -t -g 256 actions2-m.csv
