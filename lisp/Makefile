

%.bin: %.ros
	ros build $<
	mv $* $@

all: domain.bin problem.bin parse-plan.bin

clean:
	-rm *.bin *.pddl

test: all
	python3 test.py
