
all: $(patsubst %.ros,%.bin,$(wildcard *.ros))

%.bin: %.ros
	LANG=en_US.utf8 ros dynamic-space-size=8000 build $<
	mv $* $@

clean:
	-rm *.bin *.pddl

test: all
	python3 test.py
	ros ./msdd.ros -t -g 256 actions2-m.csv
