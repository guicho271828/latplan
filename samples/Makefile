
all: accuracy.tex

accuracy.tex: table
	./table logs/* > $@

table: table.ros
	ros build table.ros
