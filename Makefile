
.PHONY: sync

target = $(HOME)/repos/papers/strips-sae/img/static/

all: 
	parallel $(MAKE) {1}-{2}.pdf \
		::: problem-instances problem-instances-16 problem-instances-16-korf problem-instances-ama2 \
		::: expanded evaluated generated search total total2 initialization \
                    ood-histogram-states ood-histogram-transitions

sync:
	-parallel scp -p ccc016:repos/latplan-strips/latplan/{}.csv . ::: table4 table7 table8
	-parallel scp -p ccc016:repos/latplan-strips/latplan/{1}-{2}.csv . \
		::: problem-instances problem-instances-16 problem-instances-16-korf problem-instances-ama2 \
		::: expanded evaluated generated search total total2 initialization \
                    coverage ood-histogram-states ood-histogram-transitions

# hanoi

# ood-result-states.pdf ood-result-transitions.pdf: ood-plot.ros $(wildcard ood-result-*.csv)
# 	./ood-plot.ros $(wildcard ood-result-*.csv)

%-ood-histogram-states.pdf: %-ood-histogram-states.csv ood-histogram.ros
	./ood-histogram.ros $< $@
	-cp -t $(target) $@

%-ood-histogram-transitions.pdf: %-ood-histogram-transitions.csv ood-histogram.ros
	./ood-histogram.ros $< $@
	-cp -t $(target) $@

%.pdf: %.csv plot.ros
	./plot.ros $@
	-cp -t $(target) $@

