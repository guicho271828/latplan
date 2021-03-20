

db = results-planning-10min.sqlite


ids   = $(shell sqlite3 -quote $(db) "select id from $(1) order by id")
keys  = $(shell sqlite3 -quote $(db) "select key from $(1) order by id")
names = $(shell sqlite3 -quote $(db) "select name from $(1) order by id")
split = $(subst -, ,$1)
elem = $(word $2,$(call split,$1))

splitdot = $(subst ., ,$1)
elemdot = $(word $2,$(call splitdot,$1))

upcase = $(shell echo '$1' | tr '[:lower:]' '[:upper:]')
is_noisy = $(shell if [ "$(call elem,$1,$2)" != "0.0" ] ; then echo ", corrupted" ; fi  )
config = $(call upcase,$(call elem,$1,$2))+$(call is_noisy,$1,$3)
# config = $(call upcase,$(call elem,$1,$2)), σ=$(call elem,$1,$3), c=$(call elem,$1,$4)



hand_written_tables = $(patsubst tables/%.sql,%.csv,$(wildcard tables/*.sql))

generated_tables = $(shell echo metrics-{ama3,ama4}-{kltune,nozsae}.csv)

hand_written_plots  = $(patsubst plots/%.sql,%.pdf,$(wildcard plots/*.sql))

generated_plots = \
	$(shell echo ama3-ama4-elbo-{kltune,nozsae}.pdf) \
	$(shell echo ama3-ama4-{applicability,effect}-{kltune,nozsae}.pdf) \
	$(shell echo {ama3,ama4}-stability-{constantzero,effective,variance}.pdf) \
	$(shell echo bestsolved-vs-nozsae-{evaluated,search}-{domains,generators,heuristics}-{ama3,ama4}-{1.0,0.0}-1.pdf) \
	$(shell echo bestsolved-vs-blind-{evaluated,search}-{domains,generators,heuristics2}-{ama3,ama4}-{1.0,0.0}-1.pdf) \
	$(shell echo all-vs-nozsae-{found,valid,optimal,exhausted}-{domains,generators,heuristics3}-{ama3,ama4}-{1.0,0.0}-1.pdf) \
	$(shell echo vs-{notune,nozsae}-{ama3,ama4}.pdf)\
	$(shell echo pareto-{ama3,ama4}-{blocks,lightsout_digital,lightsout_twisted,puzzle_mandrill,puzzle_mnist,sokoban}.pdf)\
	$(shell echo cycle-{ama3,ama4}-{blocks,lightsout_digital,lightsout_twisted,puzzle_mandrill,puzzle_mnist,sokoban}-{kltune,nozsae}.pdf)

generated_queries = \
	$(patsubst %.csv,tables-generated/%.sql,$(generated_tables)) \
	$(patsubst %.pdf,plots-generated/%.sql,$(generated_plots))


.PHONY: sync
.SECONDARY: %.csv
# .PRECIOUS: %.csv

all:  plots tables

tables: $(hand_written_tables) $(generated_tables)

plots: $(hand_written_plots) $(generated_plots)
	-rm all.pdf
	cp *.pdf ~/repos/papers/2021-latplan-journal/img/static/experiments/
	pdfunite $$(ls *.pdf | grep -v bestsolved) bestsolved*.pdf all.pdf


# database setup

db: $(db)

planning = $(shell ls results-planning-10min-noise{0.0,1.0}-cycle{1,5}.csv)

results-planning-10min.sqlite: $(wildcard names/*.csv) create.sql \
				results-training-AMA3.csv \
				results-training-AMA4.csv \
				$(planning)
	-rm -vf $@
	sqlite3 $@ < create.sql
	parallel -j 1 -v "(echo .mode csv ; echo .import --skip 1 {} planning) | sqlite3 $@" ::: $(planning)


results-training-AMA3.csv:
	./extract-training-results.sh ../samples/*AMA3*/logs/*/ > results-training-AMA3.csv

results-training-AMA4.csv:
	./extract-training-results.sh ../samples/*AMA4*/logs/*/ > results-training-AMA4.csv

results-planning%.csv:
	./extract-planning-results.sh ../problem-instances$* > results-planning$*.csv


# making tables, figures (not generated)

elbo-vs-compiled-action.pdf: elbo-vs-compiled-action.csv elbo-vs-compiled-action.ros
	./elbo-vs-compiled-action.ros $< $@ $(call names,generators)


# making tables, figures (using generated queries)

plots-generated/ama3-ama4-elbo-%.sql: templates/ama3-ama4.sql 
	sed 's/COMMENT/$*2/g;s/METRIC/"metrics.test.elbo"/g' $< > $@
plots-generated/ama3-ama4-applicability-%.sql: templates/ama3-ama4.sql
	sed 's/COMMENT/$*2/g;s/METRIC/"metrics.test.kl_a_z0"/g' $< > $@
plots-generated/ama3-ama4-effect-%.sql: templates/ama3-ama4.sql
	sed 's/COMMENT/$*2/g;s/METRIC/"z1z2.mae.vanilla.test"/g' $< > $@

ama3-ama4-elbo-%.pdf: ama3-ama4-elbo-%.csv plot.ros
	./plot.ros $< $@ "Comparison of -ELBO between AMA_3^+ / AMA_4^+, $(shell echo 'select name from comments where key = "$(call elem,$*,1)2"' | sqlite3 $(db))" "AMA_4^+" "AMA_3^+" 1000 100000 t t t t $(call names,generators)
ama3-ama4-applicability-%.pdf: ama3-ama4-applicability-%.csv plot.ros
	./plot.ros $< $@ "KL(q(a|x_0,x_1) || p(a|z_0)), $(shell echo 'select name from comments where key = "$(call elem,$*,1)2"' | sqlite3 $(db))" "AMA_4^+" "AMA_3^+" 500 4000 t t t t $(call names,generators)
ama3-ama4-effect-%.pdf: ama3-ama4-effect-%.csv plot.ros
	./plot.ros $< $@ "\E_{i,f} |z^{i,2}_f-z^{i,1}_f|, $(shell echo 'select name from comments where key = "$(call elem,$*,1)2"' | sqlite3 $(db))" "AMA_4^+" "AMA_3^+" 1e-5 1 t t t t $(call names,generators)



plots-generated/%-stability-constantzero.sql: templates/stability.sql
	sed 's/AECLASS/$*/g;s/METRIC/N-"sae.ever_1.test"/g' $< > $@
plots-generated/%-stability-effective.sql: templates/stability.sql
	sed 's/AECLASS/$*/g;s/METRIC/"sae.effective.test"/g' $< > $@
plots-generated/%-stability-variance.sql: templates/stability.sql
	sed 's/AECLASS/$*/g;s/METRIC/"sae.variance.gaussian.test.mean"/g' $< > $@

ama3-stability-constantzero.pdf: ama3-stability-constantzero.csv plot.ros
	./plot.ros $< $@ "Number of constant zero bits" "ε=0.1" "ε=0.5" 0 300 nil nil nil t $(call names,generators)
ama3-stability-effective.pdf: ama3-stability-effective.csv plot.ros
	./plot.ros $< $@ "Number of effective bits" "ε=0.1" "ε=0.5" 0 300 t t nil t $(call names,generators)
ama3-stability-variance.pdf: ama3-stability-variance.csv plot.ros
	./plot.ros $< $@ "State variance" "ε=0.1" "ε=0.5" 1e-6 1 t t t t $(call names,generators)
ama4-stability-constantzero.pdf: ama4-stability-constantzero.csv plot.ros
	./plot.ros $< $@ "Number of constant zero bits" "ε=0.1" "ε=0.5" 0 300 nil nil nil t $(call names,generators)
ama4-stability-effective.pdf: ama4-stability-effective.csv plot.ros
	./plot.ros $< $@ "Number of effective bits" "ε=0.1" "ε=0.5" 0 300 t t nil t $(call names,generators)
ama4-stability-variance.pdf: ama4-stability-variance.csv plot.ros
	./plot.ros $< $@ "State variance" "ε=0.1" "ε=0.5" 1e-6 1 t t t t $(call names,generators)



plots-generated/bestelbo-vs-nozsae-%.sql: templates/bestelbo-vs-nozsae.sql
	sed 's/METRIC/$(call elem,$*,1)/g;s/GROUP/$(call elem,$*,2)/g;s/AECLASS/$(call elem,$*,3)/g;s/NOISE/$(call elem,$*,4)/g' $< > $@

bestelbo-vs-nozsae-search-%.pdf: bestelbo-vs-nozsae-search-%.csv plot.ros
	./plot.ros $< $@ "Search time [sec] ($(call elem,$*,2), best ELBO, $(call elem,$*,3))" "ε=0.1" "ε=0.5" 1e-4 10000 t t t nil $(call names,$(call elem,$*,1))
bestelbo-vs-nozsae-%.pdf: bestelbo-vs-nozsae-%.csv plot.ros
	./plot.ros $< $@ "$(call elem,$*,1) nodes ($(call elem,$*,3), best ELBO, $(call elem,$*,4))" "ε=0.1" "ε=0.5" 1 1000000000 t t t nil $(call names,$(call elem,$*,2))


plots-generated/bestsolved-vs-blind-%.sql: templates/bestsolved-vs-blind.sql
	sed 's/METRIC/$(call elem,$*,1)/g;s/GROUP/$(call elem,$*,2)/g;s/AECLASS/$(call elem,$*,3)/g;s/NOISE/$(call elem,$*,4)/g;s/CYCLE/$(call elem,$*,5)/g' $< > $@
bestsolved-vs-blind-search-%.pdf: bestsolved-vs-blind-search-%.csv plot.ros
	./plot.ros $< $@ "Effect of heuristic on search time [sec] ($(call config,$*,2,3,4))" "blind" "LMCut, M＆S, LAMA" 1e-4 10000 nil nil t nil $(call names,$(call elem,$*,1))
bestsolved-vs-blind-%.pdf: bestsolved-vs-blind-%.csv plot.ros
	./plot.ros $< $@ "Effect of heuristic on $(call elem,$*,1) nodes ($(call config,$*,3,4,5))" "blind" "LMCut, M＆S, LAMA" 1 1000000000 nil nil t nil $(call names,$(call elem,$*,2))


plots-generated/bestsolved-vs-nozsae-%.sql: templates/bestsolved-vs-nozsae.sql
	sed 's/METRIC/$(call elem,$*,1)/g;s/GROUP/$(call elem,$*,2)/g;s/AECLASS/$(call elem,$*,3)/g;s/NOISE/$(call elem,$*,4)/g;s/CYCLE/$(call elem,$*,5)/g' $< > $@

bestsolved-vs-nozsae-search-%.pdf: bestsolved-vs-nozsae-search-%.csv plot.ros
	./plot.ros $< $@ "Search time [sec] ($(call config,$*,2,3,4))" "ε=0.1" "ε=0.5" 1e-4 10000 t t t nil $(call names,$(call elem,$*,1))
bestsolved-vs-nozsae-%.pdf: bestsolved-vs-nozsae-%.csv plot.ros
	./plot.ros $< $@ "$(call elem,$*,1) nodes ($(call config,$*,3,4,5))" "ε=0.1" "ε=0.5" 1 1000000000 t t t nil $(call names,$(call elem,$*,2))


plots-generated/all-vs-nozsae-%.sql: templates/all-vs-nozsae.sql
	sed 's/METRIC/$(call elem,$*,1)/g;s/GROUP/$(call elem,$*,2)/g;s/AECLASS/$(call elem,$*,3)/g;s/NOISE/$(call elem,$*,4)/g;s/CYCLE/$(call elem,$*,5)/g' $< > $@

all-vs-nozsae-exhausted-%.pdf: all-vs-nozsae-exhausted-%.csv plot.ros
	./plot.ros $< $@ "Exhausted instances ($(call config,$*,2,3,4))" "ε=0.1" "ε=0.5" 0 50 t t nil nil $(call names,$(call elem,$*,1))
all-vs-nozsae-%.pdf: all-vs-nozsae-%.csv plot.ros
	./plot.ros $< $@ "$(call elem,$*,1) instances ($(call config,$*,3,4,5))" "ε=0.1" "ε=0.5" 0 50 nil nil nil nil $(call names,$(call elem,$*,2))



plots-generated/vs-notune-%.sql: templates/vs-notune.sql
	sed 's/AECLASS/$*/g' $< > $@
plots-generated/vs-nozsae-%.sql: templates/vs-nozsae.sql
	sed 's/AECLASS/$*/g' $< > $@

vs-notune-%.pdf: vs-notune-%.csv plot.ros
	./plot.ros $< $@ "Comparison of -ELBO with / without tuning β_1,β_3" "Tuning F,β_1,β_3" "Tuning F only" 1000 100000 t nil t t $(call names,generators)
vs-nozsae-%.pdf: vs-nozsae-%.csv plot.ros
	./plot.ros $< $@ "Comparison of -ELBO between ε=0.1 / ε=0.5" "ε=0.1" "ε=0.5" 1000 100000 t t t t $(call names,generators)




plots-generated/pareto-ama3-%.sql: templates/pareto-ama3.sql
	sed "s/GENERATOR/'latplan.puzzles.$*'/g" $< > $@
plots-generated/pareto-ama4-%.sql: templates/pareto-ama4.sql
	sed "s/GENERATOR/'latplan.puzzles.$*'/g" $< > $@

pareto-%.pdf: pareto-%.csv pareto.ros
	./pareto.ros $< $@


plots-generated/cycle-%.sql: templates/cycle.sql
	sed "s/AECLASS/$(call elem,$*,1)/g;s/GENERATOR/'latplan.puzzles.$(call elem,$*,2)'/g;s/COMMENT/'$(call elem,$*,3)2'/g;" $< > $@

cycle-%.pdf: cycle-%.csv cycle.ros
	./cycle.ros $< $@



tables-generated/metrics-ama3-%.sql: templates/metrics-ama3.sql
	sed "s/COMMENT/'$*2'/g" $< > $@
tables-generated/metrics-ama4-%.sql: templates/metrics-ama4.sql
	sed "s/COMMENT/'$*2'/g" $< > $@





# three difference sources of csv

%.csv: tables/%.sql $(db)
	cat $< | sqlite3 -csv -header $(db) > $@
%.csv: tables-generated/%.sql $(db)
	cat $< | sqlite3 -csv -header $(db) > $@
%.csv: plots/%.sql $(db)
	cat $< | sqlite3 -csv $(db) > $@
%.csv: plots-generated/%.sql $(db)
	cat $< | sqlite3 -csv $(db) > $@

$(generated_queries): | plots-generated tables-generated

plots-generated:
	mkdir -p $@
tables-generated:
	mkdir -p $@


%.pdf: %.csv %.ros
	./$*.ros $*.csv $*.pdf

sync:
	rsync -avz --info=progress1 ccc016:repos/latplan-ijcai21/tables/*.csv .

clean:
	-rm $$(ls *.csv | grep -v results) *.pdf
	-rm -r plots-generated
distclean:
	-rm *.csv *.pdf *.sqlite plots-generated/*

