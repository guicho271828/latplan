#!/bin/bash

samples=${1:-samples/}

paste <(parallel --keep-order jq .acc.shuffled.vanilla.test                ::: $samples/*_p4/performance.json) \
      <(parallel --keep-order jq .parameters.N                             ::: $samples/*_p4/aux.json) \
      <(parallel --keep-order jq .parameters.predicates                    ::: $samples/*_p4/aux.json) \
      <(parallel --keep-order jq .parameters.arity                         ::: $samples/*_p4/aux.json) \
      <(parallel --keep-order "cat {} ; echo"                              ::: $samples/*_p4/parameter_count.json) \
      <(parallel --keep-order "du -b {} "                                  ::: $samples/*_p4/domain.pddl) \
      | sort | column -t > $samples/contour.csv

# paste <(parallel --keep-order jq .acc.vanilla.test                         ::: $samples/*_pifc/performance.json) \
#       <(parallel --keep-order jq .parameters.N                             ::: $samples/*_pifc/aux.json) \
#       <(parallel --keep-order jq .parameters.predicates                    ::: $samples/*_pifc/aux.json) \
#       <(parallel --keep-order jq 0                                         ::: $samples/*_pifc/aux.json) \
#       <(parallel --keep-order "cat {} ; echo"                              ::: $samples/*_pifc/parameter_count.json) \
#       >> $samples/contour.csv

