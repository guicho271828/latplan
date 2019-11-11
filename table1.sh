#!/bin/bash

paste <(./performance.sh .sae.MSE.vanilla.test) \
      <(echo actions ; ls -v samples/*/available_actions.csv | while read file ; do echo $(($(wc -l < $file)-1)) ; done) \
      <(./performance.sh .aae.mae.val                 | awk '{print $2}') \
      <(./rf-performance.sh accuracy/PU-BINARY-80-100 | awk '{print $6}') \
      | column -t

paste <(./performance.sh .sae.MSE.vanilla.test) \
      <(echo actions ; ls -v samples/*/available_actions.csv | while read file ; do echo $(($(wc -l < $file)-1)) ; done) \
      <(./performance.sh .aae.prob_bitwise.val        | awk '{print $2}') \
      <(./rf-performance.sh accuracy/PU-BINARY-80-100 | awk '{print $7}') \
      | column -t

