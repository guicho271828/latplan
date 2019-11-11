#!/bin/bash
# 

paste <(./performance.sh .ad.sd.recall .ad.sd.specificity .ad2.sd.recall .ad2.sd.specificity) \
      <(./rf-performance.sh accuracy-with-successors/PU-BINARY-80-100 | awk '{print $3" "$4}') \
      <(./rf-performance.sh accuracy/PU-BINARY-80-100                 | awk '{print $3" "$4}') \
      | column -t

paste <(./performance.sh .ad.sd.accuracy .ad2.sd.accuracy) \
      <(./rf-performance.sh accuracy-with-successors/PU-BINARY-80-100 | awk '{print $2}') \
      <(./rf-performance.sh accuracy/PU-BINARY-80-100                 | awk '{print $2}') \
      | column -t

      # <(./rf-performance.sh accuracy-with-successors/PU-BINARY-80-4  | awk '{print $4}') \
      # <(./rf-performance.sh accuracy-with-successors/PU-BINARY-80-7  | awk '{print $4}') \
      # <(./rf-performance.sh accuracy-with-successors/PU-BINARY-80-12 | awk '{print $4}') \
      # <(./rf-performance.sh accuracy-with-successors/PU-BINARY-80-18 | awk '{print $4}') \
      # <(./rf-performance.sh accuracy-with-successors/PU-BINARY-80-25 | awk '{print $4}') \
      # <(./rf-performance.sh accuracy-with-successors/PU-BINARY-80-50 | awk '{print $4}') \
      # <(./rf-performance.sh accuracy-with-successors/PU-BINARY-80-100 | awk '{print $4}') \
      # <(./rf-performance.sh accuracy-with-successors/PU-BINARY-80-200 | awk '{print $4}') \

paste <(./performance.sh .ad.sd.f .ad2.sd.f) \
      <(./rf-performance.sh accuracy-with-successors/PU-BINARY-80-100 | awk '{print $5}') \
      <(./rf-performance.sh accuracy/PU-BINARY-80-100                 | awk '{print $5}') \
      | column -t
