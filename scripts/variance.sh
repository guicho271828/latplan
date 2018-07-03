#!/bin/bash

paste <(jq .variance.gaussian.test[2] samples/*conv/performance.json) \
      <(jq .variance.gaussian.test[2] samples/*convz/performance.json) \
      <(dirname samples/*convz/performance.json) \
    | column -t
