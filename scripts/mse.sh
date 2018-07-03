#!/bin/bash



paste <(jq .MSE.gaussian.test samples/*conv/performance.json) \
      <(jq .MSE.gaussian.test samples/*convz/performance.json) \
      <(dirname samples/*convz/performance.json) \
    | column -t

