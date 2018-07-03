#!/bin/bash

paste <(jq .variance.gaussian.test[2] samples/*/performance.json) \
      <(jq .inactive.both.test samples/*/performance.json) \
      <(jq .MSE.gaussian.test samples/*/performance.json) \
      <(dirname samples/*/performance.json) \
    | column -t
