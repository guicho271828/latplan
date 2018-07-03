#!/bin/bash

paste <(jq .inactive.both.test samples/*conv/performance.json) \
      <(jq .inactive.both.test samples/*convz/performance.json) \
      <(dirname samples/*convz/performance.json) \
    | column -t
