#!/bin/bash

parallel dsama dump-tsv {1}/action+ids.csv {1}/action+ids.fasl ::: samples/*
