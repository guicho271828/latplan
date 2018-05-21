#!/bin/bash

find -name "*.gz" | parallel gunzip
