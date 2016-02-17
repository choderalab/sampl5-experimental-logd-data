#!/usr/bin/env bash
# Generate documentation for scripts
# Depends on docco.
docco -l linear ../raw_data/process_data.py -o ./
docco -l linear ../analysis.py -o ./
