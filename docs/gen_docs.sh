#!/usr/bin/env bash
# Generate documentation for scripts
# Depends on docco.
docco -l linear --template docco_math.jst -c public/css/docco.css ../raw_data/process_data.py -o ./
docco -l linear --template docco_math.jst -c public/css/docco.css  ../analysis.py -o ./
docco --template docco_math.jst -c public/css/docco.css ../wip_code/bootstrap_model.py -o ./
