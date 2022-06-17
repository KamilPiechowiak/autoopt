#!/bin/bash

gsutil -m rsync -x ".*pt" -r gs://autoopt/ results