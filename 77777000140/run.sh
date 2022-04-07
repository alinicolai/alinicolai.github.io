#!/usr/bin/env bash


set -e

venv=$1
model=$2

if [ ! -d "$venv" ]; then
  echo "Virtualenv not found" > demo_failure.txt
  exit 0
else
  source $venv/bin/activate
fi

execute.py $model
