#!/bin/bash

FLAG="--execute"

pixi run quarto render 01_data_ingestion.ipynb $FLAG --to html

