#!/bin/bash

FLAG="--execute"

pixi run quarto render 01_data_ingestion.ipynb $FLAG --to html
pixi run quarto render 02_data_exploration.ipynb $FLAG --to html
pixi run quarto render 03_preprocessing.ipynb $FLAG --to html
pixi run quarto render 04_feature_selection.ipynb $FLAG --to html