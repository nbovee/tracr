# This is a PowerShell script

. .\venv\Scripts\activate
# Run pipreqs against subdirectories of /src
pipreqs ./src --savepath requirements-all.txt --force
pipreqs ./src/experiment_design/ --savepath requirements-experiment.txt --force
pipreqs ./src/app_api/ --savepath requirements-api.txt --force
