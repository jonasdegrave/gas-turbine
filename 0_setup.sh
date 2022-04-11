# Updates local repository with changes from remote repository;
git checkout main
git pull

# Installs conda to create coding environment;
pip install conda

# Creates conda environment;
conda create --name embraer python=3.10

# Activates project environment;
conda activate embraer

# Installs project requirements;
pip install -r requirements.txt