# Updates local repository with changes from remote repository;
git checkout main
git pull

# Installs conda to create coding environment;
pip install conda

# Creates conda environment;
conda create --name embraer python=3.10

# Activates project environment;
## OBS: Isso aqui dá pau às vezes.. nesse caso, comentar essa linha e digitar ela no terminal manualmente!
conda activate embraer

# Installs project requirements;
pip install -r requirements.txt

# Configure git user.name and user.email;

echo ""
echo "Escreva seu 'Nome Sobrenome':"
read username
git config user.name "$username"

echo ""
echo "Escreva seu 'email':"
read email
git config user.email "$email"

echo ""
echo "Ambiente Python e GIT configurados. Pronto!"
echo ""