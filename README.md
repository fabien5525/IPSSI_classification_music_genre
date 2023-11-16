# IPSSI - Classificaton de genre de musique

Initialisation des dossiers
~~~
cd notebooks

mkdir clean_data
mkdir data
mkdir models
mkdr temp
mkdir data/temp
mkdir data/genre_3s
~~~

Lancement du docker
~~~
docker compose up -d
~~~

Installation ffmepg
~~~
docker compose exec tensorflow bash -c "apt-get install ffmpeg -y" 
~~~

dataset de base à mettre dans le dossier data
https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification/

