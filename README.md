# RT change detection (Dégats & SAR)

Ce dossier contient les scripts de toutes les méthodes de détection de changement développés dans le cadre des R&T change detection.

Tous les fichiers de configuration sont documentées.

## Préparation des données

Les données utilisées par ce code doivent respecter plusieurs exigences.

Concernant les images, n'importe quel format d'image peut être utilisé tant qu'il est lisible par la librairie rasterio.

Le code gère la base de données en utilisant un csv fourni en entrée et spécifié dans le fichier de configuration. Pour chaque jeu de données, un csv doit doit contenir :

- une colonne 'in', elle contient le chemin relatif à l'image (chemin relatif au csv).
- une colonne 'out' contenant le chemin relatif au masque de VT (s'il existe).
- une colonne 'RLE' contenant une liste de RLE pour créer le masque de VT de l'image.
- une colonne 'classes' contenant la liste des classes présentes dans l'image.
- une colonne 'range_max' contenant la valeur maximale présente dans les images (utilisée pour les rescales).
- d'autres colonnes spécifiques à chacunes des R&T.

Ces csv utilisent un point-virgule comme séparateur.

## Containers

Les containers utilisables pour exécuter les codes se trouvent dans le dossier containers/.
La solution préconisée sur le HPC du CNES est celle utilisant l'image singularity.
Des explications plus détaillées se trouvent dans le read_me du dossier containers/.


## Entrainement 

Le script train.py permet d'entrainer un modèle.

Pour exécuter :
```
python train.py config_files/config_train.yaml -OF /somewhere
```

## Évaluation 

Le script eval.py permet d'évaluer un modèle.

Pour exécuter :
```
python eval.py config_files/config_eval.yaml -OF /somewhere
```