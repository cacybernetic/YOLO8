# YOLOv8 From Scratch - Les concepts pour débutants (Français)

Bienvenue ! Ce document explique, avec des mots simples, comment
fonctionnent le modèle et le pipeline d'entraînement de ce dépôt. Pas
besoin d'être un expert. Prenez votre temps, lisez les exemples, et
tout va s'éclairer.

> Le même document existe en anglais : `docs/en_concepts.md`.

---

## 1. C'est quoi, la détection d'objets ?

Imaginez que vous montrez une photo à un ami et que vous demandez :
« Qu'est-ce que tu vois, et où ? ». Il répond : « Une chaise ici, une
table là ». C'est exactement la détection d'objets : trouver **quoi**
(la classe) et **où** (la boîte englobante).

Une boîte englobante, c'est juste un rectangle. Au format YOLO, on la
décrit avec 4 nombres entre 0 et 1 :

```
<classe> <centre_x> <centre_y> <largeur> <hauteur>
```

Exemple : `0 0.5 0.5 0.4 0.4` veut dire « un objet de classe 0, au
centre de l'image, qui occupe 40% de la largeur et 40% de la
hauteur ».

**Question naïve : pourquoi des nombres entre 0 et 1 ?**
Parce qu'ainsi le label ne dépend pas de la taille de l'image. Le
même label marche pour une image 640x640 et pour une image 1920x1080.

## 2. Les trois parties du modèle

Le modèle est comme une petite usine avec trois ateliers :

1. **Backbone** (`src/yolov8/modules/backbone.py`)
   Il regarde l'image et en extrait des « features » : contours,
   textures, formes. Il travaille à trois niveaux de zoom appelés P3,
   P4 et P5. P3 voit les petits détails, P5 voit la vue d'ensemble.

2. **Neck** (`src/yolov8/modules/neck.py`)
   Il mélange les trois niveaux de zoom entre eux, pour que chaque
   niveau sache ce que les autres ont vu. C'est le PAN-FPN. Pensez à
   trois amis qui partagent leurs notes avant un examen.

3. **Head** (`src/yolov8/modules/head.py`)
   Elle prend la décision finale : pour plein de petites positions
   sur l'image (les « anchors »), elle prédit une boîte et un score
   par classe.

**Question : ça veut dire quoi, « anchor-free » ?**
Les anciens YOLO utilisaient des formes de boîtes prédéfinies (les
anchors) qu'ils ajustaient. YOLOv8 n'en a pas besoin : chaque
position prédit directement la distance vers les quatre côtés de la
boîte. Moins de réglages, entraînement plus simple.

### L'astuce spéciale : la DFL

Au lieu de prédire un seul nombre pour « distance au bord gauche »,
le modèle prédit une petite distribution de probabilités sur 16
valeurs possibles, et on prend la moyenne. C'est l'idée de la
Distribution Focal Loss (DFL). Les boîtes deviennent plus précises,
comme répondre « la distance est probablement entre 3 et 4, plutôt
proche de 4 » au lieu de juste « 4 ».

## 3. La fonction de perte (comment le modèle apprend)

La perte (loss) est un nombre qui dit « à quel point le modèle se
trompe en ce moment ». S'entraîner, c'est faire baisser ce nombre.
Notre perte a trois morceaux (`src/yolov8/lossfn.py`) :

- **Perte de boîte (CIoU)** : les boîtes prédites sont-elles au bon
  endroit, à la bonne taille ?
- **Perte de classification (BCE)** : les classes sont-elles bonnes ?
- **Perte DFL** : les distributions de distances sont-elles nettes et
  correctes ?

Avant de calculer la perte, il faut décider quelles prédictions
correspondent à quels objets réels. C'est le travail de l'**assigneur
TAL** : pour chaque objet réel, il choisit les 10 meilleures positions
candidates, selon le score de classe ET le recouvrement de boîte.

## 4. Le pipeline de données

### Sources : dossier ou zip

Un split de dataset peut être un dossier ou une archive `.zip`. Dans
les deux cas, il doit contenir :

```
images/    les photos
labels/    un fichier .txt par photo
data.yaml  la liste des classes (champ `names:`)
```

### Le scan et son cache

Avant l'entraînement, on vérifie chaque fichier de label et chaque
image. Les échantillons cassés (label manquant, image corrompue,
valeurs impossibles) sont écartés avec un avertissement. Le résultat
est sauvegardé à côté du dataset dans `train.cache.json` : le
prochain démarrage est instantané.

**Question : et si je modifie mon dataset ?**
Le cache garde une empreinte du dataset (taille, date). Si le dataset
a changé, le scan se relance tout seul.

### Le split de validation

Il n'y a pas de dossier `val/` séparé. La valeur `val_prob` de la
config (0.5 par défaut) prélève une fraction du set de **test** pour
la validation de chaque epoch. L'évaluation finale, elle, tourne sur
le test complet à la fin de l'entraînement.

### HDF5 : des données précuites

Décoder les images et les augmenter coûte du CPU à chaque step. Avec
`buildh5ds`, vous « précuisez » le dataset dans des fichiers
`train.h5` et `test.h5` : les échantillons sont déjà redimensionnés
(et éventuellement augmentés). L'entraînement avec `use_hdf5: true`
ne fait plus que lire des tableaux.

### Les augmentations

Pendant l'entraînement, on modifie les images au hasard pour que le
modèle ne voie jamais deux fois exactement la même photo
(`src/yolov8/dataset/augment.py`) :

- **Mosaïque** : coller 4 images sur un grand canevas, puis recadrer.
  Le modèle voit des objets à plein d'échelles et de positions.
- **MixUp** : fondre deux images l'une dans l'autre.
- **Jitter HSV** : changer un peu les couleurs.
- **Flips, rotations, échelle** : déplacer les choses.
- **Cutout, flou, bruit, niveaux de gris** : compliquer la vie du
  modèle exprès.

Vers la fin de l'entraînement (`close_mosaic`), mosaïque et mixup
sont coupés pour finir sur des images réalistes.

## 5. L'entraînement tolérant aux pannes

C'est le super-pouvoir de ce pipeline. Imaginez une coupure de
courant après 3 jours d'entraînement. Avec la plupart des projets,
vous perdez l'epoch en cours. Ici, vous perdez au pire quelques
minutes.

### Le DataLoaderAdapter

`src/yolov8/dataset/adapter.py` enveloppe le DataLoader de PyTorch.
Il retient trois nombres : la graine (seed), l'epoch, et le nombre de
batches déjà consommés dans l'epoch en cours. L'ordre de mélange ne
dépend que de (seed + epoch) : après un crash, on peut reconstruire
exactement le même ordre et **sauter** les batches déjà faits. Chaque
échantillon est quand même vu exactement une fois par epoch.

### Les checkpoints

Tous les `ckpt_step` pas d'optimiseur, un instantané complet est
écrit : poids du modèle, optimiseur, EMA, scaler AMP, la position des
trois loaders, les compteurs de pertes partiels, l'accumulateur de
métriques partiel, et même l'état des générateurs aléatoires. Les
fichiers s'appellent :

```
checkpoint_e0001c0012.pth   -> epoch 1, pas 12
```

Les vieux fichiers sont supprimés automatiquement (`max_checkpoint`).
Au redémarrage avec `resume: true`, le trainer recharge le checkpoint
le plus récent et continue exactement où il était, même au milieu
d'une passe de validation.

## 6. La boucle d'entraînement, pas à pas

Une epoch ressemble à ça (`src/yolov8/training/trainer.py`) :

1. Pour chaque batch : forward, perte, backward.
2. Tous les `grad_accum` batches : un pas d'optimiseur (cela simule
   un batch plus gros), puis la mise à jour de l'EMA.
3. Tous les `ckpt_step` pas d'optimiseur : un checkpoint.
4. Après le dernier batch : validation sur le split val, tableau de
   métriques, courbe d'historique, `last.pt`, et `best.pt` si la
   métrique choisie s'est améliorée.

Après la dernière epoch, une évaluation finale tourne sur le test
**complet** et écrit `test_results.csv`.

### Le warmup du learning rate

Au début, le learning rate est minuscule et grandit pendant environ
3 epochs. Pourquoi ? Un modèle tout neuf fait des prédictions au
hasard ; un grand learning rate à ce moment pousserait les poids dans
des directions aléatoires. Détail amusant : les biais démarrent avec
un GRAND learning rate (0.1) qui redescend ; les biais ne coûtent pas
cher à bouger et aident le modèle à se calibrer vite.

### L'EMA (moyenne mobile exponentielle)

On garde une deuxième copie lissée des poids : à chaque pas,
`ema = 0.9999 * ema + 0.0001 * modele`. Cette copie lissée est moins
bruitée et gagne souvent 1 à 2 points de mAP. La validation et
`best.pt` utilisent les poids EMA.

## 7. Les métriques

- **Précision** : parmi mes alarmes, combien étaient vraies ? (peu de
  fausses alertes = précision haute)
- **Rappel** : parmi les objets réels, combien ai-je trouvés ? (peu
  d'oublis = rappel haut)
- **F1** : l'équilibre des deux.
- **AP@0.5** : l'aire sous la courbe précision-rappel, quand une
  boîte est « bonne » si elle recouvre la vraie boîte à au moins 50%
  (IoU 0.5).
- **mAP@0.5:0.95** : pareil, mais moyenné sur 10 seuils de
  recouvrement de plus en plus stricts. C'est le chiffre COCO
  principal.

Le programme d'évaluation dessine aussi les courbes précision-rappel,
la courbe F1-confiance (avec le meilleur seuil de confiance marqué)
et la matrice de confusion.

## 8. Où vont mes fichiers ?

```
runs/
  mon_run/
    train/            premier entraînement
    train2/           deuxième, et ainsi de suite
      weights/best.pt et last.pt
      checkpoints/    les instantanés de tolérance aux pannes
      plotes/         training_history.png
      logs/           un fichier de log par démarrage
      history.csv     une ligne par epoch
      config_used.yaml
      test_results.csv
    eval/, eval2/     les évaluations
      results.csv, per_class.csv, plotes/, renders/, logs/
```

## 9. Les cinq programmes

| Commande     | Rôle                                            |
|--------------|--------------------------------------------------|
| `buildh5ds`  | Précuire un dataset en fichiers HDF5             |
| `trainyolo8` | Entraîner (train + val + test final)             |
| `evalyolo8`  | Évaluation complète avec courbes et matrices     |
| `exportw`    | Exporter le modèle en ONNX                       |
| `runyolo8`   | Inférence ONNX autonome sur une image            |
| `ftyolo8`    | Préparer un checkpoint pour de nouvelles classes |

Chacun prend `--config chemin/vers/fichier.yaml`. Des configs prêtes
à l'emploi se trouvent dans `cpu/configs/` et `gpu/configs/`.

Bon entraînement !
