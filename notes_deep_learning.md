# Continuité du machine learning

Travail sur **TensorFlow** afin de construire des réseaux de neurones. 
Cet outil est dédié au deep learning.

Il est utilisable pour entraîner des modèles se trouvant sur le web (NodeJS), sur les
applications mobiles, ou sur des environnements de production (cloud, serveur...)

On va l'utiliser en local.

Pour l'installer : `pip install tensorflow`

**Keras** a été intégré à TensorFlow : il s'agit d'une API qui permet d'effectuer des
opérations de hauts niveaux. Avant cette intégration, TensorFlow ne pouvait effectuer
des opérations que de bas niveau.

Il est nécessaire de bénéficier d'un GPU pour faire du traitement d'images en grande
quantité, pour le deep learning. Avoir une carte graphique séparé du processeur
(NVIDIA, AMD, etc...donc hors-chipset), est important ici.

On peut passer par _colaboratory_ afin de bénéficier de notebooks qui s'exécutent via
des machines virtuelles offertes par google, offrant des ressources systèmes plus
poussées. _colaboratory_ est intéressant si on ne possède pas de carte graphique sur sa
machine.

Il faut préciser à côté _colaboratory_ qu'on souhaite utiliser le GPU qui est offert par la VM : `Exécution > Modifier le type d'exécution >
mettre la valeur à "GPU"`

Pour le moment on va étudier un jeu de données classique (dataframe via pandas) :
artificial_generator.xlsx Ce fichier contient deux jeux de données (2 feuilles :
data2 et data4) : tous les deux évoquent des problèmes de classification.

- X1 et X2 font référence aux données entrantes
- Y fait référence à l'output désiré / au label

Étant donné que nous avons deux feuilles dans ce fichier excel, la syntaxe pour lire
une feuille pour ensuite la convertir en dataframe est la suivante :

```py
data4 = pd.read_excel('artificial_generator.xlsx', sheet_name="data4")
```
Comme on a affaire à un problème de classification, il faut labelliser / numériser les
labels : en l'occurrence ici la colonne Y :

```py
le = LabelEncoder()
data2['Y'] = le.fit_transform(data2['Y'])
data4['Y'] = le.fit_transform(data4['Y'])
```
On peut observer la répartition des données en fonction de leur label via un **un nuage
de point**.

```py
# Nous spécifions X1 en abscisse, X2 en ordonnée, et ns utilisons Y pour distinguer
les points.

plt.figure(figsize = ( 20 , 10 ))
plt.subplot( 2 , 1 , 1 )
plt.scatter(data2['X1'], data2['X2'], c = data2['Y'], edgecolors='red')

plt.title("data2") # affichage du graphique pour la feuille "data2"
plt.subplot( 2 , 1 , 2 )
plt.scatter(data4['X1'], data4['X2'], c = data4['Y'], edgecolors='red')
plt.title('data4') # affichage du graphique pour la feuille "data4"
```
Les réseaux de neurones permettent de répondre à des problèmes de classification
complexes. Lorsque la répartition du jeu de données est trop complexe (répartition en
écocentrique)...

# Construction d'un réseau de neurones

Définition des termes :

- Couche d'entrée : les neurones qui recoivent l'information, on a x neurones
pour x colonnes dans le dataframe Chaque neurone possède un poids ( w ) qui est
réparti aléatoirement.

- Fonction d'activation (= fonction déterministe) : intermédiaire entre les
données en entrée et en sortie, cet intermédiaire détermine les données qui
doivent être envoyés en d'une couche à une autre ou pas. Il s'agit d'une fonction mathématique
(tangente, sigmoïd...). La fonction d'activation est très utile pour résoudre des problèmes non-linéaires, comme un problème de classification.

- Couche cachée (= couche intermédiaire) : des neurones qui recoivent
l'information transmise par la couche d'entrée. Il est possible rajouter autant
de couches cachées qu'on veut : plus il y a des couches cachées, plus on parle
de réseau de neurone profond ( deep ). Dans le cas contraire, on parle de réseau
de neurones creux ( shallow )

- Couche de sortie : couche de neurones finale qui va envoyer l'output final de
la donnée envoyée

**Mécanisme de calcul :**

- On reçoit les données en entrée
- On applique l'opération x*w
- La fonction d'activation est ensuite appliquée à ce résultat
- On passe à la couche de neurone suivante

Comme on a affaire à un **apprentissage supervisé** on peut faire la différence entre le résultat qui est **souhaité** et le résultat qui est **calculé** par le réseau de neurones.
Cette différence est calculé par le réseau de neurones à la fin, et le résultat de cette différence sera propagée ( _backdrop_ : rétro-propagation) sur les poids des
neurones : certains seront mis à jour, d'autres non. 

Le réseau de neurones permet donc, grâce à cette mise à jour, d'apprendre de ses erreurs, donc de progresser.

Il y a trois façons de construire un réseau de neurones : **séquentielle** , **fonctionnelle** ,
et par **sous-classes**.

Construction d'un réseau de neurones fonctionnel (permet de réaliser des modèles plus
complexes):

```py
# Exemple de création d'un NN avec `Functional API`
from tensorflow.keras.models import Model

from tensorflow.keras.layers import Input, Dense

# Couche d'entrée avec deux neurones
inputs = Input(shape=( 2 ,))
# inputs.get_shape()
# inputs.name
# Couches cachées comme fonction d'activation 'relu' :
x = Dense( 3 , activation='relu')(inputs)
# la variable x sera appliqué à la couche cachée suivante :
x = Dense( 3 , activation='relu')(x)
```
```
# Comme il s'agit de la dernière instance Dense, il s'agit de la couche de sortie
outputs = Dense( 3 , activation='softmax')(x)

# Création du modèle
model = Model(inputs=inputs, outputs=outputs)
```
Construction d'un réseau de neurones de manière séquentielle (plus simple, mais plus
limitée):

```py
model = Sequential([
Dense( 3 , activation='relu', input_shape=( 4 ,)), # couche d'entrée car possède
input_shape
Dense( 10 , activation='relu'),
Dense( 3 , activation='softmax')
]) # on n'écrit pas la donnée qui est insérée dans chaque couche
```
Répartition du réseau de neurones avec la fonction get_weights() du modèle :

```
[array([[-0.14119095, 0.0544771 , -0.0071274 ], # poids associés au neurone x
[-0.9484316 , -0.31077462, 0.204157 ]], dtype=float32), # poids associés au
neurone x
# tableau de biais : ici le tableau du biais est zéro car la modèle n'a pas encore été
entraîné
array([0., 0., 0.], dtype=float32),
# pour rappel, le biais désigne le taux d'erreur entre 0 et 1
array([[-0.8759022 ],
[-0.2409901 ],
[-0.40759063]], dtype=float32),
array([0.], dtype=float32)]
```
Exécution de la fonction get_summary() :

```
Layer (type) Output Shape Param #
=================================================================
dense_10 (Dense) (None, 3) 9
_________________________________________________________________
dense_11 (Dense) (None, 1) 4
=================================================================
Total params: 13 # nous avons 13 paramètres, donc 13 neurones dans notre réseau
Trainable params: 13
Non-trainable params: 0
```

Le `get_summary()` peut avoir une représentation différente si la construction du
réseau de neurones est différente. Ici pour un réseau de neurones fonctionnel :

```py
# Exemple de création d'un NN avec `Functional API`
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# Couche d'entrée
inputs = Input(shape=( 2 ,))
# inputs.get_shape()
# inputs.name

# Couches cachées :
x = Dense( 3 , activation='relu')(inputs)
x = Dense( 3 , activation='relu')(x)

# Couche de sortie
outputs = Dense( 3 , activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs)
```
On obtient la représentation suivante, qui est un peu plus aisée à lire :

```
# on peut déterminer la couche d'entrée grâce à InputLayer
_________________________________________________________________
Layer (type) Output Shape Param #
=================================================================
input_4 (InputLayer) [(None, 2)] 0
_________________________________________________________________
dense_11 (Dense) (None, 3) 9
_________________________________________________________________
dense_12 (Dense) (None, 3) 12
_________________________________________________________________
dense_13 (Dense) (None, 3) 12
=================================================================
Total params: 33
Trainable params: 33
Non-trainable params: 0
```
Il est possible qu'une machine puisse partir sur le même réseau de neurones en
**chargeant un fichier contenant les poids de ce réseau de neurones**.

Pour sauvegarder les poids : <model>.save_weights('path')

Pour compiler le modèle afin qu'il soit prêt à utilisation en production :

```py
# optmizer = manière d'actualiser les poids après backdrop
# loss = calculer l'erreur : ici il s'agit d'une classification binaire ->
binary_crossentropy
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
# construction du modèle
# epochs = nombre d'itérations

model.fit(data,labels,epochs= 10 ,batch_size= 32 )
model.predict(data)
```
# Réseau de neurones à convolution (traitement d'images)

La **convolution** en traitement d'image, est une technique permettant d'extraire des informations d'une image.
Il s'agit d'une opération mathématique, effectuée **sur la matrice de l'image**.
Une nouvelle matrice est donnée en résultat, et cette matrice permet d'effectuer des opérations pertinentes pour la récupération d'informations d'une image : détection des contours, amélioration de la netteté...

**Reprise du TP cats VS Dogs**

Sources :

[Notebook](https://colab.research.google.com/github/google/eng-edu/blob/master/ml/pc/exercises/image_classification_part1.ipynb)

[Jeu de données](https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip)

[S'est inspiré de ce tutoriel](https://deeplizard.com/learn/video/LhEMXbjGV_4)

```py
import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.image import imread
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, Model
import zipfile

# lecture des différentes images
# Une boucle pr afficher plusieurs images dans une figure

for i in range(9):
    plt.subplot(3, 3, i+1)
    filename = 'train/' + 'cats/' + 'cat.' + str(i) + '.jpg'
    image = imread(filename)
    plt.imshow(image)
    plt.title(image.shape)

# S : On constate que les images sont en mode paysage, mode portrait et de différentes tailles  
# => Ceci peut poser des Pb pour la perfermonce du classifieur. Il nous faut un classifieur robuste !

# Manip avec PIL : pkg Python de réf pr le traitement d'image
# dog1 = Image.open('train/dogs/dog.1.jpg')
# dog1.show()
# dog1.size
# dog1.mode
# dog1.format
# np.array(dog1)
# imread('train/dogs/dog.1.jpg')

# traitement de l'image directement avec keras
dog1 = tf.keras.preprocessing.image.load_img('train/dogs/dog.1.jpg')
dog1_ = tf.keras.preprocessing.image.img_to_array(dog1)

```

Pour faciliter le traitement, on met les images à la même échelle :

```py
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
```

Puis ensuite, pour les groupes d'image d'entraînement et de test, on indique qu'on souhaite obtenir des images de 150x150, toujours dans l'optique de faciliter le traitement des images.
**Plus l'homogénéité est grande**, mieux le traitement sera performant.

```py
train_gen = train_datagen.flow_from_directory(
    'train', # nom du dossier
    target_size=(150, 150), 
    batch_size=20, # on précise qu'on souhaite lire les images par groupe de 20 (batch)
    class_mode='binary' # il s'agit de résoudre un problème de classification binaire
)

val_gen = val_datagen.flow_from_directory(
    'validation', 
    target_size=(150, 150), 
    batch_size=20, 
    class_mode='binary'
)
```

Voici une fonction utilitaire qui nous sera utile pour afficher les différentes images :

```py
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 10, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
```

Les objets retournés par `ImageDataGenerator` sont de type `tensorflow.python.keras.preprocessing.image.DirectoryIterator`.
Pour pouvoir lire les images, il faut invoquer la fonction `next` :

```py
# Visualiser les images resizés et rescalés
# On a déjà les labels de chaque image.
# En effet, un label = le dossier contenant l'image
imgs, labels = next(val_gen)
imgs # affichera 20 matrices numpy (une matrice = une image)
```

Construction du CNN (méthode fonctionnelle -> on ne passe pas par `add` qui est séquentielle) :

```py
from tensorflow.keras import layers
from tensorflow.keras import Model

# signifie qu'on souhaite insérer des images qui font 150 x 150
# et qui est composé d'une matrice de 3 couleurs (RGB) 
img_input = layers.input(shape=(150,150,3))

# Couche de convolution appliquant 16 filtres sous une matrice de 3x3
# Cette matrice balaye l'image
# La fonction d'activation est relu
x = layers.Conv2D(16, 3, activation='relu')(img_input)
# Couche pour réaliser un MaxPooling : opération permettant de réduire
# la taille de la matrice de l'image tout en récupérant un maximum d'infos
# ici on redimensionne la matrice en une 2x2
x = layers.MaxPooling2D(2)(x)

# Couche de convolution appliquant 16 filtres sous une matrice de 3x3
x = layers.Conv2D(32, 3, activation='relu')(x)
x = layers.MaxPooling2D(2)(x)

# Couche de convolution appliquant 64 filtres sous une matrice de 3x3
x = layers.Conv2D(64, 3, activation='relu')(x)
x = layers.MaxPooling2D(2)(x)
# Aplatissement de x pour préparer la sortie
x = layers.Flatten()(x)
# Couche Dense pour le output = pleinement connecté aux neurones la couche précédente
x = layers.Dense(512, activation='relu')(x)

outputs = layers.Dense(1, activation='sigmoid')(x)
```

Instanciation du modèle :

```py
model = layers.models.Model(inputs=inputs, outputs=outputs)
models.summary() # résumé du modèle
```

Compilation du modèle :

```py
# metrics = accuracy
model.compile(loss='binary_crossentropy', metrics=['acc'], optimizer='rmsprop')
```

Entraînement :

```py
history = model.fit_generator(train_gen,
					# combien d'images doivent être traitées à la fois ?
                    # opération : batch_size * steps = nombre d'images
					steps_per_epoch=100, 
                    validation_data=val_gen, # préciser l'échantillon de test
                    # le nombre d'étapes pour la validation
                    # opération : batch_size * steps = nombre d'images
                    validation_steps=50,
                    verbose=2)
```

Le code suivant permet de visualiser les différentes transformations des images après application de chaque couche (maxpooling2d, conv2d):

```py
import numpy as np
import random
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Let's define a new Model that will take an image as input, and will output
# intermediate representations for all layers in the previous model after
# the first.
successive_outputs = [layer.output for layer in model.layers[1:]]
visualization_model = Model(img_input, successive_outputs)

# Let's prepare a random input image of a cat or dog from the training set.
cat_img_files = [os.path.join(train_cats_dir, f) for f in train_cat_fnames]
dog_img_files = [os.path.join(train_dogs_dir, f) for f in train_dog_fnames]
img_path = random.choice(cat_img_files + dog_img_files)

img = load_img(img_path, target_size=(150, 150))  # this is a PIL image
x = img_to_array(img)  # Numpy array with shape (150, 150, 3)
x = x.reshape((1,) + x.shape)  # Numpy array with shape (1, 150, 150, 3)

# Rescale by 1/255
x /= 255

# Let's run our image through our network, thus obtaining all
# intermediate representations for this image.
successive_feature_maps = visualization_model.predict(x)

# These are the names of the layers, so can have them as part of our plot
layer_names = [layer.name for layer in model.layers]

# Now let's display our representations
for layer_name, feature_map in zip(layer_names, successive_feature_maps):
  if len(feature_map.shape) == 4:
    # Just do this for the conv / maxpool layers, not the fully-connected layers
    n_features = feature_map.shape[-1]  # number of features in feature map
    # The feature map has shape (1, size, size, n_features)
    size = feature_map.shape[1]
    # We will tile our images in this matrix
    display_grid = np.zeros((size, size * n_features))
    for i in range(n_features):
      # Postprocess the feature to make it visually palatable
      x = feature_map[0, :, :, i]
      x -= x.mean()
      x /= x.std()
      x *= 64
      x += 128
      x = np.clip(x, 0, 255).astype('uint8')
      # We'll tile each filter into this big horizontal grid
      display_grid[:, i * size : (i + 1) * size] = x
    # Display the grid
    scale = 20. / n_features
    plt.figure(figsize=(scale * n_features, scale))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
```

### Diagnostic du modèle à l'aide de courbes

Pour obtenir les performances du modèle, il est souvent nécessaire de passer par une représentation en courbes de l'évolution en précision, ainsi que de l'évolution de la perte d'information / erreur/

```py
# Retrieve a list of accuracy results on training and validation data
# sets for each training epoch
acc = history.history['acc']
val_acc = history.history['val_acc']

# Retrieve a list of list results on training and validation data
# sets for each training epoch
loss = history.history['loss']
val_loss = history.history['val_loss']

# Get number of epochs
epochs = range(len(acc))

# Plot training and validation accuracy per epoch
plt.plot(epochs, acc)
plt.plot(epochs, val_acc)
plt.title('Training and validation accuracy')

plt.figure()

# Plot training and validation loss per epoch
plt.plot(epochs, loss)
plt.plot(epochs, val_loss)
plt.title('Training and validation loss')
```

Pour que le modèle soit valide, il faut que pour que chaque graphique, on ait une **convergence** de courbes, et pas une divergence.
S'il y a divergence, on a affaire à un **problème de surapprentissage**.

[Notebook](https://colab.research.google.com/github/google/eng-edu/blob/master/ml/pc/exercises/image_classification_part2.ipynb)

**Surapprentissage** (_overfitting_) : ce problème se produit lorsqu'un modèle ne possède pas suffisamment de données pour pouvoir mémoriser les différentes caractéristiques qui distinguent une classe d'une autre.
Ce défaut amène de mauvais résultats au moment de l'évaluation.

Deux méthodes pour prévenir ce souci :

-  **Injecter beaucoup plus de données**, mais comment faire s'il nous n'est pas possible d'en fournir davantage ? Il faut passer par une **augmentation des données** (_Data augmentation_) existantes.
Les "nouvelles données" seront en réalité des copies des données existantes, **mais avec des transformations aléatoires** (rotation, zoom...).
Comme il y a des différences dans les copies, le moèdle considèrera qu'il s'agit de données nouvelles.

- **Régularisation par abandon** (_dropout_) : on choisit arbitrairement les neurones qui ne devront pas traiter la donnée --> alléger le réseau de neurones. Moins on exécute de neurones, plus les données fournies seront suffisantes.

**Application de _Data augmentation_ :**


```py
# Data augmentation sur les données d'apprentissage

train_datagen = ImageDataGenerator(
    rescale=1./255, # mise à échelle
    rotation_range=40, # intervalle de rotation aléatoire en degrés
    width_shift_range=0.2, # redimensionnement de l'image (20 % de la largeur totale ici)
    height_shift_range=0.2, # redimensionnement de l'image (20 % de la hauteur totale ici)
    shear_range=0.2, # transvection aléatoire de 20 % de l'image ("skew", exemple : rectangle + skew = parallélogramme)
    zoom_range=0.2, # zoom aléatoire de l'image à 20%
    horizontal_flip=True, # mettre l'image à l'horizontale ?
    fill_mode='nearest' # mode de reproduction des pixels l'image ('nearest' : reproduction fidèle de l'originale
    )

# On n'applique pas la data augmentation sur nos données de test !
# Après tout le modèle ne s'entraîne pas avec, mais s'évalue
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir, 
        target_size=(150, 150), 
        batch_size=20,
        class_mode='binary')


validation_generator = val_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')
```

On peut ensuite refaire l'apprentissage du modèle à partir de ce nouveau jeu de données :

```py
history = model.fit_generator(train_generator,
					# combien d'images doivent être traitées à la fois ?
                    # opération : batch_size * steps = nombre d'images
					steps_per_epoch=100, 
                    validation_data=validation_generator, # préciser l'échantillon de test
                    # le nombre d'étapes pour la validation
                    # opération : batch_size * steps = nombre d'images
                    validation_steps=50,
                    verbose=2)

```

Il faudra ensuite observer si la performance du modèle est concluant ou pas (toujours avec les plots).

**Application de dropout :**

Cela s'applique juste avant la création de **la couche de l'output** du CNN :

```py
# 0.5 : probabilité qu'un neurone à l'entrée soit ignoré
# le neurone est choisi aléatoirement, à chaque étape de l'apprentissage (steps_per_epoch)
# cette valeur est totalement arbitraire, et il faut déterminer le bon taux de dropout
x = layers.output(0.5)(x)
```

### Évaluer le modèle

Permet d'obtenir un score du modèle

```py
# evaluate_generator est obsolète : préférer evaluate
# renvoie un tableau de deux éléments : loss, accuracy
model.evaluate_generator(validation_generator)

```

### Prédictions du modèle

On peut vérifier si les résultats que notre modèle nous a renvoyé pour nos données de tests, sont corrects ou non.

```py
# renvoie plusieurs tableaux de numpy à dimension 1
# chaque élément représente une probabilité (si supérieur à 0.5, appartient à la classe / label 1, sinon 0)
y_pred = model.predict_generator(validation_generator)

#on peut améliorer l'affichage en le convertissant y_pred en Series
import pandas as pd

# flatten : fusionner les différents tableaux en un seul
pd.Series(y_pred.flatten()).head()

```

On peut récupérer les différents labels toujours grâce à une instance de `tensorflow.python.keras.preprocessing.image.DirectoryIterator` 
:

```py
imgs, labels = next(validation_generator)
labels
```

Et on peut comparer le contenu de chaque label, avec le contenu de chaque image (pouvoir distinguer si la classe 1 représente celle des chiens ou des chats).

## Matrice de confusion

Utile pour avoir une répartition des différentes classes de notre prédiction.
En second paramètre de `confusion_matrix`, il faut passer un tableau d'entiers binaire (0 ou 1).
Cependant, nos données de prédiction sont en décimales.

Il faut convertir le tout en des valeurs entières :

```py

y_pred = (y_pred > 0.5).astype('int32')
from sklearn.metrics import confusion_matrix

# validation_generator.classes : tableau représentant chaque image par sa classe
conf_mat = confusion_matrix(validation_generator.classes, y_pred)
```

On peut styliser la matrice à l'aide de `mlxtend` (l'installer si pas déjà fait) :

```py
from mlxtend.plotting import plot_confusion_matrix

plot_confusion_matrix(conf_mat=conf_mat, colorbar=True, show_absolute=True, show_normed=True, class_names=['dog','cats'])
```
## Transfer Learning : réutiliser un modèle en production

Le principe est de pouvoir **réutiliser un modèle** réseau de neurones, qui est **déjà entraîné** sans forcément en bâtir un.

Des exemples de réseaux de neurones fameux pour leur réutilisation :

- AlexNet
- VGG16

Chacun a ses avantages et ses inconvénients.

VGG16 utilise un algorithme de réseau de neurones dont **la profondeur / le nombre de couches** est très élevée, et de grosses quantités de données lui ont été passé.
Son apprentissage a duré **2 semaines, mais avec de très bons résultats en retour !**

Tensorflow possède dans ses librairies, une **implémentation du VGG16** :

```py
import tensorflow as tf

tf.keras.applications.vgg16

# décode la prédiction d'un jeu de données provenant de ImageNet (base de données d'images publique)
# accepte en premier argument une matrice
# création d'un tableau de 1000 valeurs : np.arange(1000)
# création d'une matrice à une dimension, et avec une ligne de longueur 1000
tf.keras.applications.vgg16.decode_predictions(np.arange(1000).reshape(1, 1000), top=1000)

# attribue à chaque groupe d'image existant sur ImageNet, un entier (entre 1 et 1000)
```

### Importer le modèle VGG16 + instanciation

```py
from tensorflow.keras.applications.vgg16 import VGG16
model = VGG16()

# l'importation du VGG16 télécharge les poids du modèle (528 MB) 
# dans le répertoire .keras/models
```

On peut avoir une représentation du VGG16 via la fonction `summary()` :

```py
model.summary()
```

[Voici une représentation schématisée par couche](https://neurohive.io/wp-content/uploads/2018/11/vgg16.png)

L'exécution de la fonction nous affichera environ **140 millions de paramètres** (soit 140 millions de poids).

### Préprocessing pour format VGG

Afin d'appliquer de la classification d'images à partir du VGG16, il faut convertir les images **en format VGG**.


```py
from IPython.display import Image
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.vgg16  import preprocess_import

# Chargement et redimension (VGG16 étant entraîné sur des images 224 x 224)
image = load_img('chemin vers image', target_size(224,224))

# Préparation en VGG : soustraction de la moyenne des pixels
image = preprocess_input(image)
image.shape # affichera (1, 224, 224, 3)

# pour afficher l'image préprocessée

tf.keras.preprocessing.image_to_array(img)

```

### Prédiction

Maintenant que l'image a un format compatible, on peut demander à VGG, de deviner la classe.

```py
Renvoie un tableau de probabilités
proba = model.predict(image_pr)
proba

"""
   On peut convertir le tableau de probabilités
   en un dataframe pour que cela soit plus visible
"""
pd.Dataframe(proba).T.sort_values(0, ascending=False)
```

### Reconnaissance de l'image

On peut commencer à déterminer la classe de l'image grâce à la probabilité.

```py
from tensorflow.keras.applications.vgg16 import decode_predictions
label = decode_predictions(proba)
label # renverra un tableau contenant des labels se rapprochant de la probabilité
```

## Adapter un modèle de production à notre besoin

Le VGG16 contient un gros nombre de paramètres (140 millions), car il est prêt à travailler sur **1000 classes (des millions d'images)**.

Pour éviter de gaspiller des ressources inutiles, il faut justement **adapter la structure de ce modèle à  notre besoin** -> récupérer uniquement les labels concernant les chats et chiens ici.

```py
target = decode_predictions(np.arrage(1000).reshape(1,1000), top=1000) # récupération de tous les labels
dog = target.label.str.contains('_dog')

target[dog] # labels contenant "dog"

cat = target.label.str.contains('_cat')
```

Cependant cette façon de filtrer les labels n'est pas la meilleure, car il contient un nombre restreint de races de chiens et de chats.

**Il faut aller plus loin dans l'adaptation du modèle.**

L'idée est de récupérer les couches du modèle, dans un nouveau modèle :

```py

cnn = Sequential()
for layer in model.layers[:-1]:
    cnn.add(layer)
    
cnn.summary()

"""
   On ne récupère pas la dernière couche (output)
   Car l'output envoie des résultats pour 1000 classes
   Nous, comme nous voulons savoir s'il s'agit d'un chien ou d'un chat,
   deux classes nous suffit.
"""
```

Comme notre modèle se base sur un modèle déjà entraîné, il est inutile que notre modèle **se soumettre à un nouvel entraînement.**

Il faut donc préciser pour chaque couche, qu'on ne souhaite pas effectuer un entraînement.
Il s'agit de mettre l'attribut `trainable` à `False`.

```py

for layer in cnn.layers:
    layer.trainable = False
    
```

On peut maintenant rajouter la dernière couche, avec le bon nombre de classes à déterminer :

```py
cnn.add(Dense(units=2, activation='softmax'))
```

[Notebook pour déconstruire un autre modèle afin de l'appliquer sur un cas plus restreint](developers.google.com/machine_learning/practica/image_classification/exercice_3)

[Ressources du prof (notebooks pour VGG, data)](https://onedrive.live.com/?authkey=%21AEpov3AcXs2dSqY&id=6C6D756296D4662%21783226&cid=06C6D756296D4662)
