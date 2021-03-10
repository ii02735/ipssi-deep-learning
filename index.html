<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, minimal-ui">
    <title>Continuité du machine learning</title>
    <link type="text/css" rel="stylesheet" href="assets/css/github-markdown.css">
    <link type="text/css" rel="stylesheet" href="assets/css/pilcrow.css">
    <link type="text/css" rel="stylesheet" href="assets/css/hljs-github.min.css"/>
  </head>
  <body>
    <article class="markdown-body"><h1 id="continuité-du-machine-learning"><a class="header-link" href="#continuité-du-machine-learning"></a>Continuité du machine learning</h1>
<p>Travail sur <strong>TensorFlow</strong> afin de construire des réseaux de neurones. 
Cet outil est dédié au deep learning.</p>
<p>Il est utilisable pour entraîner des modèles se trouvant sur le web (NodeJS), sur les
applications mobiles, ou sur des environnements de production (cloud, serveur...)</p>
<p>On va l&#39;utiliser en local.</p>
<p>Pour l&#39;installer : <code>pip install tensorflow</code></p>
<p><strong>Keras</strong> a été intégré à TensorFlow : il s&#39;agit d&#39;une API qui permet d&#39;effectuer des
opérations de hauts niveaux. Avant cette intégration, TensorFlow ne pouvait effectuer
des opérations que de bas niveau.</p>
<p>Il est nécessaire de bénéficier d&#39;un GPU pour faire du traitement d&#39;images en grande
quantité, pour le deep learning. Avoir une carte graphique séparé du processeur
(NVIDIA, AMD, etc...donc hors-chipset), est important ici.</p>
<p>On peut passer par <em>colaboratory</em> afin de bénéficier de notebooks qui s&#39;exécutent via
des machines virtuelles offertes par google, offrant des ressources systèmes plus
poussées. <em>colaboratory</em> est intéressant si on ne possède pas de carte graphique sur sa
machine.</p>
<p>Il faut préciser à côté <em>colaboratory</em> qu&#39;on souhaite utiliser le GPU qui est offert par la VM : <code>Exécution &gt; Modifier le type d&#39;exécution &gt; mettre la valeur à &quot;GPU&quot;</code></p>
<p>Pour le moment on va étudier un jeu de données classique (dataframe via pandas) :
artificial_generator.xlsx Ce fichier contient deux jeux de données (2 feuilles :
data2 et data4) : tous les deux évoquent des problèmes de classification.</p>
<ul class="list">
<li>X1 et X2 font référence aux données entrantes</li>
<li>Y fait référence à l&#39;output désiré / au label</li>
</ul>
<p>Étant donné que nous avons deux feuilles dans ce fichier excel, la syntaxe pour lire
une feuille pour ensuite la convertir en dataframe est la suivante :</p>
<pre class="hljs"><code>data4 = pd.read_excel(<span class="hljs-string">&#x27;artificial_generator.xlsx&#x27;</span>, sheet_name=<span class="hljs-string">&quot;data4&quot;</span>)</code></pre><p>Comme on a affaire à un problème de classification, il faut labelliser / numériser les
labels : en l&#39;occurrence ici la colonne Y :</p>
<pre class="hljs"><code>le = LabelEncoder()
data2[<span class="hljs-string">&#x27;Y&#x27;</span>] = le.fit_transform(data2[<span class="hljs-string">&#x27;Y&#x27;</span>])
data4[<span class="hljs-string">&#x27;Y&#x27;</span>] = le.fit_transform(data4[<span class="hljs-string">&#x27;Y&#x27;</span>])</code></pre><p>On peut observer la répartition des données en fonction de leur label via un <strong>un nuage
de point</strong>.</p>
<pre class="hljs"><code><span class="hljs-comment"># Nous spécifions X1 en abscisse, X2 en ordonnée, et ns utilisons Y pour distinguer</span>
les points.

plt.figure(figsize = ( <span class="hljs-number">20</span> , <span class="hljs-number">10</span> ))
plt.subplot( <span class="hljs-number">2</span> , <span class="hljs-number">1</span> , <span class="hljs-number">1</span> )
plt.scatter(data2[<span class="hljs-string">&#x27;X1&#x27;</span>], data2[<span class="hljs-string">&#x27;X2&#x27;</span>], c = data2[<span class="hljs-string">&#x27;Y&#x27;</span>], edgecolors=<span class="hljs-string">&#x27;red&#x27;</span>)

plt.title(<span class="hljs-string">&quot;data2&quot;</span>) <span class="hljs-comment"># affichage du graphique pour la feuille &quot;data2&quot;</span>
plt.subplot( <span class="hljs-number">2</span> , <span class="hljs-number">1</span> , <span class="hljs-number">2</span> )
plt.scatter(data4[<span class="hljs-string">&#x27;X1&#x27;</span>], data4[<span class="hljs-string">&#x27;X2&#x27;</span>], c = data4[<span class="hljs-string">&#x27;Y&#x27;</span>], edgecolors=<span class="hljs-string">&#x27;red&#x27;</span>)
plt.title(<span class="hljs-string">&#x27;data4&#x27;</span>) <span class="hljs-comment"># affichage du graphique pour la feuille &quot;data4&quot;</span></code></pre><p>Les réseaux de neurones permettent de répondre à des problèmes de classification
complexes. Lorsque la répartition du jeu de données est trop complexe (répartition en
écocentrique)...</p>
<h1 id="construction-dun-réseau-de-neurones"><a class="header-link" href="#construction-dun-réseau-de-neurones"></a>Construction d&#39;un réseau de neurones</h1>
<p>Définition des termes :</p>
<ul class="list">
<li><p>Couche d&#39;entrée : les neurones qui recoivent l&#39;information, on a x neurones
pour x colonnes dans le dataframe Chaque neurone possède un poids ( w ) qui est
réparti aléatoirement.</p>
</li>
<li><p>Fonction d&#39;activation (= fonction déterministe) : intermédiaire entre les
données en entrée et en sortie, cet intermédiaire détermine les données qui
doivent être envoyés en d&#39;une couche à une autre ou pas. Il s&#39;agit d&#39;une fonction mathématique
(tangente, sigmoïd...). La fonction d&#39;activation est très utile pour résoudre des problèmes non-linéaires, comme un problème de classification.</p>
</li>
<li><p>Couche cachée (= couche intermédiaire) : des neurones qui recoivent
l&#39;information transmise par la couche d&#39;entrée. Il est possible rajouter autant
de couches cachées qu&#39;on veut : plus il y a des couches cachées, plus on parle
de réseau de neurone profond ( deep ). Dans le cas contraire, on parle de réseau
de neurones creux ( shallow )</p>
</li>
<li><p>Couche de sortie : couche de neurones finale qui va envoyer l&#39;output final de
la donnée envoyée</p>
</li>
</ul>
<p><strong>Mécanisme de calcul :</strong></p>
<ul class="list">
<li>On reçoit les données en entrée</li>
<li>On applique l&#39;opération x*w</li>
<li>La fonction d&#39;activation est ensuite appliquée à ce résultat</li>
<li>On passe à la couche de neurone suivante</li>
</ul>
<p>Comme on a affaire à un <strong>apprentissage supervisé</strong> on peut faire la différence entre le résultat qui est <strong>souhaité</strong> et le résultat qui est <strong>calculé</strong> par le réseau de neurones.
Cette différence est calculé par le réseau de neurones à la fin, et le résultat de cette différence sera propagée ( <em>backdrop</em> : rétro-propagation) sur les poids des
neurones : certains seront mis à jour, d&#39;autres non. </p>
<p>Le réseau de neurones permet donc, grâce à cette mise à jour, d&#39;apprendre de ses erreurs, donc de progresser.</p>
<p>Il y a trois façons de construire un réseau de neurones : <strong>séquentielle</strong> , <strong>fonctionnelle</strong> ,
et par <strong>sous-classes</strong>.</p>
<p>Construction d&#39;un réseau de neurones fonctionnel (permet de réaliser des modèles plus
complexes):</p>
<pre class="hljs"><code><span class="hljs-comment"># Exemple de création d&#x27;un NN avec `Functional API`</span>
<span class="hljs-keyword">from</span> tensorflow.keras.models <span class="hljs-keyword">import</span> Model

<span class="hljs-keyword">from</span> tensorflow.keras.layers <span class="hljs-keyword">import</span> Input, Dense

<span class="hljs-comment"># Couche d&#x27;entrée avec deux neurones</span>
inputs = Input(shape=( <span class="hljs-number">2</span> ,))
<span class="hljs-comment"># inputs.get_shape()</span>
<span class="hljs-comment"># inputs.name</span>
<span class="hljs-comment"># Couches cachées comme fonction d&#x27;activation &#x27;relu&#x27; :</span>
x = Dense( <span class="hljs-number">3</span> , activation=<span class="hljs-string">&#x27;relu&#x27;</span>)(inputs)
<span class="hljs-comment"># la variable x sera appliqué à la couche cachée suivante :</span>
x = Dense( <span class="hljs-number">3</span> , activation=<span class="hljs-string">&#x27;relu&#x27;</span>)(x)</code></pre><pre class="hljs"><code># Comme il s&#x27;agit de la dernière instance Dense, il s&#x27;agit de la couche de sortie
outputs = <span class="hljs-constructor">Dense( 3 , <span class="hljs-params">activation</span>=&#x27;<span class="hljs-params">softmax</span>&#x27;)</span>(x)

# Création du <span class="hljs-keyword">mod</span>èle
model = <span class="hljs-constructor">Model(<span class="hljs-params">inputs</span>=<span class="hljs-params">inputs</span>, <span class="hljs-params">outputs</span>=<span class="hljs-params">outputs</span>)</span></code></pre><p>Construction d&#39;un réseau de neurones de manière séquentielle (plus simple, mais plus
limitée):</p>
<pre class="hljs"><code>model = Sequential([
Dense( <span class="hljs-number">3</span> , activation=<span class="hljs-string">&#x27;relu&#x27;</span>, input_shape=( <span class="hljs-number">4</span> ,)), <span class="hljs-comment"># couche d&#x27;entrée car possède</span>
input_shape
Dense( <span class="hljs-number">10</span> , activation=<span class="hljs-string">&#x27;relu&#x27;</span>),
Dense( <span class="hljs-number">3</span> , activation=<span class="hljs-string">&#x27;softmax&#x27;</span>)
]) <span class="hljs-comment"># on n&#x27;écrit pas la donnée qui est insérée dans chaque couche</span></code></pre><p>Répartition du réseau de neurones avec la fonction get_weights() du modèle :</p>
<pre class="hljs"><code>[array([[<span class="hljs-number">-0.14119095</span>, <span class="hljs-number">0.0544771</span> , <span class="hljs-number">-0.0071274</span> ], # poids associés au neurone x
[<span class="hljs-number">-0.9484316</span> , <span class="hljs-number">-0.31077462</span>, <span class="hljs-number">0.204157</span> ]], dtype=float32), # poids associés au
neurone x
# tableau de biais : ici le tableau du biais est zéro car la modèle n&#x27;a pas encore été
entraîné
array([<span class="hljs-number">0.</span>, <span class="hljs-number">0.</span>, <span class="hljs-number">0.</span>], dtype=float32),
# pour rappel, le biais désigne le taux d&#x27;erreur entre <span class="hljs-number">0</span> et <span class="hljs-number">1</span>
array([[<span class="hljs-number">-0.8759022</span> ],
[<span class="hljs-number">-0.2409901</span> ],
[<span class="hljs-number">-0.40759063</span>]], dtype=float32),
array([<span class="hljs-number">0.</span>], dtype=float32)]</code></pre><p>Exécution de la fonction get_summary() :</p>
<pre class="hljs"><code><span class="hljs-section">Layer (type) Output Shape Param #
=================================================================</span>
dense<span class="hljs-emphasis">_10 (Dense) (None, 3) 9
<span class="hljs-strong">____</span><span class="hljs-strong">____</span><span class="hljs-strong">____</span><span class="hljs-strong">____</span><span class="hljs-strong">____</span><span class="hljs-strong">____</span><span class="hljs-strong">____</span><span class="hljs-strong">____</span><span class="hljs-strong">____</span><span class="hljs-strong">____</span><span class="hljs-strong">____</span><span class="hljs-strong">____</span><span class="hljs-strong">____</span><span class="hljs-strong">____</span><span class="hljs-strong">____</span><span class="hljs-strong">____</span>_</span>
<span class="hljs-section">dense<span class="hljs-emphasis">_11 (Dense) (None, 1) 4
=================================================================
Total params: 13 # nous avons 13 paramètres, donc 13 neurones dans notre réseau
Trainable params: 13
Non-trainable params: 0</span></span></code></pre><p>Le <code>get_summary()</code> peut avoir une représentation différente si la construction du
réseau de neurones est différente. Ici pour un réseau de neurones fonctionnel :</p>
<pre class="hljs"><code><span class="hljs-comment"># Exemple de création d&#x27;un NN avec `Functional API`</span>
<span class="hljs-keyword">from</span> tensorflow.keras.models <span class="hljs-keyword">import</span> Model
<span class="hljs-keyword">from</span> tensorflow.keras.layers <span class="hljs-keyword">import</span> Input, Dense

<span class="hljs-comment"># Couche d&#x27;entrée</span>
inputs = Input(shape=( <span class="hljs-number">2</span> ,))
<span class="hljs-comment"># inputs.get_shape()</span>
<span class="hljs-comment"># inputs.name</span>

<span class="hljs-comment"># Couches cachées :</span>
x = Dense( <span class="hljs-number">3</span> , activation=<span class="hljs-string">&#x27;relu&#x27;</span>)(inputs)
x = Dense( <span class="hljs-number">3</span> , activation=<span class="hljs-string">&#x27;relu&#x27;</span>)(x)

<span class="hljs-comment"># Couche de sortie</span>
outputs = Dense( <span class="hljs-number">3</span> , activation=<span class="hljs-string">&#x27;softmax&#x27;</span>)(x)

model = Model(inputs=inputs, outputs=outputs)</code></pre><p>On obtient la représentation suivante, qui est un peu plus aisée à lire :</p>
<pre class="hljs"><code><span class="hljs-section"># on peut déterminer la couche d&#x27;entrée grâce à InputLayer</span>
<span class="hljs-strong">____</span><span class="hljs-strong">____</span><span class="hljs-strong">____</span><span class="hljs-strong">____</span><span class="hljs-strong">____</span><span class="hljs-strong">____</span><span class="hljs-strong">____</span><span class="hljs-strong">____</span><span class="hljs-strong">____</span><span class="hljs-strong">____</span><span class="hljs-strong">____</span><span class="hljs-strong">____</span><span class="hljs-strong">____</span><span class="hljs-strong">____</span><span class="hljs-strong">____</span><span class="hljs-strong">____</span><span class="hljs-emphasis">_
Layer (type) Output Shape Param #
=================================================================
input_</span>4 (InputLayer) [(None, 2)] 0
<span class="hljs-strong">____</span><span class="hljs-strong">____</span><span class="hljs-strong">____</span><span class="hljs-strong">____</span><span class="hljs-strong">____</span><span class="hljs-strong">____</span><span class="hljs-strong">____</span><span class="hljs-strong">____</span><span class="hljs-strong">____</span><span class="hljs-strong">____</span><span class="hljs-strong">____</span><span class="hljs-strong">____</span><span class="hljs-strong">____</span><span class="hljs-strong">____</span><span class="hljs-strong">____</span><span class="hljs-strong">____</span><span class="hljs-emphasis">_
dense_</span>11 (Dense) (None, 3) 9
<span class="hljs-strong">____</span><span class="hljs-strong">____</span><span class="hljs-strong">____</span><span class="hljs-strong">____</span><span class="hljs-strong">____</span><span class="hljs-strong">____</span><span class="hljs-strong">____</span><span class="hljs-strong">____</span><span class="hljs-strong">____</span><span class="hljs-strong">____</span><span class="hljs-strong">____</span><span class="hljs-strong">____</span><span class="hljs-strong">____</span><span class="hljs-strong">____</span><span class="hljs-strong">____</span><span class="hljs-strong">____</span><span class="hljs-emphasis">_
dense_</span>12 (Dense) (None, 3) 12
<span class="hljs-strong">____</span><span class="hljs-strong">____</span><span class="hljs-strong">____</span><span class="hljs-strong">____</span><span class="hljs-strong">____</span><span class="hljs-strong">____</span><span class="hljs-strong">____</span><span class="hljs-strong">____</span><span class="hljs-strong">____</span><span class="hljs-strong">____</span><span class="hljs-strong">____</span><span class="hljs-strong">____</span><span class="hljs-strong">____</span><span class="hljs-strong">____</span><span class="hljs-strong">____</span><span class="hljs-strong">____</span><span class="hljs-emphasis">_
dense_</span>13 (Dense) (None, 3) 12
=================================================================
Total params: 33
Trainable params: 33
Non-trainable params: 0</code></pre><p>Il est possible qu&#39;une machine puisse partir sur le même réseau de neurones en
<strong>chargeant un fichier contenant les poids de ce réseau de neurones</strong>.</p>
<p>Pour sauvegarder les poids : <model>.save_weights(&#39;path&#39;)</p>
<p>Pour compiler le modèle afin qu&#39;il soit prêt à utilisation en production :</p>
<pre class="hljs"><code><span class="hljs-comment"># optmizer = manière d&#x27;actualiser les poids après backdrop</span>
<span class="hljs-comment"># loss = calculer l&#x27;erreur : ici il s&#x27;agit d&#x27;une classification binaire -&gt;</span>
binary_crossentropy
model.<span class="hljs-built_in">compile</span>(optimizer=<span class="hljs-string">&#x27;rmsprop&#x27;</span>, loss=<span class="hljs-string">&#x27;binary_crossentropy&#x27;</span>, metrics=[<span class="hljs-string">&#x27;accuracy&#x27;</span>])
<span class="hljs-comment"># construction du modèle</span>
<span class="hljs-comment"># epochs = nombre d&#x27;itérations</span>

model.fit(data,labels,epochs= <span class="hljs-number">10</span> ,batch_size= <span class="hljs-number">32</span> )
model.predict(data)</code></pre><h1 id="réseau-de-neurones-à-convolution-traitement-dimages"><a class="header-link" href="#réseau-de-neurones-à-convolution-traitement-dimages"></a>Réseau de neurones à convolution (traitement d&#39;images)</h1>
<p>La <strong>convolution</strong> en traitement d&#39;image, est une technique permettant d&#39;extraire des informations d&#39;une image.
Il s&#39;agit d&#39;une opération mathématique, effectuée <strong>sur la matrice de l&#39;image</strong>.
Une nouvelle matrice est donnée en résultat, et cette matrice permet d&#39;effectuer des opérations pertinentes pour la récupération d&#39;informations d&#39;une image : détection des contours, amélioration de la netteté...</p>
<p><strong>Reprise du TP cats VS Dogs</strong></p>
<p>Sources :</p>
<p><a href="https://colab.research.google.com/github/google/eng-edu/blob/master/ml/pc/exercises/image_classification_part1.ipynb">Notebook</a></p>
<p><a href="https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip">Jeu de données</a></p>
<p><a href="https://deeplizard.com/learn/video/LhEMXbjGV_4">S&#39;est inspiré de ce tutoriel</a></p>
<pre class="hljs"><code><span class="hljs-keyword">import</span> os
<span class="hljs-keyword">import</span> numpy <span class="hljs-keyword">as</span> np
<span class="hljs-keyword">from</span> matplotlib <span class="hljs-keyword">import</span> pyplot <span class="hljs-keyword">as</span> plt
<span class="hljs-keyword">from</span> matplotlib.image <span class="hljs-keyword">import</span> imread
<span class="hljs-keyword">from</span> PIL <span class="hljs-keyword">import</span> Image
<span class="hljs-keyword">import</span> tensorflow <span class="hljs-keyword">as</span> tf
<span class="hljs-keyword">from</span> tensorflow.keras <span class="hljs-keyword">import</span> layers, Model
<span class="hljs-keyword">import</span> zipfile

<span class="hljs-comment"># lecture des différentes images</span>
<span class="hljs-comment"># Une boucle pr afficher plusieurs images dans une figure</span>

<span class="hljs-keyword">for</span> i <span class="hljs-keyword">in</span> <span class="hljs-built_in">range</span>(<span class="hljs-number">9</span>):
    plt.subplot(<span class="hljs-number">3</span>, <span class="hljs-number">3</span>, i+<span class="hljs-number">1</span>)
    filename = <span class="hljs-string">&#x27;train/&#x27;</span> + <span class="hljs-string">&#x27;cats/&#x27;</span> + <span class="hljs-string">&#x27;cat.&#x27;</span> + <span class="hljs-built_in">str</span>(i) + <span class="hljs-string">&#x27;.jpg&#x27;</span>
    image = imread(filename)
    plt.imshow(image)
    plt.title(image.shape)

<span class="hljs-comment"># S : On constate que les images sont en mode paysage, mode portrait et de différentes tailles  </span>
<span class="hljs-comment"># =&gt; Ceci peut poser des Pb pour la perfermonce du classifieur. Il nous faut un classifieur robuste !</span>

<span class="hljs-comment"># Manip avec PIL : pkg Python de réf pr le traitement d&#x27;image</span>
<span class="hljs-comment"># dog1 = Image.open(&#x27;train/dogs/dog.1.jpg&#x27;)</span>
<span class="hljs-comment"># dog1.show()</span>
<span class="hljs-comment"># dog1.size</span>
<span class="hljs-comment"># dog1.mode</span>
<span class="hljs-comment"># dog1.format</span>
<span class="hljs-comment"># np.array(dog1)</span>
<span class="hljs-comment"># imread(&#x27;train/dogs/dog.1.jpg&#x27;)</span>

<span class="hljs-comment"># traitement de l&#x27;image directement avec keras</span>
dog1 = tf.keras.preprocessing.image.load_img(<span class="hljs-string">&#x27;train/dogs/dog.1.jpg&#x27;</span>)
dog1_ = tf.keras.preprocessing.image.img_to_array(dog1)
</code></pre><p>Pour faciliter le traitement, on met les images à la même échelle :</p>
<pre class="hljs"><code>train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=<span class="hljs-number">1.</span>/<span class="hljs-number">255</span>)
val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=<span class="hljs-number">1.</span>/<span class="hljs-number">255</span>)</code></pre><p>Puis ensuite, pour les groupes d&#39;image d&#39;entraînement et de test, on indique qu&#39;on souhaite obtenir des images de 150x150, toujours dans l&#39;optique de faciliter le traitement des images.
<strong>Plus l&#39;homogénéité est grande</strong>, mieux le traitement sera performant.</p>
<pre class="hljs"><code>train_gen = train_datagen.flow_from_directory(
    <span class="hljs-string">&#x27;train&#x27;</span>, <span class="hljs-comment"># nom du dossier</span>
    target_size=(<span class="hljs-number">150</span>, <span class="hljs-number">150</span>), 
    batch_size=<span class="hljs-number">20</span>, <span class="hljs-comment"># on précise qu&#x27;on souhaite lire les images par groupe de 20 (batch)</span>
    class_mode=<span class="hljs-string">&#x27;binary&#x27;</span> <span class="hljs-comment"># il s&#x27;agit de résoudre un problème de classification binaire</span>
)

val_gen = val_datagen.flow_from_directory(
    <span class="hljs-string">&#x27;validation&#x27;</span>, 
    target_size=(<span class="hljs-number">150</span>, <span class="hljs-number">150</span>), 
    batch_size=<span class="hljs-number">20</span>, 
    class_mode=<span class="hljs-string">&#x27;binary&#x27;</span>
)</code></pre><p>Voici une fonction utilitaire qui nous sera utile pour afficher les différentes images :</p>
<pre class="hljs"><code><span class="hljs-function"><span class="hljs-keyword">def</span> <span class="hljs-title">plotImages</span>(<span class="hljs-params">images_arr</span>):</span>
    fig, axes = plt.subplots(<span class="hljs-number">1</span>, <span class="hljs-number">10</span>, figsize=(<span class="hljs-number">20</span>,<span class="hljs-number">20</span>))
    axes = axes.flatten()
    <span class="hljs-keyword">for</span> img, ax <span class="hljs-keyword">in</span> <span class="hljs-built_in">zip</span>( images_arr, axes):
        ax.imshow(img)
        ax.axis(<span class="hljs-string">&#x27;off&#x27;</span>)
    plt.tight_layout()
    plt.show()</code></pre><p>Les objets retournés par <code>ImageDataGenerator</code> sont de type <code>tensorflow.python.keras.preprocessing.image.DirectoryIterator</code>.
Pour pouvoir lire les images, il faut invoquer la fonction <code>next</code> :</p>
<pre class="hljs"><code><span class="hljs-comment"># Visualiser les images resizés et rescalés</span>
<span class="hljs-comment"># On a déjà les labels de chaque image.</span>
<span class="hljs-comment"># En effet, un label = le dossier contenant l&#x27;image</span>
imgs, labels = <span class="hljs-built_in">next</span>(val_gen)
imgs <span class="hljs-comment"># affichera 20 matrices numpy (une matrice = une image)</span></code></pre><p>Construction du CNN (méthode fonctionnelle -&gt; on ne passe pas par <code>add</code> qui est séquentielle) :</p>
<pre class="hljs"><code><span class="hljs-keyword">from</span> tensorflow.keras <span class="hljs-keyword">import</span> layers
<span class="hljs-keyword">from</span> tensorflow.keras <span class="hljs-keyword">import</span> Model

<span class="hljs-comment"># signifie qu&#x27;on souhaite insérer des images qui font 150 x 150</span>
<span class="hljs-comment"># et qui est composé d&#x27;une matrice de 3 couleurs (RGB) </span>
img_input = layers.<span class="hljs-built_in">input</span>(shape=(<span class="hljs-number">150</span>,<span class="hljs-number">150</span>,<span class="hljs-number">3</span>))

<span class="hljs-comment"># Couche de convolution appliquant 16 filtres sous une matrice de 3x3</span>
<span class="hljs-comment"># Cette matrice balaye l&#x27;image</span>
<span class="hljs-comment"># La fonction d&#x27;activation est relu</span>
x = layers.Conv2D(<span class="hljs-number">16</span>, <span class="hljs-number">3</span>, activation=<span class="hljs-string">&#x27;relu&#x27;</span>)(img_input)
<span class="hljs-comment"># Couche pour réaliser un MaxPooling : opération permettant de réduire</span>
<span class="hljs-comment"># la taille de la matrice de l&#x27;image tout en récupérant un maximum d&#x27;infos</span>
<span class="hljs-comment"># ici on redimensionne la matrice en une 2x2</span>
x = layers.MaxPooling2D(<span class="hljs-number">2</span>)(x)

<span class="hljs-comment"># Couche de convolution appliquant 16 filtres sous une matrice de 3x3</span>
x = layers.Conv2D(<span class="hljs-number">32</span>, <span class="hljs-number">3</span>, activation=<span class="hljs-string">&#x27;relu&#x27;</span>)(x)
x = layers.MaxPooling2D(<span class="hljs-number">2</span>)(x)

<span class="hljs-comment"># Couche de convolution appliquant 64 filtres sous une matrice de 3x3</span>
x = layers.Conv2D(<span class="hljs-number">64</span>, <span class="hljs-number">3</span>, activation=<span class="hljs-string">&#x27;relu&#x27;</span>)(x)
x = layers.MaxPooling2D(<span class="hljs-number">2</span>)(x)
<span class="hljs-comment"># Aplatissement de x pour préparer la sortie</span>
x = layers.Flatten()(x)
<span class="hljs-comment"># Couche Dense pour le output = pleinement connecté aux neurones la couche précédente</span>
x = layers.Dense(<span class="hljs-number">512</span>, activation=<span class="hljs-string">&#x27;relu&#x27;</span>)(x)

outputs = layers.Dense(<span class="hljs-number">1</span>, activation=<span class="hljs-string">&#x27;sigmoid&#x27;</span>)(x)</code></pre><p>Instanciation du modèle :</p>
<pre class="hljs"><code>model = layers.models.Model(inputs=inputs, outputs=outputs)
models.summary() <span class="hljs-comment"># résumé du modèle</span></code></pre><p>Compilation du modèle :</p>
<pre class="hljs"><code><span class="hljs-comment"># metrics = accuracy</span>
model.<span class="hljs-built_in">compile</span>(loss=<span class="hljs-string">&#x27;binary_crossentropy&#x27;</span>, metrics=[<span class="hljs-string">&#x27;acc&#x27;</span>], optimizer=<span class="hljs-string">&#x27;rmsprop&#x27;</span>)</code></pre><p>Entraînement :</p>
<pre class="hljs"><code>history = model.fit_generator(train_gen,
                    <span class="hljs-comment"># combien d&#x27;images doivent être traitées à la fois ?</span>
                    <span class="hljs-comment"># opération : batch_size * steps = nombre d&#x27;images</span>
                    steps_per_epoch=<span class="hljs-number">100</span>, 
                    validation_data=val_gen, <span class="hljs-comment"># préciser l&#x27;échantillon de test</span>
                    <span class="hljs-comment"># le nombre d&#x27;étapes pour la validation</span>
                    <span class="hljs-comment"># opération : batch_size * steps = nombre d&#x27;images</span>
                    validation_steps=<span class="hljs-number">50</span>,
                    verbose=<span class="hljs-number">2</span>)</code></pre><p>Le code suivant permet de visualiser les différentes transformations des images après application de chaque couche (maxpooling2d, conv2d):</p>
<pre class="hljs"><code><span class="hljs-keyword">import</span> numpy <span class="hljs-keyword">as</span> np
<span class="hljs-keyword">import</span> random
<span class="hljs-keyword">from</span> tensorflow.keras.preprocessing.image <span class="hljs-keyword">import</span> img_to_array, load_img

<span class="hljs-comment"># Let&#x27;s define a new Model that will take an image as input, and will output</span>
<span class="hljs-comment"># intermediate representations for all layers in the previous model after</span>
<span class="hljs-comment"># the first.</span>
successive_outputs = [layer.output <span class="hljs-keyword">for</span> layer <span class="hljs-keyword">in</span> model.layers[<span class="hljs-number">1</span>:]]
visualization_model = Model(img_input, successive_outputs)

<span class="hljs-comment"># Let&#x27;s prepare a random input image of a cat or dog from the training set.</span>
cat_img_files = [os.path.join(train_cats_dir, f) <span class="hljs-keyword">for</span> f <span class="hljs-keyword">in</span> train_cat_fnames]
dog_img_files = [os.path.join(train_dogs_dir, f) <span class="hljs-keyword">for</span> f <span class="hljs-keyword">in</span> train_dog_fnames]
img_path = random.choice(cat_img_files + dog_img_files)

img = load_img(img_path, target_size=(<span class="hljs-number">150</span>, <span class="hljs-number">150</span>))  <span class="hljs-comment"># this is a PIL image</span>
x = img_to_array(img)  <span class="hljs-comment"># Numpy array with shape (150, 150, 3)</span>
x = x.reshape((<span class="hljs-number">1</span>,) + x.shape)  <span class="hljs-comment"># Numpy array with shape (1, 150, 150, 3)</span>

<span class="hljs-comment"># Rescale by 1/255</span>
x /= <span class="hljs-number">255</span>

<span class="hljs-comment"># Let&#x27;s run our image through our network, thus obtaining all</span>
<span class="hljs-comment"># intermediate representations for this image.</span>
successive_feature_maps = visualization_model.predict(x)

<span class="hljs-comment"># These are the names of the layers, so can have them as part of our plot</span>
layer_names = [layer.name <span class="hljs-keyword">for</span> layer <span class="hljs-keyword">in</span> model.layers]

<span class="hljs-comment"># Now let&#x27;s display our representations</span>
<span class="hljs-keyword">for</span> layer_name, feature_map <span class="hljs-keyword">in</span> <span class="hljs-built_in">zip</span>(layer_names, successive_feature_maps):
  <span class="hljs-keyword">if</span> <span class="hljs-built_in">len</span>(feature_map.shape) == <span class="hljs-number">4</span>:
    <span class="hljs-comment"># Just do this for the conv / maxpool layers, not the fully-connected layers</span>
    n_features = feature_map.shape[-<span class="hljs-number">1</span>]  <span class="hljs-comment"># number of features in feature map</span>
    <span class="hljs-comment"># The feature map has shape (1, size, size, n_features)</span>
    size = feature_map.shape[<span class="hljs-number">1</span>]
    <span class="hljs-comment"># We will tile our images in this matrix</span>
    display_grid = np.zeros((size, size * n_features))
    <span class="hljs-keyword">for</span> i <span class="hljs-keyword">in</span> <span class="hljs-built_in">range</span>(n_features):
      <span class="hljs-comment"># Postprocess the feature to make it visually palatable</span>
      x = feature_map[<span class="hljs-number">0</span>, :, :, i]
      x -= x.mean()
      x /= x.std()
      x *= <span class="hljs-number">64</span>
      x += <span class="hljs-number">128</span>
      x = np.clip(x, <span class="hljs-number">0</span>, <span class="hljs-number">255</span>).astype(<span class="hljs-string">&#x27;uint8&#x27;</span>)
      <span class="hljs-comment"># We&#x27;ll tile each filter into this big horizontal grid</span>
      display_grid[:, i * size : (i + <span class="hljs-number">1</span>) * size] = x
    <span class="hljs-comment"># Display the grid</span>
    scale = <span class="hljs-number">20.</span> / n_features
    plt.figure(figsize=(scale * n_features, scale))
    plt.title(layer_name)
    plt.grid(<span class="hljs-literal">False</span>)
    plt.imshow(display_grid, aspect=<span class="hljs-string">&#x27;auto&#x27;</span>, cmap=<span class="hljs-string">&#x27;viridis&#x27;</span>)</code></pre><h3 id="diagnostic-du-modèle-à-laide-de-courbes"><a class="header-link" href="#diagnostic-du-modèle-à-laide-de-courbes"></a>Diagnostic du modèle à l&#39;aide de courbes</h3>
<p>Pour obtenir les performances du modèle, il est souvent nécessaire de passer par une représentation en courbes de l&#39;évolution en précision, ainsi que de l&#39;évolution de la perte d&#39;information / erreur/</p>
<pre class="hljs"><code><span class="hljs-comment"># Retrieve a list of accuracy results on training and validation data</span>
<span class="hljs-comment"># sets for each training epoch</span>
acc = history.history[<span class="hljs-string">&#x27;acc&#x27;</span>]
val_acc = history.history[<span class="hljs-string">&#x27;val_acc&#x27;</span>]

<span class="hljs-comment"># Retrieve a list of list results on training and validation data</span>
<span class="hljs-comment"># sets for each training epoch</span>
loss = history.history[<span class="hljs-string">&#x27;loss&#x27;</span>]
val_loss = history.history[<span class="hljs-string">&#x27;val_loss&#x27;</span>]

<span class="hljs-comment"># Get number of epochs</span>
epochs = <span class="hljs-built_in">range</span>(<span class="hljs-built_in">len</span>(acc))

<span class="hljs-comment"># Plot training and validation accuracy per epoch</span>
plt.plot(epochs, acc)
plt.plot(epochs, val_acc)
plt.title(<span class="hljs-string">&#x27;Training and validation accuracy&#x27;</span>)

plt.figure()

<span class="hljs-comment"># Plot training and validation loss per epoch</span>
plt.plot(epochs, loss)
plt.plot(epochs, val_loss)
plt.title(<span class="hljs-string">&#x27;Training and validation loss&#x27;</span>)</code></pre><p>Pour que le modèle soit valide, il faut que pour que chaque graphique, on ait une <strong>convergence</strong> de courbes, et pas une divergence.
S&#39;il y a divergence, on a affaire à un <strong>problème de surapprentissage</strong>.</p>
<p><a href="https://colab.research.google.com/github/google/eng-edu/blob/master/ml/pc/exercises/image_classification_part2.ipynb">Notebook</a></p>
<p><strong>Surapprentissage</strong> (<em>overfitting</em>) : ce problème se produit lorsqu&#39;un modèle ne possède pas suffisamment de données pour pouvoir mémoriser les différentes caractéristiques qui distinguent une classe d&#39;une autre.
Ce défaut amène de mauvais résultats au moment de l&#39;évaluation.</p>
<p>Deux méthodes pour prévenir ce souci :</p>
<ul class="list">
<li><p> <strong>Injecter beaucoup plus de données</strong>, mais comment faire s&#39;il nous n&#39;est pas possible d&#39;en fournir davantage ? Il faut passer par une <strong>augmentation des données</strong> (<em>Data augmentation</em>) existantes.
Les &quot;nouvelles données&quot; seront en réalité des copies des données existantes, <strong>mais avec des transformations aléatoires</strong> (rotation, zoom...).
Comme il y a des différences dans les copies, le moèdle considèrera qu&#39;il s&#39;agit de données nouvelles.</p>
</li>
<li><p><strong>Régularisation par abandon</strong> (<em>dropout</em>) : on choisit arbitrairement les neurones qui ne devront pas traiter la donnée --&gt; alléger le réseau de neurones. Moins on exécute de neurones, plus les données fournies seront suffisantes.</p>
</li>
</ul>
<p><strong>Application de <em>Data augmentation</em> :</strong></p>
<pre class="hljs"><code><span class="hljs-comment"># Data augmentation sur les données d&#x27;apprentissage</span>

train_datagen = ImageDataGenerator(
    rescale=<span class="hljs-number">1.</span>/<span class="hljs-number">255</span>, <span class="hljs-comment"># mise à échelle</span>
    rotation_range=<span class="hljs-number">40</span>, <span class="hljs-comment"># intervalle de rotation aléatoire en degrés</span>
    width_shift_range=<span class="hljs-number">0.2</span>, <span class="hljs-comment"># redimensionnement de l&#x27;image (20 % de la largeur totale ici)</span>
    height_shift_range=<span class="hljs-number">0.2</span>, <span class="hljs-comment"># redimensionnement de l&#x27;image (20 % de la hauteur totale ici)</span>
    shear_range=<span class="hljs-number">0.2</span>, <span class="hljs-comment"># transvection aléatoire de 20 % de l&#x27;image (&quot;skew&quot;, exemple : rectangle + skew = parallélogramme)</span>
    zoom_range=<span class="hljs-number">0.2</span>, <span class="hljs-comment"># zoom aléatoire de l&#x27;image à 20%</span>
    horizontal_flip=<span class="hljs-literal">True</span>, <span class="hljs-comment"># mettre l&#x27;image à l&#x27;horizontale ?</span>
    fill_mode=<span class="hljs-string">&#x27;nearest&#x27;</span> <span class="hljs-comment"># mode de reproduction des pixels l&#x27;image (&#x27;nearest&#x27; : reproduction fidèle de l&#x27;originale</span>
    )

<span class="hljs-comment"># On n&#x27;applique pas la data augmentation sur nos données de test !</span>
<span class="hljs-comment"># Après tout le modèle ne s&#x27;entraîne pas avec, mais s&#x27;évalue</span>
val_datagen = ImageDataGenerator(rescale=<span class="hljs-number">1.</span>/<span class="hljs-number">255</span>)

train_generator = train_datagen.flow_from_directory(
        train_dir, 
        target_size=(<span class="hljs-number">150</span>, <span class="hljs-number">150</span>), 
        batch_size=<span class="hljs-number">20</span>,
        class_mode=<span class="hljs-string">&#x27;binary&#x27;</span>)


validation_generator = val_datagen.flow_from_directory(
        validation_dir,
        target_size=(<span class="hljs-number">150</span>, <span class="hljs-number">150</span>),
        batch_size=<span class="hljs-number">20</span>,
        class_mode=<span class="hljs-string">&#x27;binary&#x27;</span>)</code></pre><p>On peut ensuite refaire l&#39;apprentissage du modèle à partir de ce nouveau jeu de données :</p>
<pre class="hljs"><code>history = model.fit_generator(train_generator,
                    <span class="hljs-comment"># combien d&#x27;images doivent être traitées à la fois ?</span>
                    <span class="hljs-comment"># opération : batch_size * steps = nombre d&#x27;images</span>
                    steps_per_epoch=<span class="hljs-number">100</span>, 
                    validation_data=validation_generator, <span class="hljs-comment"># préciser l&#x27;échantillon de test</span>
                    <span class="hljs-comment"># le nombre d&#x27;étapes pour la validation</span>
                    <span class="hljs-comment"># opération : batch_size * steps = nombre d&#x27;images</span>
                    validation_steps=<span class="hljs-number">50</span>,
                    verbose=<span class="hljs-number">2</span>)
</code></pre><p>Il faudra ensuite observer si la performance du modèle est concluant ou pas (toujours avec les plots).</p>
<p><strong>Application de dropout :</strong></p>
<p>Cela s&#39;applique juste avant la création de <strong>la couche de l&#39;output</strong> du CNN :</p>
<pre class="hljs"><code><span class="hljs-comment"># 0.5 : probabilité qu&#x27;un neurone à l&#x27;entrée soit ignoré</span>
<span class="hljs-comment"># le neurone est choisi aléatoirement, à chaque étape de l&#x27;apprentissage (steps_per_epoch)</span>
<span class="hljs-comment"># cette valeur est totalement arbitraire, et il faut déterminer le bon taux de dropout</span>
x = layers.output(<span class="hljs-number">0.5</span>)(x)</code></pre><h3 id="évaluer-le-modèle"><a class="header-link" href="#évaluer-le-modèle"></a>Évaluer le modèle</h3>
<p>Permet d&#39;obtenir un score du modèle</p>
<pre class="hljs"><code><span class="hljs-comment"># evaluate_generator est obsolète : préférer evaluate</span>
<span class="hljs-comment"># renvoie un tableau de deux éléments : loss, accuracy</span>
model.evaluate_generator(validation_generator)
</code></pre><h3 id="prédictions-du-modèle"><a class="header-link" href="#prédictions-du-modèle"></a>Prédictions du modèle</h3>
<p>On peut vérifier si les résultats que notre modèle nous a renvoyé pour nos données de tests, sont corrects ou non.</p>
<pre class="hljs"><code><span class="hljs-comment"># renvoie plusieurs tableaux de numpy à dimension 1</span>
<span class="hljs-comment"># chaque élément représente une probabilité (si supérieur à 0.5, appartient à la classe / label 1, sinon 0)</span>
y_pred = model.predict_generator(validation_generator)

<span class="hljs-comment">#on peut améliorer l&#x27;affichage en le convertissant y_pred en Series</span>
<span class="hljs-keyword">import</span> pandas <span class="hljs-keyword">as</span> pd

<span class="hljs-comment"># flatten : fusionner les différents tableaux en un seul</span>
pd.Series(y_pred.flatten()).head()
</code></pre><p>On peut récupérer les différents labels toujours grâce à une instance de <code>tensorflow.python.keras.preprocessing.image.DirectoryIterator</code> 
:</p>
<pre class="hljs"><code>imgs, labels = <span class="hljs-built_in">next</span>(validation_generator)
labels</code></pre><p>Et on peut comparer le contenu de chaque label, avec le contenu de chaque image (pouvoir distinguer si la classe 1 représente celle des chiens ou des chats).</p>
<h2 id="matrice-de-confusion"><a class="header-link" href="#matrice-de-confusion"></a>Matrice de confusion</h2>
<p>Utile pour avoir une répartition des différentes classes de notre prédiction.
En second paramètre de <code>confusion_matrix</code>, il faut passer un tableau d&#39;entiers binaire (0 ou 1).
Cependant, nos données de prédiction sont en décimales.</p>
<p>Il faut convertir le tout en des valeurs entières :</p>
<pre class="hljs"><code>
y_pred = (y_pred &gt; <span class="hljs-number">0.5</span>).astype(<span class="hljs-string">&#x27;int32&#x27;</span>)
<span class="hljs-keyword">from</span> sklearn.metrics <span class="hljs-keyword">import</span> confusion_matrix

<span class="hljs-comment"># validation_generator.classes : tableau représentant chaque image par sa classe</span>
conf_mat = confusion_matrix(validation_generator.classes, y_pred)</code></pre><p>On peut styliser la matrice à l&#39;aide de <code>mlxtend</code> (l&#39;installer si pas déjà fait) :</p>
<pre class="hljs"><code><span class="hljs-keyword">from</span> mlxtend.plotting <span class="hljs-keyword">import</span> plot_confusion_matrix

plot_confusion_matrix(conf_mat=conf_mat, colorbar=<span class="hljs-literal">True</span>, show_absolute=<span class="hljs-literal">True</span>, show_normed=<span class="hljs-literal">True</span>, class_names=[<span class="hljs-string">&#x27;dog&#x27;</span>,<span class="hljs-string">&#x27;cats&#x27;</span>])</code></pre><h2 id="transfer-learning--réutiliser-un-modèle-en-production"><a class="header-link" href="#transfer-learning--réutiliser-un-modèle-en-production"></a>Transfer Learning : réutiliser un modèle en production</h2>
<p>Le principe est de pouvoir <strong>réutiliser un modèle</strong> réseau de neurones, qui est <strong>déjà entraîné</strong> sans forcément en bâtir un.</p>
<p>Des exemples de réseaux de neurones fameux pour leur réutilisation :</p>
<ul class="list">
<li>AlexNet</li>
<li>VGG16</li>
</ul>
<p>Chacun a ses avantages et ses inconvénients.</p>
<p>VGG16 utilise un algorithme de réseau de neurones dont <strong>la profondeur / le nombre de couches</strong> est très élevée, et de grosses quantités de données lui ont été passé.
Son apprentissage a duré <strong>2 semaines, mais avec de très bons résultats en retour !</strong></p>
<p>Tensorflow possède dans ses librairies, une <strong>implémentation du VGG16</strong> :</p>
<pre class="hljs"><code><span class="hljs-keyword">import</span> tensorflow <span class="hljs-keyword">as</span> tf

tf.keras.applications.vgg16

<span class="hljs-comment"># décode la prédiction d&#x27;un jeu de données provenant de ImageNet (base de données d&#x27;images publique)</span>
<span class="hljs-comment"># accepte en premier argument une matrice</span>
<span class="hljs-comment"># création d&#x27;un tableau de 1000 valeurs : np.arange(1000)</span>
<span class="hljs-comment"># création d&#x27;une matrice à une dimension, et avec une ligne de longueur 1000</span>
tf.keras.applications.vgg16.decode_predictions(np.arange(<span class="hljs-number">1000</span>).reshape(<span class="hljs-number">1</span>, <span class="hljs-number">1000</span>), top=<span class="hljs-number">1000</span>)

<span class="hljs-comment"># attribue à chaque groupe d&#x27;image existant sur ImageNet, un entier (entre 1 et 1000)</span></code></pre><h3 id="importer-le-modèle-vgg16--instanciation"><a class="header-link" href="#importer-le-modèle-vgg16--instanciation"></a>Importer le modèle VGG16 + instanciation</h3>
<pre class="hljs"><code><span class="hljs-keyword">from</span> tensorflow.keras.applications.vgg16 <span class="hljs-keyword">import</span> VGG16
model = VGG16()

<span class="hljs-comment"># l&#x27;importation du VGG16 télécharge les poids du modèle (528 MB) </span>
<span class="hljs-comment"># dans le répertoire .keras/models</span></code></pre><p>On peut avoir une représentation du VGG16 via la fonction <code>summary()</code> :</p>
<pre class="hljs"><code>model.summary()</code></pre><p><a href="https://neurohive.io/wp-content/uploads/2018/11/vgg16.png">Voici une représentation schématisée par couche</a></p>
<p>L&#39;exécution de la fonction nous affichera environ <strong>140 millions de paramètres</strong> (soit 140 millions de poids).</p>
<h3 id="préprocessing-pour-format-vgg"><a class="header-link" href="#préprocessing-pour-format-vgg"></a>Préprocessing pour format VGG</h3>
<p>Afin d&#39;appliquer de la classification d&#39;images à partir du VGG16, il faut convertir les images <strong>en format VGG</strong>.</p>
<pre class="hljs"><code><span class="hljs-keyword">from</span> IPython.display <span class="hljs-keyword">import</span> Image
<span class="hljs-keyword">from</span> tensorflow.keras.preprocessing.image <span class="hljs-keyword">import</span> load_img
<span class="hljs-keyword">from</span> tensorflow.keras.applications.vgg16  <span class="hljs-keyword">import</span> preprocess_import

<span class="hljs-comment"># Chargement et redimension (VGG16 étant entraîné sur des images 224 x 224)</span>
image = load_img(<span class="hljs-string">&#x27;chemin vers image&#x27;</span>, target_size(<span class="hljs-number">224</span>,<span class="hljs-number">224</span>))

<span class="hljs-comment"># Préparation en VGG : soustraction de la moyenne des pixels</span>
image = preprocess_input(image)
image.shape <span class="hljs-comment"># affichera (1, 224, 224, 3)</span>

<span class="hljs-comment"># pour afficher l&#x27;image préprocessée</span>

tf.keras.preprocessing.image_to_array(img)
</code></pre><h3 id="prédiction"><a class="header-link" href="#prédiction"></a>Prédiction</h3>
<p>Maintenant que l&#39;image a un format compatible, on peut demander à VGG, de deviner la classe.</p>
<pre class="hljs"><code>Renvoie un tableau de probabilités
proba = model.predict(image_pr)
proba

<span class="hljs-string">&quot;&quot;&quot;
   On peut convertir le tableau de probabilités
   en un dataframe pour que cela soit plus visible
&quot;&quot;&quot;</span>
pd.Dataframe(proba).T.sort_values(<span class="hljs-number">0</span>, ascending=<span class="hljs-literal">False</span>)</code></pre><h3 id="reconnaissance-de-limage"><a class="header-link" href="#reconnaissance-de-limage"></a>Reconnaissance de l&#39;image</h3>
<p>On peut commencer à déterminer la classe de l&#39;image grâce à la probabilité.</p>
<pre class="hljs"><code><span class="hljs-keyword">from</span> tensorflow.keras.applications.vgg16 <span class="hljs-keyword">import</span> decode_predictions
label = decode_predictions(proba)
label <span class="hljs-comment"># renverra un tableau contenant des labels se rapprochant de la probabilité</span></code></pre><h2 id="adapter-un-modèle-de-production-à-notre-besoin"><a class="header-link" href="#adapter-un-modèle-de-production-à-notre-besoin"></a>Adapter un modèle de production à notre besoin</h2>
<p>Le VGG16 contient un gros nombre de paramètres (140 millions), car il est prêt à travailler sur <strong>1000 classes (des millions d&#39;images)</strong>.</p>
<p>Pour éviter de gaspiller des ressources inutiles, il faut justement <strong>adapter la structure de ce modèle à  notre besoin</strong> -&gt; récupérer uniquement les labels concernant les chats et chiens ici.</p>
<pre class="hljs"><code>target = decode_predictions(np.arrage(<span class="hljs-number">1000</span>).reshape(<span class="hljs-number">1</span>,<span class="hljs-number">1000</span>), top=<span class="hljs-number">1000</span>) <span class="hljs-comment"># récupération de tous les labels</span>
dog = target.label.<span class="hljs-built_in">str</span>.contains(<span class="hljs-string">&#x27;_dog&#x27;</span>)

target[dog] <span class="hljs-comment"># labels contenant &quot;dog&quot;</span>

cat = target.label.<span class="hljs-built_in">str</span>.contains(<span class="hljs-string">&#x27;_cat&#x27;</span>)</code></pre><p>Cependant cette façon de filtrer les labels n&#39;est pas la meilleure, car il contient un nombre restreint de races de chiens et de chats.</p>
<p><strong>Il faut aller plus loin dans l&#39;adaptation du modèle.</strong></p>
<p>L&#39;idée est de récupérer les couches du modèle, dans un nouveau modèle :</p>
<pre class="hljs"><code>
cnn = Sequential()
<span class="hljs-keyword">for</span> layer <span class="hljs-keyword">in</span> model.layers[:-<span class="hljs-number">1</span>]:
    cnn.add(layer)
    
cnn.summary()

<span class="hljs-string">&quot;&quot;&quot;
   On ne récupère pas la dernière couche (output)
   Car l&#x27;output envoie des résultats pour 1000 classes
   Nous, comme nous voulons savoir s&#x27;il s&#x27;agit d&#x27;un chien ou d&#x27;un chat,
   deux classes nous suffit.
&quot;&quot;&quot;</span></code></pre><p>Comme notre modèle se base sur un modèle déjà entraîné, il est inutile que notre modèle se soumettre à un entraînement.</p>
<p>Il faut donc préciser pour chaque couche, qu&#39;on ne souhaite pas effectuer un entraînement.
Il s&#39;agit de mettre l&#39;attribut <code>trainable</code> à <code>False</code>.</p>
<pre class="hljs"><code>
<span class="hljs-keyword">for</span> layer <span class="hljs-keyword">in</span> cnn.layers:
    layer.trainable = <span class="hljs-literal">False</span>
    </code></pre><p>On peut maintenant rajouter la dernière couche, avec le bon nombre de classes à déterminer :</p>
<pre class="hljs"><code>cnn.add(Dense(units=<span class="hljs-number">2</span>, activation=<span class="hljs-string">&#x27;softmax&#x27;</span>))</code></pre><p><a href="developers.google.com/machine_learning/practica/image_classification/exercice_3">Notebook pour déconstruire un autre modèle afin de l&#39;appliquer sur un cas plus restreint</a></p>
<p><a href="https://onedrive.live.com/?authkey=%21AEpov3AcXs2dSqY&id=6C6D756296D4662%21783226&cid=06C6D756296D4662">Ressources du prof (notebooks pour VGG, data)</a></p>
    </article>
  </body>
</html>
