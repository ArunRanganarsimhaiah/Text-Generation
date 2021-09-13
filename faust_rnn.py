#!/usr/bin/env python
# coding: utf-8

# # Projekt 3: Rekurrente neuronale Netze
# 
# Herzlich willkommen zu *Projekt 3: Rekurrente neuronale Netze*!
# 
# In diesem Projekt werden Sie mit einer grundlegenden Architektur rekurrenter neuronaler Netze zur automatisierten Generierung von Texten arbeiten. In diesem Kontext kommen sogenannte *Sequence-to-Sequence*-Modelle zum Einsatz, welche eine Sequenz als Input erhalten und ebenfalls eine Sequenz als Output produzieren (siehe (Abb. 1)). *Sequence-to-Vector*-Modelle funktionieren ähnlich, geben aber nur einen einzelnen Output im finalen Zeitschritt zurück (siehe (Abb. 2)). Diese Architektur ist zum Beispiel für Klassifikationsaufgaben geeignet, bei denen die Inputobjekte Texte sind.
# 
# <table style="width:100%">
#     <tr>
#         <th><img src="rnn_seq_2_seq.png?666" alt="" style="width: 475px;"></th>
#         <th><img src="rnn_seq_2_vec.png?666" alt="" style="width: 475px;"></th>
#     </tr>
#     <tr>
#         <th>(Abb. 1) Sequence-to-Sequence RNN mit einer verdeckten Schicht.</th>
#         <th>(Abb. 2) Sequence-to-Vector RNN mit einer verdeckten Schicht.</th>
#     </tr>
# </table>
# 
# Der Zeitraum für die Bearbeitung des Projektes dauert bis zum **18. Dezember 2019** um 9:45 Uhr. Senden Sie bitte bis zu diesem Termin alle abzugebenden Dateien an **matthias.neumann-brosig@tu-bs.de** sowie **m.lahmann@tu-bs.de**.

# In[4]:


import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import io
from tensorflow import keras


# ## 3.1 Ein zeichenbasiertes RNN für Textgenerierung

# 1. Lesen Sie die Dateien ```faust_1.txt``` und ```faust_2.txt``` ein und schreiben Sie den jeweiligen Inhalt in Variablen ```faust_1_text``` bzw. ```faust_2_text```. Berücksichtigen Sie, dass beide Dateien in UTF-8 kodiert sind. Hilfreiche Informationen finden Sie unter https://docs.python.org/3/tutorial/inputoutput.html#reading-and-writing-files.

# In[5]:


# IHR CODE
#faust_1_text=io.open('faust_1.txt', 'rb').read().decode(encoding='utf-8')
#faust_2_text=io.open('faust_2.txt', 'rb').read().decode(encoding='utf-8')

with io.open('faust_1.txt',mode='r',encoding='utf-8') as f:
    faust_1_text=f.read()
    
with io.open('faust_2.txt',mode='r',encoding='utf-8') as f:
    faust_2_text=f.read()


# 2. Lassen Sie die ersten 434 Zeichen von ```faust_1_text```ausgeben.
# 
# **Checkpoint:** Sie sollten folgende Ausgabe erhalten:
# 
# ```
# Faust: Der Tragödie erster Teil
# 
# Johann Wolfgang von Goethe
# 
# 
# Zueignung.
# 
# Ihr naht euch wieder, schwankende Gestalten,
# Die früh sich einst dem trüben Blick gezeigt.
# Versuch ich wohl, euch diesmal festzuhalten?
# Fühl ich mein Herz noch jenem Wahn geneigt?
# Ihr drängt euch zu!  nun gut, so mögt ihr walten,
# Wie ihr aus Dunst und Nebel um mich steigt;
# Mein Busen fühlt sich jugendlich erschüttert
# Vom Zauberhauch, der euren Zug umwittert.
# 
# ```

# In[6]:


# IHR CODE
print(faust_1_text[:434])


# 3. Instanziieren Sie ein Objekt ```tokenizer``` vom Typ ```tf.keras.preprocessing.text.Tokenizer``` mit dem Argument ```char_level=True```. Hilfreiche Informationen finden Sie unter https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/text/Tokenizer.

# In[7]:


# IHR CODE
tokenizer=tf.keras.preprocessing.text.Tokenizer(char_level=True)


# Das gerade erzeugte Objekt werden wir letztlich dazu verwenden, die Strings ```faust_1_text``` und ```faust_2_text``` als ganzzahlige Sequenzen zu kodieren, in denen jede Zahl einem einzelnen Zeichen entspricht.
# 
# 4. Rufen Sie die Methode ```fit_on_texts``` von ```tokenizer``` auf. Übergeben Sie der Methode eine Liste, die genau die zwei Strings ```faust_1_text``` und ```faust_2_text``` enthält.

# In[8]:


# IHR CODE
texts=[faust_1_text,faust_2_text]
tokenizer.fit_on_texts(texts)


# Durch den Aufruf von ```fit_on_texts``` wurde in ```tokenizer``` ein Vokabular angelegt, welches jedem in ```faust_1_text``` oder ```faust_2_text``` vorkommenden Zeichen eine positive ganze Zahl zuordnet.
# 
# 5. Lassen Sie das Attribut ```word_index``` von ```tokenizer``` ausgeben, welches das angelegte Vokabular enthält.
# 
# **Checkpoint:** Sie sollten folgende Ausgabe erhalten:
# 
# ```
# {' ': 1, 'e': 2, 'n': 3, 'i': 4, 'r': 5, 's': 6, 'h': 7, 't': 8, 'a': 9, 'd': 10, '\n': 11, 'l': 12, 'c': 13, 'u': 14, 'g': 15, 'm': 16, 'o': 17, ',': 18, 'w': 19, 'b': 20, 'f': 21, 'k': 22, 'z': 23, '.': 24, 'p': 25, 'ü': 26, 'v': 27, ':': 28, '!': 29, 'ä': 30, 'ß': 31, "'": 32, 'ö': 33, ';': 34, '?': 35, '-': 36, 'j': 37, '(': 38, ')': 39, 'y': 40, 'q': 41, 'x': 42, '+': 43, '"': 44, '2': 45, '1': 46, '4': 47, '/': 48, '3': 49, '5': 50, '[': 51, '6': 52, '*': 53, '=': 54}
# ```

# In[9]:


# IHR CODE
print(tokenizer.word_index)


# 6. Berechnen Sie die Länge des Vokabulars, schreiben Sie diese in ```max_id``` und lassen Sie ```max_id``` ausgeben.

# In[10]:


# IHR CODE
max_id=len(tokenizer.word_index)
print(max_id)


# 7. Nutzen Sie nun die Methode ```texts_to_sequences``` von ```tokenizer```, um ```faust_1_text``` und ```faust_2_text``` zu kodieren. Schreiben Sie die kodierten Versionen in ```faust_1_encoded``` bzw. ```faust_2_encoded```. Konvertieren Sie beide Listen in das Format ```numpy.array``` und subtrahieren Sie von allen Einträgen ```1```, sodass die Werte anschließend zwischen```0``` und ```max_id-1``` liegen (diesen Wertebereich werden wir später benötigen).

# In[11]:


# IHR CODE
faust_1_encoded=tokenizer.texts_to_sequences(faust_1_text)
faust_2_encoded=tokenizer.texts_to_sequences(faust_2_text)
faust_1_encoded=np.array(faust_1_encoded,dtype=np.int64)-1
faust_2_encoded=np.array(faust_2_encoded,dtype=np.int64)-1
faust_1_encoded=faust_1_encoded.reshape(faust_1_encoded.size)
faust_2_encoded=faust_2_encoded.reshape(faust_2_encoded.size)


# 8. Lassen Sie die ersten 434 Einträge von ```faust_1_encoded```, sowie jeweils den kleinsten und größten Eintrag von ```faust_1_encoded``` und ```faust_2_encoded``` ausgeben.
# 
# **Checkpoint:** Sie sollten folgende Ausgabe erhalten:
# 
# ```
# [20  8 13  5  7 27  0  9  1  4  0  7  4  8 14 32  9  3  1  0  1  4  5  7
#   1  4  0  7  1  3 11 10 10 36 16  6  8  2  2  0 18 16 11 20 14  8  2 14
#   0 26 16  2  0 14 16  1  7  6  1 10 10 10 22 13  1  3 14  2 13  2 14 23
#  10 10  3  6  4  0  2  8  6  7  0  1 13 12  6  0 18  3  1  9  1  4 17  0
#   5 12  6 18  8  2 21  1  2  9  1  0 14  1  5  7  8 11  7  1  2 17 10  9
#   3  1  0 20  4 25  6  0  5  3 12  6  0  1  3  2  5  7  0  9  1 15  0  7
#   4 25 19  1  2  0 19 11  3 12 21  0 14  1 22  1  3 14  7 23 10 26  1  4
#   5 13 12  6  0  3 12  6  0 18 16  6 11 17  0  1 13 12  6  0  9  3  1  5
#  15  8 11  0 20  1  5  7 22 13  6  8 11  7  1  2 34 10 20 25  6 11  0  3
#  12  6  0 15  1  3  2  0  6  1  4 22  0  2 16 12  6  0 36  1  2  1 15  0
#  18  8  6  2  0 14  1  2  1  3 14  7 34 10  3  6  4  0  9  4 29  2 14  7
#   0  1 13 12  6  0 22 13 28  0  0  2 13  2  0 14 13  7 17  0  5 16  0 15
#  32 14  7  0  3  6  4  0 18  8 11  7  1  2 17 10 18  3  1  0  3  6  4  0
#   8 13  5  0  9 13  2  5  7  0 13  2  9  0  2  1 19  1 11  0 13 15  0 15
#   3 12  6  0  5  7  1  3 14  7 33 10 15  1  3  2  0 19 13  5  1  2  0 20
#  25  6 11  7  0  5  3 12  6  0 36 13 14  1  2  9 11  3 12  6  0  1  4  5
#  12  6 25  7  7  1  4  7 10 26 16 15  0 22  8 13 19  1  4  6  8 13 12  6
#  17  0  9  1  4  0  1 13  4  1  2  0 22 13 14  0 13 15 18  3  7  7  1  4
#   7 23]
# 
# 0	43	0	53
# ```

# In[12]:


# IHR CODE
print(faust_1_encoded[:434])
print(np.min(faust_1_encoded))
print(np.max(faust_1_encoded))
print(np.min(faust_2_encoded))
print(np.max(faust_2_encoded))


# Die ursprünglichen Texte lassen sich nun mittels der Methode ```sequences_to_texts``` von ```tokenizer``` zurück gewinnen. Dekodierte Texte enthalten ausschließlich Kleinbuchstaben und auf jedes Zeichen folgt ein Leerzeichen.
# 
# 9. Wenden Sie ```sequences_to_texts``` auf ```faust_1_encoded + 1``` und schreiben Sie das Ergebnis in einen String ```faust_1_decoded```. Lassen Sie anschließend die ersten 867 Zeichen von ```faust_1_decoded``` ausgeben.
# 
# **Checkpoint:** Sie sollten folgende Ausgabe erhalten:
# 
# ```
# f a u s t :   d e r   t r a g ö d i e   e r s t e r   t e i l 
#  
#  j o h a n n   w o l f g a n g   v o n   g o e t h e 
#  
#  
#  z u e i g n u n g . 
#  
#  i h r   n a h t   e u c h   w i e d e r ,   s c h w a n k e n d e   g e s t a l t e n , 
#  d i e   f r ü h   s i c h   e i n s t   d e m   t r ü b e n   b l i c k   g e z e i g t . 
#  v e r s u c h   i c h   w o h l ,   e u c h   d i e s m a l   f e s t z u h a l t e n ? 
#  f ü h l   i c h   m e i n   h e r z   n o c h   j e n e m   w a h n   g e n e i g t ? 
#  i h r   d r ä n g t   e u c h   z u !     n u n   g u t ,   s o   m ö g t   i h r   w a l t e n , 
#  w i e   i h r   a u s   d u n s t   u n d   n e b e l   u m   m i c h   s t e i g t ; 
#  m e i n   b u s e n   f ü h l t   s i c h   j u g e n d l i c h   e r s c h ü t t e r t 
#  v o m   z a u b e r h a u c h ,   d e r   e u r e n   z u g   u m w i t t e r t .
# ```

# In[13]:


# IHR CODE
faust_1_decoded=''.join(tokenizer.sequences_to_texts([faust_1_encoded+1]))
print(faust_1_decoded[:867])


# 10. Erzeugen Sie nun zwei Objekte ```faust_1_dataset``` und ```faust_2_dataset``` vom Typ ```tf.data.Dataset```, indem Sie die Methode ```tf.data.Dataset.from_tensor_slices``` (siehe https://www.tensorflow.org/api_docs/python/tf/data/Dataset#from_tensor_slices) auf ```faust_1_encoded``` und ```faust_2_encoded```anwenden.

# In[14]:


# IHR CODE
faust_1_dataset=tf.data.Dataset.from_tensor_slices(faust_1_encoded)
faust_2_dataset=tf.data.Dataset.from_tensor_slices(faust_2_encoded)


# 11. Lassen Sie die ersten zehn Elemente von ```faust_1_dataset``` ausgeben.
# 
# **Checkpoint:** Sie sollten folgende Ausgabe erhalten:
# 
# ```
# tf.Tensor(20, shape=(), dtype=int64)
# tf.Tensor(8, shape=(), dtype=int64)
# tf.Tensor(13, shape=(), dtype=int64)
# tf.Tensor(5, shape=(), dtype=int64)
# tf.Tensor(7, shape=(), dtype=int64)
# tf.Tensor(27, shape=(), dtype=int64)
# tf.Tensor(0, shape=(), dtype=int64)
# tf.Tensor(9, shape=(), dtype=int64)
# tf.Tensor(1, shape=(), dtype=int64)
# tf.Tensor(4, shape=(), dtype=int64)
# ```

# In[15]:


# IHR CODE

for x in faust_1_dataset.take(10):
    print(x)
    
   


# Wie Sie sehen, ist jedes Element von ```faust_1_dataset``` ein ganzzahliger Tensor mit einem einzelnen Eintrag. Entsprechend hat ```faust_1_dataset``` insgesamt ```len(faust_1_encoded)``` Elemente (entsprechendes gilt für ```faust_2_dataset```). Wir wollen im Folgenden ein rekurrentes neuronales Netz trainieren, welches in jedem Zeitschritt einen kodierten String der Länge ```T = 100``` erhält und daraufhin das nächste Zeichen vorhersagt. Entsprechend wollen wir ```faust_1_dataset``` und ```faust_2_dataset``` nun zunächst derart transformieren, dass die Elemente anschließend eindimensionale Tensoren der Länge ```window_length = 101``` sind.
# 
# 12. Initialisieren Sie ```T``` und ```window_length``` wie oben beschrieben.

# In[16]:


# IHR CODE
T = 100
window_length = 101


# 13. Verwenden Sie nun die Methode ```tf.data.Dataset.window``` (siehe https://www.tensorflow.org/api_docs/python/tf/data/Dataset#window), um jeweils Datensätze mit Elementen der oben beschriebenen Länge zu erhalten. Rufen Sie die Methode mit optionalen Argumenten ```shift=1``` und ```drop_remainder=True``` auf.
# 
# Das erstgenannte Argument ```shift=1``` sorgt dafür, dass das erste Element eines transformierten Datensatzes die Elemente ```0,...,window_length-1``` des ursprünglichen Datensatzes enthält, danach ```1,...,window_length``` und so weiter. Mit anderen Worten: Ein *Fenster* der Länge ```window_length``` gleitet mit Vorschub ```shift``` über den ursprünglichen Datensatz und extrahiert Sequenz für Sequenz bis das Ende erreicht ist. Das zweitgenannte Argument ```drop_remainder=True``` sorgt dafür, dass am Ende des Datensatzes keine kleineren Sequenzen extrahiert werden, indem das Fenster über das Ende des Datensatzes hinausgleitet.

# In[17]:


# IHR CODE
faust_1_dataset=faust_1_dataset.window(size=window_length,shift=1,drop_remainder=True)
faust_2_dataset=faust_2_dataset.window(size=window_length,shift=1,drop_remainder=True)


# 14. Führende Sie den Code in der folgenden Zelle aus.
# 
# **Checkpoint:** Sie sollten folgende Ausgabe erhalten:
# 
# ```
# <_VariantDataset shapes: (), types: tf.int64>
# tf.Tensor(20, shape=(), dtype=int64)
# tf.Tensor(8, shape=(), dtype=int64)
# tf.Tensor(13, shape=(), dtype=int64)
# tf.Tensor(5, shape=(), dtype=int64)
# tf.Tensor(7, shape=(), dtype=int64)
# tf.Tensor(27, shape=(), dtype=int64)
# tf.Tensor(0, shape=(), dtype=int64)
# tf.Tensor(9, shape=(), dtype=int64)
# tf.Tensor(1, shape=(), dtype=int64)
# tf.Tensor(4, shape=(), dtype=int64)
# ```

# In[18]:


for window in faust_1_dataset.take(1):
    print(window)
for item in window.take(10):
    print(item)


# Wir sehen also, dass die transformierten Datensätze ```faust_1_dataset``` und ```faust_2_dataset``` nunmehr Elemente enthalten, die selbst Objekte vom Typ ```tf.data.Dataset``` (bzw. einer davon abgeleiteten Klasse) sind. Jeder einzelne dieser Datensätze ```window``` enthält nun ```window_size``` viele Tensoren mit jeweils einem einzelnen Eintrag.

# 15. Wenden Sie die Methode ```tf.data.Dataset.flat_map``` (siehe https://www.tensorflow.org/api_docs/python/tf/data/Dataset#flat_map) auf ```faust_1_dataset``` und ```faust_2_dataset``` an, um die  inneren Datensätze ```window``` in eindimensionale Tensoren der Länge ```window_length``` zu transformieren. Übergeben Sie der Methode dafür jeweils eine Funktion, welche ```window``` auf ```window.batch(window_length)``` abbildet.

# In[19]:


# IHR CODE
faust_1_dataset = faust_1_dataset.flat_map(lambda window: window.batch(window_length))
faust_2_dataset = faust_2_dataset.flat_map(lambda window: window.batch(window_length))


# 16. Lassen Sie das erste Element von ```faust_1_dataset``` ausgeben.
# 
# **Checkpoint:** Sie sollten folgende Ausgabe erhalten:
# 
# ```
# tf.Tensor(
# [20  8 13  5  7 27  0  9  1  4  0  7  4  8 14 32  9  3  1  0  1  4  5  7
#   1  4  0  7  1  3 11 10 10 36 16  6  8  2  2  0 18 16 11 20 14  8  2 14
#   0 26 16  2  0 14 16  1  7  6  1 10 10 10 22 13  1  3 14  2 13  2 14 23
#  10 10  3  6  4  0  2  8  6  7  0  1 13 12  6  0 18  3  1  9  1  4 17  0
#   5 12  6 18  8], shape=(101,), dtype=int64)
# ```

# In[20]:


# IHR CODE
for item in faust_1_dataset.take(1):
    print(item)
    


# 17. Verwenden Sie nun die Methode ```tf.data.Dataset.concatenate``` von ```faust_1_dataset``` (siehe https://www.tensorflow.org/api_docs/python/tf/data/Dataset#concatenate), um ```faust_1_dataset``` und ```faust_2_dataset``` zu einem Datensatz zu vereinen.

# In[21]:


# IHR CODE

faust_dataset=faust_1_dataset.concatenate(faust_2_dataset)


# 18. Setzen Sie nun ```batch_size = 32``` und wenden sich zunächst ```tf.data.Dataset.repeat``` (ohne Argument), dann ```tf.data.Dataset.shuffle``` mit ```buffer_size=10000``` und anschließend ```tf.data.Dataset.batch``` auf ```faust_dataset``` an.

# In[22]:


tf.random.set_seed(0)
# IHR CODE
batch_size = 32
faust_dataset=faust_dataset.repeat().shuffle(buffer_size=10000).batch(batch_size)


# Unser Datensatz enthält nun zweidimensionale Tensoren ```window_batch``` der Größe ```(32, 101)```. Jede Schicht ```window_batch[i, :]``` enthält ein Trainingsbeispiel (hier ist jedes Trainingsbeispiel eine Sequenz von Inputs und Outputs), welches allerdings noch nicht fertig in Inputs und Outputs unterteilt ist. Jedes einzelne kodierte Zeichen ```window_batch[i, j]``` für ```j=0,...,99``` ist ein Input $\mathbf{x}^{<t>(i)}$ in einem Zeitschritt mit zugehörigem Output ```window_batch[i, j+1]```, welcher $\mathbf{y}^{<t>(i)}$ entspricht.
# 
# 19. Wenden Sie die Methode ```tf.data.Dataset.map``` (siehe https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map) auf ```faust_dataset``` an. Jeder Stapel ```window_batch``` soll auf ein Tupel aus zwei Tensoren der Größe ```(32, 100)``` abgebildet werden. Die ```[i, :]```-te Schicht des ersten Tensors soll die Einträge ```window_batch[i, 0:100]``` enthalten. Die entsprechende Schicht des zweiten Tensors im Tupel soll die Einträge ```window_batch[i, 1:101]``` enthalten.

# In[23]:


# IHR CODE

faust_dataset=faust_dataset.map(lambda window_batch: (window_batch[:, 0:100], window_batch[:, 1:101]))


# 20. Führen Sie den Code in der folgenden Zelle aus.
# 
# **Checkpoint:** Sie sollten folgende Ausgabe erhalten:
# 
# ```
#  w e n n   d i c h   i n   s c h l a c h t e n   f e i n d e   d r ä n g e n , 
#  w e n n   m i t   g e w a l t   a n   d e i n e n   h a l s 
#  s i c h   a l l e r l i e b s t e   m ä d c h e n   h ä
# 
# w e n n   d i c h   i n   s c h l a c h t e n   f e i n d e   d r ä n g e n , 
#  w e n n   m i t   g e w a l t   a n   d e i n e n   h a l s 
#  s i c h   a l l e r l i e b s t e   m ä d c h e n   h ä n
# 
# ```

# In[24]:


for window_batch in faust_dataset.take(1):
    [x] = tokenizer.sequences_to_texts([window_batch[0][0, :].numpy() + 1])
    [y] = tokenizer.sequences_to_texts([window_batch[1][0, :].numpy() + 1])
    print(x)
    print()
    print(y)


# 21. Wenden Sie nun nochmals ```tf.data.Dataset.map``` auf ```faust_dataset```, um jedes Element ```(X_batch, Y_batch)``` auf ein neues zweielementiges Tupel abzubilden, wobei ```Y_batch``` unverändert bleibt und ```X_batch``` mittels ```tf.one_hot``` (siehe https://www.tensorflow.org/api_docs/python/tf/one_hot) kodiert wird. Finden Sie heraus, welchen Wert ```depth``` Sie ```tf.one_hot``` zusätzlich zu ```X_batch``` konsequenterweise übergeben müssen. Hinweis: Sie haben den passenden Wert oben bereits berechnet.

# In[25]:


# IHR CODE

faust_dataset=faust_dataset.map(lambda X_batch, Y_batch:(tf.one_hot(X_batch, depth=max_id),Y_batch))


# 22. Wenden Sie schließlich die Methode ```tf.data.Dataset.prefetch``` mit ```buffer_size=1``` auf ```faust_dataset``` an, um den fertigen Datensatz für das Training zu erhalten.

# In[26]:


# IHR CODE
faust_dataset=faust_dataset.prefetch(buffer_size=1)


# 23. Leiten Sie eine Formel für die Anzahl der Elemente in ```faust_dataset``` her und schreiben Sie das Ergebnis in ```steps_per_epoch```. Lassen Sie anschließend ```steps_per_epoch``` ausgeben.
# 
# **Checkpoint:** Sie sollten folgende Ausgabe erhalten:
# 
# ```
# 14829
# ```

# In[30]:


# IHR CODE
steps_per_epoch=np.ceil((len(faust_1_encoded)+len(faust_2_encoded)-2*T)/32)
print(steps_per_epoch)


# 24. Definieren Sie nun mittels ```keras.models.Sequential``` ein Modell ```model```, welches über zwei verdeckte GRU-Schichten (siehe https://www.tensorflow.org/api_docs/python/tf/keras/layers/GRU) mit jeweils ```128``` Neuronen sowie eine vollständig verbundene Outputschicht (https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense) mit ```max_id``` Neuronen und ```softmax```-Aktivierungsfunktion verfügt. Dieses Modell entspricht schematisch dem in (Abb. 1) dargestellten RNN mit einer zusätzlichen verdeckten GRU-Schicht. Damit das Modell den Output der vollständig verbundenen Schicht in jedem Zeitschritt erzeugt, müssen Sie diese Schicht von außen mit einem ```keras.layers.TimeDistributed```-Wrapper (siehe https://www.tensorflow.org/api_docs/python/tf/keras/layers/TimeDistributed) umschließen. Wenn Sie den Wrapper weglassen, dann wird der Output der vollständig verbundenen Schicht ausschließlich im letzten Zeitschritt erzeugt, wie in (Abb. 2) dargestellt. Im Fall von GRU-Schichten müssen Sie zum selben Zweck das Argument ```return_sequences=True``` übergeben. Denken Sie außerdem daran, der Inputschicht das Argument ```input_shape=[None, max_id]``` zur Verfügung zu stellen. Der erste Eintrag ```None``` steht dabei für die zeitliche Dimension der Inputsequenzen. Während des Trainings könnten wir ```None``` hier durch ```T``` ersetzen, da alle verwendeten Sequenzen von einheitlicher Länge sind. Die Verwendung von ```None``` stellt allerdings sicher, dass wir später auch Sequenzen mit mehr oder weniger als ```T``` Elementen als Input verwenden können.

# In[31]:


# IHR CODE
model=keras.models.Sequential([keras.layers.GRU(128,return_sequences=True, input_shape=[None,max_id]),
                                keras.layers.GRU(128,return_sequences=True),
                            keras.layers.TimeDistributed(keras.layers.Dense(max_id,activation="softmax"))])


# 25. Kompilieren Sie ```model``` unter Verwendung von ```"sparse_categorical_crossentropy"``` und ```"adam"``` (mit Standardargumenten).

# In[32]:


# IHR CODE
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")


# 26. Trainieren Sie das Modell zunächst für eine Epoche. Denken sie dabei daran, das Argument ```steps_per_epoch``` zu übergeben.

# In[33]:


# IHR CODE
#history = model.fit(faust_dataset, epochs=1,steps_per_epoch=steps_per_epoch)


# In[ ]:

callback = keras.callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
history = model.fit(faust_dataset, epochs=20,steps_per_epoch=steps_per_epoch,callbacks=[callback])
model.save('faust_model.h5')
#history = model.fit(faust_dataset, epochs=20,steps_per_epoch=steps_per_epoch,callbacks=[keras.callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True]))

# 27. Sie haben nun vermutlich festgestellt, dass bereits eine einzelne Epoche verhältnismäßig viel Rechenzeit benötigt. Lagern Sie den bisher erstellten Code in eine Datei ```faust_rnn.py``` aus und trainieren Sie das Modell auf dem GPU-Cluster für 20 Epochen unter Verwendung eines Callbacks:
# 
# ```keras.callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)```
# 
# Speichern Sie das Modell als ```faust_model.h5``` ab und laden Sie es im folgenden Schritt in ```model``` (das oben für eine Epoche trainierte Modell wird dabei überschrieben).
# 
# Solange Ihr Programm auf dem GPU-Cluster läuft, können Sie mit Ihrem für eine Epoche trainierten Modell weiter im Notebook arbeiten. Überspringen Sie dafür einfach vorerst das Laden in diesem Schritt.

# In[ ]:


# IHR CODE
#model = keras.models.load_model('faust_model.h5')


# 28. Schreiben Sie eine Funktion ```preprocess```, welche als Argument eine Liste ```texts``` mit einem einzigen String-Element erhält. Nutzen Sie innerhalb der Funktion wieder ```tokenizer```, um den in ```texts``` enthaltenen String letztlich als NumPy-Array ```X``` zu kodieren. Der Rückgabewert von ```preprocess``` soll anschließend die One-Hot-kodierte Version von ```X``` sein.

# In[34]:


# IHR CODE
def preprocess(texts):
    X = np.array(tokenizer.texts_to_sequences(texts)) - 1
    return tf.one_hot(X, max_id)


# 29. Die Funktion ```next_char``` erhält als Argument einen String ```text``` sowie eine positive Zahl ```temperature``` mit dem Ziel, ein auf ```text``` folgendes Zeichen zu generieren. Ersetzen Sie alle Platzhalter ```None``` folgendermaßen: Wenden Sie zunächst ```preprocess``` an, um ```text``` zu kodieren und schreiben Sie das Ergebnis in ```X_new```. Wenden Sie dann ```model.predict``` an, um die Modelloutputs zu der eingegebenen Sequenz ```X_new``` zu generieren. Extrahieren Sie aus dem Ergebnis den Output für den letzten Zeitschritt (bzw. für das letzte Zeichen in ```text```) und schreiben Sie diesen in ```y_proba```. Über ```rescaled_logits``` und ```char_id``` wird letztlich eine einelementige Stichprobe aus $\{1,\dots,\mathrm{max\_id}\}$ generiert, welche die kodierte Version des generierten Zeichens darstellt. Dekodieren Sie ```char_id.numpy()``` und geben Sie das resultierende Zeichen als String zurück.

# In[35]:


def next_char(text, temperature=1):
    X_new = preprocess(text)
    y_proba = model.predict(X_new)[0,-1:,:]
    rescaled_logits = tf.math.log(y_proba) / temperature
    char_id = tf.random.categorical(rescaled_logits, num_samples=1) + 1
    return tokenizer.sequences_to_texts(char_id.numpy())[0]


# 30. Die Funktion ```complete_text``` erhält als Argument einen String ```text``` und soll diesen um ```n_chars``` neue Zeichen ergänzen. Ergänzen Sie die Funktion entsprechend. Das Argument ```temperature``` soll lediglich an ```next_char``` weitergegeben werden.

# In[ ]:


def complete_text(text, n_chars=50, temperature=1):
    for _ in range(n_chars):
        text += next_char(text,temperature)
    return text


# 31. Nutzen Sie nun ```complete_text```, um ausgehend von dem String ```"Faust"``` einen Text mit ```1000``` neuen Zeichen zu generieren und lassen Sie diesen Text ausgeben. Das Argument ```temperature``` können Sie variieren, um die Wahrscheinlichkeitsverteilung, gemäß derer neue Zeichen gezogen werden, manuell zu beeinflussen. Werte nahe Null begünstigen Zeichen, die gemäß der ursprünglichen (durch das RNN generierten) Verteilung bereits wahrscheinlich sind. Sehr hohe Werte führen dazu, dass alle Zeichen im Vokabular mit gleicher Wahrscheinlichkeit gezogen werden (nicht wünschenswert). Sie können zum Beispiel verschiedene Werte zwischen ```0``` und ```2``` ausprobieren und jeweils die Plausibilität der generierten Texte bewerten.

# In[ ]:


# IHR CODE
print(complete_text('Faust',1000,temperature=1))

# 32. Laden Sie einen Text Ihrer Wahl und trainieren Sie darauf basierend ein weiteres Modell für zeichenbasierte Textgenerierung. Der gewählte Text muss nicht deutschsprachig sein, Sie sollten aber darauf achten, dass er keinen zu geringen Umfang hat. Sie dürfen obige Modellarchitektur und das verwendete Trainingssetting modifizieren, zum Beispiel können Sie weitere ```GRU```-Schichten einfügen oder die Anzahl der Neuronen modifizieren.
