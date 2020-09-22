{
 "metadata": {
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5-final"
  },
  "orig_nbformat": 2,
  "kernelspec": {
   "name": "python_defaultSpec_1600594530413",
   "display_name": "Python 3.8.5 64-bit"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2,
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "output_type": "stream",
     "name": "stderr",
     "text": "[nltk_data] Downloading package punkt to\n[nltk_data]     C:\\Users\\abi_r\\AppData\\Roaming\\nltk_data...\n[nltk_data]   Package punkt is already up-to-date!\n[nltk_data] Downloading package wordnet to\n[nltk_data]     C:\\Users\\abi_r\\AppData\\Roaming\\nltk_data...\n[nltk_data]   Package wordnet is already up-to-date!\n"
    }
   ],
   "source": [
    "import nltk\n",
    "nltk.download('punkt')\n",
    "nltk.download('wordnet')\n",
    "from nltk.stem import WordNetLemmatizer\n",
    "lemmatizer = WordNetLemmatizer()\n",
    "import json\n",
    "import pickle\n",
    "\n",
    "import numpy as np\n",
    "from keras.models import Sequential\n",
    "from keras.layers import Dense, Activation, Dropout\n",
    "from keras.optimizers import SGD\n",
    "import random"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "words=[]\n",
    "classes = []\n",
    "documents = []\n",
    "ignore_words = ['?', '!']\n",
    "data_file = open('intents.json').read()\n",
    "intents = json.loads(data_file)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "for intent in intents['intents']:\n",
    "    for pattern in intent['patterns']:\n",
    "\n",
    "        # take each word and tokenize it\n",
    "        w = nltk.word_tokenize(pattern)\n",
    "        words.extend(w)\n",
    "        # adding documents\n",
    "        documents.append((w, intent['tag']))\n",
    "\n",
    "        # adding classes to our class list\n",
    "        if intent['tag'] not in classes:\n",
    "            classes.append(intent['tag'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "output_type": "stream",
     "name": "stdout",
     "text": "Training data created\n"
    }
   ],
   "source": [
    "# initializing training data\n",
    "training = []\n",
    "output_empty = [0] * len(classes)\n",
    "for doc in documents:\n",
    "    # initializing bag of words\n",
    "    bag = []\n",
    "    # list of tokenized words for the pattern\n",
    "    pattern_words = doc[0]\n",
    "    # lemmatize each word - create base word, in attempt to represent related words\n",
    "    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]\n",
    "    # create our bag of words array with 1, if word match found in current pattern\n",
    "    for w in words:\n",
    "        bag.append(1) if w in pattern_words else bag.append(0)\n",
    "\n",
    "    # output is a '0' for each tag and '1' for current tag (for each pattern)\n",
    "    output_row = list(output_empty)\n",
    "    output_row[classes.index(doc[1])] = 1\n",
    "\n",
    "    training.append([bag, output_row])\n",
    "# shuffle our features and turn into np.array\n",
    "random.shuffle(training)\n",
    "training = np.array(training)\n",
    "# create train and test lists. X - patterns, Y - intents\n",
    "train_x = list(training[:,0])\n",
    "train_y = list(training[:,1])\n",
    "print(\"Training data created\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {
    "tags": [
     "outputPrepend"
    ]
   },
   "outputs": [
    {
     "output_type": "stream",
     "name": "stdout",
     "text": "acy: 0.1277\nEpoch 3/200\n10/10 [==============================] - 0s 2ms/step - loss: 2.0340 - accuracy: 0.2766\nEpoch 4/200\n10/10 [==============================] - 0s 2ms/step - loss: 1.6355 - accuracy: 0.4894\nEpoch 5/200\n10/10 [==============================] - 0s 1ms/step - loss: 1.6323 - accuracy: 0.4894\nEpoch 6/200\n10/10 [==============================] - 0s 2ms/step - loss: 1.3990 - accuracy: 0.7021\nEpoch 7/200\n10/10 [==============================] - 0s 2ms/step - loss: 1.2778 - accuracy: 0.6809\nEpoch 8/200\n10/10 [==============================] - 0s 2ms/step - loss: 1.1470 - accuracy: 0.6596\nEpoch 9/200\n10/10 [==============================] - 0s 2ms/step - loss: 1.0907 - accuracy: 0.5957\nEpoch 10/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.8437 - accuracy: 0.7447\nEpoch 11/200\n10/10 [==============================] - 0s 2ms/step - loss: 1.0137 - accuracy: 0.5319\nEpoch 12/200\n10/10 [==============================] - 0s 3ms/step - loss: 0.9532 - accuracy: 0.7234\nEpoch 13/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.7527 - accuracy: 0.7872\nEpoch 14/200\n10/10 [==============================] - 0s 3ms/step - loss: 0.8784 - accuracy: 0.6383\nEpoch 15/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.7623 - accuracy: 0.7660\nEpoch 16/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.6827 - accuracy: 0.7660\nEpoch 17/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.6571 - accuracy: 0.7234\nEpoch 18/200\n10/10 [==============================] - 0s 1ms/step - loss: 0.5956 - accuracy: 0.8511\nEpoch 19/200\n10/10 [==============================] - 0s 1ms/step - loss: 0.6704 - accuracy: 0.8085\nEpoch 20/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.5626 - accuracy: 0.8085\nEpoch 21/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.5702 - accuracy: 0.7872\nEpoch 22/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.5338 - accuracy: 0.8085\nEpoch 23/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.5752 - accuracy: 0.8298\nEpoch 24/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.4490 - accuracy: 0.8085\nEpoch 25/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.4394 - accuracy: 0.8085\nEpoch 26/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.4071 - accuracy: 0.8511\nEpoch 27/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.3108 - accuracy: 0.9149\nEpoch 28/200\n10/10 [==============================] - 0s 1ms/step - loss: 0.5778 - accuracy: 0.8085\nEpoch 29/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.4105 - accuracy: 0.8723\nEpoch 30/200\n10/10 [==============================] - 0s 1ms/step - loss: 0.4323 - accuracy: 0.7660\nEpoch 31/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.3804 - accuracy: 0.8298\nEpoch 32/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.4291 - accuracy: 0.8511\nEpoch 33/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.3088 - accuracy: 0.8936\nEpoch 34/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.3472 - accuracy: 0.9149\nEpoch 35/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.2740 - accuracy: 0.9362\nEpoch 36/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.3057 - accuracy: 0.8936\nEpoch 37/200\n10/10 [==============================] - 0s 1ms/step - loss: 0.2479 - accuracy: 0.9362\nEpoch 38/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.3357 - accuracy: 0.9149\nEpoch 39/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.2649 - accuracy: 0.9574\nEpoch 40/200\n10/10 [==============================] - 0s 1ms/step - loss: 0.3723 - accuracy: 0.8511\nEpoch 41/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.1997 - accuracy: 0.9787\nEpoch 42/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.1958 - accuracy: 0.9149\nEpoch 43/200\n10/10 [==============================] - 0s 1ms/step - loss: 0.3154 - accuracy: 0.8723\nEpoch 44/200\n 1/10 [==>...........................] - ETA: 0s - loss: 0.0579 - accuracy: 1.0010/10 [==============================] - 0s 2ms/step - loss: 0.4241 - accuracy: 0.8936\nEpoch 45/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.2526 - accuracy: 0.9149\nEpoch 46/200\n10/10 [==============================] - 0s 1ms/step - loss: 0.3671 - accuracy: 0.9149\nEpoch 47/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.2694 - accuracy: 0.8511\nEpoch 48/200\n10/10 [==============================] - 0s 1ms/step - loss: 0.3114 - accuracy: 0.9149\nEpoch 49/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.3324 - accuracy: 0.8936\nEpoch 50/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.2806 - accuracy: 0.8936\nEpoch 51/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.2561 - accuracy: 0.8936\nEpoch 52/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.2899 - accuracy: 0.8723\nEpoch 53/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.1846 - accuracy: 0.9149\nEpoch 54/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.3036 - accuracy: 0.8723\nEpoch 55/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.2416 - accuracy: 0.8723\nEpoch 56/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.2995 - accuracy: 0.8511\nEpoch 57/200\n10/10 [==============================] - 0s 1ms/step - loss: 0.2015 - accuracy: 0.9362\nEpoch 58/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.2372 - accuracy: 0.8936\nEpoch 59/200\n10/10 [==============================] - 0s 1ms/step - loss: 0.2577 - accuracy: 0.8936\nEpoch 60/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.1790 - accuracy: 0.9362\nEpoch 61/200\n10/10 [==============================] - 0s 1ms/step - loss: 0.2429 - accuracy: 0.8723\nEpoch 62/200\n10/10 [==============================] - 0s 1ms/step - loss: 0.1833 - accuracy: 0.9149\nEpoch 63/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.2204 - accuracy: 0.9149\nEpoch 64/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.2745 - accuracy: 0.8723\nEpoch 65/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.2542 - accuracy: 0.9362\nEpoch 66/200\n10/10 [==============================] - 0s 1ms/step - loss: 0.3414 - accuracy: 0.9149\nEpoch 67/200\n10/10 [==============================] - 0s 1ms/step - loss: 0.1928 - accuracy: 0.9149\nEpoch 68/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.1093 - accuracy: 0.9574\nEpoch 69/200\n10/10 [==============================] - 0s 1ms/step - loss: 0.1672 - accuracy: 0.9574\nEpoch 70/200\n10/10 [==============================] - 0s 1ms/step - loss: 0.1673 - accuracy: 0.9362\nEpoch 71/200\n10/10 [==============================] - 0s 1ms/step - loss: 0.1776 - accuracy: 0.9574\nEpoch 72/200\n10/10 [==============================] - 0s 1ms/step - loss: 0.3487 - accuracy: 0.8936\nEpoch 73/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.2150 - accuracy: 0.9362\nEpoch 74/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.1603 - accuracy: 0.9574\nEpoch 75/200\n10/10 [==============================] - 0s 3ms/step - loss: 0.2534 - accuracy: 0.8936\nEpoch 76/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.1407 - accuracy: 0.9574\nEpoch 77/200\n10/10 [==============================] - 0s 3ms/step - loss: 0.2094 - accuracy: 0.8936\nEpoch 78/200\n10/10 [==============================] - 0s 3ms/step - loss: 0.1582 - accuracy: 0.9362\nEpoch 79/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.1916 - accuracy: 0.9362\nEpoch 80/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.1754 - accuracy: 0.9149\nEpoch 81/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.1441 - accuracy: 0.9574\nEpoch 82/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.1028 - accuracy: 0.9362\nEpoch 83/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.0984 - accuracy: 0.9787\nEpoch 84/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.1500 - accuracy: 0.9362\nEpoch 85/200\n10/10 [==============================] - 0s 3ms/step - loss: 0.1990 - accuracy: 0.9362\nEpoch 86/200\n10/10 [==============================] - 0s 3ms/step - loss: 0.1515 - accuracy: 0.9574\nEpoch 87/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.1734 - accuracy: 0.9362\nEpoch 88/200\n10/10 [==============================] - 0s 3ms/step - loss: 0.1974 - accuracy: 0.9149\nEpoch 89/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.1480 - accuracy: 0.9574\nEpoch 90/200\n10/10 [==============================] - 0s 3ms/step - loss: 0.1436 - accuracy: 0.9362\nEpoch 91/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.1634 - accuracy: 0.9574\nEpoch 92/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.1089 - accuracy: 0.9787\nEpoch 93/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.2486 - accuracy: 0.8723\nEpoch 94/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.1053 - accuracy: 0.9574\nEpoch 95/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.2001 - accuracy: 0.8936\nEpoch 96/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.1233 - accuracy: 0.9362\nEpoch 97/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.1093 - accuracy: 0.9574\nEpoch 98/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.1748 - accuracy: 0.9149\nEpoch 99/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.2161 - accuracy: 0.9574\nEpoch 100/200\n10/10 [==============================] - 0s 1ms/step - loss: 0.0856 - accuracy: 0.9787\nEpoch 101/200\n10/10 [==============================] - 0s 1ms/step - loss: 0.1501 - accuracy: 0.9574\nEpoch 102/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.1065 - accuracy: 0.9787\nEpoch 103/200\n10/10 [==============================] - 0s 1ms/step - loss: 0.1371 - accuracy: 0.9362\nEpoch 104/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.3092 - accuracy: 0.9149\nEpoch 105/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.1037 - accuracy: 0.9574\nEpoch 106/200\n10/10 [==============================] - 0s 1ms/step - loss: 0.2219 - accuracy: 0.9149\nEpoch 107/200\n10/10 [==============================] - 0s 1ms/step - loss: 0.1189 - accuracy: 0.9362\nEpoch 108/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.1466 - accuracy: 0.9362\nEpoch 109/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.1621 - accuracy: 0.9362\nEpoch 110/200\n10/10 [==============================] - 0s 3ms/step - loss: 0.1209 - accuracy: 0.9574\nEpoch 111/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.1270 - accuracy: 0.9574\nEpoch 112/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.2007 - accuracy: 0.9574\nEpoch 113/200\n10/10 [==============================] - 0s 3ms/step - loss: 0.1094 - accuracy: 0.9574\nEpoch 114/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.1347 - accuracy: 0.9362\nEpoch 115/200\n10/10 [==============================] - 0s 6ms/step - loss: 0.1649 - accuracy: 0.9149\nEpoch 116/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.1167 - accuracy: 0.9362\nEpoch 117/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.0727 - accuracy: 0.9787\nEpoch 118/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.1196 - accuracy: 0.9149\nEpoch 119/200\n10/10 [==============================] - 0s 1ms/step - loss: 0.1069 - accuracy: 0.9574\nEpoch 120/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.2497 - accuracy: 0.8511\nEpoch 121/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.0742 - accuracy: 0.9787\nEpoch 122/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.0910 - accuracy: 0.9787\nEpoch 123/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.1783 - accuracy: 0.9149\nEpoch 124/200\n10/10 [==============================] - 0s 1ms/step - loss: 0.2483 - accuracy: 0.9149\nEpoch 125/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.0895 - accuracy: 0.9787\nEpoch 126/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.1444 - accuracy: 0.9362\nEpoch 127/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.1593 - accuracy: 0.9149\nEpoch 128/200\n10/10 [==============================] - 0s 1ms/step - loss: 0.1280 - accuracy: 0.9787\nEpoch 129/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.1651 - accuracy: 0.9362\nEpoch 130/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.1266 - accuracy: 0.9362\nEpoch 131/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.1518 - accuracy: 0.9149\nEpoch 132/200\n10/10 [==============================] - 0s 1ms/step - loss: 0.1029 - accuracy: 0.9574\nEpoch 133/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.1380 - accuracy: 0.9574\nEpoch 134/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.1122 - accuracy: 0.9574\nEpoch 135/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.1078 - accuracy: 0.9574\nEpoch 136/200\n10/10 [==============================] - 0s 1ms/step - loss: 0.1077 - accuracy: 0.9362\nEpoch 137/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.1390 - accuracy: 0.9362\nEpoch 138/200\n10/10 [==============================] - 0s 1ms/step - loss: 0.1016 - accuracy: 0.9574\nEpoch 139/200\n10/10 [==============================] - 0s 1ms/step - loss: 0.1171 - accuracy: 0.9574\nEpoch 140/200\n10/10 [==============================] - 0s 1ms/step - loss: 0.1783 - accuracy: 0.9362\nEpoch 141/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.1713 - accuracy: 0.9362\nEpoch 142/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.0919 - accuracy: 0.9787\nEpoch 143/200\n10/10 [==============================] - 0s 1ms/step - loss: 0.0704 - accuracy: 0.9787\nEpoch 144/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.1110 - accuracy: 0.9574\nEpoch 145/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.0810 - accuracy: 0.9787\nEpoch 146/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.0789 - accuracy: 0.9787\nEpoch 147/200\n10/10 [==============================] - 0s 1ms/step - loss: 0.1340 - accuracy: 0.9362\nEpoch 148/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.1040 - accuracy: 0.9362\nEpoch 149/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.1296 - accuracy: 0.9574\nEpoch 150/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.0943 - accuracy: 0.9574\nEpoch 151/200\n10/10 [==============================] - 0s 1ms/step - loss: 0.1075 - accuracy: 0.9574\nEpoch 152/200\n10/10 [==============================] - 0s 1ms/step - loss: 0.1220 - accuracy: 0.9574\nEpoch 153/200\n10/10 [==============================] - 0s 1ms/step - loss: 0.0697 - accuracy: 0.9787\nEpoch 154/200\n10/10 [==============================] - 0s 1ms/step - loss: 0.0945 - accuracy: 0.9574\nEpoch 155/200\n10/10 [==============================] - 0s 1ms/step - loss: 0.1314 - accuracy: 0.9574\nEpoch 156/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.0862 - accuracy: 0.9787\nEpoch 157/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.1351 - accuracy: 0.9362\nEpoch 158/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.1345 - accuracy: 0.9362\nEpoch 159/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.1069 - accuracy: 0.9787\nEpoch 160/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.0843 - accuracy: 0.9787\nEpoch 161/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.0616 - accuracy: 0.9787\nEpoch 162/200\n10/10 [==============================] - 0s 6ms/step - loss: 0.1845 - accuracy: 0.9149\nEpoch 163/200\n10/10 [==============================] - 0s 1ms/step - loss: 0.1234 - accuracy: 0.9787\nEpoch 164/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.0718 - accuracy: 0.9574\nEpoch 165/200\n10/10 [==============================] - 0s 1ms/step - loss: 0.1522 - accuracy: 0.9149\nEpoch 166/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.0881 - accuracy: 0.9787\nEpoch 167/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.0823 - accuracy: 0.9787\nEpoch 168/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.0998 - accuracy: 0.9787\nEpoch 169/200\n10/10 [==============================] - 0s 1ms/step - loss: 0.1372 - accuracy: 0.9362\nEpoch 170/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.1398 - accuracy: 0.9362\nEpoch 171/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.0984 - accuracy: 0.9787\nEpoch 172/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.2297 - accuracy: 0.8936\nEpoch 173/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.1949 - accuracy: 0.9149\nEpoch 174/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.0931 - accuracy: 0.9574\nEpoch 175/200\n10/10 [==============================] - 0s 1ms/step - loss: 0.1185 - accuracy: 0.9574\nEpoch 176/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.1885 - accuracy: 0.9362\nEpoch 177/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.0502 - accuracy: 1.0000\nEpoch 178/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.0885 - accuracy: 0.9787\nEpoch 179/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.1083 - accuracy: 0.9574\nEpoch 180/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.1022 - accuracy: 0.9574\nEpoch 181/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.1032 - accuracy: 0.9787\nEpoch 182/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.1388 - accuracy: 0.9362\nEpoch 183/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.1059 - accuracy: 0.9787\nEpoch 184/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.1043 - accuracy: 0.9362\nEpoch 185/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.1058 - accuracy: 0.9787\nEpoch 186/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.0721 - accuracy: 0.9787\nEpoch 187/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.1151 - accuracy: 0.9787\nEpoch 188/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.0664 - accuracy: 0.9787\nEpoch 189/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.0902 - accuracy: 0.9362\nEpoch 190/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.0879 - accuracy: 0.9574\nEpoch 191/200\n10/10 [==============================] - 0s 1ms/step - loss: 0.0688 - accuracy: 0.9787\nEpoch 192/200\n10/10 [==============================] - 0s 1ms/step - loss: 0.0835 - accuracy: 0.9787\nEpoch 193/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.1029 - accuracy: 0.9787\nEpoch 194/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.0820 - accuracy: 0.9787\nEpoch 195/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.0689 - accuracy: 0.9787\nEpoch 196/200\n10/10 [==============================] - 0s 1ms/step - loss: 0.0988 - accuracy: 0.9787\nEpoch 197/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.2190 - accuracy: 0.9362\nEpoch 198/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.1164 - accuracy: 0.9787\nEpoch 199/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.1513 - accuracy: 0.9362\nEpoch 200/200\n10/10 [==============================] - 0s 2ms/step - loss: 0.1411 - accuracy: 0.9574\nmodel created\n"
    }
   ],
   "source": [
    "model = Sequential()\n",
    "model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))\n",
    "model.add(Dropout(0.5))\n",
    "model.add(Dense(64, activation='relu'))\n",
    "model.add(Dropout(0.5))\n",
    "model.add(Dense(len(train_y[0]), activation='softmax'))\n",
    "\n",
    "# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model\n",
    "sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)\n",
    "model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])\n",
    "\n",
    "#fitting and saving the model\n",
    "hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)\n",
    "model.save('chatbot_model.h5', hist)\n",
    "\n",
    "print(\"model created\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "output_type": "error",
     "ename": "FileNotFoundError",
     "evalue": "[Errno 2] No such file or directory: 'words.pkl'",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mFileNotFoundError\u001b[0m                         Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-8-ca17e716d03a>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[0;32m      4\u001b[0m \u001b[1;32mimport\u001b[0m \u001b[0mrandom\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      5\u001b[0m \u001b[0mintents\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mjson\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mloads\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mopen\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m'intents.json'\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mread\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m----> 6\u001b[1;33m \u001b[0mwords\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mpickle\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mload\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mopen\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m'words.pkl'\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;34m'rb'\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m      7\u001b[0m \u001b[0mclasses\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mpickle\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mload\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mopen\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m'classes.pkl'\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;34m'rb'\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mFileNotFoundError\u001b[0m: [Errno 2] No such file or directory: 'words.pkl'"
     ]
    }
   ],
   "source": [
    "from keras.models import load_model\n",
    "model = load_model('chatbot_model.h5')\n",
    "import json\n",
    "import random\n",
    "intents = json.loads(open('intents.json').read())\n",
    "words = pickle.load(open('words.pkl','rb'))\n",
    "classes = pickle.load(open('classes.pkl','rb'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ]
}