# Sentiment-Analysis-on-Movie-Reviews
Sentiment Analysis on The Rotten Tomatoes Movie Reviews

--
PREPROCESS

-tokenize text (parse them as single words)
	-parse from spaces and separating punctuations
	-specific word grups shouldt be parsed from space in it like "San Francisco"

-lemmatize tokens (put into standard form)
	-number should be written (instead of "7" convert it to "seven")
	-all should be capital or all uncapital
	
-set a vocabulary from the whole set and give them all a unique id
	-to create a vocabulary set delete uninformative words like "the", "an" ...
	-most frequent words could be selected to create a vocabulary list (sense list)
	-all remaining words that couldt be found in vocabulary list will take another specific id so add one more id for the remaning ones to the list
	-id's in convention is noted as "w" (categorical feature for the original word)

--	
WORD REPRESENTATIONS
	
-one hot encoding (to vectorize the word to feed through the any machine learning algorithm)
	-let say vocabulary size is D=10 (means 9 words are in the list and 1 id is specified for remaning out of vocabulary representation) 
		for example for word id 4 (w=4) in the list the corresponding one-hot vector is e(w=4)=[0001000000]
	-one-hot encoding makes no assumption about word similarities such that |e(w)-e(w')|^(2)=0 if ws (words) are same or 2 if words are different
		which means words are equally different from each other everytime
		if we use w's directly the distance brtween for example w=5 and w=100 wouldnt be same 
		one hot gives same distance oppotunity without considering similarities between meanings of the words
	-the problem is one hot coding is very high dimentional since each e() for specific w has the size of whole vocabulary
		for example if vocabulary size is 100 000 for 10 word window size we need 1 000 000 units to extract a meaning from it
	-moreover although one hot vectors are sparse and vector multipication for example done efficiently, 
		computations from the second level when non sparse vectors appears will expensive in terms of logical power
		and reconstruction of the sentence vector forms are hard to do
	-vulnerability to overfitting since high dimentional
		millions of inputs means millions of parameters to train in regular nural network
		
--

-continuous word representation (learn continuous representation of words, each w is associated with a real valued vector C(w))
	-for example the word "the" has id w=1 and C(w=1)=[0.67,0.9,...] such that similar words in meaning has similar vectors means close each other in euclidian distance
		the thing is trying to learn the vectors of words (using neural network)
	-let say window of 10 words [w1,w2...w10] concatenate the vector representation of each word as x=[C(w0)(transpose),C(w1)(transpose),...] and feed that to neural network
	-use stocastic (?) gradient descent
		-dont only update neural net parameters also update C(w) in the x with gradient step as [ C(w) <= C(w)-(alpha)(gradientC(w))l ] l is the loss function optimised by neural net
		
-C matrix has all C(w) vectors as its row 
	-C(w) = e(w)(transpose).C one hot of w and C multiplied to get C(w) 
		bu instead of vector multiplication C(w)s are located to a lookup table seperately

--
LANGUAGE MODELING

-probabilistic model that assign probabilities to any sequence of words