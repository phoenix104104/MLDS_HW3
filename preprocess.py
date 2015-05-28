from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import re
import gensim
import os
import pickle

st = LancasterStemmer()
le = WordNetLemmatizer()

regex_eol = re.compile(r'[.?!]')
regex_symbol = re.compile(r'\W')
regex_line = re.compile(r'[^a-zA-Z0-9 ]')

# terms = pickle.load(open('../test_terms.pkl', 'rb'))

def trim(term) :
	term = st.stem(le.lemmatize(term)).encode('utf-8')
	return term 

def trim_line(line) :
	line = regex_line.sub('', line)
	return [trim(term) for term in line.split()]

def trim_term(term, terms) :
	term = trim(term)
	if term in terms :
		return term
	else :
		return 'others' 

def segmentation(filename, terms) :
	file = open(filename)

	# content start after the line *END*...*END*
	for line in file :
		if 'END' in line[0:5]:
			break 

	lines = [line.split() for line in file if len(line)>1]

	sentences = [[]]
	flag = False

	for i in xrange(len(lines)) :
		for j in xrange(len(lines[i])) :
			# maybe a new sentence
			if flag and 'A' <= lines[i][j][0] <= 'Z' :
				sentences.append([])

			flag = False
			sentences[-1].append(lines[i][j])

			# maybe the end of a sentence
			if regex_eol.match(lines[i][j][-1]):
				flag = True

	# stemming and remove all symbol
	sentences = [[trim_term(regex_symbol.sub('', term), terms) for term in line if regex_symbol.sub('', term) != ''] for line in sentences]
	sentences = [line for line in sentences if len(line) >= 3]

	return sentences

def vectorize(sentences, min_count, size, iteration) :
	model = gensim.models.Word2Vec(sentences, min_count=min_count, size=size, iter=iteration)
	return model

if __name__ == '__main__' :
	import argparse, pickle

	folder = '../'

	parser = argparse.ArgumentParser()
	parser.add_argument('-f', dest='folder', type=str, default=folder, help='path contains data folder, which contains Holmes_Training_Data and test and testing_data.txt, default %s' %folder)
	parser.add_argument('-min', type=int, default=1, help='word2vec min_count, default 1')
	parser.add_argument('-size', type=int, default=100, help='word2vec feature size, default 100')
	parser.add_argument('-iter', type=int, default=2, help='word2vec training iteration, default 2')
	args = parser.parse_args()

	folder = args.folder

	# only consider terms that appear in testing data
	print 'Loading testing data......'
	f_test = open(os.path.join(folder, 'testing_data.txt'))
	lines_test = [trim_line(line)[1:] for line in f_test]
	terms = list(set([t for line in lines_test for t in line]))

	dir_data = os.path.join(folder, 'Holmes_Training_Data')

	dir_feature = os.path.join(folder, 'feature_%d_%d_%d' %(args.min, args.size, args.iter))
	if not os.path.exists(dir_feature) :
		os.makedirs(dir_feature)

	dir_feature_N = os.path.join(dir_feature, 'N')
	if not os.path.exists(dir_feature_N) :
		os.makedirs(dir_feature_N)

	dir_feature_Vec = os.path.join(dir_feature, 'Vec')
	if not os.path.exists(dir_feature_Vec) :
		os.makedirs(dir_feature_Vec)

	# segment data and save with pickle
	f_sentences = os.path.join(folder, 'data_segments_test.pkl')
	if os.path.exists(f_sentences) :
		print 'Loading data segments......'
		sentences = pickle.load(open(f_sentences, 'rb'))
	else :
		print 'Generating data segments......'
		sentences = []
		for filename in os.listdir(dir_data) :
			sentences.extend(segmentation(os.path.join(dir_data, filename), terms))
		pickle.dump(sentences, open(f_sentences, 'wb'))

	# generate word2vec model
	f_model = os.path.join(dir_feature, 'model')
	if os.path.exists(f_model) :
		print 'Loading word2vec model......'
		model = gensim.models.Word2Vec.load(f_model)
	else :
		print 'Generating word2vec model......'
		model = vectorize(sentences, args.min, args.size, args.iter)
		model.save(f_model)

	vocab = model.__dict__['vocab'].keys()

	# generate training feature
	dir_train_N = os.path.join(dir_feature_N, 'train')
	if not os.path.exists(dir_train_N) :
		os.makedirs(dir_train_N)

	dir_train_Vec = os.path.join(dir_feature_Vec, 'train')
	if not os.path.exists(dir_train_Vec) :
		os.makedirs(dir_train_Vec)

	idx = 1
	tmp = 0
	f_train_N = open(os.path.join(dir_train_N, 'train_%03d' %idx), 'w+')
	f_train_Vec = open(os.path.join(dir_train_Vec, 'train_%03d' %idx), 'w+')
	eol = False
	for line in sentences+[[]] :
		if tmp == 20000 :
			tmp = 0
			idx += 1
			f_train_N = open(os.path.join(dir_train_N, 'train_%03d' %idx), 'w+')
			f_train_Vec = open(os.path.join(dir_train_Vec, 'train_%03d' %idx), 'w+')

		for term in line :
			if term in vocab :
				eol = False

				index = vocab.index(term) + 1
				vec = model[term]
				f_train_N.write('%d\n' %index)
				f_train_Vec.write('%d %s\n' %(index, ' '.join([str(x) for x in vec])))

		if not eol :
			tmp += 1
			eol = True
			f_train_N.write('0\n')
			f_train_Vec.write('0 %s\n' %(' '.join(['0'] * args.size)))

	# generate testing feature
	dir_test_N = os.path.join(dir_feature_N, 'test')
	if not os.path.exists(dir_test_N) :
		os.makedirs(dir_test_N)

	dir_test_Vec = os.path.join(dir_feature_Vec, 'test')
	if not os.path.exists(dir_test_Vec) :
		os.makedirs(dir_test_Vec)

	for i in xrange(1, 1041) :
		f_test_N = open(os.path.join(dir_test_N, 'test%04d' %(i)), 'w+')
		f_test_Vec = open(os.path.join(dir_test_Vec, 'test%04d' %(i)), 'w+')

		for j in range(5) :
			line = lines_test[i*5+j-5]
			for term in line :
				if term in vocab :
					index = vocab.index(term) + 1
					vec = model[term]
					f_test_N.write('%d\n' %index)
					f_test_Vec.write('%d %s\n' %(index, ' '.join([str(x) for x in vec])))
			f_test_N.write('0\n')
			f_test_Vec.write('0 %s\n' %' '.join(['0']*len(vec)))
