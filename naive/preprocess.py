from nltk.stem.lancaster import LancasterStemmer
import re

st = LancasterStemmer()
regex0 = re.compile(r'[.?!:;]')
regex1 = re.compile(r'\W')

def segmentation(filename) :
	file = open(filename)

	# content start after the line *END*...*END*
	for line in file :
		if 'END' in line[0:5]:
			break 

	lines = [line.split() for line in file if len(line)>1]

	seg = [[]]
	flag = False

	for i in xrange(len(lines)) :
		for j in xrange(len(lines[i])) :
			# maybe a new sentence
			if flag and 'A' <= lines[i][j][0] <= 'Z' :
				seg.append([])

			flag = False
			seg[-1].append(lines[i][j])

			# maybe the end of a sentence
			if regex0.match(lines[i][j][-1]):
				flag = True

	# stemming and remove all symbol
	seg = [[st.stem(regex1.sub('', term)).encode('utf-8') for term in line if regex1.sub('', term) != ''] for line in seg]

	return seg

def ngram_add_count(ngram, key) :
	if len(key) == 1 :
		if key[0] not in ngram :
			ngram[key[0]] = 1
		else :
			ngram[key[0]] += 1
		return
	else :
		if key[0] not in ngram :
			ngram[key[0]] = {}

		ngram_add_count(ngram[key[0]], key[1:])

def generate_ngram(ngram, seg, n) :
	for line in seg :
		for i in xrange(len(line)-n+1) :
			ngram_add_count(ngram, line[i:i+n])

if __name__ == '__main__' :
	import os, pickle, argparse

	path = '../../Holmes_Training_Data'
	output = '../../'
	n = 2

	parser = argparse.ArgumentParser()
	parser.add_argument('-f', dest='training_folder', type=str, help='training data folder, default %s' %path)
	parser.add_argument('-o', dest='output_folder', type=str, help='output folder, default %s' %output)
	parser.add_argument('-n', type=int, help='n of ngram, default %d' %n)
	args = parser.parse_args()

	if args.training_folder :
		path = args.training_folder
	if args.output_folder :
		output = args.output_folder
	if args.n :
		n = args.n
	
	ngram = {}

	for filename in os.listdir(path) :
		seg = segmentation(os.path.join(path, filename))
		generate_ngram(ngram, seg, n)

	pickle.dump(ngram, open(os.path.join(output, 'ngram_%d.pkl' %n), 'wb'))
