from nltk.stem.lancaster import LancasterStemmer
import re

st = LancasterStemmer()
regex = re.compile(r'[^a-zA-Z0-9 ]')

def extract_query(line, n) :
	terms = []
	[former, latter] = line.split('[', 1)
	former = regex.sub('', former).split()
	terms.extend(former[max(-len(former), -n+1):])

	[word, latter] = latter.split(']', 1)
	terms.append(word)

	latter = regex.sub('', latter).split()
	terms.extend(latter[:min(len(latter), n-1)])

	return terms

def ngram_get_count(ngram, terms) :
	if len(terms) == 0 :
		return 1
	if len(terms) == 1 :
		if terms[0] in ngram :
			return ngram[terms[0]]
		else :
			return 1
	if terms[0] not in ngram :
		return 1
	else :
		return ngram_get_count(ngram[terms[0]], terms[1:])

def compute_score(ngram, terms, n) :
	score = 1
	for i in xrange(len(terms) - n + 1) :
		score *= ngram_get_count(ngram, terms[i:i+n])
	return score

if __name__ == '__main__' :
	import pickle, argparse

	n = 2
	testfile = 'testing_data.txt'
	outputfile = 'predict_%d' %n

	parser = argparse.ArgumentParser()
	parser.add_argument('-i', dest='testfile', type=str, help='path of test file, default %s' %testfile)
	parser.add_argument('-o', dest='outputfile', type=str, help='output file, default %s' %outputfile)
	parser.add_argument('-n', type=int, help='n of ngram, default %d' %n)
	args = parser.parse_args()

	if args.testfile :
		testfile = args.testfile
	if args.outputfile :
		outputfile = args.outputfile
	if args.n :
		n = args.n

	ngram = pickle.load(open('ngram_%d.pkl' %n, 'rb'))
	test = open(testfile)
	output = open(outputfile, 'w+')

	output.write('Id,Answer\n')

	for i in xrange(1, 1041) :
		best = ('a', 0)
		for j in ['a','b','c','d','e'] :
			line = test.readline()
			terms = extract_query(line, n)

			score = compute_score(ngram, terms, n)
			if score > best[1] :
				best = (j, score)

		output.write('%d,%s\n' %(i,best[0]))
