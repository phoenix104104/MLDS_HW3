from xgoogle.search import GoogleSearch, SearchError

def extract_query(file) :
	lines = [file.readline().rstrip().split(None, 1)[1] for i in xrange(5)]

	[former, latter] = lines[0].split('[', 1)
	[_, latter] = latter.split(']', 1)

	terms = [x[x.index('[')+1 : x.index(']')] for x in lines]

	return former+latter, terms

file = open('testing_data.txt')

choice = ['a','b','c','d','e']

out = open('answer')
skip = len(out.readlines())
print 'skip %d' %skip

out = open('answer', 'a+')

for i in xrange(1, 1041) :
	query, terms = extract_query(file)
        if i <= skip :
                continue

	try:
		gs = GoogleSearch(query+' holmes')
		gs.results_per_page = 10
		results = gs.get_results()
	except SearchError, e:
		print "Search failed: %s" % e
		break

	score = [0] * 5
	for result in results :
		result = str(result)
		for j in range(5) :
			if terms[j] in result :
				score[j] += 1

	m = max(score)
	if m == 0:
		out.write('\n')
	else :
		out.write(','.join([choice[x] for x in range(5) if score[x] == m])+'\n')

        print 'Answering question %d '%i, score
