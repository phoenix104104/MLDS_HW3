import sys, os
from random import randint, random

idx = 1

state_num = 4
train_num = 25
min_train_len = 20
max_train_len = 50
test_num = 10
min_test_len = 15
max_test_len = 20
output_path = './'

p = 0.75 #regular_behavior_prob

choice_num = 2

argc = len(sys.argv)

while idx < argc:
    if(sys.argv[idx] == '--train'):
        if(idx+1 < argc):
            train_num = int(sys.argv[idx+1])
            if(train_num>999):
                print 'num too big...\n'
                exit()
            idx += 2
        else:
            print 'insufficient argument\n'
            exit()
    elif(sys.argv[idx] == '--test'):
        if(idx+1 < argc):
            test_num = int(sys.argv[idx+1])
            if(test_num>999):
                print 'num too big...\n'
                exit()
            idx += 2
        else:
            print 'insufficient argument\n'
            exit()
    elif(sys.argv[idx] == '-o'):
        if(idx+1 < argc):
            output_path = sys.argv[idx+1]
            idx+=2
        else:
            print 'insufficient argument\n'
            exit()
    else:
        print "unknown argument: " + sys.argc[idx] + '\n'
        exit() 

def print_state(f, state):
    for i in range(0, state_num):
        if(i != 0):
            f.write(' ')
        if(i==state):
            f.write('1')
        else:
            f.write('0')
    f.write('\n')

if(output_path[-1] != '/'):
    output_path += '/'

os.system('mkdir ' + output_path + "train")

for i in range(0, train_num):
    tlen = randint(min_train_len, max_train_len)
    filename = 'train%03d' %i
    filepath = output_path + 'train/' + filename
    f = open(filepath, 'w')
    state1 = randint(0, state_num-1)
    state2 = randint(0, state_num-1)
    print_state(f, state1)
    for j in range(0, tlen-1):
        print_state(f, state2)
        if(random() > p):
            next_state = randint(0, state_num-1)
        else:
            next_state = (state1+state2+1) % state_num
        state1 = state2
        state2 = next_state
    f.close()

answers = []

for i in range(0, test_num):
    tlen = randint(min_test_len, max_test_len)
    dirname = 'test%03d' %i
    os.system('mkdir ' + output_path + dirname)
    
    ans = randint(0, choice_num-1)
    answers.append(ans)
    
    for j in range(0, choice_num):
        filename = '_%02d' %j
        filename = dirname + filename
        filepath = output_path + dirname + '/' + filename
        f = open(filepath, 'w')
        state1 = randint(0, state_num-1)
        state2 = randint(0, state_num-1)
        if(j == ans):
            p2 = p
        else:
            p2 = 0.0
        print_state(f, state1)
        for k in range(0, tlen-1): 
            print_state(f, state2)
            if(random() > p2):
                next_state = randint(0, state_num-1)
            else:
                next_state = (state1+state2+1) % state_num
            state1 = state2
            state2 = next_state
        f.close()

f = open(output_path + 'testing_ans', 'w')
for i in range(0, len(answers)):
    f.write(str(answers[i]))
    f.write('\n')
f.close()
