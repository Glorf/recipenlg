# yeah, that could be parallelised
from os import listdir
import json

source = 'recipes'
target = 'packed'
step = 100000

filenames = listdir(source)
filenames.remove('.ipynb_checkpoints')
print("Total files: ", len(filenames))

number = 0
for i in range(0, len(filenames), step):
    rec_list = []
    print("Packing ", number, " from: ", i, " to: ", min(len(filenames), i+step))
    for j in range(i, min(len(filenames), i+step)):
        with open(source + '/' + filenames[j]) as f:
            data = json.load(f)
            # data['link'] = 'http://www.cookbooks.com/Recipe-Details.aspx?id=' + str(int(filenames[j].split('-')[0]))
            rec_list.append(data)
    with open(target + '/' + 'packed-' + str(number) + '.json', 'w+') as g:
        json.dump(rec_list, g)
    print("Packed ", number)
    number += 1
