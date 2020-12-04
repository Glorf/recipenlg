import urllib.request

import sys
import time
import random
import json

from bs4 import BeautifulSoup

import multiprocessing as mp

# find and parse ingredients
def ingredients_cookbooks(soup):
    ingredients_tag = soup.find_all('span', 'H2', string='ingredients')[0]
    # print(ingredients_tag)
    return ingredients_tag.parent.p.get_text('|').strip().split('|')

# find and parse directions
def directions_cookbooks(soup):
    directions_tag = soup.find_all('span', 'H2', string='preparation')[0]
    # print(directions_tag)
    return directions_tag.parent.p.get_text('|').strip().split('  ')

# title
def title_cookbooks(soup):
    return soup.find_all('p', 'H2')[0].get_text().lower()

## the proper function to be used in future and in framework
def save_recipe(step, url, retrieve_ingredients, retrieve_directions, retrieve_title, path="./", filename_prefix="", filename_suffix=""):
    try:
        start = time.time()
        
        # get page
        html_doc = urllib.request.urlopen(url).read().decode('utf-8')
        # parse
        soup = BeautifulSoup(html_doc, 'html.parser')
        # retrive information and save to dictionary
        title = retrieve_title(soup)
        recipe = dict()
        recipe['title'] = title
        recipe['ingredients'] = retrieve_ingredients(soup)
        recipe['directions'] = retrieve_directions(soup)
        # save to file
        with open(path+'/'+filename_prefix+title.lower().replace(' ','_')+filename_suffix+'.json', 'w+') as f:
            f.write(json.dumps(recipe))
            
        end = time.time()
        elapsed = end-start
        
        print("Step:\t", step, "Time:\t", elapsed)
    
    except:
        print("Unable to get recipe from:\t", url)
        #for s in sys.exc_info():
        #    print(s)
        
if __name__ == '__main__':
    #
    # only downloads links from the list
    #
    with open('./cookbooks/missing.txt') as f:
        missing = f.readlines()

    missing = list(map(lambda x: int(x.strip()), missing))

    sleep_time = 0.8
    max_processes = 8
    counter = 0

    seed = 'http://www.cookbooks.com/Recipe-Details.aspx?id='

    processes = []
    #for i in range(1050000, 1090000):
    for i in missing[180000:]:
        # url selection or other custom shit here
        str_id = '0' * (8 - len(str(i))) + str(i) # since names/titles are not unique, id is added

        arguments = (i, seed+str(i), ingredients_cookbooks, directions_cookbooks, title_cookbooks, 'cookbooks/recipes', str_id+'-')
    
        target_func = save_recipe
    
        ### try not to edit code below this line
    
        counter += 1
        inactive = []
        # visit list of processes
        for proc in processes:
            # when process is no longer active, join it and add to list of inactive processes
            if not proc.is_alive():
                proc.join()
                inactive.append(proc)
        # remove inactive processes from processes list
        while inactive:
            processes.remove(inactive.pop())
        # print("Number of active processes:\t", len(processes))
    
        # if number of active processes is acceptable, we can start new process
        if len(processes) < max_processes: 
            p = mp.Process(target = target_func, args = arguments)
            p.start()
            processes.append(p)
        else:
            print("List of processes is full")
    
        # sleep, to avoid ddos attack or to fit in robots.txt rules
        time.sleep(sleep_time)

    # join remaining processes
    while processes:
        temp = processes.pop()
        temp.join()
    
