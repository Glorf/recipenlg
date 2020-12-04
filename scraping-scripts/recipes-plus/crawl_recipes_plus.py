### imports
import urllib.request

import sys
import time
import json

from bs4 import BeautifulSoup

import multiprocessing as mp

def recipesplus_generator(begin, end):
    seed = 'http://recipes-plus.com/api/v2.0/recipes/'
    with open('ids.txt') as f:
        ids = list(map(lambda x: x.strip(), f.readlines()))
    for i in range(min(begin, len(ids)), min(end, len(ids))):
        yield seed + ids[i], i

# recipesplus get ingredients
def ingredients_recipesplus(recipe):
    func = lambda x: str(x["amount"]) + ' ' + str(x["unit"]) + ' ' + str(x['ingredient'])
    return list(map(func, recipe['ingredients']))

def directions_recipesplus(recipe):
    return recipe['steps']

def title_recipesplus(recipe):
    return recipe['title']

str_id = lambda i: '0' * (8 - len(str(i))) + str(i) + "-"

def save_recipe(step, url, retrieve_ingredients, retrieve_directions, retrieve_title, path="./", filename_prefix="", filename_suffix=""):
    try:
        start = time.time()
        
        # get page
        html_doc = urllib.request.urlopen(url).read().decode('utf-8')
        # parse
        # soup = BeautifulSoup(html_doc, 'html.parser')
        soup = json.loads(html_doc)['data']
        # retrive information and save to dictionary
        title = retrieve_title(soup)
        recipe = dict()
        recipe['link'] = url
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
        for s in sys.exc_info():
            print(s)

def scrapping_core(begin, end, generator, 
                   ingredients, directions, title, 
                   target_func = save_recipe, path='./', prefix = None, suffix = None, 
                   sleep_time = 1, max_processes = 8):
    #
    # only downloads links from the list
    #
    
    processes = []
    for url, i in generator(begin, end):
        
        p = ""
        s = ""
        if prefix:
            p = prefix(i)
        if suffix:
            s = suffix(i)

        arguments = (i, url, ingredients, directions, title, path, p, s)
            
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

if __name__ == '__main__':
    scrapping_core(0, 25000, recipesplus_generator, ingredients_recipesplus, directions_recipesplus, title_recipesplus, prefix=str_id, path='./recipes', sleep_time=0.7)