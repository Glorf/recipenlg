import urllib.request

import sys
import time
import random
import json

from bs4 import BeautifulSoup

import multiprocessing as mp

# allrecipes get directions
def directions_allrecipes(soup):
    temp = list(map(lambda x: x.get_text().strip(), soup.find_all('span', 'recipe-directions__list--item')))
    temp = list(filter(lambda x: x != '', temp))
    return temp

# allrecipes get ingredients
def ingredients_allrecipes(soup):
    temp = list(map(lambda x: x.get_text().strip(), soup.find_all('span', 'recipe-ingred_txt')))
    temp = list(filter(lambda x: x != '' and x != 'Add all ingredients to list', temp))
    return temp

# allrecipes get title
def title_allrecipes(soup):
    return list(map(lambda x: x.get_text().strip(), soup.find_all('h1','recipe-summary__h1')))[0]

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
        recipe['link'] = url
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
    links = []
    with open('links.txt') as f:
        links = list(map(lambda x: x.strip(), f.readlines()))
        
    #
    # only downloads links from the list
    #
    
    sleep_time = 1.1
    max_processes = 8
    counter = 0
    
    processes = []
    for i in range(0, len(links)):
        # url selection or other custom shit here
        str_id = '0' * (8 - len(str(i))) + str(i) # since names/titles are not unique, id is added
        
        arguments = (i, links[i], ingredients_allrecipes, directions_allrecipes, title_allrecipes, './recipes', str_id+'-')
        
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