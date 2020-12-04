### imports
import urllib.request

import sys
import time
import json

from bs4 import BeautifulSoup

import multiprocessing as mp

def foodcom_generator(begin, end):
    with open('links.txt') as f:
        links = list(map(lambda x: x.strip(), f.readlines()))
    for i in range(begin, end):
        yield links[i], i

def title_foodcom(soup):
    return soup.find_all('h1')[0].get_text()

def ingredients_foodcom(soup):
    return list(map(lambda x: x.get_text(), soup.find_all('li', 'recipe-ingredients__item')))

def directions_foodcom(soup):
    return list(map(lambda x: x.get_text(), soup.find_all('li', 'recipe-directions__step')))

str_id = lambda i: '0' * (8 - len(str(i))) + str(i) + "-"

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
    scrapping_core(0, 298699, foodcom_generator, ingredients_foodcom, directions_foodcom, title_foodcom, prefix=str_id, path='./recipes')