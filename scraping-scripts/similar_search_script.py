import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.metrics as metrics

import math
import json

from multiprocessing import Queue, Process, Lock
import time

from numba import jit

import tqdm
import argparse


# loops the subset of data in search of values exceeding given treshold
@jit
def faster_loop(submatrix, treshold):
    result = []
    for index, val in np.ndenumerate(submatrix):
        if val > treshold:
            result.append(index + (val,))
    return result


# @jit
def adjust_results_with_bucket_index(bucket_x, bucket_y, result, bucket_size):
    result = list(map(lambda x: (x[0] + bucket_x * bucket_size, x[1] + bucket_y * bucket_size, x[2]), result))
    if bucket_x == bucket_y:
        result = list(filter(lambda x: x[0] < x[1], result))
    return result


# generates submatrix of
def get_similar_in_subset(bucket_x: int, bucket_y: int, tfidf_mat, treshold, bucket_size):
    subset1 = tfidf_mat[bucket_x * bucket_size: (bucket_x + 1) * bucket_size]
    subset2 = tfidf_mat[bucket_y * bucket_size: (bucket_y + 1) * bucket_size]
    part_result = metrics.pairwise.cosine_similarity(subset1, subset2)
    result = faster_loop(part_result, treshold)
    return adjust_results_with_bucket_index(bucket_x, bucket_y, result, bucket_size)


def worker(input_connection: Queue, output_connection: Queue, tfidf_matrix, syncLock: Lock, treshold=0.8, bucket_size=10000):
    syncLock.acquire()
    while not input_connection.empty():
        bucket = input_connection.get()
        syncLock.release()

        # start = time.time()

        # result = get_similar_in_subset(bucket[0], bucket[1], tfidf_matrix_all, treshold=0.8)
        result = get_similar_in_subset(bucket[0], bucket[1], tfidf_matrix, treshold, bucket_size)

        # end = time.time()

        for r in result:
            output_connection.put(r)

        # print(bucket, (end-start))

        syncLock.acquire()
    syncLock.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Deduplication step 3 (cosine similarity upon TF-IDF matrix) for recipes deduplication process")
    parser.add_argument('source', metavar='S', type=str, help='source filename')
    parser.add_argument('target', metavar='T', type=str, help='target filename')
    parser.add_argument('-w', '--workers', default=4, type=int, help='number of workers')
    parser.add_argument('-t', '--treshold', default=0.8, type=float, help='similarity treshold value')
    parser.add_argument('-b', '--bucket-size', default=10000, type=int, help='size of bucket')

    args = parser.parse_args()

    df = pd.read_csv(args.source)
    vectorizer = TfidfVectorizer()

    start = time.time()

    corpus_all = list(map(lambda x, y: x + ' ' + y, df.ingredients.map(lambda x: ' '.join(json.loads(x))), df.directions.map(lambda x: ' '.join(json.loads(x)))))
    tfidf_matrix_all = vectorizer.fit_transform(corpus_all)
    tfidf_matrix_all

    end = time.time()

    print('TF-IDF matrix computation time', (end-start)/60.0, 'min')

    bucket_size = args.bucket_size
    n_buckets = tfidf_matrix_all.shape[0]//bucket_size + 1

    buckets = []
    for i in range(n_buckets):
        for j in range(i, n_buckets):
            buckets.append((i, j))

    input_queue = Queue()
    output_queue = Queue()

    for b in buckets:
        input_queue.put(b)

    lock = Lock()

    n_processes = args.workers
    processes = []
    for i in range(n_processes):
        p = Process(target=worker, args=(input_queue, output_queue, tfidf_matrix_all, lock, args.treshold, bucket_size))
        p.start()
        processes.append(p)
        time.sleep(1)

    results = []
    processed_blocks = set()

    pbar = tqdm.tqdm(total=len(buckets))
    while True:
        all_finished = True
        dead = 0
        for p in processes:
            all_finished = all_finished and (not p.is_alive())
            if not p.is_alive():
                dead += 1
        # print('Dead: ', dead)
            # if queue not empty and process is dead, than restart [beta]
            # if not input_queue.empty() and not p.is_alive():
            #     print('restarting process')
            #     p.join()
            #     p = Process(target=worker, args=(input_queue, output_queue, tfidf_matrix_all, lock))
            #     p.start()

        if not all_finished:
            time.sleep(5)

        while not output_queue.empty():
            record = output_queue.get()
            b = (record[0]//bucket_size, record[1]//bucket_size)
            if b not in processed_blocks:
                processed_blocks.add(b)
                pbar.update(1)

            results.append(record)

        if all_finished:
            break

    pbar.close()

    for p in processes:
        p.join()

    rdf = pd.DataFrame(results, columns=['I1', 'I2', 'similarity'])
    rdf.to_csv(args.target, index=False)

