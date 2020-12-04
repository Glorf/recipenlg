import pandas as pd

if __name__ == '__main__':

    dataset_name = 'train-positive-pairs.csv'
    
    all_data = pd.read_csv(dataset_name)

    # split and check if number of urls is diferent
    gathered = all_data.loc[all_data.source == 'Gathered']
    rec1m = all_data.loc[all_data.source == 'Recipes1M']
    print('Gathered shape', gathered.shape,'Recipes1M shape' ,rec1m.shape)

    # generate common urls
    common_urls = set(gathered.link) & set(rec1m.link)
    print('Number of common urls', len(common_urls))

    # filter out not common urls
    gathered = gathered.loc[gathered.link.map(lambda x: x in common_urls)]
    rec1m = rec1m.loc[rec1m.link.map(lambda x: x in common_urls)]
    print('Gathered shape', gathered.shape,'Recipes1M shape' ,rec1m.shape)

    # append dataset
    df = gathered.append(rec1m)

    # assert that results are valid
    assert df.groupby('link').count().title.map(lambda x: x==2).all()

    # overrite dataset
    df.to_csv(dataset_name, index=False)
