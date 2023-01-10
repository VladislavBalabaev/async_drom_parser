# pip freeze > requirements.txt
# import heartrate
# heartrate.trace(browser=True)

import os
import pickle
import datetime
from functools import wraps


data_path = '/home/balabaev/All/WORK/1_at_NSU/2_Market_of_cars/Drom/data'
# data_path = os.environ['DATA_PATH']  # link, where you would store you cash files and results of parsing


if not os.path.exists(data_path):
    os.makedirs(data_path)

    directories = ['/urls', '/urls/all', '/urls/current',
                   '/proxies',
                   '/dataframes', '/dataframes/all', '/dataframes/current', '/dataframes/final'
                   '/parameters', '/parameters/pages',
                   '/cache']

    for d in directories:
        os.makedirs(f'{data_path}{d}')

    print('### Your NEW directory for files is set. ###\n')


def rewrite_pkl(path_to_file, what_to_dump):
    open(path_to_file, "w").close()
    with open(path_to_file, "wb") as fp:
        pickle.dump(what_to_dump, fp)


def timer(func):
    def wrapper(*args, **kwargs):
        start = datetime.datetime.now()
        print(f"""\n### Start of {func.__name__} - {start:%Y-%m-%d %H:%M} ###""")
        result = func(*args, **kwargs)
        print(f"""\n### End of {func.__name__}, taken time - {datetime.datetime.now() - start} ###""")

        return result

    return wrapper


def gen_cache(dict_name):
    def decorate(func):
        if not os.path.exists(dict_name):
            rewrite_pkl(dict_name, {})

        with open(dict_name, 'rb') as file:
            dict_ = pickle.load(file)

        @wraps(func)
        def wrapper(*args, **kwargs):
            all_args = f'{args} {kwargs.keys()} {kwargs.values()}'

            if all_args not in dict_:
                dict_[all_args] = func(*args, **kwargs)

            rewrite_pkl(dict_name, dict_)
            return dict_[all_args]
        return wrapper
    return decorate
