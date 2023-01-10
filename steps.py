import geopy.exc

from utils import data_path, rewrite_pkl, timer, gen_cache
from classes import UrlsParser, CarsParser
import httplib2
import proxy_parser
import enchant
import time
import os
import sys
import itertools
import requests
import pickle
import datetime
import re
import pymorphy2
import pandas as pd
from deep_translator import GoogleTranslator
from bs4 import BeautifulSoup
from random import random
from hideme.proxy_collector import ProxiesList
from tqdm import tqdm as standard_tqdm
from random import sample, choice
from geopy.geocoders import Nominatim


# PAGES PARSING #
url0 = 'https://auto.drom.ru/'
h = httplib2.Http()
ech = enchant.Dict('en_US')


@gen_cache(f'{data_path}/cache/translator.pkl')
def translator(x, source='russian', target='english'):
    global ech
    if ech.check(x):
        return x
    else:
        return GoogleTranslator(source=source, target=target).translate(x)


def check_exist(url, http):
    for _ in range(3):
        resp = http.request(url, 'HEAD')
        if int(resp[0]['status']) == 200:
            return True
    return False


def check_class_and_exists(urlfor, h_, class_needed):
    def check_class(url, class_):
        soup = BeautifulSoup(requests.get(url=url).text, 'html.parser')
        return bool(soup.find_all(class_=class_))

    return check_exist(url=urlfor, http=h_) and check_class(url=urlfor, class_=class_needed)


@timer
def execute_pages_parsing(continue_session=False):
    global url0
    global h
    global ech

    # PARAMETERS
    if continue_session:
        with open(f'{data_path}/parameters/pages/brands_more100.pkl', 'rb') as f:
            brands_more100 = pickle.load(f)

        with open(f'{data_path}/parameters/pages/regions.pkl', 'rb') as f:
            regions = pickle.load(f)

        with open(f'{data_path}/urls/current/urls_of_pages_parsed.pkl', 'rb') as f:
            urls = pickle.load(f)
    else:
        # 1. brand
        soup_brands = BeautifulSoup(requests.get(url=url0).text, 'html.parser')

        elements_brands = [k.text for k in soup_brands.find_all('a')]

        brands_raw = elements_brands[(elements_brands.index('Подать объявление') + 1):
                                     elements_brands.index('Прочие авто')]
        brands = list(map(lambda x: translator(x).lower().replace(' ', '_'), brands_raw))

        brands_replace = [('exeed', 'cheryexeed'),
                          ('iran_khadro', 'iran_khodro'),
                          ('rolls_royce', 'rolls-royce'),
                          ('ssangyong', 'ssang_yong'),
                          ('gas', 'gaz'),
                          ('moskvich', 'moskvitch'),
                          ('land_wind', 'landwind')]

        for i in brands_replace:
            if i[0] in brands:
                brands[brands.index(i[0])] = i[1]

        for b in brands:
            if not check_class_and_exists(urlfor=url0 + b, h_=h, class_needed='css-hqbmxg'):
                raise TypeError(
                    f"""
                    Please, inspect brand [ {b} ], there is no such link: {url0 + b}.
                    Actual name of [ {b} ] is [ {brands_raw[brands.index(b)]} ].
                    You should add to list 'brands_replace' next tuple: ({b}, *actual.brand.from.link*).
                    """)
        else:
            print('All brands are valid.')

        brands_more100 = [str(b) for b in brands if
                          check_class_and_exists(urlfor=url0 + b + '/all/page100/', h_=h, class_needed='css-4gbnjj')]
        brands_less100 = list(set(brands) - set(brands_more100))

        # 2. region
        regions = [str(i) for i in range(1, 90) if check_exist(url=url0 + 'region' + str(i), http=h)]
        print('Regions are collected.')

        rewrite_pkl(f'{data_path}/parameters/pages/brands_more100.pkl', brands_more100)
        rewrite_pkl(f'{data_path}/parameters/pages/regions.pkl', regions)

        # URLS
        urls = []

        for b in brands_less100:
            urls.append(f'{url0}{b}/')

        rewrite_pkl(f'{data_path}/urls/current/urls_of_pages_parsed.pkl', urls)

    # 3. condition
    conditions = ['used', 'new']

    # 4. privod
    privods = {'1': 'front-wheel',
               '2': 'rear-wheel',
               '3': '4WD'}

    # 5. colors
    colors = {'1': 'black',
              '2': 'purple',
              '3&colorid[]=14': 'blue',
              '4&colorid[]=16': 'gray',
              '6&colorid[]=11&colorid[]=5&colorid[]=15': 'red',
              '7': 'brown',
              '13&colorid[]=10&colorid[]=8': 'yellow',
              '9': 'green',
              '12': 'white'}

    # 6. pts
    pts = {'1': 'problems',
           '2': 'ok'}

    # 7. damaged
    damaged = {'1': 'repair is needed',
               '2': 'no repair is needed'}

    class_needed = 'css-4gbnjj'

    def elifline(url):
        nonlocal urls
        nonlocal class_needed
        global h
        if check_class_and_exists(url.replace('page100', 'page1'), h, class_needed):
            urls.append(url.replace('page100/', ''))

    pbar = standard_tqdm(regions)
    regions_remains = regions[:]

    # I HATE THIS PART OF CODE, HOPE YOU TOO
    for r in pbar:
        pbar.set_description(f'Processing region - {r}')
        urlfor = f'{url0}region{r}/all/page100/'
        time.sleep(1)

        rewrite_pkl(f'{data_path}/urls/current/urls_of_pages_parsed.pkl', urls)
        rewrite_pkl(f'{data_path}/parameters/pages/regions.pkl', regions_remains)
        regions_remains.remove(r)

        if check_class_and_exists(urlfor, h, class_needed):
            for b in brands_more100:
                pbar.set_description(f'Processing region - {r}, brand - {b}')

                urlfor = f'{url0}region{r}/{b}/all/page100/'
                time.sleep(random() / 2)
                if check_class_and_exists(urlfor, h, class_needed):
                    for cd in conditions:
                        urlfor = f'{url0}region{r}/{b}/{cd}/all/page100/'

                        if check_class_and_exists(urlfor, h, class_needed):
                            for pr in privods:
                                urlfor = f'{url0}region{r}/{b}/{cd}/all/page100/?privod={pr}'

                                if check_class_and_exists(urlfor, h, class_needed):
                                    for clr in colors:
                                        urlfor = f'{url0}region{r}/{b}/{cd}/all/page100/?privod={pr}&colorid[]={clr}'

                                        if check_class_and_exists(urlfor, h, class_needed):
                                            for p in pts:
                                                urlfor = f'{url0}region{r}/{b}/{cd}/all/page100/?privod={pr}&colorid[]={clr}&pts={p}'

                                                if check_class_and_exists(urlfor, h, class_needed):
                                                    for d in damaged:
                                                        urlfor = f'{url0}region{r}/{b}/{cd}/all/page100/?privod={pr}&colorid[]={clr}&pts={p}&damaged={d}'

                                                        urls.append(urlfor.replace('page100/', ''))
                                                else:
                                                    elifline(url=urlfor)
                                        else:
                                            elifline(url=urlfor)
                                else:
                                    elifline(url=urlfor)
                        else:
                            elifline(url=urlfor)
                else:
                    elifline(url=urlfor)
        else:
            elifline(url=urlfor)

        rewrite_pkl(f'{data_path}/urls/all/urls_of_pages.pkl', urls)

    return 1


# URLS PARSING # -------------------------------------------------------------------------------------------------------
@timer
def execute_urls_parsing(continue_session=True, tqdm_=True):
    if continue_session:
        with open(f'{data_path}/urls/current/urls_of_cars.pkl', 'rb') as f:
            urls_of_cars = pickle.load(f)

        with open(f'{data_path}/urls/current/urls_of_pages.pkl', 'rb') as f:
            urls_of_pages = pickle.load(f)
    else:
        urls_of_cars = []

        with open(f'{data_path}/urls/all/urls_of_pages.pkl', 'rb') as f:
            urls_of_pages = list(set(pickle.load(f)))

    # PARSING OF ALL EXISTING URLS
    while len(urls_of_pages) > 1:
        sys.stdout.write(f'\rLength of the remaining page URLs: {len(urls_of_pages)}')
        sys.stdout.flush()

        drom = UrlsParser(urls=urls_of_pages, tqdm_=tqdm_)

        urls_of_cars.extend([*itertools.chain(*drom.urls_ads)])
        urls_of_pages = drom.urls_pages

        rewrite_pkl(f'{data_path}/urls/current/urls_of_pages.pkl', urls_of_pages)  # cut from current urls added urls
        rewrite_pkl(f'{data_path}/urls/current/urls_of_cars.pkl', urls_of_cars)  # add urls of cars to the file

        time.sleep(max(min(len(urls_of_pages) / 50, 20), 3))
    else:
        urls_of_cars = [*dict.fromkeys(urls_of_cars)]
        rewrite_pkl(f'{data_path}/urls/all/urls_of_cars.pkl', urls_of_cars)

    # GET NEW URLS
    if os.path.isfile(f'{data_path}/urls/all/old_urls_of_cars.pkl'):
        with open(f'{data_path}/urls/all/old_urls_of_cars.pkl', 'rb') as f:
            old_urls_of_cars = pickle.load(f)

        urls_of_cars_to_use = list(set(urls_of_cars) - set(old_urls_of_cars))

        rewrite_pkl(f'{data_path}/urls/all/urls_of_cars_to_use.pkl', urls_of_cars_to_use)
    else:
        rewrite_pkl(f'{data_path}/urls/all/urls_of_cars_to_use.pkl', urls_of_cars)

    rewrite_pkl(f'{data_path}/urls/all/old_urls_of_cars.pkl', urls_of_cars)

    with open(f'{data_path}/urls/all/history_urls_of_cars.pkl', 'ab') as fp:
        pickle.dump({datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'): urls_of_cars}, fp)

    return 1


# PROXIES PARSING # ----------------------------------------------------------------------------------------------------
def get_proxies_from_hideme(**kwargs):
    PL = ProxiesList(**kwargs)  # port='3128'
    return [f"{i['ip_address']}:{i['port']}" for i in PL.get()]


def get_proxies_from_free_proxy_cz(n):
    try:
        pp = proxy_parser.proxy_parser()
        pp.get_proxies('https', n)
        return [f"{i['ip']}:{i['port']}" for i in pp.server]
    except:
        return get_proxies_from_free_proxy_cz(n - 1)


def get_proxies_to_use(urls, proxies_raw, n_cars=200, n_proxies=15):
    proxies_to_use = []

    pbar = standard_tqdm(proxies_raw)
    for prx in pbar:
        pbar.set_description(f'\rNumber of proxies to use: {len(proxies_to_use)} / {n_proxies}')

        if len(proxies_to_use) >= n_proxies:
            break

        l_connect = []
        for k in range(4):
            try:
                r = requests.get(choice(urls),
                                 proxies={'http': prx, 'https': prx}, timeout=3)
                l_connect.append(r.status_code == 200)
            except (requests.exceptions.ProxyError,
                    requests.exceptions.ConnectTimeout,
                    requests.exceptions.ConnectionError,
                    requests.exceptions.ReadTimeout,
                    ConnectionResetError):
                l_connect.append(False)

        if True in l_connect:
            try:
                dp = CarsParser(urls=sample(urls, 1000),
                                proxy=f'http://{prx}',
                                timeout=0.7 * 60)
                if len(dp.cars) > n_cars:
                    proxies_to_use.append(prx)
            except ConnectionResetError:
                pass

    return proxies_to_use


@timer
def execute_proxies_parsing(search_with_previous=True, n_proxies=15):
    with open(f'{data_path}/urls/all/urls_of_cars_to_use.pkl', 'rb') as f:
        urls = pickle.load(f)

    if search_with_previous:
        with open(f'{data_path}/proxies/proxies_to_use.pkl', 'rb') as f:
            proxies_raw = pickle.load(f) + \
                          get_proxies_from_hideme(https=True) + \
                          get_proxies_from_free_proxy_cz(5)
    else:
        proxies_raw = get_proxies_from_hideme(https=True) + get_proxies_from_free_proxy_cz(5)

    rewrite_pkl(f'{data_path}/proxies/proxies_to_use.pkl',
                get_proxies_to_use(urls, proxies_raw, n_cars=200, n_proxies=n_proxies))

    return 1


# CARS PARSING # -------------------------------------------------------------------------------------------------------
@timer
def execute_cars_parsing(continue_session=True, timeout=2 * 60, n_workers=49, tqdm_=True, print_=True):
    start_time = datetime.datetime.now()

    # START WITH PREVIOUS SESSION OR NEW?
    if continue_session:
        with open(f'{data_path}/urls/current/urls_of_cars_to_use.pkl', 'rb') as f:
            urls = pickle.load(f)
    else:
        if os.path.isfile(f'{data_path}/dataframes/current/cars.tsv'):
            os.remove(f'{data_path}/dataframes/current/cars.tsv')

        with open(f'{data_path}/urls/all/urls_of_cars_to_use.pkl', 'rb') as f:
            urls = pickle.load(f)

    with open(f'{data_path}/proxies/proxies_to_use.pkl', "rb") as f:
        proxies_to_use = pickle.load(f)

    print(f'Number of proxies to use: {len(proxies_to_use)}')
    print(f'Number of urls to use: {len(urls)}')

    # PARSING PROCESS
    n = 1000

    while len(urls) > 1:
        for proxy in proxies_to_use:
            try:
                r = requests.get(choice(urls), proxies={'http': proxy, 'https': proxy}, timeout=4)
                if not (r.status_code in [200, 404]):
                    continue
            except (requests.exceptions.ProxyError,
                    requests.exceptions.ConnectTimeout,
                    requests.exceptions.ConnectionError,
                    requests.exceptions.ReadTimeout,
                    ConnectionResetError):
                continue

            drom = CarsParser(urls=urls[:n],
                              proxy=f'http://{proxy}',
                              timeout=timeout,
                              n=n_workers,
                              tqdm_=tqdm_)

            pd.DataFrame(drom.cars).to_csv(f'{data_path}/dataframes/current/cars.tsv',
                                           header=not os.path.exists(f'{data_path}/dataframes/current/cars.tsv'),
                                           mode='a',
                                           index=False,
                                           encoding='utf-16',
                                           sep='\t')

            urls = urls[n:]
            urls.extend(drom.urls_of_cars)
            rewrite_pkl(f'{data_path}/urls/current/urls_of_cars_to_use.pkl', urls)

            if not urls:
                break
            if print_:
                print(f'Length of collected cars: {len(drom.cars)}. '
                      f'Time of work: {datetime.datetime.now() - start_time} '
                      f'Length of urls to parse: {len(urls)}')
    else:
        pd.read_csv(f'{data_path}/dataframes/current/cars.tsv', encoding='utf-16', sep='\t') \
            .to_csv(f'{data_path}/dataframes/all/cars_ready.tsv', index=False, encoding='utf-16', sep='\t')

    return 1


# CARS PROCESSING # ----------------------------------------------------------------------------------------------------
@timer
def execute_cars_processing():
    standard_tqdm.pandas()

    df = pd.read_csv(f'{data_path}/dataframes/all/cars_ready.tsv', encoding='utf-16', sep='\t')

    df['Пробег, км'] = df['Пробег, км'].fillna(df['Пробег'])
    df.drop('Пробег', axis=1, inplace=True)

    keys_rus = ['Двигатель', 'Мощность', 'Коробка передач', 'Привод', 'Тип кузова', 'Цвет', 'Пробег, км', 'Руль',
                'Поколение', 'Комплектация', 'Особые отметки']
    keys_eng = ['engine', 'power', 'gearbox', 'drive', 'body_type', 'color', 'mileage, km', 'wheel',
                'generation', 'configuration', 'notes']
    df.rename(columns=dict(zip(keys_rus, keys_eng)), inplace=True)

    df = df[~df['url'].duplicated()]
    df.reset_index(drop=True, inplace=True)

    df['year_of_car'] = df['ad_name'].apply(lambda x: x[(x.index(' год') - 4):x.index(' год')])

    for i in df['year_of_car']:
        if not str(i).isdigit():
            raise ValueError(f'{i} is not an year of car producing')

    # BRAND
    print('I\'ve started [brand] part')

    df['brand_model'] = df['ad_name'].apply(lambda x: x[:x.index(',')].replace('Продажа ', ''))
    df = df.rename({'brand_and_model': 'brand_model_rus'}, axis=1)
    df['brand'] = df['brand_model'].apply(lambda x: x.split(' ')[0])
    d_brand = {'Alfa': 'Alfa Romeo',
               'Aston': 'Aston Martin',
               'DW': 'DW Hower',
               'Great': 'Great Wall',
               'Iran': 'Iran Khodro',
               'Land': 'Land Rover'}
    pattern = re.compile(r'\b(' + '|'.join(d_brand.keys()) + r')\b')

    df['brand'] = df['brand'].apply(lambda x: pattern.sub(lambda y: d_brand[y.group()], x))
    df['model'] = df.apply(lambda row: row['brand_model'].replace(row['brand'], ''), axis=1)

    # LOCATION
    print('I\'ve started [location] part')

    df['city_from_title'] = df['ad_name'].apply(lambda x: x[x.index('год в') + 6:])
    df.drop('ad_name', axis=1, inplace=True)

    nomin = Nominatim(user_agent="GetLoc")

    @gen_cache(f'{data_path}/cache/get_geocode.pkl')
    def get_geocode(location):
        nonlocal nomin

        try:
            return nomin.geocode(location)
        except geopy.exc.GeocoderServiceError:
            get_geocode(location)

    morph = pymorphy2.MorphAnalyzer()

    @gen_cache(f'{data_path}/cache/normal_form.pkl')
    def normal_form(x):
        nonlocal morph
        return morph.parse(x)[0].normal_form

    df['location'] = df['city'] \
        .apply(lambda x: x[x.index(':') + 2:]) \
        .apply(lambda x: x[x.index(' в ') + len(' в '):] if ' в ' in x else x) \
        .progress_apply(lambda x: normal_form(x)) \
        .progress_apply(lambda x: get_geocode(x + ', Россия'))

    print(f'city is checked')

    df.loc[df['location'].isna(), 'location'] = df.loc[df['location'].isna(), 'city_from_title'] \
        .apply(lambda x: x[x.index(' в ') + len(' в '):] if ' в ' in x else x) \
        .progress_apply(lambda x: normal_form(x)) \
        .progress_apply(lambda x: get_geocode(x + ', Россия'))

    print(f'city_from_title is checked')

    df = df[~df['location'].isna()]
    df = df[df.location.apply(lambda x: 'Россия' in x[0])]
    df.reset_index(drop=True, inplace=True)

    df['location_latitude'] = df['location'].apply(lambda x: x[1][0])
    df['location_longitude'] = df['location'].apply(lambda x: x[1][1])
    df['location'] = df['location'].apply(lambda x: x[0])

    df['city_ad'] = df['city']
    df['city'] = df['location'].apply(lambda x: x.split(',')[0])

    # ENGINE
    print('I\'ve started [engine] part')
    def type_engine(x):
        if 'бензин' in x:
            return 'бензин'
        elif 'дизель' in x:
            return 'дизель'
        elif 'электро' in x:
            return 'электро'
        else:
            return None

    df['engine_type'] = df['engine'].apply(lambda x: type_engine(str(x)))
    df['engine_gas_equipment'] = df['engine'].apply(lambda x: True if 'ГБО' in str(x) else False).astype(bool)
    df['engine_hybrid'] = df['engine'].apply(lambda x: True if 'гибрид' in str(x) else False).astype(bool)
    df['engine_liters'] = df['engine'].apply(
        lambda x: str(x)[str(x).index(' л') - 3:str(x).index(' л')] if ' л' in str(x) else None)
    df.drop('engine', axis=1, inplace=True)

    # POWER
    print('I\'ve started [power] part')

    df['power'] = df['power'].apply(
        lambda x: int(str(x).replace('л.с.,налог', '')) if str(x).replace('л.с.,налог', '') != 'nan' else None)

    # MILEAGE
    print('I\'ve started [mileage] part')

    def km(x):
        x = str(x)

        if x == 'nan':
            return None

        l1 = [i.replace(' ', '') for i in x.split(',')]
        l2 = [i for i in l1 if i.isdigit()]
        if len(l2) > 0:
            return int(l2[0])
        else:
            return None

    df['mileage'] = df['mileage, km'].apply(lambda x: km(x))
    df['no_mileage_in_RF'] = df['mileage, km'].apply(lambda x: True if 'безпробегапоРФ' in str(x) else False)
    df['mileage_new_car'] = df['mileage, km'].apply(lambda x: True if 'новый автомобиль' in str(x) else False)
    df.drop('mileage, km', axis=1, inplace=True)

    # DATE locOF PARSING
    df['date_of_parsing'] = pd.to_datetime('today').strftime('%Y-%m-%d %H:%M:%S')

    # ORDER IN DF
    df = df[['price', 'estimate_of_price',
             'brand_model', 'brand_model_rus', 'brand', 'model', 'estimate_of_the_model',
             'year_of_car', 'date_of_ad', 'num_of_views',
             'location', 'city', 'location_latitude', 'location_longitude', 'city_ad', 'city_from_title',
             'power', 'gearbox', 'drive', 'color', 'wheel', 'body_type',
             'engine_type', 'engine_gas_equipment', 'engine_hybrid', 'engine_liters',
             'mileage', 'no_mileage_in_RF', 'mileage_new_car',
             'generation', 'configuration', 'notes',
             'report', 'url', 'date_of_parsing']]

    # SAVE
    df.to_csv(f'{data_path}/dataframes/all/cars_final_from_ready.tsv',
              encoding='utf-8', sep='\t', index=False)

    # df.to_csv(f'{data_path}/dataframes/final/cars_final.tsv',
    #           header= not os.path.exists(f'{data_path}/dataframes/final/cars_final.tsv'),
    #           mode='a', index=False, encoding='utf-8', sep='\t')
    df.to_csv(f'{data_path}/dataframes/final/cars_final_{datetime.datetime.now().strftime("%Y_%m_%d__%H_%M")}.tsv',
              encoding='utf-8', sep='\t', index=False)

    return 1
