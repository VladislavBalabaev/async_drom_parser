from bs4 import BeautifulSoup
import asyncio
import aiohttp
from random import random
from tqdm.asyncio import tqdm
from aiosocksy.connector import ProxyConnector, ProxyClientRequest
import itertools
import re


class UrlsParser:
    def __init__(self, urls, tqdm_=True):
        self.urls = urls
        self.urls_pages = []
        self.urls_ads = []
        self.tqdm_ = tqdm_

        asyncio.run(self.main())

    async def get_soup_without_timeout(self, url, session):
        try:
            for j in range(5):
                await asyncio.sleep(max(random() * 2, 0.1))

                if j > 1:
                    await asyncio.sleep(j + random() * j * 3)

                async with session.get(url, timeout=False) as resp:
                    if resp.status == 200:
                        await asyncio.sleep(max(random(), 0.1))
                        soup = BeautifulSoup(await resp.text(), 'html.parser')

                        return soup
            else:
                return 0

        except (aiohttp.ServerConnectionError,
                aiohttp.ClientPayloadError,
                aiohttp.ClientConnectorError,
                aiohttp.ClientOSError) as error:
            # print(f'{error}, returning url')
            return 0
            # await asyncio.sleep(10 + random() * 20)
            # return await self.get_soup(url, session)

    async def get_soup(self, url, session):
        try:
            result = await asyncio.wait_for(self.get_soup_without_timeout(url, session), timeout=60.0)
            return result
        except asyncio.TimeoutError:
            return 0

    async def get_page_ads(self, url, session):
        soup = await self.get_soup(url, session)

        if soup == 0:
            return [], url

        list_of_href = []
        for link in soup.find_all('a'):
            if len(link.find_all('span')) >= 7:
                list_of_href.append(link.get('href'))

        continue_link = soup.find_all(class_='css-4gbnjj')
        if continue_link:
            href = continue_link[0].get('href')
        else:
            href = 0

        return list_of_href, href

    async def main(self):
        tasks = []

        async with aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(limit=49, force_close=True)) as session:
            for url in self.urls:
                tasks.append(self.get_page_ads(url, session))

            if self.tqdm_:
                urls_of_pages = [await task_ for task_ in tqdm.as_completed(tasks, total=len(tasks))]
            else:
                urls_of_pages = await asyncio.gather(*tasks)
            # urls_of_pages = [await task_ for task_ in tqdm.as_completed(tasks, total = len(tasks))]
            self.urls_pages.extend([i[1] for i in urls_of_pages if i[1] != 0])
            self.urls_ads.extend([i[0] for i in urls_of_pages])


class CarsParser:
    def __init__(self, urls, proxy, timeout, n=49,
                 tqdm_=False, print_error=False, print_url=False):
        self.urls = urls
        self.proxy = proxy
        self.timeout = timeout
        self.print_error = print_error
        self.print_url = print_url
        self.tqdm_ = tqdm_
        self.n = n

        self.cars = []
        self.urls_of_cars = []

        self.all_attributes = {'Двигатель': None,
                               'Мощность': None,
                               'Коробка передач': None,
                               'Привод': None,
                               'Тип кузова': None,
                               'Цвет': None,
                               'Пробег, км': None,
                               'Пробег': None,
                               'Руль': None,
                               'Поколение': None,
                               'Комплектация': None,
                               'Особые отметки': None}

        asyncio.run(self.main())

    async def get_soup_without_timeout(self, url, session):
        try:
            for j in range(5):
                await asyncio.sleep(max(random() * 4, 0.5))

                if j > 1:
                    await asyncio.sleep(j + random() * j * 3)

                async with session.get(url, proxy=self.proxy) as resp:
                    if resp.status == 200:
                        await asyncio.sleep(max(random(), 0.1))
                        soup = BeautifulSoup(await resp.text(), 'html.parser')

                        return soup

                    elif resp.status == 404:
                        return 404
            else:
                return 0

        except (aiohttp.ServerConnectionError,
                aiohttp.ClientPayloadError,
                aiohttp.ClientConnectorError,
                aiohttp.ClientOSError,
                aiohttp.ClientHttpProxyError) as error:
            if self.print_error:
                print(error)
            await asyncio.sleep(random() * 10)
            return 0

    async def get_soup(self, url, session):
        try:
            result = await asyncio.wait_for(self.get_soup_without_timeout(url, session),
                                            timeout=self.timeout)
            return result
        except (asyncio.TimeoutError,
                aiohttp.ServerConnectionError,
                aiohttp.ClientPayloadError,
                aiohttp.ClientConnectorError,
                aiohttp.ClientOSError,
                aiohttp.ClientHttpProxyError):
            return 0

    async def get_car(self, url, session):
        soup = await self.get_soup(url, session)

        if soup == 0:
            return [], url

        if soup == 404:
            return [], 0

        def element(x, lambd):
            if x is None:
                return x
            else:
                return lambd(x)

        if element(x=soup.find('h1', class_='b-title b-title_type_h1'),
                   lambd=lambda x: x.text) == 'Запрошенная вами страница не существует!':
            return [], 0

        if element(x=soup.find('div', class_='css-1bw6vfx'),
                   lambd=lambda x: x.text) == 'Автомобиль снят с продажи':
            return [], 0

        if element(x=soup.find('span', class_='css-1kb7l9z'),
                   lambd=lambda x: x.text) in ['Объявление удалено!', 'Объявление не опубликовано.']:
            return [], 0

        def take_float(x):
            match_ = re.match(r'^-?\d+(?:\.\d+)$', x)
            if match_:
                return match_.group(0)
            else:
                return None

        properties = {
            'ad_name': element(x=soup.find('span', class_='css-1kb7l9z'),
                               lambd=lambda x: x.text),
            'price': element(x=soup.find('div', class_='css-eazmxc'),
                             lambd=lambda x: x.text.replace('\xa0', '').replace('₽', '')),
            'city': element(x=soup.find_all('div', class_='css-inmjwf'),
                            lambd=lambda x: x[-1].text if len(x) > 0 else False),
            'estimate_of_price': element(x=soup.find('div', class_='css-1nbcgqx'),
                                         lambd=lambda x: x.text),
            'date_of_ad': element(x=soup.find('div', class_='css-pxeubi'),
                                  lambd=lambda x: re.search(r'\d{2}.\d{2}.\d{4}', x.text).group()),
            'num_of_views': element(x=soup.find('div', class_='css-14wh0pm'),
                                    lambd=lambda x: x.text.replace(' ', '')),
            'brand_and_model': element(x=soup.find('div', class_='css-1tux9ri'),
                                       lambd=lambda x: x.find('a').text),
            'estimate_of_the_model': element(x=soup.find('div', class_='css-1tux9ri'),
                                             lambd=lambda x: take_float(x.text[-3:]))
        }

        for att in self.all_attributes.keys():
            select_one = soup.select_one(f'table:-soup-contains("{att}")')
            if select_one:
                break
        else:
            return [], url

        rows = select_one.find('tbody').find_all('tr')

        if rows[-1].find_all('th')[0].text.strip() == 'Проверено':
            rows = rows[:-3]

        attributes = {row.find_all('th')[0].text.strip(): row.find_all('td')[0].text.strip().replace('\xa0', '')
                      for row in rows if len(row) >= 2}

        for k in list(set(attributes) - set(self.all_attributes.keys())):
            attributes.pop(k, None)

        report = {'report': ', '.join([*itertools.chain(*[j.text.replace('\xa0', '').split(', ')
                                                          for j in soup.find_all('div', class_='css-13qo6o5')])][:4])}
        url_dict = {'url': url}

        car = {}
        for part in [properties, self.all_attributes, attributes, report, url_dict]:
            car.update(part)

        if self.print_url:
            print(url)

        return car, 0

    async def main(self):
        tasks = []
        async with aiohttp.ClientSession(trust_env=True,
                                         connector=ProxyConnector(limit=self.n),
                                         request_class=ProxyClientRequest) as session:
            # for url in tqdm.as_completed(self.urls, total = len(urls)):
            for url in self.urls:
                tasks.append(self.get_car(url, session))

            if self.tqdm_:
                result = [await task_ for task_ in tqdm.as_completed(tasks, total=len(tasks))]
            else:
                result = await asyncio.gather(*tasks)
            self.cars.extend([i[0] for i in result if i[0]])
            self.urls_of_cars.extend([i[1] for i in result if i[1] != 0])
