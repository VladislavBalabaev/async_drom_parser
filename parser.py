from steps import *
import nest_asyncio
nest_asyncio.apply()


# execution = [bool(int(os.environ[env])) for env in ['PAGES_P', 'URLS_P', 'PROXIES_P', 'CARS_P', 'CARS_PR']]
execution = [0, 0, 0, 0, 1]

print(
    f"""
    ###
    PLAN OF PARSING:
    0. Execute parsing of pages - {bool(execution[0])}
    1. Execute parsing of urls - {bool(execution[1])}
    2. Execute parsing of proxies - {bool(execution[2])}
    3. Execute parsing of cars - {bool(execution[3])}
    4. Execute processing of cars - {bool(execution[4])}
    ###
    """)
# 0. pages_parsing------------------------------------------------------------------------------------------------------
if execution[0]:
    execute_pages_parsing(continue_session=False)

# 1. urls_parsing-------------------------------------------------------------------------------------------------------
if execution[1]:
    execute_urls_parsing(continue_session=False, tqdm_=True)

# 2. proxies_parsing----------------------------------------------------------------------------------------------------
if execution[2]:
    execute_proxies_parsing(search_with_previous=True, n_proxies=15)

with open(f'{data_path}/proxies/proxies_to_use.pkl', 'rb') as f:
    if len(pickle.load(f)) < 5:
        raise ValueError('### Can\'t continue, number of proxies to use is too small. ###')

# 3. cars_parsing-------------------------------------------------------------------------------------------------------
if execution[3]:
    execute_cars_parsing(continue_session=False, timeout=1.5 * 60, n_workers=49, tqdm_=True, print_=True)

# 4. cars_processing----------------------------------------------------------------------------------------------------
if execution[4]:
    execute_cars_processing()

print(f"""
    ###
    END OF PARSING.
    ###
    """)
sys.exit()
