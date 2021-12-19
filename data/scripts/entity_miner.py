data_loc = ''
import requests
from bs4 import BeautifulSoup as bs
import re
from urllib.parse import unquote
import time
import pandas as pd


def mine_entity_names(query_url):
    ent_list = set()
    offset = 0
    failure_count = 0
    while(failure_count<=100):
        r = requests.get(query_url.replace('|!|',str(offset)), auth=('user', 'pass'))
        soup = bs(r.text, 'html.parser')
        table = soup.find("table", {"class": "table table-striped table-sm table-borderless"})
        if table:
            failure_count = 0
            table_lines = table.find_all('tr')
            for line in table_lines:
                temp_link = line.find('a', href=True)
                if temp_link:
                    temp_result = unquote(re.sub(r'\(.*\)', '',temp_link['href'].replace('http://dbpedia.org/resource/','').replace('_',' '))).strip()
                    if len(temp_result.split())>1 and len(temp_result.split())<5:
                        ent_list.add(temp_result)
        if not table:
            failure_count+=1
            print('failure', failure_count, end='\r')
            time.sleep(10)
            continue
            
        offset+=1000
        print(f'offset={offset}', f'ent_count={len(ent_list)}')
    return ent_list


person_query = "https://dbpedia.org/sparql?default-graph-uri=http%3A%2F%2Fdbpedia.org&query=prefix+dbpedia-owl%3A+%3Chttp%3A%2F%2Fdbpedia.org%2Fontology%2F%3E%0D%0A%0D%0Aselect+%3Fperson+%7B%7B%0D%0A++select+%3Fperson+%7B+%0D%0A++++%3Fperson+a+dbpedia-owl%3APerson%0D%0A++%7D%0D%0A++order+by+%3Fperson%0D%0A%7D%7D+%0D%0Aoffset+|!|%0D%0ALIMIT+1000&format=text%2Fhtml&timeout=30000&signal_void=on&signal_unconnected=on"
person_set = mine_entity_names(person_query)
pd.DataFrame(person_set, columns=['name']).to_csv(data_loc+'person_ent.csv', index=False)


location_query = "https://dbpedia.org/sparql?default-graph-uri=http%3A%2F%2Fdbpedia.org&query=prefix+dbpedia-owl%3A+%3Chttp%3A%2F%2Fdbpedia.org%2Fontology%2F%3E%0D%0A%0D%0Aselect+%3Flocation+%7B%7B%0D%0A++select+%3Flocation+%7B+%0D%0A++++%3Flocation+a+dbpedia-owl%3ALocation%0D%0A++%7D%0D%0A++order+by+%3Flocation%0D%0A%7D%7D+%0D%0Aoffset+|!|%0D%0ALIMIT+1000&format=text%2Fhtml&timeout=30000&signal_void=on&signal_unconnected=on"
location_set = mine_entity_names(location_query)
pd.DataFrame(location_set, columns=['name']).to_csv(data_loc+'location_ent.csv', index=False)