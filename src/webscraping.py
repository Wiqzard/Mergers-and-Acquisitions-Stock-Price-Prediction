

import requests
import urllib
from bs4 import BeautifulSoup

def make_url(base_url , comp):
    
    url = base_url
    
    # add each component to the base url
    for r in comp:
        url = '{}/{}'.format(url, r)
        
    return url


base_url = r"https://www.sec.gov/Archives/edgar/data"

# define a company to search (GOLDMAN SACHS), this requires a CIK number that is defined by the SEC.
cik_num = '886982'

filings_url = make_url(base_url, [cik_num, 'index.json'])

# Get the filings and then decode it into a dictionary object.
content = requests.get(filings_url)
decoded_content = content.json()

# Get a single filing number, this way we can request all the documents that were submitted.
filing_number = decoded_content['directory']['item'][0]['name']
filing_url = make_url(base_url, [cik_num, filing_number, 'index.json'])

content = requests.get(filing_url)
document_content = content.json()

# get a document name
for document in document_content['directory']['item']:
    if document['type'] != 'image2.gif':
        document_name = document['name']
        filing_url = make_url(base_url, [cik_num, filing_number, document_name])
        print(filing_url


 #Get a filing number, this way we can request all the documents that were submitted.
for filing_number in decoded_content['directory']['item'][3:5]:    
    
    filing_num = filing_number['name']
    print('-'*100)
    print('Grabbing filing : {}'.format(filing_num))
    
    # define the filing url, again I want all the data back as JSON.
    filing_url = make_url(base_url, [cik_num, filing_num, 'index.json'])

    # Get the documents submitted for that filing.
    content = requests.get(filing_url)
    document_content = content.json()


file_url = r"https://www.sec.gov/Archives/edgar/daily-index/2019/QTR2/master.20190401.idx"

# request that new content, this will not be a JSON STRUCTURE!
content = requests.get(file_url).content

# we can always write the content to a file, so we don't need to request it again.
with open('master_20190102.txt', 'wb') as f:
     f.write(content)


with open('master_20190102.txt','rb') as f:
     byte_data = f.read()

# Now that we loaded the data, we have a byte stream that needs to be decoded and then split by double spaces.
data = byte_data.decode("utf-8").split('  ')

# We need to remove the headers, so look for the end of the header and grab it's index
for index, item in enumerate(data):
    if "ftp://ftp.sec.gov/edgar/" in item:
        start_ind = index

# define a new dataset with out the header info.
data_format = data[start_ind + 1:]

master_data = []

# now we need to break the data into sections, this way we can move to the final step of getting each row value.
for index, item in enumerate(data_format):
    
    # if it's the first index, it won't be even so treat it differently
    if index == 0:
        clean_item_data = item.replace('\n','|').split('|')
        clean_item_data = clean_item_data[8:]
    else:
        clean_item_data = item.replace('\n','|').split('|')
        
    for index, row in enumerate(clean_item_data):
        
        # when you find the text file.
        if '.txt' in row:

            # grab the values that belong to that row. It's 4 values before and one after.
            mini_list = clean_item_data[(index - 4): index + 1]
            
            if len(mini_list) != 0:
                mini_list[4] = "https://www.sec.gov/Archives/" + mini_list[4]
                master_data.append(mini_list)
                
# grab the first three items
master_data[:3]