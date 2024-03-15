# Ryuryu's Bybit Historical Data Downloader 
# (Production Mode 6973)
# -------------------------------------
# (c) 2022 Ryan Hayabusa 
# Github: https://github.com/ryu878 
# Web: https://aadresearch.xyz/
# Discord: ryuryu#4087
# -------------------------------------
# pip install beautifulsoup4
# pip install requests

import urllib.request
import os
import re
import gzip
import time
import requests
from bs4 import BeautifulSoup
import pandas as pd
from pathlib import Path


# Define the base URL
base_url = 'https://public.bybit.com/kline_for_metatrader4/'
headers={"User-Agent": "Mozilla/5.0 (X11; CrOS x86_64 12871.102.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.141 Safari/537.36"}
coin = "BTCUSDT"
minutes_period = 15
year = 2000
dir_name = f"data/bybit/M{minutes_period}/RAW/{coin}/"

# Create a function to download the files
def download_file(url, local_path):
    with urllib.request.urlopen(url) as response, open(local_path, 'wb') as out_file:
        data = response.read()
        out_file.write(data)


# Create a function to check if a file exists
def file_exists(local_path):
    return os.path.exists(local_path)


def stack_csv_files(directory):
    # Get list of CSV files in the directory
    csv_files = sorted([file for file in os.listdir(directory) if file.endswith('.csv')])

    # Create an empty DataFrame to store the stacked data
    stacked_data = pd.DataFrame()

    # Iterate through each CSV file and stack the data
    for file in csv_files:
        data = pd.read_csv(os.path.join(directory, file), header=None)
        stacked_data = pd.concat([stacked_data, data], ignore_index=True)
        print(file, data.shape, "->", stacked_data.shape)

    stacked_data.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
    # Save the stacked data to a new CSV file
    stacked_data.to_csv(os.path.join(Path(directory).parent.parent, 
                                     "_".join(csv_files[0].split("_")[:2] + [csv_files[0].split("_")[2], csv_files[-1].split("_")[2]]) + ".csv"), index=False)
    print('CSV files stacked successfully!')   
   
     
# Make a GET request to the base URL and parse the HTML
response = requests.get(base_url, headers=headers)
soup = BeautifulSoup(response.text, 'html.parser')

# Find all the links on the page
links = soup.find_all('a')

# Loop through all the links
for coin_link in links:
    # Get the href attribute of the link
    href = coin_link.get('href')    
    if not coin in href:
        continue
    # Check if the href attribute is a directory
    if not href.endswith('/'):
        continue
    # Get the directory name
    # dir_name = href[:-1]
    # Create the directory locally if it doesn't exist
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    dir_url = base_url + href
    dir_response = requests.get(dir_url, headers=headers)
    dir_soup = BeautifulSoup(dir_response.text, 'html.parser')   
    links = dir_soup.find_all('a')    
    for link in links:
        href = link.get('href')    
        if not href.endswith('/'):
            continue  
        # Make a GET request to the directory URL and parse the HTML
        year_url = dir_url + href
        dir_response = requests.get(year_url, headers=headers)
        dir_soup = BeautifulSoup(dir_response.text, 'html.parser')
        # Find all the CSV files in the directory
        csv_links = dir_soup.find_all(href=re.compile('.csv.gz$'))
        # Loop through all the CSV files
        for csv_link in csv_links:
            # Get the CSV file name
            csv_name = csv_link.text
            if f"_{minutes_period}_" not in csv_name:
                continue
            # Extract the date from the CSV file name
            csv_date = re.findall(r'\d{4}-\d{2}-\d{2}', csv_name)[0]
            # Construct the full URL of the CSV file
            csv_url = year_url + csv_name
            # Construct the local path of the extracted file
            extracted_path = os.path.join(dir_name, csv_name[:-3])
            # Check if the extracted file exists locally
            if file_exists(extracted_path):
                print('Skipping download of', csv_name, '- extracted file already exists.')
            else:
                # Construct the local path of the archive file
                archive_path = os.path.join(dir_name, csv_name)
                # Download the archive file if it doesn't exist locally
                if not file_exists(archive_path):
                    download_file(csv_url, archive_path)
                    print('Downloaded:', archive_path)
                    time.sleep(5)
                # Check if the file is a gzip archive
                if csv_name.endswith('.gz'):
                    # Open the gzip archive and extract the contents
                    with gzip.open(archive_path, 'rb') as f_in:
                        with open(extracted_path, 'wb') as f_out:
                            f_out.write(f_in.read())
                            print('Extracted:', extracted_path)
                    # Remove the archive file
                    os.remove(archive_path)
                    print('Removed:', archive_path)
                else:
                    # Rename the file to remove the .csv extension
                    os.rename(archive_path, extracted_path)
                    print('Renamed:', archive_path, 'to', extracted_path)
                    
                    
stack_csv_files(dir_name)  
