from bs4 import BeautifulSoup
import csv
import numpy as np
import pandas as pd

file_path = r"G:\GlitchDetect\data\steadystate.html"

with open(file_path, "r") as file:
    soup = BeautifulSoup(file, "html.parser")

table_list = soup.find_all("table")
table = table_list[1]

table_headers = table.find_all('thead')[0].find_all('tr')[2:4]

headers = ['time',]
for col in zip(*[row.find_all("td")[1:] for row in table_headers]):
    header_text = "".join(td.get_text(strip=True) for td in col if td.get_text(strip=True))
    header_text = header_text.replace(" ","")
    headers.append(header_text)
    

data_rows = table.find("tbody").find_all("tr")
data = []
for row in data_rows:
    cells = [td.get_text(strip=True) for td in row.find_all("td")]
    data.append(cells)
    
with open("steady.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(headers)
    writer.writerows(data)

print('CSV Created')