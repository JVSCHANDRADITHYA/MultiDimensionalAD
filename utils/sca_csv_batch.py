from matplotlib.pyplot import sca
import pandas as pd
import csv
import os
import sys
import time
from datetime import datetime, timedelta, timezone

def sca_to_csv(sca_path, csv_path):
    with open(sca_path, 'r') as file:
        lines = file.readlines()

    lines = lines[1:]  # skip version line

    headers = []
    for line in lines:
        if line == '\n':
            break
        headers.append(line.strip().split()[1].strip("'"))

    # print(f"Number of headers: {len(headers)}")
    # print("Headers:", headers)

    data = []
    i = len(headers) + 1  
    start_time = lines[i].strip()
    # print(f"Start time (unix timestamp): {start_time}")
    start_time = int(start_time) 

    while i < len(lines):
        block = []
        while i < len(lines) and lines[i].strip() != '':
            if ' ' not in lines[i]:
                block.append(lines[i].strip().strip('0').strip())
            else:
                block.append(lines[i].strip().split()[1].strip())
            i += 1
        block = block[2:]
        data.append(block)
        i += 1


    with open(csv_path, 'w', newline='') as out_file:
        writer = csv.writer(out_file)
        writer.writerow(headers)
        for item in data:
            writer.writerow(item)


    with open(csv_path, 'r', newline='') as f:
        reader = list(csv.reader(f))
    
    original_headers = reader[0]
    data_rows = reader[1:]
    num_rows = len(data_rows)

    # IST timezone
    IST = timezone(timedelta(hours=5, minutes=30))


    new_headers = ['Seconds', 'Timestamp_IST'] + original_headers


    new_data_rows = []
    for i, row in enumerate(data_rows):
        seconds = i * 5
        utc_dt = datetime.fromtimestamp(start_time + seconds, timezone.utc)
        ist_dt = utc_dt.astimezone(IST)
        timestamp_str = ist_dt.strftime('%Y-%m-%d %H:%M:%S')

        new_row = [seconds, timestamp_str] + row
        new_data_rows.append(new_row)

    # Write final CSV with added columns
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(new_headers)
        writer.writerows(new_data_rows)

    print(f"CSV saved at : {csv_path}")


if __name__ == "__main__":
    sca_batch_path = r"C:\Users\adith\Downloads\Archive"
    csv_batch_path = r'G:\GlitchDetect\data\real_time_MKPL'
    
    for file in os.listdir(sca_batch_path):
        if file.endswith('.sca'):
            sca_path = os.path.join(sca_batch_path, file)
            csv_path = os.path.join(csv_batch_path, file.replace(".sca", ".csv"))
            
            if not os.path.exists(os.path.join(csv_batch_path)):
                os.makedirs(os.path.join(csv_batch_path))
            
            sca_to_csv(sca_path=sca_path, csv_path=csv_path)
            

    # for dis_leak in os.listdir(sca_batch_path):
    #     for percent_leak in os.listdir(os.path.join(sca_batch_path, dis_leak)):
    #         for sca_file in os.listdir(os.path.join(sca_batch_path, dis_leak, percent_leak)):
    #             if sca_file.endswith(".sca"):
    #                 sca_path = os.path.join(sca_batch_path, dis_leak, percent_leak, sca_file)
    #                 csv_path = os.path.join(csv_batch_path, dis_leak, percent_leak, sca_file.replace(".sca", ".csv"))
                    
    #                 if not os.path.exists(os.path.join(csv_batch_path, dis_leak, percent_leak)):
    #                     os.makedirs(os.path.join(csv_batch_path, dis_leak, percent_leak))
                    
    #                 print(f"Converting {os.path.basename(sca_path)} to {os.path.basename(csv_path)}", end="\n", flush=True)

    #                 # for _ in range(5): 
    #                 #     time.sleep(0.2)
    #                 #     sys.stdout.write(".")
    #                 #     sys.stdout.flush()
    #                 # print() 
                    
    #                 # start_time = time.time()
    #                 sca_to_csv(sca_path, csv_path)
    #                 # end_time = time.time()
    #                 # duration = end_time - start_time
                    
    #                 # print(f"TASK COMPLETED IN {duration:.2f} seconds\n")
    #                 # print("WAITING TO FLUSH OUT RESIDUAL TASKS", end="", flush=True)
                    
    #                 # for _ in range(3):
    #                 #     time.sleep(0.2)
    #                 #     sys.stdout.write(".")
    #                 #     sys.stdout.flush()
    #                 # print("\n") 
    #                 # time.sleep(1)
                    
    # print("ALL TASKS COMPLETED. CHECK THE FINAL CSV FOLDER FOR FILES\n")
    # print("TERMINATING PROGRAM", end="")                
    # for _ in range(7):
    #     time.sleep(0.5)
    #     sys.stdout.write(".")
    #     sys.stdout.flush()
    # print("\n") 
    