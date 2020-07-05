#importing libraries
import pandas as pd

#importing datas
data1 = pd.read_csv('Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv')
data2 = pd.read_csv('train_mosaic.csv')
X1 = data1.iloc[:,:]
X2 = data2.iloc[:,:]


X1_ddos_80 = 0
X1_ddos_other = 0
X1_benign_80 = 0
X1_benign_other = 0

for i in range(len(data1)):
    if data1['Label'][i] == 'DDoS' and data1['Destination_Port'][i] == 80:
        X1_ddos_80 += 1
    
    elif data1['Label'][i] == 'DDoS' and data1['Destination_Port'][i] != 80:
        X1_ddos_other += 1
    
    elif data1['Label'][i] != 'DDoS' and data1['Destination_Port'][i] == 80:
        X1_benign_80 += 1

    elif data1['Label'][i] != 'DDoS' and data1['Destination_Port'][i] != 80:
        X1_benign_other += 1
        
print(X1_ddos_80)
print(X1_ddos_other)
print(X1_benign_80)
print(X1_benign_other)


X2_ddos_80 = 0
X2_ddos_other = 0
X2_benign_80 = 0
X2_benign_other = 0

for i in range(len(data2)):
    if data2['Label'][i] == 'DoS Hulk' and data2['Destination_Port'][i] == 80:
        X2_ddos_80 += 1
        
    elif data2['Label'][i] == 'DoS slowloris' and data2['Destination_Port'][i] == 80:
        X2_ddos_80 += 1
    
    elif data2['Label'][i] == 'DoS Hulk' and data2['Destination_Port'][i] != 80:
        X2_ddos_other += 1
        
    elif data2['Label'][i] == 'DoS slowloris' and data2['Destination_Port'][i] != 80:
        X2_ddos_other += 1
    
    elif data2['Label'][i] != 'DoS Hulk' and data2['Destination_Port'][i] == 80:
        X2_benign_80 += 1
        
    elif data2['Label'][i] != 'DoS slowloris' and data2['Destination_Port'][i] == 80:
        X2_benign_80 += 1

    elif data2['Label'][i] != 'DoS Hulk' and data2['Destination_Port'][i] != 80:
        X2_benign_other += 1
        
    elif data2['Label'][i] != 'DoS slowloris' and data2['Destination_Port'][i] != 80:
        X2_benign_other += 1
        
print(X2_ddos_80)
print(X2_ddos_other)
print(X2_benign_80)
print(X2_benign_other)
