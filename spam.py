import re
import email
import json
from bs4 import BeautifulSoup

with open(r'C:\Users\admin\Desktop\ING_1.ročník\ZS_2020-2021\VINF\trec07p\trec07p\full\index', 'r') as ifile:
    raw_labels = ifile.readlines()

all_mails = []
for label in raw_labels:
    mail = {}
    match_0 = re.search(r'((?:sp|h)am) ../data/inmail.(\d{1,})', label)
    if match_0:
        class_ = match_0.group(1)
        email_num = match_0.group(2)
        mail['Email_number'] = email_num
        mail['Class'] = class_
        x = mail['Email_number']
        if email_num == 5:
            break
        
    with open(f'C:/Users/admin/Desktop/ING_1.ročník/ZS_2020-2021/VINF/trec07p/trec07p/data/inmail.{x}', 'rb') as email_file:
        message = email.message_from_binary_file(email_file)
        body = ''
        
        if message.is_multipart():
            for part in message.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get('Content-Disposition'))
                
                if (content_type in ['text/html', 'text/txt'] and 'attachment' not in content_disposition):
                    body = part.get_payload(decode=True)
                    break
        else:
            body = message.get_payload(decode=True)

        mail['Body'] = BeautifulSoup(body, 'html.parser').get_text(strip=True)

    with open(f'C:/Users/admin/Desktop/ING_1.ročník/ZS_2020-2021/VINF/trec07p/trec07p/data/inmail.76', 'r', encoding='ISO-8859-1') as email_file:
        for riadok in email_file.readlines():
            match_1 = re.search(r'^From: (.*)$', riadok)
            if match_1:
                mail['Sender'] = match_1.group(1)
            match_2 = re.search(r'^To: (.*)$', riadok)
            if match_2:
                mail['Receiver'] = match_2.group(1)
            match_3 = re.search(r'^Date: (.*)$', riadok)
            if match_3:
                mail['Date'] = match_3.group(1)
            match_4 = re.search(r'^Subject: (.*)$', riadok)
            if match_4:
                mail['Subject'] = match_4.group(1)
    
    all_mails.append(mail)

with open("data.json", 'w') as file:
    json.dump(all_mails, file)