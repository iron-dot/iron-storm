# 지식 시냅스
import datetime
import csv
class know_synapse():
    f = open('feedback.txt', 'r', encoding = 'utf-8')
    
    f2 = f.readlines()

    import csv
    import datetime

    # 현재 시간
    now = datetime.datetime.now()
    # csv 파일을 만들기 위한 함수
    def csv_writer(time):
        with open('know_synapse.csv', mode='w', newline='', encoding='utf-8') as RESULT_writer_file:
            RESULT_writer = csv.writer(RESULT_writer_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            RESULT_writer.writerow(
                ["file name", "loss", "EPOCH", "working time", "lr", "bs", 'accuracy', 'optimizer'])
            

    def csv_writer2(time, name_list):
        with open('know_synapse.csv', mode='a', newline='', encoding='utf-8') as RESULT_writer_file:
            RESULT_writer = csv.writer(RESULT_writer_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            '''RESULT_writer.writerow(
                ["file name", "loss", "EPOCH", "working time"])'''
            for row in name_list: # 위의 name_list를 순차적으로 순회
                RESULT_writer.writerow([row[0],row[1],row[2],row[3]]) # 각 행을 순차적으로 .csv 파일에 저장
