
f = open('storm_test2/feedback.txt', 'r', encoding = 'utf-8')   
print((f.read().count("\n")+1))
f3 = f.read().count("\n")+1

def listToString(str_list):
    
    result = ""

    for s in str_list:

        result += s + ""

    return result.strip()

def replace_line(file_name, line_num, text):
    lines = open(file_name, 'r', encoding = 'utf-8').readlines()
    lines[line_num] = text
    out = open(file_name, 'w', encoding = 'utf-8')
    out.writelines(lines)
    out.close()
f = open('storm_test2/num.txt', 'r', encoding='utf-8')
num = int(listToString(f.readlines()))
print(num)

num5 = num + 1
class data_augmentation():
    while True:
    
        
        
        def data_augmentation():
            global num
            global num5
            
            f = open('storm_test2/feedback.txt', 'r', encoding = 'utf-8')
            f2 = f.readlines()
            print((f.read().count("\n")+1))
            f3 = f.read().count("\n")+1
            num += 1#972가 됨
            num2 = num
            print(num2)
            num5 += 2
            num6 = num5
            print(num6)

            print(f3)

            
            f = open('storm_test2/feedback.txt', 'r', encoding = 'utf-8')
            f3 = f.read().count("\n")+1
            num6 = num5
            print(num6)
            print(str(f3)+"입니다")
            f = open('storm_test2/num.txt', 'r', encoding='utf-8')
            num6 = int(listToString(f.readlines()))
            for i in range(f3):
                num6 += 1
                
                f = open('storm_test2/num.txt','w',encoding='utf-8')
                f.write(str(num6))
                print(num6)
                print(num)
                num7 = num6 - 1
                print(num7)

                num6 = num5
                print(num6)
                replace_line('storm_test2/storm.py', num6+1, "        replace_me('answer_way' + ' = data_augmentation.f2' + num + '\\n')\n\n\n\n")
                import time
                time.sleep(5)

            for i in range(f3):
                num6 += 1
                
                f = open('storm_test2/num.txt','w',encoding='utf-8')
                f.write(str(num6))
                print(num6)
                print(num)
                num7 = num6 - 1
                print(num7)

                num6 = num5
                print(num6)

                replace_line('storm_test2/storm.py', num6+2, "        replace_me('answer_waylist.append(' + 'answer_way' + num + '\\n')\n\n\n\n")
                import time
                time.sleep(5)



                

            f1 = open('storm_test2/feedback.txt', 'r', encoding = 'utf-8')

            f2 = f1.readlines()

            f2 = listToString(f2)
            f1.close()
            f1 = open('storm_test2/feedback.txt', 'a', encoding = 'utf-8')
            f1.truncate()
            f1.close()
            

            
            
            f3 = open('storm_test2/opinion.txt', 'a', encoding = 'utf-8')
            f3.write(f2)
            f3.write('\n')





                

            
        f = open('storm_test2/feedback.txt', 'r', encoding = 'utf-8')
        f3 = f.read().count("\n")+1
        
        data_augmentation()
        print(num)
        break
        
