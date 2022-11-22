import pickle
import socket
import logging
def main():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # 建立socket物件

    addr = ('192.168.50.5', 9999)
    s.bind(addr)  # 繫結地址和埠

    logging.info('UDP Server on %s:%s...', addr[0], addr[1])

    user = {}  # 存放字典{addr:name}
    while True:
        try:
            data, addr = s.recvfrom(1024)  # 等待接收客戶端訊息存放在2個變數data和addr裡
            if not addr in user:  # 如果addr不在user字典裡則執行以下程式碼
                for address in user:  # 從user遍歷資料出來address
                    s.sendto(data + ' 進入聊天室...'.encode('utf-8'), address)  # 傳送user字典的data和address到客戶端
                user[addr] = data.decode(encoding='utf-8',errors='ignore')  # 接收的訊息解碼成utf-8並存在字典user裡,鍵名定義為addr
                continue  # 如果addr在user字典裡，跳過本次迴圈

            # if 'EXIT'.lower() in data.decode('utf-8'):#如果EXIT在傳送的data裡
            #     name = user[addr]   #user字典addr鍵對應的值賦值給變數name
            #     user.pop(addr)      #刪除user裡的addr
            #     for address in user:    #從user取出address
            #         s.sendto((name + ' 離開了聊天室...').encode(), address)     #傳送name和address到客戶端
            else:
                print('from %s:%s' ,data[0],data[2], addr[0], addr[1])
                print('data=',type(data))
                for address in user:    #從user遍歷出address
                    if address != addr:  #address不等於addr時間執行下面的程式碼
                        s.sendto(data, address)     #傳送data和address到客戶端
                        # print(type(pickle.dumps(data)))

        except ConnectionResetError:
            logging.warning('Someone left unexcept.')

if __name__ == '__main__':
    main()
