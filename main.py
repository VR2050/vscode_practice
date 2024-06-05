from head import*
from City import *
from test1 import *

if __name__=="__main__":
    image = Image.open('/root/vscode/玩/image.png')  
      
    # 自定义字符画的宽度  
    ascii_art_width = 60
      
    # 转换为字符画  
    ascii_art = img_to_ascii(image, ascii_art_width)  
    while True:
        print("1-单个城市分析")
        print("2-多个城市分析")
        print("3-相关数据结果")
        print("4-退出")
        print("break quit")
        num=int(input("root@kali:~# :"))
        match num:
            case 1:
                while True:
                    city=str(input("输入城市:"))
                    match city:
                        case "zhangjiakou":
                            #zhangjiakou()
                            CITY(file=file_path_lists[0])
                        case "sjiazhuang":
                            CITY(file_path_lists[1])
                        case "Tianjin":
                            CITY(file_path_lists[2])
                        case "break":
                            print("相关数据结果")
                            break   
            case 2:
                CITYS()
            case 3:
                deal()                              
            case 4:
                print(ascii_art)
                print()
                print("shit code sadness!")
                break