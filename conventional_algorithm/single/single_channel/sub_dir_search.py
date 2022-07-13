import os
import glob
# 함수 정의 방법
def search(dirname):
    try:
        filenames = os.listdir(dirname)  # 해당 디렉터리에 있는 파일들의 리스트
        for filename in filenames:
            full_filename = os.path.join(dirname, filename)  # 디렉터리를 포함한 전체 경로
            if os.path.isdir(full_filename):  # 디렉터리인지 파일인지 구분, 디렉터리일 경우 다시 파일을 찾도록 함
                search(full_filename)
            else:  # .wav 파일만 출력하기
                ext = os.path.splitext(full_filename)[-1]  # 파일명을 확장자를 기준으로 두 부분으로 나눠준다. -1은 확장자명 표시
                if ext == '.wav':
                    print(full_filename)
    except PermissionError: # 권한이 없는 디렉터리라도 프로그램이 오류로 종료되지 않고 그냥 수행되도록 하기 위함
        pass

# search("C:/Users/이유상/Desktop/논문/미팅발표/TIMIT")

# os.walk 이용한 TIMIT DB 폴더에 있는 모든 .wav 파일 가져오기
f = open("/home/leesunghyun/Downloads/TIMIT/TEST/all_wav.txt", 'w')
for(path, dir, files) in os.walk("/home/leesunghyun/Downloads/TIMIT/TEST"):
    for filename in files:
        ext = os.path.splitext(filename)[-1]
        if ext == '.wav':
            data = "%s/%s\n" %(path, filename)  # .wav 파일을 한줄씩 all_wav.txt 파일에 저장
            f.writelines([data])
f.close()



