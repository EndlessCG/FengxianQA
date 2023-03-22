#-*- coding : utf-8-*-
import pandas as pd
DATA_BASE = "input/data/faq/"

# 生成std_data，类别和标准问题对应关系，包含类别ID、标准问题ID、标准问题文本三列
def generate_std(file_path):
    df = pd.read_excel(file_path)
    df = df.drop_duplicates(subset=['问题'], keep='first')
    num = 1
    with open(f'{DATA_BASE}/std_data', 'w') as f:
        with open(f'{DATA_BASE}/ans_data', 'w') as f1:
            data_str = ""
            ans_str = ""
            for i in range(len(df)):
                data_str += "__label__0\t" + str(num) + "\t" + " ".join(list(str(df.iloc[i, 0]).replace('\n',''))) + "\n"
                ans_str += str(num) + "\t" + (str(df.iloc[i, 1]).replace('\n','').replace(' ','')) + "\n"
                num += 1
            f.write(data_str)
            f1.write(ans_str)

if __name__ == "__main__":
    generate_std(f'{DATA_BASE}/file/qa100.xls')
