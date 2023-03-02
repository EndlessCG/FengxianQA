import pandas as pd
# 处理风险点
counter = 1
fengxiandian_path = "preprocess/风控实体关系表/entity_fengxiandian.csv"
output_path = "input/data/fengxian/qa/QA_data.txt"
fengxiandian = pd.read_csv(fengxiandian_path)
with open(output_path, "w") as wirter:
    for i in range(fengxiandian.shape[0]):
        wirter.write("<question id="+str(counter)+"> " +
                     fengxiandian.iloc[i, 1]+"的含义是什么？" + "\n")
        wirter.write("<triple id="+str(counter)+"> " +
                     fengxiandian.iloc[i, 1]+"\t含义\t" + fengxiandian.iloc[i, 4] + "\n")
        wirter.write("<answer id="+str(counter) +
                     "> "+fengxiandian.iloc[i, 4]+"\n")
        counter += 1
        wirter.write("<question id="+str(counter)+"> " +
                     fengxiandian.iloc[i, 1]+"的定义是什么？" + "\n")
        wirter.write("<triple id="+str(counter)+"> " +
                     fengxiandian.iloc[i, 1]+"\t含义\t" + fengxiandian.iloc[i, 4] + "\n")
        wirter.write("<answer id="+str(counter) +
                     "> "+fengxiandian.iloc[i, 4]+"\n")
        counter += 1
        wirter.write("<question id="+str(counter)+"> " +
                     fengxiandian.iloc[i, 1]+"是什么意思？" + "\n")
        wirter.write("<triple id="+str(counter)+"> " +
                     fengxiandian.iloc[i, 1]+"\t含义\t" + fengxiandian.iloc[i, 4] + "\n")
        wirter.write("<answer id="+str(counter) +
                     "> "+fengxiandian.iloc[i, 4]+"\n")
        counter += 1
        wirter.write("<question id="+str(counter)+"> " +
                     fengxiandian.iloc[i, 1]+"代表什么？" + "\n")
        wirter.write("<triple id="+str(counter)+"> " +
                     fengxiandian.iloc[i, 1]+"\t含义\t" + fengxiandian.iloc[i, 4] + "\n")
        wirter.write("<answer id="+str(counter) +
                     "> "+fengxiandian.iloc[i, 4]+"\n")
        counter += 1
        wirter.write("<question id="+str(counter)+"> " +
                     fengxiandian.iloc[i, 1]+"指什么？" + "\n")
        wirter.write("<triple id="+str(counter)+"> " +
                     fengxiandian.iloc[i, 1]+"\t含义\t" + fengxiandian.iloc[i, 4] + "\n")
        wirter.write("<answer id="+str(counter) +
                     "> "+fengxiandian.iloc[i, 4]+"\n")
        counter += 1
        wirter.write("<question id="+str(counter)+"> " +"怎么解释"+
                     fengxiandian.iloc[i, 1] + "？" + "\n")
        wirter.write("<triple id="+str(counter)+"> " +
                     fengxiandian.iloc[i, 1]+"\t含义\t" + fengxiandian.iloc[i, 4] + "\n")
        wirter.write("<answer id="+str(counter) +
                     "> "+fengxiandian.iloc[i, 4]+"\n")
        counter += 1
        wirter.write("<question id="+str(counter)+"> " +"怎么理解"+
                     fengxiandian.iloc[i, 1] + "？" + "\n")
        wirter.write("<triple id="+str(counter)+"> " +
                     fengxiandian.iloc[i, 1]+"\t含义\t" + fengxiandian.iloc[i, 4] + "\n")
        wirter.write("<answer id="+str(counter) +
                     "> "+fengxiandian.iloc[i, 4]+"\n")
        counter += 1
        wirter.write("<question id="+str(counter)+"> " +
                     fengxiandian.iloc[i, 1]+"的风险等级是多少？" + "\n")
        wirter.write("<triple id="+str(counter)+"> " +
                     fengxiandian.iloc[i, 1]+"\t风险等级\t" + fengxiandian.iloc[i, 3] + "\n")
        wirter.write("<answer id="+str(counter) +
                     "> "+fengxiandian.iloc[i, 3]+"\n")
        counter += 1
        wirter.write("<question id="+str(counter)+"> " +
                     fengxiandian.iloc[i, 1]+"是什么等级的风险？" + "\n")
        wirter.write("<triple id="+str(counter)+"> " +
                     fengxiandian.iloc[i, 1]+"\t风险等级\t" + fengxiandian.iloc[i, 3] + "\n")
        wirter.write("<answer id="+str(counter) +
                     "> "+fengxiandian.iloc[i, 3]+"\n")
        counter += 1


# 处理专有名词
mingci_path = "preprocess/风控实体关系表/entity_mingci.csv"
output_path = "input/data/fengxian/qa/QA_data.txt"
mingci = pd.read_csv(mingci_path)
with open(output_path, "a+") as wirter:
    for i in range(mingci.shape[0]):
        wirter.write("<question id="+str(counter)+"> " +
                     mingci.iloc[i, 1]+"的含义是什么？" + "\n")
        wirter.write("<triple id="+str(counter)+"> " +
                     mingci.iloc[i, 1]+"\t含义\t" + mingci.iloc[i, 4] + "\n")
        wirter.write("<answer id="+str(counter) +
                     "> "+mingci.iloc[i, 4]+"\n")
        counter += 1
        wirter.write("<question id="+str(counter)+"> " +
                     mingci.iloc[i, 1]+"的定义是什么？" + "\n")
        wirter.write("<triple id="+str(counter)+"> " +
                     mingci.iloc[i, 1]+"\t含义\t" + mingci.iloc[i, 4] + "\n")
        wirter.write("<answer id="+str(counter) +
                     "> "+mingci.iloc[i, 4]+"\n")
        counter += 1
        wirter.write("<question id="+str(counter)+"> " +
                     mingci.iloc[i, 1]+"是什么意思？" + "\n")
        wirter.write("<triple id="+str(counter)+"> " +
                     mingci.iloc[i, 1]+"\t含义\t" + mingci.iloc[i, 4] + "\n")
        wirter.write("<answer id="+str(counter) +
                     "> "+mingci.iloc[i, 4]+"\n")
        counter += 1
        wirter.write("<question id="+str(counter)+"> " +
                     mingci.iloc[i, 1]+"代表什么？" + "\n")
        wirter.write("<triple id="+str(counter)+"> " +
                     mingci.iloc[i, 1]+"\t含义\t" + mingci.iloc[i, 4] + "\n")
        wirter.write("<answer id="+str(counter) +
                     "> "+mingci.iloc[i, 4]+"\n")
        counter += 1
        wirter.write("<question id="+str(counter)+"> " +
                     mingci.iloc[i, 1]+"指什么？" + "\n")
        wirter.write("<triple id="+str(counter)+"> " +
                     mingci.iloc[i, 1]+"\t含义\t" + mingci.iloc[i, 4] + "\n")
        wirter.write("<answer id="+str(counter) +
                     "> "+mingci.iloc[i, 4]+"\n")
        counter += 1
        wirter.write("<question id="+str(counter)+"> " +"怎么解释"+
                     mingci.iloc[i, 1] + "？" + "\n")
        wirter.write("<triple id="+str(counter)+"> " +
                     mingci.iloc[i, 1]+"\t含义\t" + mingci.iloc[i, 4] + "\n")
        wirter.write("<answer id="+str(counter) +
                     "> "+mingci.iloc[i, 4]+"\n")
        counter += 1
        wirter.write("<question id="+str(counter)+"> " +"怎么理解"+
                     mingci.iloc[i, 1] + "？" + "\n")
        wirter.write("<triple id="+str(counter)+"> " +
                     mingci.iloc[i, 1]+"\t含义\t" + mingci.iloc[i, 4] + "\n")
        wirter.write("<answer id="+str(counter) +
                     "> "+mingci.iloc[i, 4]+"\n")
        counter += 1
        wirter.write("<question id="+str(counter)+"> " +
                     mingci.iloc[i, 1]+"属于什么类型？" + "\n")
        wirter.write("<triple id="+str(counter)+"> " +
                     mingci.iloc[i, 1]+"\t类型\t" + mingci.iloc[i, 3] + "\n")
        wirter.write("<answer id="+str(counter) +
                     "> "+mingci.iloc[i, 3]+"\n")
        counter += 1
        wirter.write("<question id="+str(counter)+"> " +
                     mingci.iloc[i, 1]+"属于什么类别？" + "\n")
        wirter.write("<triple id="+str(counter)+"> " +
                     mingci.iloc[i, 1]+"\t类型\t" + mingci.iloc[i, 3] + "\n")
        wirter.write("<answer id="+str(counter) +
                     "> "+mingci.iloc[i, 3]+"\n")
        counter += 1
        wirter.write("<question id="+str(counter)+"> " +
                     mingci.iloc[i, 1]+"的类型是什么？" + "\n")
        wirter.write("<triple id="+str(counter)+"> " +
                     mingci.iloc[i, 1]+"\t类型\t" + mingci.iloc[i, 3] + "\n")
        wirter.write("<answer id="+str(counter) +
                     "> "+mingci.iloc[i, 3]+"\n")
        counter += 1
        wirter.write("<question id="+str(counter)+"> " +
                     mingci.iloc[i, 1]+"的类别是什么？" + "\n")
        wirter.write("<triple id="+str(counter)+"> " +
                     mingci.iloc[i, 1]+"\t类型\t" + mingci.iloc[i, 3] + "\n")
        wirter.write("<answer id="+str(counter) +
                     "> "+mingci.iloc[i, 3]+"\n")
        counter += 1

# 处理业务环节
yewuhuanjie_path = "preprocess/风控实体关系表/entity_yewuhuanjie.csv"
output_path = "input/data/fengxian/qa/QA_data.txt"
yewuhuanjie = pd.read_csv(yewuhuanjie_path, keep_default_na=False)
with open(output_path, "a+") as wirter:
    for i in range(yewuhuanjie.shape[0]):
        # if yewuhuanjie.iloc[i, 1]
        wirter.write("<question id="+str(counter)+"> " +
                     yewuhuanjie.iloc[i, 1]+"是哪个部门负责的？" + "\n")
        wirter.write("<triple id="+str(counter)+"> " +
                     yewuhuanjie.iloc[i, 1]+"\t负责角色\t" + yewuhuanjie.iloc[i, 3] + "\n")
        wirter.write("<answer id="+str(counter) +
                     "> "+yewuhuanjie.iloc[i, 3]+"\n")
        counter += 1

        wirter.write("<question id="+str(counter)+"> " + "谁负责" +
                     yewuhuanjie.iloc[i, 1]+"？" + "\n")
        wirter.write("<triple id="+str(counter)+"> " +
                     yewuhuanjie.iloc[i, 1]+"\t负责角色\t" + yewuhuanjie.iloc[i, 3] + "\n")
        wirter.write("<answer id="+str(counter) +
                     "> "+yewuhuanjie.iloc[i, 3]+"\n")
        counter += 1

# 处理包含
baohan_path = "preprocess/风控实体关系表/relationship_baohan.csv"
output_path = "input/data/fengxian/qa/QA_data.txt"
baohan = pd.read_csv(baohan_path, keep_default_na=False)
with open(output_path, "a+") as wirter:
    for i in range(baohan.shape[0]):
        wirter.write("<question id="+str(counter)+"> " +
                     baohan.iloc[i, 0]+"包含哪些内容？" + "\n")
        wirter.write("<triple id="+str(counter)+"> " +
                     baohan.iloc[i, 0]+"\t包含\t" + baohan.iloc[i, 1] + "\n")
        wirter.write("<answer id="+str(counter) +
                     "> "+baohan.iloc[i, 1]+"\n")
        counter += 1

        wirter.write("<question id="+str(counter)+"> " +
                     baohan.iloc[i, 0]+"包括哪些内容？" + "\n")
        wirter.write("<triple id="+str(counter)+"> " +
                     baohan.iloc[i, 0]+"\t包含\t" + baohan.iloc[i, 1] + "\n")
        wirter.write("<answer id="+str(counter) +
                     "> "+baohan.iloc[i, 1]+"\n")
        counter += 1
        
        wirter.write("<question id="+str(counter)+"> " +
                     baohan.iloc[i, 0]+"涉及哪些内容？" + "\n")
        wirter.write("<triple id="+str(counter)+"> " +
                     baohan.iloc[i, 0]+"\t包含\t" + baohan.iloc[i, 1] + "\n")
        wirter.write("<answer id="+str(counter) +
                     "> "+baohan.iloc[i, 1]+"\n")
        counter += 1

# 处理预警
yujing_path = "preprocess/风控实体关系表/entity_yujing.csv"
output_path = "input/data/fengxian/qa/QA_data.txt"
yujing = pd.read_csv(yujing_path, keep_default_na=False)
with open(output_path, "a+") as wirter:
    for i in range(yujing.shape[0]):
        wirter.write("<question id="+str(counter)+"> " +
                     yujing.iloc[i, 1]+"的含义是什么？" + "\n")
        wirter.write("<triple id="+str(counter)+"> " +
                     yujing.iloc[i, 1]+"\t含义\t" + yujing.iloc[i, 4] + "\n")
        wirter.write("<answer id="+str(counter) +
                     "> "+yujing.iloc[i, 4]+"\n")
        counter += 1
        wirter.write("<question id="+str(counter)+"> " +
                     yujing.iloc[i, 1]+"的定义是什么？" + "\n")
        wirter.write("<triple id="+str(counter)+"> " +
                     yujing.iloc[i, 1]+"\t含义\t" + yujing.iloc[i, 4] + "\n")
        wirter.write("<answer id="+str(counter) +
                     "> "+yujing.iloc[i, 4]+"\n")
        counter += 1
        wirter.write("<question id="+str(counter)+"> " +
                     yujing.iloc[i, 1]+"是什么意思？" + "\n")
        wirter.write("<triple id="+str(counter)+"> " +
                     yujing.iloc[i, 1]+"\t含义\t" + yujing.iloc[i, 4] + "\n")
        wirter.write("<answer id="+str(counter) +
                     "> "+yujing.iloc[i, 4]+"\n")
        counter += 1
        wirter.write("<question id="+str(counter)+"> " +
                     yujing.iloc[i, 1]+"代表什么？" + "\n")
        wirter.write("<triple id="+str(counter)+"> " +
                     yujing.iloc[i, 1]+"\t含义\t" + yujing.iloc[i, 4] + "\n")
        wirter.write("<answer id="+str(counter) +
                     "> "+yujing.iloc[i, 4]+"\n")
        counter += 1
        wirter.write("<question id="+str(counter)+"> " +
                     yujing.iloc[i, 1]+"指什么？" + "\n")
        wirter.write("<triple id="+str(counter)+"> " +
                     yujing.iloc[i, 1]+"\t含义\t" + yujing.iloc[i, 4] + "\n")
        wirter.write("<answer id="+str(counter) +
                     "> "+yujing.iloc[i, 4]+"\n")
        counter += 1
        wirter.write("<question id="+str(counter)+"> " +"怎么解释"+
                     yujing.iloc[i, 1] + "？" + "\n")
        wirter.write("<triple id="+str(counter)+"> " +
                     yujing.iloc[i, 1]+"\t含义\t" + yujing.iloc[i, 4] + "\n")
        wirter.write("<answer id="+str(counter) +
                     "> "+yujing.iloc[i, 4]+"\n")
        counter += 1
        wirter.write("<question id="+str(counter)+"> " +"怎么理解"+
                     yujing.iloc[i, 1] + "？" + "\n")
        wirter.write("<triple id="+str(counter)+"> " +
                     yujing.iloc[i, 1]+"\t含义\t" + yujing.iloc[i, 4] + "\n")
        wirter.write("<answer id="+str(counter) +
                     "> "+yujing.iloc[i, 4]+"\n")
        counter += 1
        
        wirter.write("<question id="+str(counter)+"> " +
                     yujing.iloc[i, 1]+"在什么时候会发生红色预警？" + "\n")
        wirter.write("<triple id="+str(counter)+"> " +
                     yujing.iloc[i, 1]+"\t红色预警\t" + yujing.iloc[i, 8] + "\n")
        wirter.write("<answer id="+str(counter) +
                     "> "+yujing.iloc[i, 8]+"\n")
        counter += 1
        
        wirter.write("<question id="+str(counter)+"> " +
                     yujing.iloc[i, 1]+"的红色预警是什么？" + "\n")
        wirter.write("<triple id="+str(counter)+"> " +
                     yujing.iloc[i, 1]+"\t红色预警\t" + yujing.iloc[i, 8] + "\n")
        wirter.write("<answer id="+str(counter) +
                     "> "+yujing.iloc[i, 8]+"\n")
        counter += 1
        
        wirter.write("<question id="+str(counter)+"> " +
                     yujing.iloc[i, 1]+"在什么时候会发生橙色预警？" + "\n")
        wirter.write("<triple id="+str(counter)+"> " +
                     yujing.iloc[i, 1]+"\t橙色预警\t" + yujing.iloc[i, 7] + "\n")
        wirter.write("<answer id="+str(counter) +
                     "> "+yujing.iloc[i, 7]+"\n")
        counter += 1
        
        wirter.write("<question id="+str(counter)+"> " +
                     yujing.iloc[i, 1]+"的橙色预警是什么？" + "\n")
        wirter.write("<triple id="+str(counter)+"> " +
                     yujing.iloc[i, 1]+"\t橙色预警\t" + yujing.iloc[i, 7] + "\n")
        wirter.write("<answer id="+str(counter) +
                     "> "+yujing.iloc[i, 7]+"\n")
        counter += 1
        
        wirter.write("<question id="+str(counter)+"> " +
                     yujing.iloc[i, 1]+"在什么时候会发生蓝色预警？" + "\n")
        wirter.write("<triple id="+str(counter)+"> " +
                     yujing.iloc[i, 1]+"\t蓝色预警\t" + yujing.iloc[i, 5] + "\n")
        wirter.write("<answer id="+str(counter) +
                     "> "+yujing.iloc[i, 5]+"\n")
        counter += 1
        
        wirter.write("<question id="+str(counter)+"> " +
                     yujing.iloc[i, 1]+"的蓝色预警是什么？" + "\n")
        wirter.write("<triple id="+str(counter)+"> " +
                     yujing.iloc[i, 1]+"\t蓝色预警\t" + yujing.iloc[i, 5] + "\n")
        wirter.write("<answer id="+str(counter) +
                     "> "+yujing.iloc[i, 5]+"\n")
        counter += 1

        wirter.write("<question id="+str(counter)+"> " +
                     yujing.iloc[i, 1]+"在什么时候会发生黄色预警？" + "\n")
        wirter.write("<triple id="+str(counter)+"> " +
                     yujing.iloc[i, 1]+"\t黄色预警\t" + yujing.iloc[i, 6] + "\n")
        wirter.write("<answer id="+str(counter) +
                     "> "+yujing.iloc[i, 6]+"\n")
        counter += 1
        wirter.write("<question id="+str(counter)+"> " +
                     yujing.iloc[i, 1]+"的黄色预警是什么？" + "\n")
        wirter.write("<triple id="+str(counter)+"> " +
                     yujing.iloc[i, 1]+"\t黄色预警\t" + yujing.iloc[i, 6] + "\n")
        wirter.write("<answer id="+str(counter) +
                     "> "+yujing.iloc[i, 6]+"\n")
        counter += 1
        wirter.write("<question id="+str(counter)+"> " +
                     yujing.iloc[i, 1]+"属于什么类型？" + "\n")
        wirter.write("<triple id="+str(counter)+"> " +
                     yujing.iloc[i, 1]+"\t类型\t" + yujing.iloc[i, 3] + "\n")
        wirter.write("<answer id="+str(counter) +
                     "> "+yujing.iloc[i, 3]+"\n")
        counter += 1
        wirter.write("<question id="+str(counter)+"> " +
                     yujing.iloc[i, 1]+"属于什么类别？" + "\n")
        wirter.write("<triple id="+str(counter)+"> " +
                     yujing.iloc[i, 1]+"\t类型\t" + yujing.iloc[i, 3] + "\n")
        wirter.write("<answer id="+str(counter) +
                     "> "+yujing.iloc[i, 3]+"\n")
        counter += 1
        wirter.write("<question id="+str(counter)+"> " +
                     yujing.iloc[i, 1]+"的类型是什么？" + "\n")
        wirter.write("<triple id="+str(counter)+"> " +
                     yujing.iloc[i, 1]+"\t类型\t" + yujing.iloc[i, 3] + "\n")
        wirter.write("<answer id="+str(counter) +
                     "> "+yujing.iloc[i, 3]+"\n")
        counter += 1
        wirter.write("<question id="+str(counter)+"> " +
                     yujing.iloc[i, 1]+"的类别是什么？" + "\n")
        wirter.write("<triple id="+str(counter)+"> " +
                     yujing.iloc[i, 1]+"\t类型\t" + yujing.iloc[i, 3] + "\n")
        wirter.write("<answer id="+str(counter) +
                     "> "+yujing.iloc[i, 3]+"\n")
        counter += 1
