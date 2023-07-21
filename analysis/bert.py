from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from transformers import AutoModel, AutoTokenizer, BertTokenizer
import csv
import pymysql
devices = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(devices)
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
setup_seed(50)
batch_size = 20
# 所有的样本循环多少次
epoches = 50
# 模型名称，有可以换其他模型，如albert_chinese_small
model = "bert-base-chinese"
# 这个参数一般和模型有关系，一般在模型的config.json中设置
hidden_size = 768
# 分类的个数
n_class = 4
# 每个段落最大字符个数，一般不足maxlen ，就用一个字符代替，多余maxlen 就直接截断
maxlen = 150

class MyDataset(Dataset):
    def __init__(self, sentences, labels=None, with_labels=True):
        self.tokenizer = BertTokenizer.from_pretrained(model)
        self.with_labels = with_labels
        self.sentences = sentences
        self.labels = labels

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        sent = self.sentences[index]
        #print(sent)
        encoded_pair = self.tokenizer(sent,
                                      padding='max_length',  # 填充到最大值
                                      truncation=True,  # 截断到最大值
                                      max_length=maxlen,# 最大值
                                      return_tensors='pt')  # 返回pytorch类型

        token_ids = encoded_pair['input_ids'].squeeze(0)  
        # # input_ids为讲文本转化为数字，前后加一个特殊字符101(开始字符)，102(结束字符)，0(填充字符)
        #print(token_ids)
        if self.with_labels:
            label = self.labels[index]
            return token_ids,label
        else:
            return token_ids
class BertClassify(nn.Module):
    def __init__(self):
        super(BertClassify, self).__init__()
        #self.bert = AutoModel.from_pretrained(model, output_hidden_states=True, return_dict=True)
        self.bert = AutoModel.from_pretrained(model)
        self.linear = nn.Linear(hidden_size, n_class)
        self.dropout = nn.Dropout(0.5)

    def forward(self, X):
        input_ids=X[0]
        #print(input_ids)
        outputs = self.bert(input_ids=input_ids)  # 返回一个output字典
        # 用最后一层cls向量做分类
        # outputs.pooler_output: [bs, hidden_size]
        logits = self.linear(self.dropout(outputs.pooler_output))

        return logits
def train_Function_weibo():
    content_list=[]
    label_list=[]
    with open('./csv/weibo_content.csv',encoding="utf-8") as f:#此处修改数据集
         f_csv = csv.reader(f)
         header = next(f_csv)
         print(header)
         for row in f_csv:
              content,label=row  #cata暂时不会用到
              content_list.append(content)
              label_list.append(label)
    train = DataLoader(dataset=MyDataset(content_list, label_list), batch_size=batch_size, shuffle=True, num_workers=0)
    my_module = BertClassify().to(device=devices)
    my_module.train()
    loss_fn = nn.CrossEntropyLoss()
    loss_fn = loss_fn.to(device=devices)
    optimizer = torch.optim.SGD(params=my_module.parameters(),lr=0.001)
    carve_train_x=[]
    carve_train_y=[]
    forward_loss=0
    flag_stop=0
    for i in range(epoches):
        sum_loss=0
        for x, batch in enumerate(train):
                optimizer.zero_grad()
                batch[0] = torch.LongTensor(batch[0]).to(device=devices)
                #print([batch[0]])
                batch1_list=[]
                for item in batch[1]:
                    batch1_list.append(int(item))
                #print(batch1_list)
                batch1_list = torch.LongTensor(batch1_list).to(device=devices)
                #batch[1] = torch.LongTensor(batch[1]).to(device=devices)
                #batch = tuple(p.to(device=devices) for p in batch)
                pred = my_module([batch[0]])
                pred = pred.to(device=devices)
                loss = loss_fn(pred,batch1_list)
                sum_loss += loss.item()
                loss.backward()
                optimizer.step()
                if x%2==0:
                    #print("迭代轮次{}，训练轮次{},损失为{}".format(i,x,loss))
                    pass
        print("本轮总损失为{}".format(sum_loss))
        carve_train_x.append(i)
        carve_train_y.append(sum_loss)
        if(forward_loss>sum_loss and forward_loss-sum_loss<0.05):
            print("第{}次epoch出现低增长".format(i))
            flag_stop=flag_stop+1
        if(flag_stop>3 or i>40):
            print("此时停止")
            torch.save(my_module,"./model/my_1_weibo_model.pth")
            break
        forward_loss = sum_loss
    plt.plot(carve_train_x, carve_train_y, 'b*--', alpha=0.5, linewidth=1, label='acc')
    plt.show()

def train_Function_twitter():
    content_list=[]
    label_list=[]
    n_class = 4#微博和推特仅此处有较大不同
    with open('分类/data_result/twitter_content_with_topic.csv',encoding="utf-8") as f:
         f_csv = csv.reader(f)
         header = next(f_csv)
         print(header)
         for row in f_csv:
              content,label=row  #cata暂时不会用到
              content_list.append(content)
              label_list.append(label)
    train = DataLoader(dataset=MyDataset(content_list, label_list), batch_size=batch_size, shuffle=True, num_workers=0)
    my_module = BertClassify().to(device=devices)
    my_module.train()
    loss_fn = nn.CrossEntropyLoss()
    loss_fn = loss_fn.to(device=devices)
    optimizer = torch.optim.SGD(params=my_module.parameters(),lr=0.001)
    carve_train_x=[]
    carve_train_y=[]
    forward_loss=0
    flag_stop=0
    for i in range(epoches):
        sum_loss=0
        for x, batch in enumerate(train):
                optimizer.zero_grad()
                batch[0] = torch.LongTensor(batch[0]).to(device=devices)
                #print([batch[0]])
                batch1_list=[]
                for item in batch[1]:
                    #print(item)
                    batch1_list.append(int(item))
                #print(batch1_list)
                batch1_list = torch.LongTensor(batch1_list).to(device=devices)
                #batch[1] = torch.LongTensor(batch[1]).to(device=devices)
                #batch = tuple(p.to(device=devices) for p in batch)
                pred = my_module([batch[0]])
                pred = pred.to(device=devices)
                loss = loss_fn(pred,batch1_list)
                sum_loss += loss.item()
                loss.backward()
                optimizer.step()
                if x%2==0:
                    #print("迭代轮次{}，训练轮次{},损失为{}".format(i,x,loss))
                    pass
        print("本轮总损失为{}".format(sum_loss))
        carve_train_x.append(i)
        carve_train_y.append(sum_loss)
        if(forward_loss>sum_loss and forward_loss-sum_loss<0.05):
            print("第{}次epoch出现低增长".format(i))
            flag_stop=flag_stop+1
        if(flag_stop>3 or i>40):
            print("此时停止")
            torch.save(my_module,"分类/model/my_2_twitter_model.pth")
            break
        forward_loss = sum_loss
    plt.plot(carve_train_x, carve_train_y, 'b*--', alpha=0.5, linewidth=1, label='acc')
    plt.show()

def test_Function_hotline():
    my_module = torch.load("./model/my_2_twitter_model.pth")
    my_module = my_module.to(device=devices)
    list_time_01=['2022-02-01%','2022-02-02%','2022-02-03%','2022-02-04%','2022-02-05%','2022-02-06%','2022-02-07%','2022-02-08%','2022-02-09%','2022-02-10%',
               '2022-02-11%','2022-02-12%','2022-02-13%','2022-02-14%','2022-02-15%','2022-02-16%','2022-02-17%','2022-02-18%','2022-02-19%','2022-02-20%',
               '2022-02-21%','2022-02-22%','2022-02-23%','2022-02-24%','2022-02-25%','2022-02-26%','2022-02-27%','2022-02-28%','2022-03-01%','2022-03-02%','2022-03-03%','2022-03-04%','2022-03-05%','2022-03-06%','2022-03-07%','2022-03-08%','2022-03-09%','2022-03-10%',
               '2022-03-11%','2022-03-12%','2022-03-13%','2022-03-14%','2022-03-15%','2022-03-16%','2022-03-17%','2022-03-18%','2022-03-19%','2022-03-20%',
               '2022-03-21%','2022-03-22%','2022-03-23%','2022-03-24%','2022-03-25%','2022-03-26%','2022-03-27%','2022-03-28%','2022-03-29%','2022-03-30%','2022-03-31%','2022-04-01%','2022-04-02%','2022-04-03%','2022-04-04%','2022-04-05%','2022-04-06%','2022-04-07%','2022-04-08%','2022-04-09%','2022-04-10%',
               '2022-04-11%','2022-04-12%','2022-04-13%','2022-04-14%','2022-04-15%','2022-04-16%','2022-04-17%','2022-04-18%','2022-04-19%','2022-04-20%',
               '2022-04-21%','2022-04-22%','2022-04-23%','2022-04-24%','2022-04-25%','2022-04-26%','2022-04-27%','2022-04-28%','2022-04-29%','2022-04-30%','2022-05-01%','2022-05-02%','2022-05-03%','2022-05-04%','2022-05-05%','2022-05-06%','2022-05-07%','2022-05-08%','2022-05-09%','2022-05-10%',
               '2022-05-11%','2022-05-12%','2022-05-13%','2022-05-14%','2022-05-15%','2022-05-16%','2022-05-17%','2022-05-18%','2022-05-19%','2022-05-20%',
               '2022-05-21%','2022-05-22%','2022-05-23%','2022-05-24%','2022-05-25%','2022-05-26%','2022-05-27%','2022-05-28%','2022-05-29%','2022-05-30%','2022-05-31%','2022-06-01%','2022-06-02%','2022-06-03%','2022-06-04%','2022-06-05%','2022-06-06%','2022-06-07%','2022-06-08%','2022-06-09%','2022-06-10%',
               '2022-06-11%']
    db = pymysql.connect(host='localhost', user='root', passwd='jiajian233', port=3306,db="test")
    cursor = db.cursor()  # 创建游标
    list_time=list_time_01#此处修改时间
    list_result=[[] for i in range(len(list_time))]
    for i in range(len(list_time)):
        sql = "select time_of_discovery,problem_description from weibo_new where key_points_of_management='疾控防疫'and time_of_discovery like'{}'" .format(list_time[i])
        cursor.execute(sql)
        db.commit()
        sentence_list=[]
        while 1:
            res=cursor.fetchone()
            if res is None: 
                break#表示已经取完结果集
            else:
                #print(res[1]) #取文本内容，文本内容清不清洗区别不大
                sentence_list.append(res[1])
        test = MyDataset(sentences=sentence_list,with_labels=False) #此处不调用DataLoader
        my_module.eval()
        count_label=[0,0,0,0]#初始化类别计数
        for j in range(len(sentence_list)):
            #print(sentence_list[j])
            sentence_input = test.__getitem__(j)
            #print(sentence_input)
            sentence_input = torch.LongTensor(sentence_input).to(device=devices)
            sentence_input=sentence_input.unsqueeze(0) #转成二维
            #print(sentence_input)
            pred = my_module([sentence_input])
            pred = pred.data.max(dim=1, keepdim=True)[1]
            #print(f'预测为{pred[0][0]}')
            count_label[pred[0][0]-1]+=1#对应类别的数加一，注意weibo和hotline不一样。微博是五类，且从零开始
        print(f'日期为{list_time[i]}，当天数据总数为{len(sentence_list)}，分布情况为{count_label}')
        #list_result[i].append(list_time[i]).append(len(sentence_list)).append(count_label[0]).append(count_label[1]).append(count_label[2]).append(count_label[3])
        list_result[i].append(list_time[i])
        list_result[i].append(len(sentence_list))
        list_result[i].append(count_label[0]/float(len(sentence_list)))
        list_result[i].append(count_label[1]/float(len(sentence_list)))
        list_result[i].append(count_label[2]/float(len(sentence_list)))
        list_result[i].append(count_label[3]/float(len(sentence_list)))
    path = './data_result/twitter.csv'
    f = open(path,'w',encoding='utf-8',newline='')
    csv_writer = csv.writer(f)
    for row in list_result:
        print(row)
        csv_writer.writerow(row)
    f.close()


def test_Function_weibo():
    my_module = torch.load("./model/my_1_weibo_model.pth")#注意模型不同
    my_module = my_module.to(device=devices)
    list_time_01=['2022-02-01%','2022-02-02%','2022-02-03%','2022-02-04%','2022-02-05%','2022-02-06%','2022-02-07%','2022-02-08%','2022-02-09%','2022-02-10%',
               '2022-02-11%','2022-02-12%','2022-02-13%','2022-02-14%','2022-02-15%','2022-02-16%','2022-02-17%','2022-02-18%','2022-02-19%','2022-02-20%',
               '2022-02-21%','2022-02-22%','2022-02-23%','2022-02-24%','2022-02-25%','2022-02-26%','2022-02-27%','2022-02-28%','2022-03-01%','2022-03-02%','2022-03-03%','2022-03-04%','2022-03-05%','2022-03-06%','2022-03-07%','2022-03-08%','2022-03-09%','2022-03-10%',
               '2022-03-11%','2022-03-12%','2022-03-13%','2022-03-14%','2022-03-15%','2022-03-16%','2022-03-17%','2022-03-18%','2022-03-19%','2022-03-20%',
               '2022-03-21%','2022-03-22%','2022-03-23%','2022-03-24%','2022-03-25%','2022-03-26%','2022-03-27%','2022-03-28%','2022-03-29%','2022-03-30%','2022-03-31%','2022-04-01%','2022-04-02%','2022-04-03%','2022-04-04%','2022-04-05%','2022-04-06%','2022-04-07%','2022-04-08%','2022-04-09%','2022-04-10%',
               '2022-04-11%','2022-04-12%','2022-04-13%','2022-04-14%','2022-04-15%','2022-04-16%','2022-04-17%','2022-04-18%','2022-04-19%','2022-04-20%',
               '2022-04-21%','2022-04-22%','2022-04-23%','2022-04-24%','2022-04-25%','2022-04-26%','2022-04-27%','2022-04-28%','2022-04-29%','2022-04-30%','2022-05-01%','2022-05-02%','2022-05-03%','2022-05-04%','2022-05-05%','2022-05-06%','2022-05-07%','2022-05-08%','2022-05-09%','2022-05-10%',
               '2022-05-11%','2022-05-12%','2022-05-13%','2022-05-14%','2022-05-15%','2022-05-16%','2022-05-17%','2022-05-18%','2022-05-19%','2022-05-20%',
               '2022-05-21%','2022-05-22%','2022-05-23%','2022-05-24%','2022-05-25%','2022-05-26%','2022-05-27%','2022-05-28%','2022-05-29%','2022-05-30%','2022-05-31%','2022-06-01%','2022-06-02%','2022-06-03%','2022-06-04%','2022-06-05%','2022-06-06%','2022-06-07%','2022-06-08%','2022-06-09%','2022-06-10%',
               '2022-06-11%']
    list_time_02=['2022-02-20%']
    db = pymysql.connect(host='localhost', user='root', passwd='jiajian233', port=3306,db="test")
    cursor = db.cursor()  # 创建游标
    list_time=list_time_01#此处修改时间
    list_result=[[] for i in range(len(list_time))]
    for i in range(len(list_time)):
        sql = "select time,content_w from weibo_new where time like'{}'".format(list_time[i])
        db.connect()
        cursor.execute(sql)
        db.commit()
        sentence_list=[]
        while 1:
            str_result_fenci=" "
            res=cursor.fetchone()
            if res is None: 
                break#表示已经取完结果集
            else:
                #print(res[1]) #取文本内容，文本内容清不清洗区别不大
                sentence_list.append(res[1])
        test = MyDataset(sentences=sentence_list,with_labels=False) #此处不调用DataLoader
        my_module.eval()
        count_label=[0,0,0,0,0]#初始化类别计数
        for j in range(len(sentence_list)):
            #print(sentence_list[j])
            sentence_input = test.__getitem__(j)
            #print(sentence_input)
            sentence_input = torch.LongTensor(sentence_input).to(device=devices)
            sentence_input=sentence_input.unsqueeze(0) #转成二维
            #print(sentence_input)
            pred = my_module([sentence_input])
            pred = pred.data.max(dim=1, keepdim=True)[1]
            #print(f'预测为{pred[0][0]}')
            count_label[pred[0][0]]+=1#对应类别的数加一，注意weibo和hotline不一样。微博是五类，且从零开始
        print(f'日期为{list_time[i]}，当天数据总数为{len(sentence_list)}，分布情况为{count_label}')
        #list_result[i].append(list_time[i]).append(len(sentence_list)).append(count_label[0]).append(count_label[1]).append(count_label[2]).append(count_label[3])
        list_result[i].append(list_time[i])
        list_result[i].append(len(sentence_list))
        list_result[i].append(count_label[0]/float(len(sentence_list)))
        list_result[i].append(count_label[1]/float(len(sentence_list)))
        list_result[i].append(count_label[2]/float(len(sentence_list)))
        list_result[i].append(count_label[3]/float(len(sentence_list)))
        list_result[i].append(count_label[4]/float(len(sentence_list)))
    path = './data_result/weibo_2.csv'
    f = open(path,'a+',encoding='utf-8',newline='')
    csv_writer = csv.writer(f)
    for row in list_result:
        print(row)
        csv_writer.writerow(row)
    f.close()


def verify_Function_weibo(pth):
    content_list=[]
    label_list=[]
    count_error=0
    count_error_1=0
    count_error_0=0
    number_1_total=0
    number_0_total=0
    setup_seed(51)
    with open('./csv/weibo验证集.csv',encoding="utf-8") as f:
         f_csv = csv.reader(f)
         header = next(f_csv)
         print(header)
         for row in f_csv:
              content,label=row  
              content_list.append(content)
              label_list.append(label)
              #print(content)
    #选择模型 first1对应180条长度为40  first2对应180条长度为80
    my_module = torch.load(pth)
    my_module = my_module.to(device=devices)
    my_module.eval()
    test = MyDataset(sentences=content_list,with_labels=False) #此处不调用DataLoader
    for i in range(len(content_list)):
        print(content_list[i])
        print("真实类别：{}".format(label_list[i]))
        if(label_list[i]==0):
            number_0_total+=1
        if(label_list[i]==1):
            number_1_total+=1
        sentence_input = test.__getitem__(i)
        sentence_input = torch.LongTensor(sentence_input).to(device=devices)
        #print(sentence_input)
        sentence_input=sentence_input.unsqueeze(0) #转成二维
        #print(sentence_input)
        pred = my_module([sentence_input])
        pred = pred.data.max(dim=1, keepdim=True)[1]#取出最大的
        print(f'预测为{pred[0][0]}')
        print(f'真实为{label_list[i]}')
        if(int(label_list[i])!=pred[0][0]):
            count_error=count_error+1
    print("预测错误：{}，总数：{}".format(count_error,len(content_list)))
def verify_Function_hotline(pth):
    content_list=[]
    label_list=[]
    count_error=0
    count_error_1=0
    count_error_0=0
    number_1_total=0
    number_0_total=0
    setup_seed(51)
    with open('./csv/twitter验证集.csv',encoding="utf-8") as f:
         f_csv = csv.reader(f)
         header = next(f_csv)
         print(header)
         for row in f_csv:
              content,label=row  
              content_list.append(content)
              label_list.append(label)
              #print(content)
    my_module = torch.load(pth)
    my_module = my_module.to(device=devices)
    my_module.eval()
    test = MyDataset(sentences=content_list,with_labels=False) #此处不调用DataLoader
    for i in range(len(content_list)):
        print(content_list[i])
        if(label_list[i]==0):
            number_0_total+=1
        if(label_list[i]==1):
            number_1_total+=1
        sentence_input = test.__getitem__(i)
        sentence_input = torch.LongTensor(sentence_input).to(device=devices)
        #print(sentence_input)
        sentence_input=sentence_input.unsqueeze(0) #转成二维
        #print(sentence_input)
        pred = my_module([sentence_input])
        pred = pred.data.max(dim=1, keepdim=True)[1]#取出最大的
        print(f'预测为{pred[0][0]}')
        print(f'真实为{label_list[i]}')
        if(int(label_list[i])!=pred[0][0]):
            count_error=count_error+1
    print("预测错误：{}，总数：{}".format(count_error,len(content_list)))

if __name__ == "__main__":
   #train_Function_weibo()
   #test_Function(sentence_list)
   #verify_Function_weibo('./model/my_2_weibo_model.pth')
   train_Function_twitter()
   #verify_Function_hotline('./分类/model/my_1_hotline_model.pth')
   #test_Function_hotline()
   #test_Function_weibo()