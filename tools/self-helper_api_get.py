from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor
import json
import copy
from tqdm import tqdm
import queue
import time

with open ("data/get_data/helper_langgpt.txt", 'r', encoding='utf-8') as f:
    data = f.read()
    base_id_prompt=data



internlm_client = OpenAI(
    api_key="{your_api_key}",
    base_url="{your_url}",
)

glm_client = OpenAI(
    api_key="{your_api_key}",
    base_url="{your_url}",

)

deepseek_client = OpenAI(
    api_key="{your_api_key}",
    base_url="{your_url}",
)


class Get_Data_Api():
    def __init__(self,questions_path,save_path,repeat=0):
        self.client = internlm_client
        self.messages_template=[{"role": "system", "content": base_id_prompt},
                                {"role": "user", "content": "xxx"}],
        self.data_template={
                            "conversation":[
                                {
                                    "system": base_id_prompt,
                                    "input": "xxx",
                                    "output": "xxx"
                                }
                            ]
                        }
        self.questions_path=questions_path
        self.save_path=save_path
        self.repeat=repeat

    def get_answer(self,question):
        self.messages_template[0][1]["content"]=question

        item_template = self.messages_template
        item_template[0][1]["content"]=question
        chat_rsp = self.client.chat.completions.create(
            model="internlm2-latest",#"internlm2-latest",#"glm-4",
            messages=[{"role": "system", "content": base_id_prompt},
                    {"role": "user", "content": question}],
            stream=False,
        )
        train_data=self.build_data(question,chat_rsp)
        return train_data
    
    def build_data(self,question,chat_rsp):
        temp=copy.deepcopy(self.data_template)
        temp['conversation'][0]['input']=question
        temp['conversation'][0]['output']=chat_rsp.choices[0].message.content
        return temp
    
    def run(self):
        answer_queue=queue.Queue()
        # train_data_list=[]
        promptlist=self.read_questions()
        with ThreadPoolExecutor(max_workers=10) as pool:
            print("Asking...")
            for question in tqdm(promptlist):
                results=pool.submit(self.get_answer,question)
                answer_queue.put(results.result())
                # train_data_list.append(results.result())
                pool.submit(self.save,[answer_queue.get()])
                
        # return list(results.result())
        # for question in promptlist:
        #     result=self.get_answer(question)
        #     train_data_list.append(result)
        

    def save(self,train_data):
        with open(self.save_path,'a') as f:
            # print("Saving...")
            for item in train_data:
                json_item = json.dumps(item, ensure_ascii=False) 
                f.write(json_item + "\n") 
    
    def load_txt(self,path):
        with open (path, 'r', encoding='utf-8') as f:
            data = f.read()
        
        return data

    def read_questions(self):
        prompt=self.load_txt(self.questions_path)
        promptlist=prompt.split('\n')
        if self.repeat !=0:
            promptlist = [item for _ in range(self.repeat) for item in promptlist]
        print(f"Total question: {len(promptlist)}")
        return promptlist
    
if __name__ == '__main__':
    questions_path='data/get_data/questionList.txt'
    save_path='data/train_data/helper_traindata.jsonl'
    start_time = time.time()
    gda=Get_Data_Api(questions_path,save_path,repeat=10)
    gda.run()
    end_time = time.time()
    print('Done')
    print(f'use time:{str(end_time-start_time)}')
