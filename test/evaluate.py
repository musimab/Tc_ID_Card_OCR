import json


pred_path = "predictions_json/predicted_data.json"
true_path = "true_json/data.json"

def loadDict(path):  
    predicted_words = []
    with open(path, 'r', encoding='utf-8') as fp:
        data = json.load(fp)
        predicted_words.append(data)
    
    return predicted_words 


def countDictionaryItems(wordsList):
    
    lenTc = 0
    lenName = 0
    lenDateofBirth =0
    lenSurname = 0
    print("Number of samples:", len(wordsList[0]))
    for words_dict in wordsList:
        
        for img_name, person_info in words_dict.items():
            
            for key, val in person_info.items():
                
                if(key == "Surname"):
                    lenSurname = lenSurname + len(person_info[key])
                
                if(key == "Name"):
                    lenName = lenName + len(person_info[key])
                
                if(key == "DateofBirth"):
                    lenDateofBirth = lenDateofBirth + len(person_info[key])
                
                if(key == "Tc"):
                    lenTc = lenTc + len(person_info[key])


    print("lenSurname:", lenSurname )  
                
    print("lenName:", lenName )

    print("lenDateofBirth:", lenDateofBirth )

    print("lenTc:", lenTc )  



def comparisionInCharacterLevel(dict_true, dict_predict):
  
    tc_result = []
    surname_result = []
    name_result = []
    date_result = []
    
    for dict_t, dict_p in zip(dict_true,  dict_predict):
        for (img_name_t, person_info_t), (img_name_p, person_info_p) in zip(dict_t.items(), dict_p.items()):
            
            for truth, pred in zip(person_info_t["Tc"] , person_info_p["Tc"]):
                tc_result.append(truth ==  pred)
            
            for truth, pred  in zip(person_info_t["Surname"] , person_info_p["Surname"]):
                surname_result.append(truth ==  pred )            
            
            for truth, pred in zip(person_info_t["Name"] , person_info_p["Name"]):
                name_result.append(truth == pred )
            
            for truth, pred in zip(person_info_t["DateofBirth"] , person_info_p["DateofBirth"]):
                date_result.append(truth == pred )

    
    print("tc:", sum(tc_result),"/" ,len(tc_result), "%", 100.0 * sum(tc_result)/len(tc_result))

    print("surname:", sum(surname_result),"/", len(surname_result), "%", 100.0 * sum(surname_result)/len(surname_result))

    print("name:", sum(name_result),"/", len(name_result), "%", 100 * sum(name_result)/len(name_result) )

    print("dateofbirth:", sum(date_result),"/", len(date_result), "%", 100 * sum(date_result)/len(date_result))



def comparisionInWordLevel(dict_true, dict_predict):
    
    tc_result = []
    surname_result = []
    name_result = []
    date_result = []

    id_infos= ["Tc", "Surname", "Name", "DateofBirth"]

    for dict_t, dict_p in zip(dict_true,  dict_predict):
        
        for (img_name_t, person_info_t), (img_name_p, person_info_p) in zip(dict_t.items(), dict_p.items()):
            
            tc_result.append(person_info_t["Tc"] == person_info_p["Tc"])
            surname_result.append(person_info_t["Surname"] == person_info_p["Surname"])
            name_result.append(person_info_t["Name"] == person_info_p["Name"])
            date_result.append(person_info_t["DateofBirth"] == person_info_p["DateofBirth"])

  
    return tc_result, surname_result, name_result, date_result



if '__main__' == __name__:
    
    wordsListTrue = loadDict(true_path)
    wordsListPred = loadDict(pred_path)

    print(" Count prediction items")
    countDictionaryItems(wordsListPred)
    print(" Count true items")
    countDictionaryItems(wordsListTrue)

    tc_result, surname_result, name_result, date_result = comparisionInWordLevel(wordsListTrue , wordsListPred)
    
    print("##### Word Level Comparision  #### ")

    print("tc result:", sum(tc_result)/len(tc_result))
    print("surname result:", sum(surname_result)/len(surname_result))
    print("name result:", sum(name_result)/ len(name_result))
    print("date_result:", sum(date_result)/len(date_result))


    print("##### Character Level Comparision #### ")

    comparisionInCharacterLevel(wordsListTrue , wordsListPred)



            
                 
