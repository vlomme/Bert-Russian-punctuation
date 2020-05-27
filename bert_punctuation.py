import re
import pymorphy2
import torch
from pytorch_pretrained_bert import BertTokenizer, BertForMaskedLM


class Bert_punctuation(object):
    def __init__(self):
        self.model_file = "bert_punctuation.tar.gz"
        self.vocab_file = "vocab.txt"
        self.model = self.bert_model()
        self.tokenizer = self.bert_tokenizer()

    def bert_model(self):
        model = BertForMaskedLM.from_pretrained(self.model_file).eval()
        return model

    def bert_tokenizer(self):
        tokenizer = BertTokenizer.from_pretrained(self.vocab_file, do_lower_case=True)
        return tokenizer
    
    def what_mask(self, text):
        # Смотрим стоит запятая, или нет
        w = self.tokenizer.tokenize(',')
        w_i = self.tokenizer.convert_tokens_to_ids(w)
        w = self.tokenizer.tokenize('^')
        w_j = self.tokenizer.convert_tokens_to_ids(w)        
        text = '[CLS] ' + text + ' [SEP]'
        tokenized_text = self.tokenizer.tokenize(text)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        indexed_tokens = indexed_tokens[:500]
        
        mask_input = []
        for i in range(len(indexed_tokens)):
            if indexed_tokens[i] == 103:
                mask_input.append(i)
        segments_ids = [0] * len(tokenized_text)
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])
        masked_index = mask_input
        with torch.no_grad():
            predictions = self.model(tokens_tensor, segments_tensors)
        predictsx1 = []
        for i in range(len(mask_input)):
            predictsx1.append(predictions[0,mask_input[i],:])
            predicts1 = predictsx1[i].argsort()[-8:].numpy()
            out1 = self.tokenizer.convert_ids_to_tokens(predicts1)
        output = []
        a = len(mask_input)
        for i in range(a):
            if predictsx1[i][w_i] > predictsx1[i][w_j]:
                output.append(i+1)

        return output
      
    def predict(self, texts):
        words_all = texts
        par_b = [['стар','млад'],  ['жив','мертв'],  ['день','ночь']]
        sens = []
        morph = pymorphy2.MorphAnalyzer()
        for i,words in enumerate(words_all):
            words = words.strip().lower()
            choice_list = words.replace('–','-').replace('—','-').replace(',',' ,').replace('.',' .').replace('!',' !').replace('?',' ?').replace(':',' :').replace(';',' ;').replace('»',' »').replace('«','« ').split()
            pos = ([str(morph.parse(ok)[0].tag.POS) for ok in choice_list])
            case = ([str(morph.parse(ok)[0].tag.case) for ok in choice_list])
            all_cases = ([morph.parse(ok) for ok in choice_list])
            for j, p in enumerate(pos):            
                eto_NOUN = False
                bad_par = False
                if p == 'PRTF' and (j==0 or pos[j-1] !='CONJ'):
                    choice_list[j] = '[MASK] '+choice_list[j] 
                if p == 'VERB':
                    for iii in range(j+1,min(len(pos),j+12)):
                        if pos[iii] == 'NOUN':
                            for all_cases_z in all_cases[iii]:
                                if  all_cases_z.tag.case =='nomn':
                                    #eto_NOUN = True
                                    eto_NOUN = False
                                    break
                            if eto_NOUN:
                                break
                        if pos[iii] =='VERB' and pos[iii-1] !='CONJ':
                            #print('Глагол',choice_list[j],choice_list[iii])
                            choice_list[iii] = '[MASK] '+choice_list[iii]  
                if p == 'INFN':
                    for iii in range(j+1,min(len(pos),j+5)):
                        if pos[iii] =='INFN' and pos[iii-1] !='CONJ':
                            #print('Инфинитив',choice_list[j],choice_list[iii])
                            choice_list[j] = choice_list[j]+ ' [MASK]'                
                if p == 'ADVB':
                    for iii in range(j+1,min(len(pos),j+3)):
                        if pos[iii] =='ADVB' and pos[iii-1] !='CONJ':
                            #print('Наречие',choice_list[j],choice_list[iii])
                            choice_list[j] = choice_list[j]+ ' [MASK]'
                if p == 'ADJF':
                    for iii in range(j+1,min(len(pos),j+4)):
                        if pos[iii] =='ADJF' and pos[iii-1] !='CONJ':
                            if case[j] ==  case[iii]:
                                #print('Прилогательное',choice_list[j],choice_list[iii])
                                choice_list[j] = choice_list[j]+ ' [MASK]'
                                break
                eto_NOUN = False
                for all_cases_j in all_cases[j]:
                    if all_cases_j.score> 0.05 and all_cases_j.tag.POS == 'NOUN':
                        eto_NOUN = True
                if eto_NOUN:
                    odnorod = False
                    NOUN_est_ADJF = False
                    for ii in range(j+1,len(pos)):
                        if pos[ii] =='NOUN':
                            for all_cases_j in all_cases[j]:
                                for all_cases_z in all_cases[ii]:
                                    if all_cases_z.tag.POS == 'ADJF' or all_cases_j.tag.POS == 'ADJF':
                                        NOUN_est_ADJF = True
                                    #if all_cases_j.score> 0.24 and all_cases_z.score> 0.24 and all_cases_j.tag.case == all_cases_z.tag.case  and all_cases_j.tag.number == all_cases_z.tag.number:
                                    if (all_cases_j.tag.case == all_cases_z.tag.case) and (choice_list[j] != 'свет' or choice_list[ii] != 'заря'):
                                        odnorod = True
                            if odnorod:
                                if not NOUN_est_ADJF:
                                    #print('Сущ',choice_list[j],choice_list[ii])
                                    choice_list[j] = choice_list[j]+ ' [MASK]'
                                break    
                        
                        
                        if pos[ii] !='NOUN' and pos[ii] !='ADJF' and pos[ii] !='PREP' and pos[ii] !='PRCL':
                            break            
                
                if  (p == 'CONJ'or choice_list[j] =='да' or choice_list[j] =='ни')  and j>0 and pos[j-1] != 'CONJ':
                    if choice_list[j] =='то' and choice_list[j-1] =='не':
                            #print('Союз',choice_list[j-1],choice_list[j])
                            choice_list[j-1] = '[MASK] '+choice_list[j-1]  
                          
                    elif choice_list[j] !='ни' or (choice_list[j] =='ни' and (j==0 or choice_list[j-1] !='свет')and (j==len(pos) or choice_list[j+1] !='свет')):
                        #print('Союз',choice_list[j],choice_list[j-1],choice_list[j+1])
                        for pb in par_b:
                            if (j!=0 and choice_list[j-1] ==pb[0]) and (j!=len(pos) and choice_list[j+1] ==pb[1]):
                                bad_par = True
                        if j==1 and choice_list[j] == 'и':
                            bad_par = True
                        if choice_list[j] == 'ни' and (j!=len(pos) or choice_list[j+1] =='то'):
                            bad_par = True                       
                        if not bad_par:  
                            choice_list[j] = '[MASK] '+choice_list[j]
                       
            words = ' '.join(choice_list).replace(' ,',',').replace(' .','.').replace(' !','!').replace(' ?','?').replace(' :',':').replace(' ;',';').replace(' »','»').replace('« ','«')
            words = words.replace('[MASK] [MASK]','[MASK]').replace('[MASK] [MASK]','[MASK]')
            #print(words)
            if words.startswith('[MASK] '):
                words = words.replace('[MASK] ','',1)
            
            result = self.what_mask(words)
            #print(result)
            if result:
                for i in range(1, max(result)+1):
                    if i in result:
                        words = words.replace(' [MASK]', ',', 1)
                    else:
                        words = words.replace(' [MASK]', '', 1)
            words = words.replace(' [MASK]', '')
            #print(words)
            sens.append(words)
        return sens
