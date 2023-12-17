import torch
from transformers import Blip2Processor,Blip2ForConditionalGeneration, AutoTokenizer, apply_chunking_to_forward
import json
from tqdm import tqdm
from Dataloader.dataloader import *
from utils import *

VERSION = "Salesforce/blip2-flan-t5-xxl"

def get_ckpt_model(path='', load_language_model=False, load_image_encoder=True, path2=''):
    processor = Blip2Processor.from_pretrained(VERSION)
    device = "cuda" 
    if path != '':
        model = Blip2ForConditionalGeneration.from_pretrained(VERSION, device_map='auto')#, load_in_8bit = True, device_map="auto")
        checkpoint = torch.load(path, map_location='cpu')
        ckpt_pth = checkpoint['model_state_dict']
        model.qformer.load_state_dict(ckpt_pth)
    else:
        model = Blip2ForConditionalGeneration.from_pretrained(VERSION, device_map="auto") #, load_in_8bit=True, device_map="auto")\
    if load_language_model:
        checkpoint = torch.load('language_model_path_here', map_location='cpu')
        ckpt_pth = checkpoint['model_state_dict']
        model.language_model.load_state_dict(ckpt_pth)
    if load_image_encoder and path2 != '':
        checkpoint = torch.load(path2, map_location='cpu')
        ckpt_pth = checkpoint['model_vision_encoder_state_dict']
        model.vision_model.load_state_dict(ckpt_pth)
    model.eval()
    return processor, model

def convert_index_to_letter(index):
    if index == 0:
        return "(A)"
    elif index == 1:
        return "(B)"
    elif index == 2:
        return "(C)"
    return "(D)"

def model_generate(model, pixel_values, input_ids, attention_mask, cot_ids=None, cot_attns=None, img_region=None, img_mask=None):
    with torch.no_grad():
        batch_size = pixel_values.shape[0]
        image_embeds = model.vision_model(pixel_values, return_dict=True).last_hidden_state
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

        query_tokens = model.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_outputs = model.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
        )
        query_output = query_outputs.last_hidden_state
        language_model_inputs = model.language_projection(query_output) 
        language_attention_mask = torch.ones(
            language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
        )
        inputs_embeds = model.get_input_embeddings()(input_ids)
        
        inputs_embeds = torch.cat([language_model_inputs, inputs_embeds.to(language_model_inputs.device)], dim=1)
        attention_mask = torch.cat([language_attention_mask, attention_mask.to(language_attention_mask.device)], dim=1)

        if cot_ids != None:
            cot_ids = cot_ids.expand(batch_size, -1, -1)
            cot_attns = cot_attns.expand(batch_size, -1)
            inputs_embeds = torch.cat([cot_ids, inputs_embeds], dim=1)
            attention_mask = torch.cat([cot_attns, attention_mask], dim=1)
        
        if img_region != None:
            inputs_embeds = torch.cat([inputs_embeds, img_region.to(inputs_embeds.device)], dim=1)
            attention_mask = torch.cat([attention_mask, img_mask.to(attention_mask.device)], dim=1)

        outputs = model.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_length = 256,
            min_length = 8,
            # length_penalty=1,
            num_beams = 5,
        )
            
        return outputs

def blip2qformerlayer_selfforward(layer, hidden_states,attention_mask=None,head_mask=None,encoder_hidden_states=None,encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        query_length=0,):
    self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
    self_attention_outputs = layer.attention(
        hidden_states,
        attention_mask,
        head_mask,
        output_attentions=output_attentions,
        past_key_value=self_attn_past_key_value,
    )
    attention_output = self_attention_outputs[0]
    outputs = self_attention_outputs[1:-1]

    present_key_value = self_attention_outputs[-1]

    if query_length > 0:
        if layer.has_cross_attention:
            query_length = 32
        query_attention_output = attention_output[:, :query_length, :]
        
        if layer.has_cross_attention:
            if encoder_hidden_states is None:
                raise ValueError("encoder_hidden_states must be given for cross-attention layers")
            cross_attention_outputs = layer.crossattention(
                query_attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                output_attentions=output_attentions,
            )
            query_attention_output = cross_attention_outputs[0]
            # add cross attentions if we output attention weights
            outputs = outputs + cross_attention_outputs[1:-1]

        layer_output = apply_chunking_to_forward(
            layer.feed_forward_chunk_query,
            layer.chunk_size_feed_forward,
            layer.seq_len_dim,
            query_attention_output,
        )
        
        if attention_output.shape[1] > query_length:
            layer_output_text = apply_chunking_to_forward(
                layer.feed_forward_chunk_query,
                layer.chunk_size_feed_forward,
                layer.seq_len_dim,
                attention_output[:, query_length:, :],
            )
            layer_output = torch.cat([layer_output, layer_output_text], dim=1)
    else:
        layer_output = apply_chunking_to_forward(
            layer.feed_forward_chunk,
            layer.chunk_size_feed_forward,
            layer.seq_len_dim,
            attention_output,
        )
    outputs = (layer_output,) + outputs

    outputs = outputs + (present_key_value,)

    return outputs

def blip2qformerencoder_selfforward(encoder,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        query_length=0,):
    all_hidden_states = () if output_hidden_states else None
    all_self_attentions = () if output_attentions else None
    all_cross_attentions = () if output_attentions else None

    next_decoder_cache = () if use_cache else None

    for i in range(encoder.config.num_hidden_layers):
        layer_module = encoder.layer[i]
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        layer_head_mask = head_mask[i] if head_mask is not None else None
        past_key_value = past_key_values[i] if past_key_values is not None else None

        if getattr(encoder.config, "gradient_checkpointing", False) and encoder.training:
            
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs, past_key_value, output_attentions, query_length)

                return custom_forward

            layer_outputs = torch.utils.checkpoint.checkpoint(
                create_custom_forward(layer_module),
                hidden_states,
                attention_mask,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
            )
        else:
            layer_outputs = blip2qformerlayer_selfforward(layer_module, hidden_states,
                attention_mask,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_value,
                output_attentions,
                query_length,)
            
        hidden_states = layer_outputs[0]
        if use_cache:
            next_decoder_cache += (layer_outputs[-1],)
        if output_attentions:
            all_self_attentions = all_self_attentions + (layer_outputs[1],)
            if layer_module.has_cross_attention:
                all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

    if not return_dict:
        return tuple(
            v
            for v in [
                hidden_states,
                next_decoder_cache,
                all_hidden_states,
                all_self_attentions,
                all_cross_attentions,
            ]
            if v is not None
        )
    return hidden_states,
        

def blip2qformer_selfforward(qformer, query_embeds,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,):
    output_attentions = output_attentions if output_attentions is not None else qformer.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else qformer.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else qformer.config.use_return_dict

    # past_key_values_length
    past_key_values_length = (
        past_key_values[0][0].shape[2] - qformer.config.query_length if past_key_values is not None else 0
    )

    query_length = query_embeds.shape[1] if query_embeds is not None else 0

    embedding_output = qformer.layernorm(query_embeds)
    embedding_output = qformer.dropout(embedding_output)

    input_shape = embedding_output.size()[:-1]
    batch_size, seq_length = input_shape
    device = embedding_output.device

    if attention_mask is None:
        attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

    # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
    # ourselves in which case we just need to make it broadcastable to all heads.
    extended_attention_mask = qformer.get_extended_attention_mask(attention_mask, input_shape, device)

    # If a 2D or 3D attention mask is provided for the cross-attention
    # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
    if encoder_hidden_states is not None:
        if type(encoder_hidden_states) == list:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states[0].size()
        else:
            (
                encoder_batch_size,
                encoder_sequence_length,
                _,
            ) = encoder_hidden_states.size()
        encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)

        if type(encoder_attention_mask) == list:
            encoder_extended_attention_mask = [qformer.invert_attention_mask(mask) for mask in encoder_attention_mask]
        elif encoder_attention_mask is None:
            encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = qformer.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = qformer.invert_attention_mask(encoder_attention_mask)
    else:
        encoder_extended_attention_mask = None

    # Prepare head mask if needed
    # 1.0 in head_mask indicate we keep the head
    # attention_probs has shape bsz x n_heads x N x N
    # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
    # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
    head_mask = qformer.get_head_mask(head_mask, qformer.config.num_hidden_layers)

    encoder_outputs = blip2qformerencoder_selfforward(qformer.encoder, embedding_output,
        attention_mask=extended_attention_mask,
        head_mask=head_mask,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=encoder_extended_attention_mask,
        past_key_values=past_key_values,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        query_length=query_length,)
    sequence_output = encoder_outputs[0]
    pooled_output = sequence_output[:, 0, :]

    if not return_dict:
        return (sequence_output, pooled_output) + encoder_outputs[1:]

    return sequence_output


def model_generate_with_text_selfattn_only(model, pixel_values, text=None, input_ids=None, attention_mask=None, cot_ids=None, cot_attns=None, img_region=None, img_mask=None, return_mid=False):
    with torch.no_grad():
        batch_size = pixel_values.shape[0]
        image_embeds = model.vision_model(pixel_values, return_dict=True).last_hidden_state
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)
        query_tokens = model.query_tokens.expand(image_embeds.shape[0], -1, -1)
        if text != None:
            if isinstance(text, list):
                inputs_embeds_1, attention_mask_1 = get_word_embeddings(text)
                # query_tokens = torch.cat((inputs_embeds_1.to(query_tokens.device), query_tokens), dim=1)
                query_tokens = torch.cat((query_tokens, inputs_embeds_1.to(query_tokens.device)), dim=1)
            else:
                inputs_embeds_1, attention_mask_1 = get_word_embeddings(text)
                inputs_embeds_new =  inputs_embeds_1.expand(image_embeds.shape[0], -1, -1)
                query_tokens = torch.cat((inputs_embeds_new.to(query_tokens.device), query_tokens), dim=1)
        query_outputs = blip2qformer_selfforward(model.qformer, query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,)
        
        query_output = query_outputs
        language_model_inputs = model.language_projection(query_output) 
        language_model_inputs = language_model_inputs[:, :, :]
        language_attention_mask = torch.ones(
            language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
        )
        
        if input_ids == None:
            inputs_embeds = language_model_inputs#torch.cat([language_model_inputs, inputs_embeds.to(language_model_inputs.device)], dim=1)
            attention_mask = language_attention_mask#torch.cat([language_attention_mask, attention_mask.to(language_attention_mask.device)], dim=1)
        else:
            inputs_embeds = model.get_input_embeddings()(input_ids)
            inputs_embeds = torch.cat([language_model_inputs, inputs_embeds.to(language_model_inputs.device)], dim=1)
            attention_mask = torch.cat([language_attention_mask, attention_mask.to(language_attention_mask.device)], dim=1)

        if cot_ids != None:
            cot_ids = cot_ids.expand(batch_size, -1, -1)
            cot_attns = cot_attns.expand(batch_size, -1)
            inputs_embeds = torch.cat([cot_ids, inputs_embeds], dim=1)
            attention_mask = torch.cat([cot_attns, attention_mask], dim=1)
        
        if img_region != None:
            inputs_embeds = torch.cat([inputs_embeds, img_region.to(inputs_embeds.device)], dim=1)
            attention_mask = torch.cat([attention_mask, img_mask.to(attention_mask.device)], dim=1)
        if return_mid:
            return inputs_embeds, attention_mask
        outputs = model.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_length = 256,
            min_length = 8,
            # length_penalty=-1,
            num_beams = 5,
        )
            
        return outputs




def model_generate_with_text(model, pixel_values, text=None, input_ids=None, attention_mask=None, cot_ids=None, cot_attns=None, img_region=None, img_mask=None, return_mid=False):
    with torch.no_grad():
        batch_size = pixel_values.shape[0]
        image_embeds = model.vision_model(pixel_values, return_dict=True).last_hidden_state
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)
        query_tokens = model.query_tokens.expand(image_embeds.shape[0], -1, -1)
        if text != None:
            if isinstance(text, list):
                inputs_embeds_1, attention_mask_1 = get_word_embeddings(text)
                # query_tokens = torch.cat((inputs_embeds_1.to(query_tokens.device), query_tokens), dim=1)
                query_tokens = torch.cat((query_tokens, inputs_embeds_1.to(query_tokens.device)), dim=1)
            else:
                inputs_embeds_1, attention_mask_1 = get_word_embeddings(text)
                inputs_embeds_new =  inputs_embeds_1.expand(image_embeds.shape[0], -1, -1)
                query_tokens = torch.cat((inputs_embeds_new.to(query_tokens.device), query_tokens), dim=1)
        query_outputs = model.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
        )
        query_output = query_outputs.last_hidden_state
        language_model_inputs = model.language_projection(query_output) 
        language_model_inputs = language_model_inputs[:, 32:, :]
        language_attention_mask = torch.ones(
            language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
        )
        
        if input_ids == None:
            inputs_embeds = language_model_inputs#torch.cat([language_model_inputs, inputs_embeds.to(language_model_inputs.device)], dim=1)
            attention_mask = language_attention_mask#torch.cat([language_attention_mask, attention_mask.to(language_attention_mask.device)], dim=1)
        else:
            inputs_embeds = model.get_input_embeddings()(input_ids)
            inputs_embeds = torch.cat([language_model_inputs, inputs_embeds.to(language_model_inputs.device)], dim=1)
            attention_mask = torch.cat([language_attention_mask, attention_mask.to(language_attention_mask.device)], dim=1)

        if cot_ids != None:
            cot_ids = cot_ids.expand(batch_size, -1, -1)
            cot_attns = cot_attns.expand(batch_size, -1)
            inputs_embeds = torch.cat([cot_ids, inputs_embeds], dim=1)
            attention_mask = torch.cat([cot_attns, attention_mask], dim=1)
        
        if img_region != None:
            inputs_embeds = torch.cat([inputs_embeds, img_region.to(inputs_embeds.device)], dim=1)
            attention_mask = torch.cat([attention_mask, img_mask.to(attention_mask.device)], dim=1)
        if return_mid:
            return inputs_embeds, attention_mask
        outputs = model.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_length = 256,
            min_length = 8,
            # length_penalty=-1,
            num_beams = 5,
        )
            
        return outputs

def convert_to_choices(output_text):
    ans = []
    for item in output_text:
        if "(A" in item:
            ans.append(0)
        elif "(B" in item:
            ans.append(1)
        elif "(C" in item:
            ans.append(2)
        elif "(D" in item:
            ans.append(3)
        else:
            ans.append(-1)
    return torch.tensor(ans)

def esnli_generate_full_rationale(path='', load_image_encoder=True, path2='', add_rationale=False, use_boundingbox=False, step=3, path_name=''):
    cot_indexes = [401681, 325033, 140647,57811,257420]
    tokenizer = AutoTokenizer.from_pretrained(VERSION)
    processor, model = get_ckpt_model(path, load_image_encoder=load_image_encoder, path2=path2)
    eval_dataset = eSNLIEvalDataset(processor=processor, data_path="data",use_boundingbox=use_boundingbox)
    batch_size = 2
    progress_bar = tqdm(range(int(len(eval_dataset)/batch_size)))
    indexes_list = list(range(0, len(eval_dataset)))
    indexes = [tuple([indexes_list[i+j] for j in range(0, min(batch_size, len(eval_dataset)-0-i))]) for i in range(0, len(eval_dataset)-0, batch_size)]
    correct = 0
    tot_num = 0
    j = 0
    rationale_dict = {}
    rationale_data = {}
    if path != '':
        path_ = path.split('.')[0].split('/')[1]
    else:
        path_ = '_pretrained3'
    if add_rationale:
        with open('results/0.05_finetuned_step2', 'r') as f:
            rationale_dict = json.load(f)

    for index_tuple in indexes:
        inputs_image = []
        rationale_ans = []
        answer_list = []
        text_list = []
        for i in index_tuple:
            element = eval_dataset[i]
            inputs_image.append(element[0])
            if add_rationale:
                prompt = rationale_dict[str(i)].split(' because')[0] 
                text_list.append(rationale_dict[str(i)].split(prompt + " because ")[1])
            rationale_ans.append( "Based on the image can we conclude that '"+element[1]+"'?"+"\n\nAnswer: ")                                    
            # rationale_ans.append("Based on the image can we conclude that '"+element[1]+"'?"+"\n\nAnswer: ")  
            answer_list.append(element[2])
            j += 1
        inputs_image = torch.stack(inputs_image).squeeze(1)
        tokenized_rationale_ans = tokenizer(rationale_ans, padding='longest', truncation=True, return_tensors='pt')
        rationale_ans_id = tokenized_rationale_ans["input_ids"]
        rationale_ans_mask = tokenized_rationale_ans["attention_mask"]
        if add_rationale:
            output = model_generate_with_text(model, inputs_image, text_list, rationale_ans_id, rationale_ans_mask)
        else:
            output = model_generate(model, inputs_image, rationale_ans_id, rationale_ans_mask)
        output_text = processor.batch_decode(output, skip_special_tokens=True)
        print(output_text)
        print(answer_list)
        
        for num, i in enumerate(index_tuple):
            rationale_data[i] = output_text[num]
        progress_bar.update(1)       
        with open('./results/'+path_name, 'w') as f:
            json.dump(rationale_data, f, indent = 2)

def esnli_generate_selftrain(path='', load_image_encoder=True, path2='', add_rationale=False, use_boundingbox=False, step=3, path_name = ''):
    tokenizer = AutoTokenizer.from_pretrained(VERSION)
    processor, model = get_ckpt_model(path, load_image_encoder=load_image_encoder, path2=path2)
    eval_dataset = eSNLIEvalDataset(processor=processor, data_path="data",use_boundingbox=use_boundingbox)
    batch_size = 2
    progress_bar = tqdm(range(int(len(eval_dataset)/batch_size)))
    with open('./filtered_train/esnli_correct_on_2', 'r') as f:
        indexs = json.load(f)
    indexes_list = []
    for key in indexs.keys():
        indexes_list.append(key)   
    indexes = [tuple([indexes_list[i+j] for j in range(0, min(batch_size, len(indexes_list)-i))]) for i in range(0, len(indexes_list), batch_size)]
    print(indexes)
    correct = 0
    tot_num = 0
    j = 0
    rationale_dict = {}
    rationale_data = {}
    if path != '':
        path_ = path.split('.')[0].split('/')[1]
    else:
        path_ = '_pretrained3'

    for index_tuple in indexes:
        inputs_image = []
        rationale_ans = []
        answer_list = []
        text_list = []
        for i in index_tuple:
            element = eval_dataset[int(i)]
            inputs_image.append(element[0])
            if add_rationale:
                if ' because ' in rationale_dict[str(i)]: 
                    text_list.append(rationale_dict[str(i)].split(' because ')[1])
                else:
                    text_list.append(rationale_dict[str(i)]) 
                    # text_list.append(rationale_dict[str(i)].split(' because ')[1])
                rationale_ans.append("Answer the following question by reasoning step by step. Question: "+element[1]+"\n\nAnswer: ")                                    
            else:
                rationale_ans.append("Based on the image can we conclude that '"+element[1]+"'?"+"\n\nAnswer: "+element[2]+" because ")  
            # rationale_ans.append("Based on the image can we conclude that 'The answer to the question "+element[1]+" is "+rationale_dict[str(i)].split(' because ')[0] + "'?")
            answer_list.append(element[2]+" because ")
            j += 1
        inputs_image = torch.stack(inputs_image).squeeze(1)
        tokenized_rationale_ans = tokenizer(rationale_ans, padding='longest', truncation=True, return_tensors='pt')
        rationale_ans_id = tokenized_rationale_ans["input_ids"]
        rationale_ans_mask = tokenized_rationale_ans["attention_mask"]
        print(rationale_ans)
        if not add_rationale:
            output = model_generate(model, inputs_image,rationale_ans_id, rationale_ans_mask)
        else:
            output = model_generate_with_text(model, inputs_image, text_list, rationale_ans_id, rationale_ans_mask)
           
        output_text = processor.batch_decode(output, skip_special_tokens=True)
        print(output_text)
        
        for num, i in enumerate(index_tuple):
            rationale_data[i] = answer_list[num]+output_text[num]
        progress_bar.update(1)
        with open('./results/'+path_name, 'w') as f:
            json.dump(rationale_data, f, indent = 2)

def esnli_generate_wrongonly(path='', load_image_encoder=True, path2='', add_rationale=False, use_boundingbox=False, step=3, path_name = '',self_only=True):
    tokenizer = AutoTokenizer.from_pretrained(VERSION)
    processor, model = get_ckpt_model(path, load_image_encoder=load_image_encoder, path2=path2)
    eval_dataset = eSNLIEvalDataset(processor=processor, data_path="data",use_boundingbox=use_boundingbox)
    batch_size = 2
    with open('results/0.05_finetuned_step1', 'r') as f:
        data_before1 = json.load(f)
    indexes_list2 = list(range(0, len(eval_dataset)))
    indexes_list = []
    for i in indexes_list2:
        element = eval_dataset[i]
        prompt2 =  data_before1[str(i)].split(' because ')[0] 
        if prompt2 not in [element[2]]:
            indexes_list.append(i)
    indexes = [tuple([indexes_list[i+j] for j in range(0, min(batch_size, len(indexes_list)-i))]) for i in range(0, len(indexes_list), batch_size)]
    print(indexes)
    correct = 0
    tot_num = 0
    j = 0
    rationale_dict = {}
    rationale_data = {}
    if path != '':
        path_ = path.split('.')[0].split('/')[1]
    else:
        path_ = '_pretrained3'
    with open('results/esnli_step2_selfonly', 'r') as f:
        rationale_dict = json.load(f)
    progress_bar = tqdm(range(int(len(indexes)/batch_size)))
    for index_tuple in indexes:
        inputs_image = []
        rationale_ans = []
        answer_list = []
        text_list = []
        for i in index_tuple:
            element = eval_dataset[int(i)]
            inputs_image.append(element[0])
            if add_rationale:
                if ' because ' in rationale_dict[str(i)]: 
                    text_list.append(rationale_dict[str(i)].split(' because ')[1])
                else:
                    text_list.append(rationale_dict[str(i)]) 
                    # text_list.append(rationale_dict[str(i)].split(' because ')[1])
                rationale_ans.append("Based on the image can we conclude that '"+element[1]+"'?"+"\n\nAnswer: ")                                    
            else:
                rationale_ans.append("Based on the image can we conclude that '"+element[1]+"'?"+"\n\nAnswer: ")  
            # rationale_ans.append("Based on the image can we conclude that 'The answer to the question "+element[1]+" is "+rationale_dict[str(i)].split(' because ')[0] + "'?")
            answer_list.append(element[2]+" because ")
            j += 1
        inputs_image = torch.stack(inputs_image).squeeze(1)
        tokenized_rationale_ans = tokenizer(rationale_ans, padding='longest', truncation=True, return_tensors='pt')
        rationale_ans_id = tokenized_rationale_ans["input_ids"]
        rationale_ans_mask = tokenized_rationale_ans["attention_mask"]
        print(rationale_ans)
        if not add_rationale:
            output = model_generate(model, inputs_image,rationale_ans_id, rationale_ans_mask)
        else:
            if not self_only:
                output = model_generate_with_text(model, inputs_image, text_list, rationale_ans_id, rationale_ans_mask)
            else:
                 output = model_generate_with_text_selfattn_only(model, inputs_image, text_list, rationale_ans_id, rationale_ans_mask)
        output_text = processor.batch_decode(output, skip_special_tokens=True)
        print(output_text)
        
        for num, i in enumerate(index_tuple):
            rationale_data[i] = output_text[num]
        progress_bar.update(1)
        with open('./results/'+path_name, 'w') as f:
            json.dump(rationale_data, f, indent = 2)

if __name__=='__main__':

    """esnli"""
    path_qformer = 'ckpts/...' (enter your path)
    path = 'ckpts/...' (enter your path)
    esnli_generate_full_rationale(path=path_qformer, load_image_encoder=True, path2=path, add_rationale=False, use_boundingbox=False, path_name='selftrain_before')
    