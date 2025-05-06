import json
import re
from django.http import JsonResponse
from django.shortcuts import render
from transformers import AutoTokenizer, AutoModelForCausalLM
from django.views.decorators.csrf import csrf_protect

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")

tokenizer.pad_token = tokenizer.eos_token

def index(request):
    return render(request, 'index.html')

@csrf_protect 
def generate_text(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        user_input = data.get('userInput', '').strip()

        if not user_input:
            return JsonResponse({'error': 'Input text is required'}, status=400)

        keywords = ['environment', 'sustainability', 'eco', 'carbon', 'recycle', 'green', 'energy', 'climate', 'plastic', 'waste']
        if not any(word in user_input.lower() for word in keywords):
            return JsonResponse({'error': 'Please ask a question related to sustainability'}, status=400)

        inputs = tokenizer.encode(user_input, return_tensors="pt", padding=True, truncation=True)
        attention_mask = inputs.ne(tokenizer.pad_token_id).type(inputs.dtype)

        try:
            outputs = model.generate(
                inputs,
                max_length=250,  
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.95, 
                do_sample=True,
                attention_mask=attention_mask,
                pad_token_id=tokenizer.eos_token_id
            )

            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            generated_text = re.sub(r"([.!?])\s*$", r"\1", generated_text) 

            sentences = generated_text.split('. ')
            if len(sentences) > 10:
                generated_text = '. '.join(sentences[:10]) + '.'

            return JsonResponse({'generatedText': generated_text})

        except Exception as e:
            print(f"Error generating text: {e}")
            return JsonResponse({'error': 'Error generating text'}, status=500)

    return JsonResponse({'error': 'Invalid method'}, status=405)
