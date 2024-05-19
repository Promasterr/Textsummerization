
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from flask import Flask, request, jsonify

import re
from flask_cors import CORS

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("tokenizer")
# tokenizer.add_special_tokens({"pad_token":"<pad>",
#                              "bos_token":"<startofstring>",
#                              "eos_token":"<endofstring>"})
# tokenizer.add_tokens("<bot>:")
model = AutoModelForSeq2SeqLM.from_pretrained("model")

model.eval()
def predict(inp):
    inp = tokenizer(inp, padding='max_length', truncation=True, max_length=1000, return_tensors="pt")
    X = inp["input_ids"].to(device)
    length = len(X[0])
    a = inp["attention_mask"].to(device)
    output = model.generate(X, attention_mask=a, max_length=1000, min_length=20, do_sample=False)

    output = tokenizer.decode(output[0])
    if not output.endswith('.'):
        last_period_index = output.rfind('.')
        if last_period_index != -1:
            output = output[:last_period_index + 1]
        else:

            output += '.'
    return output
#print(predict("John Drayton. Lionel Messi took matters into his own hands and delivered Argentinas team talk following a goalless 90 minutes against Holland. Manager Alejandro Sabella began the chat but Messi soon took over as his team-mates listened intently. And although his side couldn’t muster a win in extra time, they edged out Holland 4-2 on penalties to reach Sunday’s final in Rio de Janeiro. VIDEO Scroll down to watch Mascherano hailed the hero as Buenos Aires celebrates. Follow the leader: Lionel Messi (centre) led Argentinas team talk before extra time. Stepping up to the plate: Javier Mascherano led the team talk at half-time during extra time. Elation: Messi is overjoyed after Argentinas shoot-out win over Holland in the World Cup semi-final. Leading the celebrations: Messi (left) runs along with his team-mates to celebrate their win. As cool as you like: Messi slots home his penalty during the shoot-out as Argentina win. Leading from the front: Messi (second left) takes on water before extra time begins. Rest: Messi sits on water cooler box as he waits for extra time to begin in Sao Paulo. Captains duty: Lionel Messi led the Argentina team talk between normal time and extra time. Treble-team: Messi is surrounded by three Holland players during the World Cup semi-final. Powers that be: Argentina coach Alejandro Sabella (left) and Messi exchange words during the match. Aggressive: Messi was tracked throughout the match by Hollands determined midfielders. Competing: Messi (left) and Sneijder (right) challenge for the ball during the semi-final. VIDEO All Star XI: Lionel Messi - highlights. Having scored four goals in the tournament before the semi-final clash in Sao Paulo, Messi had a quiet game against the Dutch. With the score line still blank after 105 minutes, the captain seemed subdued as the Barcelona forward left the next team talk in the hands of Sabella. Indeed, it was his Barcelona team-mate Javier Mascherano who was the most animated player, appearing to put his hand up in his manager’s face. He again stepped up to speak to his fellow players before penalties. Holland and Aston Villa defender Ron Vlaar had kept the dangerous Argentina front line quiet as both teams battled for a place in the final against Germany. Messi was perhaps missing the creative influence of Angel di Maria who missed the game through injury. Heat map: Messi made his way around the park but had a relatively quiet evening. Follow me: Barcelonas Messi (second right) leads his players back on to the pitch for extra time. Two minds: Sabella (left) and Messi (right) talk tactics before extra time during semi-final."))
def preprocess(text):
    removals = ['(CNN) --', 'PUBLISHED:', 'UPDATED:', 'Daily Mail Reporter . ', 'Ellie Zolfagharifard . ', 'By . ', '  ', ' . .', ' | .', ' |,', '. | ', '..', '. ,', ', .', ', ,', '\n']
    patterns = [r'\b\xa0\b', '\xa0', "\\'" , r'\d{2} \b\w+\b \d{4}', r'\d{2}:\d{2} EST']
    combined_pattern = '|'.join(patterns)
    for rem in removals:
        text = text.replace(rem, "")
    new_text = re.sub(combined_pattern, '', text)
    sentences = re.split(r'[.!?]+', new_text)

    filtered_sentences = [sentence for sentence in sentences if "CLICK HERE" and "SCROLL DOWN" not in sentence.upper()]

    new_text = ".".join(filtered_sentences)
    return new_text



app = Flask(__name__)
CORS(app)
@app.route("/get_resp", methods=["POST"])
def create_user():
    try:
        # data1 = request.get_json(force=True)
        file_paths = request.json
        file_paths = file_paths.get('files', [])
        text_list = []
        for file_path in file_paths:
            file_content = file_path['content']
            text_list.append(file_content)
        text = ".".join(text_list)
        text = preprocess(text)
        text = predict(text)
        cleaned_text = re.sub('<.*?>', '', text)
        return jsonify({'extracted_text': cleaned_text}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="192.168.10.2", port=5000, debug=True)
