from transformers import StoppingCriteria, StoppingCriteriaList
import torch
from time import perf_counter
from spacy import displacy
from transformers import TokenClassificationPipeline, AutoTokenizer, AutoModelForTokenClassification
import torch.nn.functional as F
import json

class StopAtMultipleTokensCriteria(StoppingCriteria):
    def __init__(self, tokenizer, stop_sequences):
        # Encode each sequence to token IDs using the tokenizer
        self.stop_token_ids_list = [tokenizer.encode(seq, add_special_tokens=False) for seq in stop_sequences]

    def __call__(self, input_ids, scores):
        input_len = input_ids.shape[1]
        # Check each set of stop token IDs
        for stop_token_ids in self.stop_token_ids_list:
            seq_len = len(stop_token_ids)
            if input_len >= seq_len:
                # Compare the last generated tokens against the current stop token IDs set
                if all(input_ids[0, -seq_len + i] == stop_token_ids[i] for i in range(seq_len)):
                    return True
        return False


class CustomNERPipeline(TokenClassificationPipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def preprocess(self, inputs, **tokenizer_kwargs):
        # Tokenize the input text
        model_inputs = self.tokenizer(inputs, return_tensors="pt", truncation=True, padding=True)
        return model_inputs

    def _forward(self, model_inputs):
        # Forward pass through the model
        outputs = self.model(**model_inputs)
        # Include input_ids in the outputs to pass them to postprocess
        outputs["input_ids"] = model_inputs["input_ids"]
        return outputs

    def postprocess(self, model_outputs, **kwargs):
        # Extract logits and apply softmax to get probabilities
        logits = model_outputs.logits
        probs = F.softmax(logits, dim=-1)

        # Get the input_ids from model_outputs
        input_ids = model_outputs["input_ids"]

        # Convert input IDs to tokens
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])

        # Prepare the output with probabilities
        entities = []
        for token, prob in zip(tokens, probs[0]):
            entities.append({
                "token": token,
                "probabilities": prob.tolist()
            })

        return entities

def _generate_identity(prompt, model, tokenizer, gen_config):
    tokenized_prompt = tokenizer(prompt, return_tensors='pt')['input_ids'].to(model.device)
    stop_sequences = ['###', '</s>']
    stopping_criterion = StoppingCriteriaList([StopAtMultipleTokensCriteria(tokenizer, stop_sequences)])

    with torch.inference_mode():
        t0 = perf_counter()
        output = model.generate(input_ids=tokenized_prompt,
                                stopping_criteria=stopping_criterion,
                                generation_config=gen_config)
        total_time = perf_counter() - t0
        generation_ids = output[0][len(tokenized_prompt[0]):]
        num_gen_tokens = len(generation_ids)
        generation = tokenizer.decode(generation_ids, skip_special_tokens=True)
        return dict(generation=generation,
                    generation_ids=generation_ids.tolist(),
                    total_time=total_time,
                    num_gen_tokens=num_gen_tokens)


def ner_prediction_postprocess(model, tokenizer, model_outputs, input_ids, offset_mapping):

    entities = []
    for i in range(len(model_outputs.logits)):
        logits = model_outputs.logits[i]
        token_scores = logits.softmax(dim=-1)  # 计算每个 token 的 softmax 得分
        token_labels = token_scores.argmax(dim=-1)  # 选择得分最高的标签

        for token_index, token_label in enumerate(token_labels):
            # 跳过特殊 token（例如 [CLS], [SEP] 等）
            if input_ids[i][token_index] in [tokenizer.cls_token_id, tokenizer.sep_token_id]:
                continue

            # 获取原始文本中 token 的起始和结束位置
            start, end = offset_mapping[i][token_index].tolist()
            entity = {
                "word": tokenizer.convert_ids_to_tokens([input_ids[i][token_index]])[0],
                "entity": model.config.id2label[token_label.item()],
                "score": token_scores[token_index][token_label].item(),
                "start": start,
                "end": end
            }
            entities.append(entity)

    return entities




def ner_clean_ground_truth(tokens, ner_tags, id_to_label):
    text = ' '.join(tokens)
    entities = []
    current_entity = None
    current_label = None

    start_char = 0
    for idx, (token, tag_id) in enumerate(zip(tokens, ner_tags)):
        end_char = start_char + len(token)
        if id_to_label[tag_id].startswith('B-') or (id_to_label[tag_id].startswith('I-') and current_entity is None):
            if current_entity is not None:
                entities.append({
                    "start": current_entity['start'],
                    "end": current_entity['end'],
                    "entity_group": current_label
                })
            current_entity = {"start": start_char, "end": end_char}
            current_label = id_to_label[tag_id][2:]
        elif id_to_label[tag_id].startswith('I-') and current_label == id_to_label[tag_id][2:]:
            current_entity['end'] = end_char
        else:
            if current_entity is not None:
                entities.append({
                    "start": current_entity['start'],
                    "end": current_entity['end'],
                    "entity_group": current_label
                })
            current_entity = None
            current_label = None

        start_char = end_char + 1  # +1 for the space between tokens

    if current_entity is not None:
        entities.append({
            "start": current_entity['start'],
            "end": current_entity['end'],
            "entity_group": current_label
        })

    # Handle multi-line text
    for entity in entities:
        entity['start'], entity['end'] = adjust_for_multiline(text, entity['start'], entity['end'])

    return {"text": text, "ents": entities, "title": None}




def visualize_predictions_and_ground_truth(pred_doc, gt_doc, metadata="", options=None):
    pred_html = displacy.render(pred_doc, style="ent", page=True, options=options)
    gt_html = displacy.render(gt_doc, style="ent", page=True, options=options)
    combined_html = f"""
    <div style="display: flex; justify-content: space-around; position: relative;">
        <div style="width: 45%; border: 1px solid black;">
            <h3>Predicted</h3>
            {pred_html}
        </div>
        <div style="width: 45%; border: 1px solid black;">
            <h3>Ground Truth</h3>
            {gt_html}
        </div>
        <div style="position: absolute; top: 0; right: 0; background-color: #fff; padding: 5px; border: 1px solid black;">
            <strong>Metadata:</strong> {metadata}
        </div>
    </div>
    """
    return combined_html

def visualize_token_predictions(token_outputs):
    debug_output_json = json.dumps(token_outputs)
    # HTML template
    html_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>NER Output Visualization</title>
        <style>
            .token {{
                display: inline-block;
                padding: 2px 4px;
                margin: 2px;
                cursor: pointer;
                border-radius: 3px;
            }}
            
            .highlight {{
                background-color: yellow;
            }}

            .tooltip {{
                position: absolute;
                display: none;
                background-color: white;
                border: 1px solid #ccc;
                padding: 10px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
                z-index: 10;
            }}
        </style>
    </head>
    <body>
        <h2>NER Token-level Predictions</h2>
        <p id="paragraph">
            <!-- Tokens will be inserted here -->
        </p>
        <div class="tooltip" id="tooltip">
            <canvas id="probChart" width="400" height="200"></canvas>
        </div>

        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <script>
            const labelList = [
                "O",
                "B-identity",
                "I-identity",
                "B-relation",
                "I-relation",
                "B-action",
                "I-action",
            ];

            const debugOutput = {debug_output_json};
            let currentChart;
            let tooltipTimeout;
    
            // Function to insert tokens into the paragraph
            function insertTokens() {{
                const paragraph = document.getElementById('paragraph');
                debugOutput.forEach((item, index) => {{
                    const tokenSpan = document.createElement('span');
                    tokenSpan.classList.add('token');
                    tokenSpan.textContent = item.token;
                    tokenSpan.dataset.index = index;
                    paragraph.appendChild(tokenSpan);
                }});
            }}
    
            // Function to show tooltip with chart
            function showTooltip(event) {{
                clearTimeout(tooltipTimeout);
                tooltipTimeout = setTimeout(() => {{
                    const tooltip = document.getElementById('tooltip');
                    const index = event.target.dataset.index;
                    const probabilities = debugOutput[index].probabilities;
                    
                    // Highlight token
                    event.target.classList.add('highlight');
        
                    // Create chart data
                    const data = {{
                        labels: labelList,
                        datasets: [{{
                            label: 'Probabilities',
                            data: probabilities,
                            backgroundColor: 'rgba(75, 192, 192, 0.2)',
                            borderColor: 'rgba(75, 192, 192, 1)',
                            borderWidth: 1
                        }}]
                    }};
        
                    const config = {{
                        type: 'bar',
                        data: data,
                        options: {{
                            scales: {{
                                y: {{
                                    beginAtZero: true
                                }}
                            }}
                        }}
                    }};
        
                    // Clear existing chart if any
                    if (currentChart) {{
                        currentChart.destroy();
                    }}
        
                    // Show tooltip
                    tooltip.style.display = 'block';
                    tooltip.style.left = event.pageX + 'px';
                    tooltip.style.top = event.pageY + 'px';
        
                    // Draw chart
                    const canvas = document.getElementById('probChart');
                    currentChart = new Chart(canvas, config);
                }}, 100); // Add a small delay (100ms) before showing the tooltip
            }}
    
            // Function to hide tooltip
            function hideTooltip() {{
                clearTimeout(tooltipTimeout);
                const tooltip = document.getElementById('tooltip');
                tooltip.style.display = 'none';
                // Remove highlight from token
                event.target.classList.remove('highlight');
            }}
    
            // Insert tokens and add event listeners
            insertTokens();
            document.querySelectorAll('.token').forEach(token => {{
                token.addEventListener('mouseover', showTooltip);
                token.addEventListener('mouseout', hideTooltip);
            }});
        </script>
    </body>
    </html>
    """
    return html_template


def visualize_combined_predictions(token_outputs, pred_html, metadata="", options=None):

    debug_output_json = json.dumps(token_outputs)
    html_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>NER Output Visualization</title>
        <style>
            .token {{
                display: inline-block;
                padding: 2px 4px;
                margin: 2px;
                cursor: pointer;
                border-radius: 3px;
            }}

            .highlight {{
                background-color: yellow;
            }}

            .tooltip {{
                position: absolute;
                display: none;
                background-color: white;
                border: 1px solid #ccc;
                padding: 10px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
                z-index: 10;
            }}
        </style>
    </head>
    <body>
        <h2>NER Output with Probabilities</h2>
        <div style="display: flex; justify-content: space-around; position: relative;">
            <div style="width: 100%; border: 1px solid black;">
                <h3>Predicted</h3>
                {pred_html}
            </div>
            <div style="position: absolute; top: 0; right: 0; background-color: #fff; padding: 5px; border: 1px solid black;">
                <strong>Metadata:</strong> {metadata}
            </div>
        </div>
        <p id="paragraph">
            <!-- Tokens will be inserted here -->
        </p>
        <div class="tooltip" id="tooltip">
            <canvas id="probChart" width="400" height="200"></canvas>
        </div>

        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <script>
            const labelList = [
                "O",
                "B-identity",
                "I-identity",
                "B-relation",
                "I-relation",
                "B-action",
                "I-action",
            ];

            const debugOutput = {debug_output_json};
            let currentChart;
            let tooltipTimeout;

            // Function to insert tokens into the paragraph
            function insertTokens() {{
                const paragraph = document.getElementById('paragraph');
                debugOutput.forEach((item, index) => {{
                    const tokenSpan = document.createElement('span');
                    tokenSpan.classList.add('token');
                    tokenSpan.textContent = item.token;
                    tokenSpan.dataset.index = index;
                    paragraph.appendChild(tokenSpan);
                }});
            }}

            // Function to show tooltip with chart
            function showTooltip(event) {{
                clearTimeout(tooltipTimeout);
                tooltipTimeout = setTimeout(() => {{
                    const tooltip = document.getElementById('tooltip');
                    const index = event.target.dataset.index;
                    const probabilities = debugOutput[index].probabilities;

                    // Highlight token
                    event.target.classList.add('highlight');

                    // Create chart data
                    const data = {{
                        labels: labelList,
                        datasets: [{{
                            label: 'Probabilities',
                            data: probabilities,
                            backgroundColor: 'rgba(75, 192, 192, 0.2)',
                            borderColor: 'rgba(75, 192, 192, 1)',
                            borderWidth: 1
                        }}]
                    }};

                    const config = {{
                        type: 'bar',
                        data: data,
                        options: {{
                            scales: {{
                                y: {{
                                    beginAtZero: true
                                }}
                            }}
                        }}
                    }};

                    // Clear existing chart if any
                    if (currentChart) {{
                        currentChart.destroy();
                    }}

                    // Show tooltip
                    tooltip.style.display = 'block';
                    tooltip.style.left = event.pageX + 'px';
                    tooltip.style.top = event.pageY + 'px';

                    // Draw chart
                    const canvas = document.getElementById('probChart');
                    currentChart = new Chart(canvas, config);
                }}, 100); // Add a small delay (100ms) before showing the tooltip
            }}

            // Function to hide tooltip
            function hideTooltip() {{
                clearTimeout(tooltipTimeout);
                const tooltip = document.getElementById('tooltip');
                tooltip.style.display = 'none';
                // Remove highlight from token
                event.target.classList.remove('highlight');
            }}

            // Insert tokens and add event listeners
            insertTokens();
            document.querySelectorAll('.token').forEach(token => {{
                token.addEventListener('mouseover', showTooltip);
                token.addEventListener('mouseout', hideTooltip);
            }});
        </script>
    </body>
    </html>
    """
    return html_template