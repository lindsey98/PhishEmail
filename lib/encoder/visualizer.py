import os
from .IdentityBert import IdentityBert
from spacy import displacy
import spacy
from typing import Tuple, Set, Dict, List, Optional
import torch
from transformers import pipeline
from transformers.pipelines import Pipeline
import html  # For escaping HTML entities
import re
from typing import List
from spacy.tokens import Doc, Span
from typing import List, Tuple
from spacy.language import Language
from ..utilities import Timer, Logger
from ..utilities.data_utils import remove_urls
from transformers import DataCollatorForTokenClassification, AutoModelForTokenClassification, AutoTokenizer
import json
import torch.nn.functional as F

class Visualizer(IdentityBert):
    _CallerPrefix = "Visualizer"

    _entity_colors: Dict[str, str] = {
        "identity": "#DFF4E1",    # Lighter Medium Slate Blue for predicted organizations
        "action": "#F4D1C1",      # Lighter Indian Red for predicted actions
        "relation": "#DFF4E1",    # Lighter Lime Green for predicted relations
    }

    _ccs_style_sheet = """
        <style>
            .entity {{
                display: inline-block;
                position: relative;
                padding: 2px 4px;
                margin: 0 2px;
                line-height: 1.5;
                border-radius: 4px;
            }}
            .entity .label {{
                position: absolute;
                bottom: -1.5em;
                left: 0;
                background: white;
                color: black;
                font-size: 0.8em;
                padding: 1px 4px;
                border-radius: 2px;
                border: 1px solid #ccc;
                box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
                white-space: nowrap;
            }}

        </style>
    """

    def __init__(self, identity_checkpoint_path: str):
        super().__init__(identity_checkpoint_path)  # Initialize the superclass
        # Remove the existing classifier_pipeline if it exists
        if hasattr(self, 'classifier_pipeline'):
            del self.classifier_pipeline
        # Initialize a new NER pipeline without aggregation
        self.classifier_pipeline: Pipeline = pipeline(
            "ner",
            model=identity_checkpoint_path,
            device=self.device,  # Ensure 'self.device' is defined in IdentityBert
            aggregation_strategy="first"  # No aggregation
        )

    @classmethod
    def get_ccs_stylesheet(cls) -> str:
        return cls._ccs_style_sheet

    @classmethod
    def get_entity_colors(cls) -> Dict[str, str]:
        return cls._entity_colors

    @classmethod
    def set_entity_colors(cls, entity_colors: Dict[str, str]) -> None:
        cls._entity_colors = entity_colors

    @staticmethod
    def sanitize_metadata(metadata: str) -> str:
        if not len(metadata):
            return metadata
        escaped_metadata = html.escape(metadata)
        escaped_metadata = escaped_metadata.replace('\t', '&emsp;')  # Replaces tab with em space
        return escaped_metadata

    @staticmethod
    def ner_create_spacy_doc(raw_text: str, entities: List[Dict], nlp: Language) -> Doc:
        """
        Creates a spaCy Doc object with the specified entities.

        Args:
            raw_text (str): The original text.
            entities (List[Dict]): A list of entities with 'start', 'end', and 'entity_group'.
            nlp (Language): A spaCy language model.

        Returns:
            Doc: A spaCy Doc object with entities.
        """
        doc = nlp.make_doc(raw_text)
        ents: List[Span] = []
        for ent in entities:
            span = doc.char_span(ent['start'], ent['end'], label=ent['entity_group'])
            if span is not None:
                ents.append(span)
            else:
                # Optionally handle cases where char_span returns None
                print(f"Warning: Could not create span for entity {ent}")
        doc.ents = ents
        return doc

    @staticmethod
    def visualize_full_predictions(pred_html: str, metadata: str = "") -> str:
        # Generate the displacy HTML with page=False to get the snippet only
        sanitized_metadata = Visualizer.sanitize_metadata(metadata)

        rendered_html = f"""
            {Visualizer.get_ccs_stylesheet()}
            
            <h2>NER Full Predictions</h2>
            <div style="display: flex; flex-wrap: wrap; gap: 20px;">
                <div style="flex: 1 1 60%; min-width: 300px; border: 1px solid black; padding: 10px; box-sizing: border-box;">
                    {pred_html}
                </div>
                
                <div style="
                    flex: 1 1 30%; 
                    min-width: 200px; 
                    border: 1px solid black; 
                    padding: 10px; 
                    box-sizing: border-box; 
                    background-color: #f9f9f9; 
                    white-space: pre-wrap;  /* Preserves whitespace */
                    max-height: 400px;      /* Optional: set a max height */
                    overflow: auto;         /* Adds scrollbar if content exceeds max-height */
                ">
                    <strong>Metadata:</strong><br>
                    {sanitized_metadata}
                </div>
            </div>
        """
        return rendered_html

    @staticmethod
    def ner_clean_predictions(predictions: List[Dict]) -> List[Dict]:
        unique_entities = []
        seen = set()
        for ent in predictions:
            key = (ent['start'], ent['end'], ent['entity_group'])
            if key not in seen:
                seen.add(key)
                unique_entities.append({
                    "start": ent['start'],
                    "end": ent['end'],
                    "entity_group": ent['entity_group']
                })
        return unique_entities

    @torch.inference_mode()
    def __call__(self, raw_text: str, metadata: str = "") -> str:
        raw_text = remove_urls(raw_text)
        raw_text = self._tokenize(raw_text)
        raw_text = raw_text[0]
        with Timer():
            # Obtain NER predictions
            output = self.classifier_pipeline(raw_text)

        # Temporary lists to store entities with their confidence scores
        temp_identities: List[Tuple[str, float]] = []
        temp_relation: List[Tuple[str, float]] = []
        seen_identities = set()

        for ent in output:
            ent_label = ent['entity_group']
            ent_text = ent['word']
            ent_score = ent.get('score', 0.0)  # Get the confidence score
            if ent_label == 'identity':
                if ent_text not in seen_identities:
                    temp_identities.append((ent_text, ent_score))
                    seen_identities.add(ent_text)
            elif ent_label == 'relation':
                temp_relation.append(ent_text)

        temp_identities.sort(key=lambda x: x[1], reverse=True)

        # Logging: Convert entities with scores to strings for better readability
        identities_str = '\n'.join(
            [f"{entity}, confidence = {score:.2f}"
             for entity, score in temp_identities])
        relation_str = f"{temp_relation}"

        metadata += '\nReported identities: ' + '\n' + identities_str + '\nReported relation: ' + '\n' + relation_str

        # Clean the predictions
        cleaned_output = self.ner_clean_predictions(output)
        nlp = spacy.blank("en")
        pred_doc = self.ner_create_spacy_doc(raw_text, cleaned_output, nlp)
        pred_html = displacy.render(pred_doc, style="ent", page=False, options={"colors": self.get_entity_colors()})

        # Render the visualization using the current entity colors
        html = self.visualize_full_predictions(
            pred_html,
            metadata=metadata,
        )

        return html


class TokenPredVisualizer(Visualizer):
    _CallerPrefix = "TokenPredVisualizer"

    def __init__(self, identity_checkpoint_path: str):
        super().__init__(identity_checkpoint_path)  # Initialize the superclass
        self.model = AutoModelForTokenClassification.from_pretrained(identity_checkpoint_path)
        self.tokenizer = AutoTokenizer.from_pretrained(identity_checkpoint_path)

    @staticmethod
    def visualize_token_predictions(token_outputs: List[Dict], other_html: str, metadata: str = "",
                                    token_idx: Optional[int] = None):
        sanitized_metadata = Visualizer.sanitize_metadata(metadata)
        debug_output_json = json.dumps(token_outputs)

        # HTML template with added token_idx
        html_template = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>NER Output Visualization</title>
            <style>
                .entity {{
                    display: inline-block;
                    position: relative;
                    padding: 2px 4px;
                    margin: 0 2px;
                    line-height: 1.5;
                    border-radius: 4px;
                }}
                .entity .label {{
                    position: absolute;
                    bottom: -1.5em;
                    left: 0;
                    background: white;
                    color: black;
                    font-size: 0.8em;
                    padding: 1px 4px;
                    border-radius: 2px;
                    border: 1px solid #ccc;
                    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
                    white-space: nowrap;
                }}
                .tooltip {{
                    position: absolute;
                    display: none;
                    background-color: white;
                    border: 1px solid #ccc;
                    padding: 10px;
                    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                    z-index: 10;
                    max-width: 400px;
                    max-height: 300px;
                    overflow: hidden;
                }}
                .token {{
                    display: inline-block;
                    padding: 2px 4px;
                    margin: 2px 0;
                    cursor: pointer;
                    border-radius: 3px;
                }}
                .highlight {{
                    background-color: yellow;
                }}
            </style>
        </head>
        <body>
            <strong>Metadata:</strong><br>
            {sanitized_metadata}

            <h2>NER Token-level Predictions</h2>

            <p id="paragraph">
                <!-- Tokens will be inserted here -->
            </p>
            <div class="tooltip" id="tooltip">
                <canvas id="probChart" width="300" height="150"></canvas>
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

                const debugOutput = {debug_output_json};  // Replace with your JSON output
                let currentChart;
                let tooltipTimeout;

                // Variable to hold the token index to highlight and show tooltip
                const initialTokenIdx = {token_idx if token_idx is not None else 'null'};

                // Function to insert tokens as a paragraph
                function insertTokens() {{
                    const paragraph = document.getElementById('paragraph');
                    debugOutput.forEach((item, index) => {{
                        const tokenSpan = document.createElement('span');
                        tokenSpan.classList.add('token');
                        tokenSpan.textContent = item.token;
                        tokenSpan.dataset.index = index;
                        paragraph.appendChild(tokenSpan);

                        // Add a space after each token except the last
                        if (index < debugOutput.length - 1) {{
                            const spaceSpan = document.createElement('span');
                            spaceSpan.textContent = ' ';
                            paragraph.appendChild(spaceSpan);
                        }}
                    }});
                }}

                // Function to show tooltip with chart
                function showTooltip(event) {{
                    clearTimeout(tooltipTimeout);
                    tooltipTimeout = setTimeout(() => {{
                        const tooltip = document.getElementById('tooltip');
                        const index = event.target.dataset.index;
                        const probabilities = debugOutput[index].probabilities;

                        // Highlight the token
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
                                responsive: true,
                                maintainAspectRatio: true,
                                scales: {{
                                    y: {{
                                        beginAtZero: true
                                    }}
                                }},
                                plugins: {{
                                    legend: {{
                                        display: false
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
                        tooltip.style.left = event.pageX + 10 + 'px';  // Offset tooltip slightly
                        tooltip.style.top = event.pageY + 10 + 'px';

                        // Draw chart
                        const canvas = document.getElementById('probChart');
                        currentChart = new Chart(canvas, config);
                    }}, 100); // Add a small delay before showing the tooltip
                }}

                // Function to hide tooltip
                function hideTooltip(event) {{
                    clearTimeout(tooltipTimeout);
                    const tooltip = document.getElementById('tooltip');
                    tooltip.style.display = 'none';
                    // Remove highlight from token
                    event.target.classList.remove('highlight');
                }}

                // Insert tokens and add event listeners
                insertTokens();
                const tokens = document.querySelectorAll('.token');
                tokens.forEach(token => {{
                    token.addEventListener('mouseover', showTooltip);
                    token.addEventListener('mouseout', hideTooltip);
                }});

                // Function to show tooltip programmatically
                function showTooltipForIndex(index) {{
                    const token = document.querySelector(`.token[data-index='${{index}}']`);
                    if (token) {{
                        // Highlight the token
                        token.classList.add('highlight');

                        const tooltip = document.getElementById('tooltip');
                        const probabilities = debugOutput[index].probabilities;

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
                                responsive: true,
                                maintainAspectRatio: true,
                                scales: {{
                                    y: {{
                                        beginAtZero: true
                                    }}
                                }},
                                plugins: {{
                                    legend: {{
                                        display: false
                                    }}
                                }}
                            }}
                        }};

                        // Clear existing chart if any
                        if (currentChart) {{
                            currentChart.destroy();
                        }}

                        // Position the tooltip near the token
                        const rect = token.getBoundingClientRect();
                        tooltip.style.display = 'block';
                        tooltip.style.left = rect.right + window.scrollX + 10 + 'px';
                        tooltip.style.top = rect.top + window.scrollY + 'px';

                        // Draw chart
                        const canvas = document.getElementById('probChart');
                        currentChart = new Chart(canvas, config);
                    }}
                }}

                // If token_idx is provided, highlight and show tooltip for it by default
                if (initialTokenIdx !== null) {{
                    showTooltipForIndex(initialTokenIdx);
                }}
            </script>

            {other_html}
        </body>
        </html>
        """
        return html_template

    @torch.inference_mode()
    def get_token_outputs(self, text: str) -> List[Dict]:
        # Tokenize the input text
        tokens = self.tokenizer(text, return_tensors="pt", truncation=True, add_special_tokens=False).to(self.model.device)

        # Run the model to get logits
        with torch.no_grad():
            outputs = self.model(**tokens)

        # Extract logits and apply softmax to get probabilities
        logits = outputs.logits  # Shape: [batch_size, seq_length, num_labels]
        probabilities = F.softmax(logits, dim=-1)  # Convert logits to probabilities

        # Map token predictions back to tokens and labels
        token_ids = tokens["input_ids"][0]  # Input token IDs
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)  # Convert to token strings
        label_ids = torch.argmax(logits, dim=-1).squeeze()  # Predicted label IDs

        # Get the list of label names (e.g., "B-identity", "I-identity")
        label_names = self.model.config.id2label  # A mapping of label ID to label name

        # Process each token and map to its corresponding probabilities
        token_outputs = []
        for i, token in enumerate(tokens):
            token_outputs.append({
                "token": token,
                "probabilities": probabilities[0, i].cpu().numpy().tolist(),  # Probabilities for each label
                "label": label_names[label_ids[i].item()]  # Predicted label
            })

        return token_outputs

    @torch.inference_mode()
    def __call__(self, raw_text: str, metadata: str = "") -> str:
        raw_text = remove_urls(raw_text)
        raw_text = self._tokenize(raw_text)
        raw_text = raw_text[0]
        with Timer():
            # Obtain NER predictions
            token_outputs = self.get_token_outputs(raw_text)
            output = self.classifier_pipeline(raw_text)

        cleaned_output = self.ner_clean_predictions(output)
        nlp = spacy.blank("en")
        pred_doc = self.ner_create_spacy_doc(raw_text, cleaned_output, nlp)
        pred_html = displacy.render(pred_doc, style="ent", page=False, options={"colors": self.get_entity_colors()})

        html_template = self.visualize_token_predictions(token_outputs=token_outputs,
                                                         other_html=pred_html,
                                                         metadata=metadata)
        return html_template


class TracInVisualizer(TokenPredVisualizer):
    def __init__(self, identity_checkpoint_path: str):
        super().__init__(identity_checkpoint_path)  # Initialize the superclass

    @staticmethod
    def ner_create_spacy_doc_from_ground_truth(
        tokens: List[str],  # List of tokens
        tags: List[str],  # List of corresponding tags (e.g., BIO or BIOES tags)
        nlp: Language  # A spaCy language model
    ) -> Doc:

        # Ensure the lengths of tokens and tags match
        if len(tokens) != len(tags):
            raise ValueError("Length of tokens and tags must match.")

        raw_text = " ".join(tokens)  # Reconstruct raw text with spaces
        char_offsets = []
        current_pos = 0

        for token in tokens:
            start = raw_text.find(token, current_pos)  # Find the token's position in the text
            end = start + len(token)  # Calculate the end position
            char_offsets.append((start, end))
            current_pos = end + 1  # Update the search position for the next token

        # Create a SpaCy Doc object
        doc = nlp.make_doc(raw_text)
        ents: List[Span] = []

        # Variables to track spans
        start_idx = None
        current_label = None

        # Iterate over tokens and tags
        for i, (tag, (start, end)) in enumerate(zip(tags, char_offsets)):
            if tag.startswith("B-"):  # Beginning of a new entity
                if start_idx is not None:  # If there's an active entity, finish it
                    span = doc.char_span(start_idx[0], start_idx[1], label=current_label)
                    if span:  # Ensure valid span
                        ents.append(span)
                start_idx = (start, end)  # Start a new entity
                current_label = tag[2:]  # Extract the entity type
            elif tag.startswith("I-") and current_label == tag[2:]:  # Continuation of the same entity
                if start_idx is not None:
                    start_idx = (start_idx[0], end)  # Extend the current entity
            else:  # 'O' or unrelated tags (End of entity)
                if start_idx is not None:  # Finish the current entity
                    span = doc.char_span(start_idx[0], start_idx[1], label=current_label)
                    if span:  # Ensure valid span
                        ents.append(span)
                    start_idx = None
                    current_label = None

        # Add any remaining active entity at the end
        if start_idx is not None:
            span = doc.char_span(start_idx[0], start_idx[1], label=current_label)
            if span:
                ents.append(span)

        # Assign the entities to the doc
        doc.ents = ents

        return doc

    @torch.inference_mode()
    def __call__(self, raw_text: str, token_idx: Optional[int]=None, metadata: str = "", helpful_examples: List = [], harmful_examples: List = []) -> str:

        raw_text = remove_urls(raw_text)
        raw_text = self._tokenize(raw_text)
        raw_text = raw_text[0]
        with Timer():
            # Obtain NER predictions
            token_outputs = self.get_token_outputs(raw_text)
            output = self.classifier_pipeline(raw_text)

        spacy_options = {"colors": self.get_entity_colors()}
        other_html = ""

        # This example
        cleaned_output = self.ner_clean_predictions(output)
        nlp = spacy.blank("en")
        pred_doc = self.ner_create_spacy_doc(raw_text, cleaned_output, nlp)
        pred_html = displacy.render(pred_doc, style="ent", page=False, options=spacy_options)
        other_html += f"""<h2>NER Full Predictions for This Example</h2>
                        {pred_html}
                    """

        # helpful examples
        for it, eg in enumerate(helpful_examples):
            tokens, tags, score, metadata = eg
            nlp = spacy.blank("en")
            pred_doc = self.ner_create_spacy_doc_from_ground_truth(tokens, tags, nlp)
            pred_html = displacy.render(pred_doc, style="ent", page=False, options=spacy_options)
            other_html += f"""<h2>Top {it + 1} Helpful Example, TracIn Score = {score}, Path = {metadata}</h2>
                            {pred_html}
                        """

        # harmful examples
        for it, eg in enumerate(harmful_examples):
            tokens, tags, score, metadata = eg
            nlp = spacy.blank("en")
            pred_doc = self.ner_create_spacy_doc_from_ground_truth(tokens, tags, nlp)
            pred_html = displacy.render(pred_doc, style="ent", page=False, options=spacy_options)
            other_html += f"""<h2>Top {it + 1} Harmful Example, TracIn Score = {score}, Path = {metadata}</h2>
                            {pred_html}
                        """

        html_template = self.visualize_token_predictions(token_outputs=token_outputs,
                                                         token_idx=token_idx,
                                                         other_html=other_html,
                                                         metadata=metadata)
        return html_template